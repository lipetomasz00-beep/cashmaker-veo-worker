import os
import time
import logging
import json
import uuid
import hmac
import hashlib
import requests
import tempfile
import sys
import subprocess
import threading
import sqlite3
import urllib.parse
import socket
import shutil
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# KONFIGURACJA I INICJALIZACJA
# ---------------------------------------------------------------------------

STORAGE_DIR = os.getenv('STORAGE_DIR', '/app/data')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def run_ffmpeg(cmd, timeout=120):
    """Run FFmpeg command with proper error logging."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")
        if e.stderr:
            logger.error(e.stderr.decode(errors='ignore'))
        raise


# Initialize HuggingFace Inference Client for Wan2.2
HF_CLIENT = None

def init_hf_client():
    """Initialize HuggingFace Inference Client for Wan2.2 video generation."""
    global HF_CLIENT
    try:
        from huggingface_hub import InferenceClient
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")
        HF_CLIENT = InferenceClient(
            provider="fal-ai",
            api_key=hf_token,
        )
        logger.info("✅ HuggingFace Inference Client initialized for Wan2.2")
    except Exception as e:
        logger.error(f"❌ Failed to initialize HF_CLIENT: {e}")
        raise

app = Flask(__name__)

DB_PATH = os.path.join(STORAGE_DIR, 'renders.db')
os.makedirs(STORAGE_DIR, exist_ok=True)
MAX_CONCURRENT_RENDERS = int(os.getenv("MAX_CONCURRENT_RENDERS", "2"))
RENDER_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_RENDERS)
WORKER_API_KEY = os.getenv("WORKER_API_KEY")
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "16384"))
MAX_NARRATION_CHARS = int(os.getenv("MAX_NARRATION_CHARS", "800"))
MAX_OUTPUT_VIDEO_MB = int(os.getenv("MAX_OUTPUT_VIDEO_MB", "500"))
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ENABLE_AUTOMATION_RULES = os.getenv("ENABLE_AUTOMATION_RULES", "true").lower() == "true"
MAX_HASHTAGS = int(os.getenv("MAX_HASHTAGS", "8"))
ENABLE_DRY_RUN = os.getenv("ENABLE_DRY_RUN", "false").lower() == "true"
FREE_TIER_MODE = os.getenv("FREE_TIER_MODE", "true").lower() == "true"
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "8"))
IDEMPOTENCY_TTL_SECONDS = int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "3600"))
SKIP_NARRATION = os.getenv("SKIP_NARRATION", "false").lower() == "true"
VEO_DURATION_SECONDS = int(os.getenv("VEO_DURATION_SECONDS", "6"))  # Per-scene GPU duration (6s × 3 scenes = 18s total)

# NAVA text-to-video configuration
NAVA_ENABLED = os.getenv("NAVA_ENABLED", "false").lower() == "true"
NAVA_SPACE_ID = os.getenv("NAVA_SPACE_ID", "prithivMLmods/NAVA-Text-to-Video")

# Auto-retry configuration for paused jobs
AUTO_RETRY_ENABLED = os.getenv("AUTO_RETRY_ENABLED", "false").lower() == "true"
AUTO_RETRY_MAX_ATTEMPTS = int(os.getenv("AUTO_RETRY_MAX_ATTEMPTS", "3"))
AUTO_RETRY_INITIAL_DELAY_SECONDS = int(os.getenv("AUTO_RETRY_INITIAL_DELAY_SECONDS", "30"))
AUTO_RETRY_MAX_DELAY_SECONDS = int(os.getenv("AUTO_RETRY_MAX_DELAY_SECONDS", "600"))  # 10 minutes

# Hard execution timeout configuration for background jobs
MAX_JOB_DURATION_SECONDS = int(os.getenv("MAX_JOB_DURATION_SECONDS", "1200"))  # 10 minutes
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
DISABLE_CLEANUP = os.getenv("DISABLE_CLEANUP", "false").lower() == "true"

RATE_LIMIT_WINDOW = {}
RATE_LIMIT_LOCK = threading.Lock()
IDEMPOTENCY_CACHE = {}
IDEMPOTENCY_LOCK = threading.Lock()

# API response cache to minimize costs
API_CACHE = {}
API_CACHE_TTL_SECONDS = 3600  # 1 hour

def cache_api_response(cache_key, response_data):
    """Cache an API response with TTL."""
    API_CACHE[cache_key] = {
        "data": response_data,
        "cached_at": datetime.utcnow(),
        "ttl": API_CACHE_TTL_SECONDS
    }
    logger.info(f"💾 Cached API response: {cache_key}")

def get_cached_api_response(cache_key):
    """Get cached API response if still valid."""
    if cache_key not in API_CACHE:
        return None

    cached = API_CACHE[cache_key]
    age = (datetime.utcnow() - cached["cached_at"]).total_seconds()

    if age > cached["ttl"]:
        del API_CACHE[cache_key]
        return None

    logger.info(f"✅ Using cached API response: {cache_key} (age: {age:.0f}s)")
    return cached["data"]

METRICS = {
    "jobs_started": 0,
    "jobs_success": 0,
    "jobs_failed": 0,
    "webhook_success": 0,
    "webhook_failed": 0,
    "last_error": None
}

def is_valid_public_url(url):
    """Sprawdź czy URL jest publiczny i bezpieczny (ochrona przed SSRF)."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False

        # Podczas testów automatycznych ignorujemy fizyczną rezolucję DNS (może nie być dostępna w piaskownicy)
        if IS_TESTING:
            if host in ("localhost", "127.0.0.1", "169.254.169.254"):
                return False
            if host.startswith("10.") or host.startswith("192.168."):
                return False
            if host.startswith("172."):
                parts = host.split('.')
                if len(parts) >= 2 and parts[0] == "172":
                    try:
                        second = int(parts[1])
                        if 16 <= second <= 31:
                            return False
                    except ValueError:
                        pass
            return True

        # Sprawdź czy host nie jest adresem IP i czy nie wskazuje na localhost/prywatną podsieć
        try:
            ip = socket.gethostbyname(host)
        except socket.gaierror:
            return False

        # Wykluczenie adresów lokalnych i prywatnych
        parts = list(map(int, ip.split('.')))
        if len(parts) != 4:
            return False
        # 127.0.0.1
        if parts[0] == 127:
            return False
        # Klasa A prywatna: 10.0.0.0/8
        if parts[0] == 10:
            return False
        # Klasa B prywatna: 172.16.0.0/12
        if parts[0] == 172 and (16 <= parts[1] <= 31):
            return False
        # Klasa C prywatna: 192.168.0.0/16
        if parts[0] == 192 and parts[1] == 168:
            return False
        # Link-local / metadata (169.254.x.x)
        if parts[0] == 169 and parts[1] == 254:
            return False
        # Multicast/Broadcast/Unspecified
        if parts[0] >= 224:
            return False

        return True
    except Exception:
        return False
        
def validate_required_env():
    """Walidacja wymaganych zmiennych środowiskowych, kluczy API i binariów systemowych."""
    # 1. Walidacja binariów ffmpeg/ffprobe
    if not shutil.which("ffmpeg"):
        raise RuntimeError("System dependency 'ffmpeg' is missing from PATH. Install it first.")
    if not shutil.which("ffprobe"):
        raise RuntimeError("System dependency 'ffprobe' is missing from PATH. Install it first.")

    # 2. Walidacja obecności wymaganych zmiennych
    required = ["HF_TOKEN", "OPENAI_API_KEY", "WORKER_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )

    # 3. Twarda walidacja kluczy przy starcie (tylko gdy nie jest to DRY_RUN / TESTING)
    if not ENABLE_DRY_RUN and not IS_TESTING:
        logger.info("🔒 Rozpoczynam twardą walidację kluczy API...")
        
        # Walidacja Hugging Face
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN not set")
            logger.info("✅ Klucz HF_TOKEN zweryfikowany pomyślnie.")
        except Exception as e:
            raise RuntimeError(f"HF_TOKEN validation failed: {e}")
            
        # Walidacja OpenAI
        try:
            from openai import OpenAI
            oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            oa_client.models.list()
            logger.info("✅ Klucz OPENAI_API_KEY zweryfikowany pomyślnie.")
        except Exception as e:
            raise RuntimeError(f"OPENAI_API_KEY validation failed: {e}")

    else:
        logger.info("🧪 DRY_RUN lub TESTING włączony - pomijam twardą walidację kluczy API.")


    # Initialize HuggingFace Inference Client for Wan2.2
    if not ENABLE_DRY_RUN and not IS_TESTING:
        try:
            init_hf_client()
        except Exception as e:
            logger.error(f"Failed to initialize HF_CLIENT: {e}")
            raise

def require_api_key():
    """Wymagaj poprawnego API key w nagłówku Authorization lub X-API-Key."""
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.headers.get("X-API-Key", "").strip()

    if not token or not WORKER_API_KEY or token != WORKER_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    return None

def _is_retryable_exception(exc):
    """Retry tylko dla błędów tymczasowych (timeout/429/5xx)."""
    text = str(exc).lower()
    non_retryable_markers = [
        "400", "invalid_argument", "401", "403", "404", "422",
        "narration must", "missing or empty", "payload too large"
    ]
    if any(m in text for m in non_retryable_markers):
        return False
    retryable_markers = ["429", "timeout", "timed out", "connection", "503", "502", "500", "rate limit"]
    return any(m in text for m in retryable_markers)

def enforce_rate_limit(api_key):
    now = time.time()
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_WINDOW.get(api_key, [])
        cutoff = now - 3600
        bucket = [t for t in bucket if t >= cutoff]
        if len(bucket) >= RATE_LIMIT_PER_HOUR:
            RATE_LIMIT_WINDOW[api_key] = bucket
            return False, int(max(1, 3600 - (now - min(bucket))))
        bucket.append(now)
        RATE_LIMIT_WINDOW[api_key] = bucket
    return True, None

def get_idempotency_response(idempotency_key):
    now = time.time()
    with IDEMPOTENCY_LOCK:
        rec = IDEMPOTENCY_CACHE.get(idempotency_key)
        if not rec:
            return None
        if now - rec["created_at"] > IDEMPOTENCY_TTL_SECONDS:
            IDEMPOTENCY_CACHE.pop(idempotency_key, None)
            return None
        return rec

def remember_idempotency(idempotency_key, response_obj):
    with IDEMPOTENCY_LOCK:
        IDEMPOTENCY_CACHE[idempotency_key] = {
            "created_at": time.time(),
            **response_obj,
        }

def retry_with_backoff(operation_name, func, max_retries=3, base_delay=2):
    """Retry helper z exponential backoff dla wywołań zewnętrznych API.

    429 RESOURCE_EXHAUSTED errors receive a longer initial wait (60 s) so that
    the Gemini RPM window has time to refill before the next attempt.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            retryable = _is_retryable_exception(exc)
            is_quota_error = "429" in str(exc) or "resource_exhausted" in str(exc).lower()

            if is_quota_error:
                # Quota errors need a much longer back-off than generic retries.
                # Start at 60 s and double on each subsequent attempt.
                wait_s = 60 * (2 ** attempt)
                logger.warning(
                    f"🚦 {operation_name} hit Gemini quota limit (429) on attempt "
                    f"{attempt+1}/{max_retries}. Waiting {wait_s}s before retry..."
                )
            else:
                wait_s = base_delay ** attempt

            if attempt < max_retries - 1 and retryable:
                if not is_quota_error:
                    logger.warning(
                        f"⚠️ {operation_name} failed (attempt {attempt+1}/{max_retries}): {exc}. "
                        f"Retry in {wait_s}s..."
                    )
                time.sleep(wait_s)
            else:
                if not retryable:
                    logger.error(f"❌ {operation_name} non-retryable error: {exc}")
                else:
                    logger.error(f"❌ {operation_name} failed after {max_retries} attempts: {exc}")
                raise

def validate_request_limits(data):
    """Twarde limity rozmiaru requestu i pól wejściowych."""
    content_length = request.content_length or 0
    if content_length > MAX_REQUEST_BYTES:
        return jsonify({
            "error": "Payload too large",
            "max_request_bytes": MAX_REQUEST_BYTES
        }), 413

    topic = (data.get("topic") or "").strip()
    if len(topic) > 200:
        return jsonify({"error": "Topic too long", "max_topic_chars": 200}), 413

    narration = data.get("narration")
    if narration is not None:
        if not isinstance(narration, dict):
            return jsonify({"error": "Narration must be an object"}), 400
        total_chars = sum(len(str(v)) for v in narration.values())
        if total_chars > MAX_NARRATION_CHARS:
            return jsonify({
                "error": "Narration too large",
                "max_narration_chars": MAX_NARRATION_CHARS
            }), 413

    hashtags = data.get("hashtags")
    if hashtags is not None:
        if not isinstance(hashtags, list):
            return jsonify({"error": "Hashtags must be an array"}), 400
        if len(hashtags) > MAX_HASHTAGS:
            return jsonify({"error": "Too many hashtags", "max_hashtags": MAX_HASHTAGS}), 413
        for tag in hashtags:
            if not isinstance(tag, str):
                return jsonify({"error": "Each hashtag must be a string"}), 400
            if not tag.startswith("#"):
                return jsonify({"error": "Each hashtag must start with #"}), 400
            if " " in tag:
                return jsonify({"error": "Hashtags cannot contain spaces"}), 400

    webhook_url = data.get("webhookUrl")
    if webhook_url:
        if not is_valid_public_url(webhook_url):
            return jsonify({"error": "Invalid or unsafe 'webhookUrl' (SSRF protection)"}), 400

    return None

def build_hashtags(topic, narration_texts=None, max_count=8):
    """
    Generuje zestaw hashtagów:
    - najpierw podstawowe finansowe i brandowe,
    - potem słowa z topic.
    """
    base_tags = [
        "#finanse",
        "#oszczedzanie",
        "#budzetdomowy",
        "#kontoosobiste",
        "#porownanieofert",
        "#raportfinansowy24",
    ]
    topic_words = []
    for token in topic.lower().replace(",", " ").replace(".", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if len(cleaned) >= 4:
            topic_words.append(f"#{cleaned}")

    tags = []
    for tag in base_tags + topic_words:
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= max_count:
            break
    return tags

def send_webhook(webhook_url, payload):
    """
    Wysyłka webhooka z opcjonalnym podpisem HMAC.
    Podpis: X-Webhook-Signature: sha256=<hex_digest>
    """
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if WEBHOOK_SECRET:
        signature = hmac.new(
            WEBHOOK_SECRET.encode("utf-8"),
            body,
            hashlib.sha256
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    return requests.post(webhook_url, data=body, headers=headers, timeout=10)

def apply_optimization_rules(raw_data, topic):
    """
    Pętla feedbacku:
    - Jeśli CTR/VTR są słabe, dostosuj CTA, tempo i narrację do strategii konkretu i ekskluzywności.
    Wejście:
        raw_data["performance"] = {"ctr": 0.01, "vtr": 0.12}
    """
    if not ENABLE_AUTOMATION_RULES:
        return raw_data

    performance = raw_data.get("performance") or {}
    ctr = float(performance.get("ctr", 0) or 0)
    vtr = float(performance.get("vtr", 0) or 0)
    narration = raw_data.get("narration") or {}
    optimizations = []

    # Rule 1: słaby CTR => Dostarczenie twardych faktów i prestiżowego wezwania (zamiast taniej agresji)
    if ctr and ctr < 0.012:
        narration["hook"] = f"Analiza rynku ujawnia nieoczywiste koszty w obszarze {topic}. Zobacz niezależne zestawienie faktów."
        raw_data["ctaText"] = "Pobierz bezpłatny raport i porównaj warunki: raport-finansowy24.pl"
        optimizations.append("low_ctr_exclusive_factual_boost")

    # Rule 2: słaby VTR => krótsza/jaśniejsza narracja + szybsze tempo
    if vtr and vtr < 0.20:
        narration["problem"] = "Najczęstszy błąd rynkowy to wybór oferty bez dokładnej weryfikacji parametrów."
        narration["rozwiązanie"] = "Dostęp do rzetelnych danych pozwala podjąć decyzję w oparciu o czyste liczby, a nie obietnice."
        raw_data["targetDuration"] = 18
        optimizations.append("low_vtr_shorter_story")

    if narration:
        raw_data["narration"] = narration
    if optimizations:
        raw_data["optimizations_applied"] = optimizations

    return raw_data

# Inicjalizacja bazy danych SQLite
def init_db():
    """Inicjalizacja tabeli historii renderów"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS renders (\n        job_id TEXT PRIMARY KEY,\n        topic TEXT,\n        status TEXT,\n        video_url TEXT,\n        error TEXT,\n        video_duration REAL,\n        created_at TIMESTAMP,\n        completed_at TIMESTAMP,\n        current_stage TEXT,\n        checkpoint_data TEXT,\n        paused_at TIMESTAMP,\n        paused_reason TEXT,\n        retry_count INTEGER DEFAULT 0,\n        next_retry_at TIMESTAMP\n    )''')
    # Migrate existing databases that are missing the checkpoint columns
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(renders)")}
    for col, col_def in [
        ("current_stage",   "TEXT"),
        ("checkpoint_data", "TEXT"),
        ("paused_at",       "TIMESTAMP"),
        ("paused_reason",   "TEXT"),
    ]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE renders ADD COLUMN {col} {col_def}")
    conn.commit()
    conn.close()

def ensure_retry_columns():
    """Add retry_count and next_retry_at columns if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if columns exist
    c.execute("PRAGMA table_info(renders)")
    columns = {row[1] for row in c.fetchall()}

    if "retry_count" not in columns:
        c.execute("ALTER TABLE renders ADD COLUMN retry_count INTEGER DEFAULT 0")
        logger.info("✅ Added retry_count column to renders table")

    if "next_retry_at" not in columns:
        c.execute("ALTER TABLE renders ADD COLUMN next_retry_at TIMESTAMP")
        logger.info("✅ Added next_retry_at column to renders table")

    conn.commit()
    conn.close()

# Initialize Database
init_db()
ensure_retry_columns()

# ---------------------------------------------------------------------------
# HISTORIA RENDERÓW (SQLite)
# ---------------------------------------------------------------------------

def save_render_to_db(job_id, topic, status, video_url=None, error=None, video_duration=None):
    """Zapis renderowania do bazy danych"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    completed_at = datetime.utcnow() if status in ['success', 'failed'] else None
    c.execute('''INSERT OR REPLACE INTO renders
                 (job_id, topic, status, video_url, error, video_duration, created_at, completed_at,
                  current_stage, checkpoint_data, paused_at, paused_reason)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                  COALESCE((SELECT current_stage  FROM renders WHERE job_id = ?), NULL),
                  COALESCE((SELECT checkpoint_data FROM renders WHERE job_id = ?), NULL),
                  COALESCE((SELECT paused_at       FROM renders WHERE job_id = ?), NULL),
                  COALESCE((SELECT paused_reason   FROM renders WHERE job_id = ?), NULL))''',
              (job_id, topic, status, video_url, error, video_duration, datetime.utcnow(), completed_at,
               job_id, job_id, job_id, job_id))
    conn.commit()
    conn.close()

def save_checkpoint(job_id, stage, data=None, error=None):
    """Save a checkpoint so the job can be resumed from this stage on error."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    checkpoint_json = json.dumps(data or {})

    if error is not None:
        # Get current retry count
        c.execute("SELECT retry_count FROM renders WHERE job_id = ?", (job_id,))
        row = c.fetchone()
        current_retry_count = (row[0] if row and row[0] is not None else 0) + 1

        # Calculate next retry time with exponential backoff
        if AUTO_RETRY_ENABLED and current_retry_count <= AUTO_RETRY_MAX_ATTEMPTS:
            delay = min(
                AUTO_RETRY_INITIAL_DELAY_SECONDS * (2 ** (current_retry_count - 1)),
                AUTO_RETRY_MAX_DELAY_SECONDS
            )
            next_retry_time = datetime.utcnow() + timedelta(seconds=delay)
            logger.info(f"⏰ Job {job_id} will auto-retry in {delay}s (attempt {current_retry_count}/{AUTO_RETRY_MAX_ATTEMPTS})")
        else:
            next_retry_time = None
            logger.warning(f"❌ Job {job_id} exceeded max retry attempts ({AUTO_RETRY_MAX_ATTEMPTS})")

        # Job hit an error – pause it
        c.execute('''UPDATE renders
                     SET current_stage   = ?,
                         checkpoint_data = ?,
                         paused_at       = ?,
                         paused_reason   = ?,
                         status          = 'paused',
                         retry_count     = ?,
                         next_retry_at   = ?
                     WHERE job_id = ?''',
                  (stage, checkpoint_json, datetime.utcnow(), error, current_retry_count, next_retry_time, job_id))
    else:
        # Successful stage – save progress
        c.execute('''UPDATE renders
                     SET current_stage   = ?,
                         checkpoint_data = ?,
                         retry_count     = 0,
                         next_retry_at   = NULL
                     WHERE job_id = ?''',
                  (stage, checkpoint_json, job_id))

    conn.commit()
    conn.close()
    logger.info(f"💾 Checkpoint saved: job={job_id} stage={stage}")

def get_checkpoint(job_id):
    """Return checkpoint info for a paused job, or None if not found / not paused."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT current_stage, checkpoint_data, paused_at, paused_reason, status, topic,
                        retry_count, next_retry_at
                 FROM renders WHERE job_id = ?''', (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "current_stage":   row[0],
        "checkpoint_data": json.loads(row[1]) if row[1] else {},
        "paused_at":       row[2],
        "paused_reason":   row[3],
        "status":          row[4],
        "topic":           row[5],
        "retry_count":     row[6] if row[6] is not None else 0,
        "next_retry_at":   row[7],
    }

def get_render_from_db(job_id):
    """Pobranie statusu renderowania z bazy"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT job_id, topic, status, video_url, error, video_duration,
                        created_at, completed_at, current_stage, checkpoint_data,
                        paused_at, paused_reason, retry_count, next_retry_at
                 FROM renders WHERE job_id = ?''', (job_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {
            "job_id":          row[0],
            "topic":           row[1],
            "status":          row[2],
            "video_url":       row[3],
            "error":           row[4],
            "video_duration":  row[5],
            "created_at":      row[6],
            "completed_at":    row[7],
            "current_stage":   row[8],
            "checkpoint_data": json.loads(row[9]) if row[9] else {},
            "paused_at":       row[10],
            "paused_reason":   row[11],
            "retry_count":     row[12] if row[12] is not None else 0,
            "next_retry_at":   row[13],
        }
    return None

# ---------------------------------------------------------------------------
# GENEROWANIE SEGMENTÓW WIDEO (Hugging Face HunyuanVideo)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GENEROWANIE WIDEO: Wan2.2 (FAL AI via HuggingFace Inference)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PROMPT OPTIMIZATION (3 warianty → scoring → best pick)
# ---------------------------------------------------------------------------

def generate_prompt_variants(topic: str, narration: dict) -> list:
    """Generate 3 prompt variants for video generation."""
    if not HF_CLIENT:
        raise RuntimeError("HF_CLIENT not initialized")
    
    hook = narration.get("hook", "")
    problem = narration.get("problem", "")
    solution = narration.get("rozwiązanie", "")
    
    logger.info(f"🎨 Generating 3 prompt variants for topic: {topic}")
    
    # Variant 1: Cinematic + emotional
    variant1 = f"Cinematic shot, dramatic lighting, emotional expression. {hook[:50]}. Problem: {problem[:50]}. Solution: {solution[:50]}"
    
    # Variant 2: Dynamic + energetic
    variant2 = f"Dynamic fast-paced scene, vibrant colors, energetic movement. {hook[:50]}. {problem[:50]}. {solution[:50]}"
    
    # Variant 3: Professional + clear
    variant3 = f"Professional presentation, clear visuals, informative. Topic: {topic}. Key message: {hook[:50]}. Details: {problem[:50]}"
    
    variants = [variant1, variant2, variant3]
    
    for i, v in enumerate(variants, 1):
        logger.info(f"  Variant {i}: {v[:60]}...")
    
    return variants


def score_prompt(prompt: str) -> float:
    """Score a prompt based on quality metrics."""
    score = 0.0
    
    # Length score (optimal 100-200 chars)
    length = len(prompt)
    if 100 <= length <= 200:
        score += 30
    elif 50 <= length <= 250:
        score += 20
    else:
        score += 10
    
    # Keyword diversity
    keywords = ["cinematic", "dynamic", "professional", "visual", "scene", "shot", "color", "movement"]
    keyword_count = sum(1 for kw in keywords if kw.lower() in prompt.lower())
    score += min(keyword_count * 5, 30)
    
    # Descriptive words
    descriptive = ["dramatic", "vibrant", "energetic", "clear", "emotional", "fast-paced", "professional"]
    desc_count = sum(1 for d in descriptive if d.lower() in prompt.lower())
    score += min(desc_count * 5, 25)
    
    # Avoid repetition
    words = prompt.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    score += unique_ratio * 15
    
    return min(score, 100.0)


def select_best_prompt(variants: list) -> tuple:
    """Select best prompt from variants based on scoring."""
    logger.info("🎯 Scoring prompt variants...")
    
    scores = []
    for i, variant in enumerate(variants, 1):
        score = score_prompt(variant)
        scores.append((i, variant, score))
        logger.info(f"  Variant {i}: score={score:.1f}")
    
    # Select best
    best_idx, best_prompt, best_score = max(scores, key=lambda x: x[2])
    logger.info(f"✅ Selected variant {best_idx} (score={best_score:.1f})")
    
    return best_prompt, best_idx, best_score



def generate_wan_video(prompt: str, output_path: str):
    """Generate video via Wan2.2 ZeroGPU Space (free) and save to output_path."""
    logger.info(f"🎬 Wan2.2 (ZeroGPU): generating video for prompt: {prompt[:60]}...")
    
    def _call_wan_space():
        start_time = time.time()
        try:
            from gradio_client import Client
            logger.info("⏳ Calling Wan2.2 ZeroGPU Space...")
            
            # Use Gradio Client to call Wan2.2 ZeroGPU Space (free)
            client = Client("zerogpu-aoti/wan2-2-fp8da-aoti-faster")
            result = client.predict(
                prompt=prompt,
                api_name="/generate"
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Wan2.2 generated video in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Wan2.2 Space error after {elapsed:.2f}s: {e}")
            raise
    
    result = retry_with_backoff("Wan2.2 ZeroGPU", _call_wan_space, max_retries=3, base_delay=30)
    
    if result is None:
        raise RuntimeError("Wan2.2 returned no result (None)")
    
    # Result is a file path from the Space
    try:
        logger.info(f"💾 Downloading video from Space...")
        if isinstance(result, str) and result.startswith("http"):
            # Download from URL
            import urllib.request
            urllib.request.urlretrieve(result, output_path)
        elif isinstance(result, str) and os.path.exists(result):
            # Local file path
            shutil.copy(result, output_path)
        else:
            raise RuntimeError(f"Invalid response from Wan2.2: expected file path or URL, got {type(result)}")
        logger.info(f"✅ Video saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save video: {e}")
        raise
    
    return output_path


def generate_hunyuan_video_segment(prompt, output_path, aspect_ratio="9:16"):
    """Generate a video segment via LTX 2.3 Gradio Space and save it directly to output_path."""

    logger.info(f"🎬 LTX 2.3 (Gradio Space): generating video for prompt: {prompt[:60]}...")
    logger.info("⏳ Calling LTX 2.3 Gradio Space /generate_video endpoint...")

    def _make_blank_image_file():
        """Create a small blank PNG in a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        img.save(tmp.name, format="PNG")
        tmp.close()
        return tmp.name

    def _call_api():
        try:
            from gradio_client import Client, handle_file
            from huggingface_hub import login
            import os

            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is not set")

            # Authenticate via huggingface_hub before creating the Client
            # (gradio_client no longer accepts hf_token in the constructor)
            login(token=hf_token)

            # Create client without auth parameter — login() above handles it
            client = Client(
                "https://r3gm-wan2-2-fp8da-aoti-preview-2.hf.space/"
            )

            # Create placeholder blank images for the image parameters
            # (this is text-to-video; the Space still requires FileData objects)
            blank_image_path = _make_blank_image_file()
            try:
                # Call /generate_video endpoint with all 16 required parameters:
                # [0]  Input Image (File)
                # [1]  Last Image Optional (File)
                # [2]  Prompt (string)
                # [3]  Inference Steps (number)
                # [4]  Negative Prompt (string)
                # [5]  Duration in seconds (number)
                # [6]  Guidance Scale - high noise (number)
                # [7]  Guidance Scale 2 - low noise (number)
                # [8]  Seed (number)
                # [9]  Randomize seed (boolean)
                # [10] Video Quality (number)
                # [11] Scheduler (string)
                # [12] Flow Shift (number)
                # [13] Video Fluidity/FPS (number)
                # [14] Safe Mode (boolean)
                # [15] Display result (boolean)
                result = client.predict(
                    handle_file(blank_image_path),   # [0] Input Image
                    handle_file(blank_image_path),   # [1] Last Image Optional
                    prompt,                          # [2] Prompt
                    6,                               # [3] Inference Steps
                    "blurry, low quality, distorted, watermark",  # [4] Negative Prompt
                    VEO_DURATION_SECONDS,            # [5] Duration in seconds
                    3.5,                             # [6] Guidance Scale - high noise
                    1,                               # [7] Guidance Scale 2 - low noise
                    42,                              # [8] Seed
                    True,                            # [9] Randomize seed
                    6,                               # [10] Video Quality
                    "UniPCMultistep",                # [11] Scheduler
                    3,                               # [12] Flow Shift
                    16,                              # [13] Video Fluidity/FPS
                    True,                            # [14] Safe Mode
                    True,                            # [15] Display result
                    api_name="/generate_video"
                )
            finally:
                # Clean up the temporary blank image
                try:
                    os.unlink(blank_image_path)
                except OSError:
                    pass

            return result
        except Exception as e:
            if "overloaded" in str(e).lower() or "busy" in str(e).lower() or "503" in str(e):
                raise RuntimeError("Space overloaded – retry")
            raise

    result = retry_with_backoff("LTX 2.3 Gradio", _call_api, max_retries=3, base_delay=30)

    # The /generate_video endpoint returns 3 elements:
    #   [0] Generated Video (displayed in UI)
    #   [1] Download Video File (the actual downloadable file path/URL)
    #   [2] Seed (number used for generation)
    # We prefer [1] (download file) and fall back to [0] (display video).
    video_path = None
    if isinstance(result, (list, tuple)) and len(result) >= 1:
        # Try [1] first (download file), then [0] (display video)
        for idx in (1, 0):
            if idx < len(result):
                candidate = result[idx]
                if isinstance(candidate, dict):
                    video_path = candidate.get("video") or candidate.get("path") or candidate.get("url")
                elif isinstance(candidate, str) and candidate:
                    video_path = candidate
                if video_path:
                    break

    if not video_path:
        raise RuntimeError(f"No video path found in LTX 2.3 /generate_video response: {result}")

    # Download video from Gradio temp URL to output_path
    if video_path.startswith("http"):
        logger.info(f"📥 Downloading video from {video_path[:80]}...")
        response = requests.get(video_path, timeout=300)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        logger.info(f"✅ Video saved to {output_path}")
    else:
        # Local file path returned by Gradio
        import shutil
        shutil.copy(video_path, output_path)
        logger.info(f"✅ Video copied to {output_path}")

    return output_path


def generate_video_segment(prompt, aspect_ratio="9:16"):
    """Generuje pojedynczy klip wideo (HunyuanVideo) i zwraca lokalną ścieżkę tymczasową."""
    logger.info(f"🎬 Generowanie segmentu HunyuanVideo: {prompt[:50]}...")
    temp_file = os.path.join(tempfile.gettempdir(), f"seg_{os.urandom(4).hex()}.mp4")
    generate_wan_video(prompt, temp_file)
    return temp_file


# ---------------------------------------------------------------------------
# GENEROWANIE WIDEO: HunyuanVideo (Hugging Face / Gradio)
# ---------------------------------------------------------------------------

def _get_bucket_client():
    """Return a boto3 S3 client configured for the Railway video-storage bucket."""
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("BUCKET_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("BUCKET_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("BUCKET_SECRET_ACCESS_KEY"),
        region_name=os.getenv("BUCKET_REGION", "auto"),
    )


def _upload_video_to_s3(local_path, object_key):
    """Upload a local video file to the S3 bucket and return its public URL.

    Returns the public URL string on success, or raises on failure.
    """
    bucket_name = os.getenv("BUCKET_NAME")
    if not bucket_name:
        raise ValueError("BUCKET_NAME environment variable is not set")

    s3 = _get_bucket_client()
    s3.upload_file(
        local_path,
        bucket_name,
        object_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    endpoint = os.getenv("BUCKET_ENDPOINT_URL", "").rstrip("/")
    if endpoint:
        public_url = f"{endpoint}/{bucket_name}/{object_key}"
    else:
        region = os.getenv("BUCKET_REGION", "us-east-1")
        public_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"

    logger.info(f"☁️  Uploaded {object_key} → {public_url}")
    return public_url


def generate_nava_video(
    prompt,
    duration_sec=6.0,
    aspect_ratio="1:1 (960×960)",
    steps=20.0,
):
    """Generate a video using the NAVA Gradio Space and upload it to S3.

    Parameters
    ----------
    prompt : str
        Text description of the video to generate.
    duration_sec : float
        Duration of the generated video in seconds (4–6 s accepted by NAVA).
    aspect_ratio : str
        Aspect ratio string as expected by the NAVA Space UI
        (e.g. "1:1 (960×960)", "16:9 (848×480)", "9:16 (480×848)").
    steps : float
        Number of inference steps (default 20).

    Returns
    -------
    str
        Public URL of the uploaded video.
    """
    if not NAVA_ENABLED:
        raise RuntimeError(
            "NAVA is disabled. Set NAVA_ENABLED=true to enable this feature."
        )

    logger.info(
        f"🎬 NAVA: generating video | prompt={prompt[:60]!r} "
        f"duration={duration_sec}s aspect={aspect_ratio} steps={steps}"
    )

    def _call_nava_api():
        from gradio_client import Client
        from huggingface_hub import login

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")

        login(token=hf_token)

        client = Client(f"https://huggingface.co/spaces/{NAVA_SPACE_ID}")

        result = client.predict(
            prompt,          # Prompt (str)
            duration_sec,    # Duration in seconds (float)
            aspect_ratio,    # Aspect ratio (str)
            steps,           # Inference steps (float)
            api_name="/generate",
        )
        return result

    result = retry_with_backoff("NAVA Gradio", _call_nava_api, max_retries=3, base_delay=30)

    # The NAVA Space returns the video file path/URL directly.
    video_path = None
    if isinstance(result, dict):
        video_path = result.get("video") or result.get("path") or result.get("url")
    elif isinstance(result, (list, tuple)) and len(result) >= 1:
        candidate = result[0]
        if isinstance(candidate, dict):
            video_path = candidate.get("video") or candidate.get("path") or candidate.get("url")
        elif isinstance(candidate, str) and candidate:
            video_path = candidate
    elif isinstance(result, str) and result:
        video_path = result

    if not video_path:
        raise RuntimeError(f"No video path found in NAVA /generate response: {result}")

    # Download or copy the video to a local temp file
    local_tmp = os.path.join(tempfile.gettempdir(), f"nava_{uuid.uuid4().hex}.mp4")
    if video_path.startswith("http"):
        logger.info(f"📥 Downloading NAVA video from {video_path[:80]}...")
        resp = requests.get(video_path, timeout=300)
        resp.raise_for_status()
        with open(local_tmp, "wb") as fh:
            fh.write(resp.content)
    else:
        shutil.copy(video_path, local_tmp)

    logger.info(f"✅ NAVA video saved locally: {local_tmp}")

    # Upload to S3 and return the public URL
    object_key = f"nava/{uuid.uuid4().hex}.mp4"
    try:
        public_url = _upload_video_to_s3(local_tmp, object_key)
    finally:
        try:
            os.unlink(local_tmp)
        except OSError:
            pass

    return public_url



def generate_silent_audio_narration(narration_texts, job_id, duration_per_scene=6.0):
    """Create silent MP3 placeholder files instead of calling ElevenLabs.

    Used when SKIP_NARRATION=true so the rest of the pipeline (assembly,
    speed-adjustment, FFmpeg concat) can proceed without audio.  Each scene
    gets a silent MP3 of `duration_per_scene` seconds so that
    calculate_video_speed() has valid durations to work with.
    """
    audio_files = {}
    for scene_key in narration_texts.keys():
        audio_file = os.path.join(tempfile.gettempdir(), f"narration_{scene_key}_{job_id}.mp3")

        if os.path.exists(audio_file):
            logger.info(f"⏩ Cicha ścieżka {scene_key} już istnieje – pomijam.")
            duration = get_audio_duration(audio_file)
            audio_files[scene_key] = {"path": audio_file, "duration": duration, "text": narration_texts[scene_key]}
            continue

        logger.info(f"🔇 Tworzę cichą ścieżkę zastępczą: {scene_key} ({duration_per_scene}s)")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration_per_scene),
            "-c:a", "libmp3lame", "-q:a", "9",
            audio_file,
        ]
        try:
            subprocess.run(
                ffmpeg_cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
            )
            logger.info(f"✅ Cicha ścieżka {scene_key} zapisana: {audio_file}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"❌ Nie udało się utworzyć cichej ścieżki {scene_key}: {e}")
            raise RuntimeError(f"Failed to create silent audio placeholder for {scene_key}: {e}") from e

        audio_files[scene_key] = {
            "path": audio_file,
            "duration": duration_per_scene,
            "text": narration_texts[scene_key],
        }

    return audio_files


# ---------------------------------------------------------------------------
# NAPISY: GENEROWANIE SRT Z AUDIO (Whisper API)
# ---------------------------------------------------------------------------

def generate_subtitles_from_audio(audio_file, job_id):
    """
    Generowanie SRT z transkrypcji audio (OpenAI Whisper API)
    
    Returns: ścieżka do pliku SRT
    """
    # 1. POPRAWKA KOSZTÓW: Twardy cache dla pliku SRT
    srt_path = os.path.join(tempfile.gettempdir(), f"subs_{job_id}.srt")
    if os.path.exists(srt_path):
        logger.info(f"⏩ Napisy dla zadania {job_id} już istnieją. Pomijam płatne API Whisper: {srt_path}")
        return srt_path

    try:
        logger.info(f"📝 Transkrypcja audio (Whisper API)...")
        
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        with open(audio_file, "rb") as f:
            transcript_obj = retry_with_backoff(
                "Whisper transcribe",
                lambda: client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="pl",  # Polski
                    response_format="verbose_json"
                )
            )
        
        if hasattr(transcript_obj, "model_dump"):
            transcript = transcript_obj.model_dump()
        elif hasattr(transcript_obj, "dict"):
            transcript = transcript_obj.dict()
        else:
            transcript = transcript_obj

        srt_content = ""
        srt_index = 1
        
        for segment in transcript.get("segments", []):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            if text:
                srt_content += f"{srt_index}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{text}\n\n"
                srt_index += 1
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        logger.info(f"✅ SRT wygenerowany: {srt_path} ({srt_index-1} napisów)")
        return srt_path
        
    except ImportError:
        logger.warning("⚠️ OpenAI library not installed. Skipping subtitles.")
        return None
    except Exception as e:
        logger.error(f"❌ Błąd przy transkrypcji: {e}")
        return None

def format_timestamp(seconds):
    """Konwersja sekund na format SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ---------------------------------------------------------------------------
# PLANSZA KOŃCOWA (PNG/MP4)
# ---------------------------------------------------------------------------

def generate_end_screen(job_id, topic, output_path):
    """Generowanie planszy końcowej (1080×1920 pioneer format)"""
    # 2. OPTYMALIZACJA: Pomijanie generowania jeśli plik końcowy już istnieje
    if os.path.exists(output_path):
        logger.info(f"⏩ Plansza końcowa już istnieje: {output_path}. Pomijam generowanie.")
        return output_path

    logger.info(f"🎨 Generowanie planszy końcowej...")
    
    width, height = 1080, 1920
    background_color = (10, 25, 50)  # Dark blue
    
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    except:
        title_font = ImageFont.load_default()
    
    try:
        cta_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 50)
    except:
        cta_font = ImageFont.load_default()
    
    title_text = topic[:30]
    draw.text((540, 800), title_text, fill=(255, 255, 255), font=title_font, anchor="mm")
    
    cta_text = "Sprawdź raport na:"
    domain_text = "raport-finansowy24.pl"
    
    draw.text((540, 1400), cta_text, fill=(200, 200, 200), font=cta_font, anchor="mm")
    draw.text((540, 1550), domain_text, fill=(0, 200, 100), font=cta_font, anchor="mm")
    
    img_path = os.path.join(tempfile.gettempdir(), f"endscreen_{job_id}.png")
    img.save(img_path)
    logger.info(f"✅ Plansza PNG: {img_path}")
    
    # 3. POPRAWKA STABILNOŚCI: Limit threads i preset ultrafast dla FFmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loop', '1', '-i', img_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-t', '3',  # 3 sekund
        '-movflags', '+faststart',
        output_path
    ]
    
    run_ffmpeg(ffmpeg_cmd, timeout=60)
    logger.info(f"✅ Plansza MP4: {output_path}")
    
    if os.path.exists(img_path):
        os.remove(img_path)
    
    return output_path

# ---------------------------------------------------------------------------
# WATERMARK
# ---------------------------------------------------------------------------

def add_watermark(video_path, output_path, watermark_text="raport-finansowy24.pl", opacity=0.7):
    """Dodanie watermarku tekstowego do wideo"""
    logger.info(f"🏷️ Dodawanie watermarku: {watermark_text}")
    
    # POPRAWKA STABILNOŚCI: Zapobieganie throttlowaniu CPU
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', (
            f"drawtext=text='{watermark_text}':"
            f"x=w-text_w-20:y=h-text_h-20:"
            f"fontsize=24:fontcolor=white@{opacity}:"
            f"box=1:boxcolor=black@0.5"
        ),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        output_path
    ]
    
    run_ffmpeg(ffmpeg_cmd, timeout=180)
    logger.info(f"✅ Watermark dodany")
    
    return output_path

# ---------------------------------------------------------------------------
# POBIERANIE CZASU TRWANIA
# ---------------------------------------------------------------------------

def get_audio_duration(audio_file):
    """Pobranie czasu trwania audio za pomocą ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"⚠️ Nie udało się pobrać czasu trwania {audio_file}: {e}")
        return 5.0


def ffmpeg_supports_subtitles():
    """Check if FFmpeg has libass support for subtitles filter."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-filters'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Check if 'subtitles' filter is available
        return 'subtitles' in result.stdout
    except Exception as e:
        logger.warning(f"⚠️ Could not check FFmpeg filters: {e}")
        return False

def get_video_duration(video_file):
    """Pobranie czasu trwania wideo za pomocą ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"⚠️ Nie udało się pobrać czasu trwania {video_file}: {e}")
        return 5.0

# ---------------------------------------------------------------------------
# AUTOMATYCZNE DOPASOWANIE DŁUGOŚCI
# ---------------------------------------------------------------------------

def calculate_video_speed(audio_files, target_duration=18):
    """Obliczenie prędkości playbacku aby zmieścić się w target_duration"""
    total_audio_duration = sum(audio["duration"] for audio in audio_files.values())
    total_audio_duration += 2  # Buffer dla CTA
    
    logger.info(f"⏱️ Całkowity czas lektora: {total_audio_duration:.2f}s")
    logger.info(f"📊 Target duration: {target_duration}s")
    
    if total_audio_duration <= target_duration:
        speed = 1.0
        logger.info(f"✅ Lektor zmieści się. Speed: {speed}x (normalnie)")
    else:
        speed = total_audio_duration / target_duration
        logger.warning(f"⚠️ Lektor za długi ({total_audio_duration:.2f}s > {target_duration}s). Przyspieszenie: {speed:.2f}x")
    
    if speed > 1.5:
        logger.warning(f"⚠️ Speed {speed:.2f}x przekracza limit 1.5x!")
        speed = 1.5
    
    return speed

def build_atempo_chain(speed):
    """Build atempo filter chain for speeds outside 0.5-2.0 range."""
    if speed < 0.5:
        return "atempo=0.5"
    elif speed <= 2.0:
        return f"atempo={speed}"
    else:
        filters = []
        remaining_speed = speed
        while remaining_speed > 2.0:
            filters.append("atempo=2.0")
            remaining_speed /= 2.0
        filters.append(f"atempo={remaining_speed:.2f}")
        return ",".join(filters)


def generate_video_with_speed_adjustment(segment_files, speed=1.0):
    """Generowanie wideo ze zmienioną prędkością"""
    if speed == 1.0:
        logger.info("✅ Brak dopasowania prędkości (1.0x)")
        return segment_files
    
    logger.info(f"⏱️ Dopasowywanie prędkości wszystkich segmentów do {speed:.2f}x...")
    
    speed_adjusted_files = []
    
    for i, video_file in enumerate(segment_files):
        output_file = os.path.join(tempfile.gettempdir(), f"speed_{i}_{os.urandom(4).hex()}.mp4")
        
        # POPRAWKA STABILNOŚCI: Zapobieganie throttlowaniu CPU na Railway
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_file,
            '-vf', f"setpts=PTS/{speed}",
            '-af', build_atempo_chain(speed),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_file
        ]
        
        logger.info(f"  ⏱️ Segment {i}: {speed:.2f}x")
        run_ffmpeg(ffmpeg_cmd, timeout=180)
        
        speed_adjusted_files.append(output_file)
    
    return speed_adjusted_files

# ---------------------------------------------------------------------------
# ŁĄCZENIE WIDEO + AUDIO + NAPISY + WATERMARK
# ---------------------------------------------------------------------------

def concat_video_with_audio_and_subtitles(video_files, audio_files, srt_file, job_id, output_path, speed=1.0):
    """
    Łączenie segmentów wideo + dodanie lektora + napisy + watermark
    """
    # 1. OPTYMALIZACJA: Jeśli finalny plik istnieje, pomiń cały ciężki proces
    if os.path.exists(output_path):
        logger.info(f"⏩ Finalne wideo {job_id} już istnieje. Pomijam renderowanie.")
        return output_path
    
    list_file_path = os.path.join(tempfile.gettempdir(), f"list_{job_id}.txt")
    with open(list_file_path, "w") as f:
        for video_file in video_files:
            f.write(f"file '{video_file}'\n")
    
    logger.info("🎬 Etap 1: Łączenie segmentów wideo (FFmpeg concat)...")
    
    concat_output = os.path.join(tempfile.gettempdir(), f"concat_{job_id}.mp4")
    ffmpeg_concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart',
        concat_output
    ]
    run_ffmpeg(ffmpeg_concat_cmd, timeout=120)
    logger.info(f"✅ Wideo połączone: {concat_output}")
    
    logger.info("🎙️ Etap 2: Miksowanie audio (lektory)...")

    combined_audio = os.path.join(tempfile.gettempdir(), f"combined_audio_{job_id}.mp3")
    audio_list_file = os.path.join(tempfile.gettempdir(), f"audio_list_{job_id}.txt")

    # Collect audio entries that actually exist on disk
    audio_entries = [
        audio_files[k]["path"]
        for k in ["hook", "problem", "rozwiązanie"]
        if k in audio_files and os.path.exists(audio_files[k]["path"])
    ]
    has_audio = bool(audio_entries)

    if has_audio:
        with open(audio_list_file, "w") as f:
            for path in audio_entries:
                f.write(f"file '{path}'\n")

        ffmpeg_audio_concat = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', audio_list_file,
            '-c:a', 'libmp3lame', '-q:a', '4',
            combined_audio
        ]
        run_ffmpeg(ffmpeg_audio_concat, timeout=120)
        logger.info(f"✅ Lektory połączone: {combined_audio}")
    else:
        logger.info("⏭️  Brak ścieżek audio – montaż wideo bez dźwięku.")

    logger.info("🎨 Etap 3: Miksowanie wideo + audio + napisy...")

    # BŁĄD LOGICZNY USUNIĘTY: Skoro speed robimy w innej funkcji, tu dajemy zwykłe kopiowanie strumienia video (bez setpts)
    # Zostawiamy po prostu wejście wideo bez modyfikacji czasu, żeby nie podwoić przyspieszenia.

    # Budujemy łańcuch filtrów
    filters = []

    if srt_file and os.path.exists(srt_file):
        if ffmpeg_supports_subtitles():
            srt_path_escaped = srt_file.replace("\\", "\\\\").replace(":", "\\:")
            # Eleganckie, wyraźne napisy dopasowane do profesjonalnego brandingu
            filters.append(f"subtitles='{srt_path_escaped}':force_style='FontSize=28,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0'")
            logger.info(f"✅ Napisy będą wypalane")

        else:
            logger.warning("⚠️ FFmpeg subtitles filter not available (libass missing). Skipping subtitles.")

    watermark_text = "raport-finansowy24.pl"
    filters.append(f"drawtext=text='{watermark_text}':x=w-text_w-20:y=h-text_h-20:fontsize=24:fontcolor=white@0.7:box=1:boxcolor=black@0.5")

    # Łączymy filtry wideo przecinkami
    final_video_filter = ",".join(filters)

    if has_audio:
        ffmpeg_final_cmd = [
            'ffmpeg', '-y',
            '-i', concat_output,
            '-i', combined_audio,
            '-filter_complex',
            f"[0:v]{final_video_filter}[vout];[1:a]volume=1.0[aout]",
            '-map', '[vout]', '-map', '[aout]',
            # STABILNOŚĆ: ultrafast i threads=2 zapobiegną zabiciu procesu przez Gunicorn
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]
    else:
        # No audio track – video-only output (SKIP_NARRATION mode)
        ffmpeg_final_cmd = [
            'ffmpeg', '-y',
            '-i', concat_output,
            '-filter_complex',
            f"[0:v]{final_video_filter}[vout]",
            '-map', '[vout]',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-crf', '23',
            '-an',
            output_path
        ]

    logger.info("🔄 Kodowanie finale (może potrwać trochę)...")
    run_ffmpeg(ffmpeg_final_cmd, timeout=300)
    logger.info(f"✅ Finalne wideo: {output_path}")

    cleanup_files = [list_file_path, concat_output]
    if has_audio:
        cleanup_files += [audio_list_file, combined_audio]
    for file in cleanup_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"⚠️ Nie udało się usunąć {file}: {e}")

    return output_path

# ---------------------------------------------------------------------------
# GŁÓWNY PROCES RENDEROWANIA
# ---------------------------------------------------------------------------

NARRATION_TEMPLATES = {
    "hook": "Większość osób traci pieniądze na złym koncie. Czy i ty?",
    "problem": "Banki promują oferty, które szybko tracą atrakcyjne warunki.",
    "rozwiązanie": "Regularne porównywanie ofert pozwala znaleźć korzystniejsze opcje i zaoszczędzić na rachunkach."
}

def render_sequence_background(job_id, raw_data, webhook_url=None, resume_from=None):
    """
    Główny proces montażu sekwencji - uruchamiany w tle
    """
    STAGES = [
        "hook_video",
        "problem_video",
        "solution_video",
        "narration",
        "assembly",
        "upload",
    ]

    def _stage_done(stage):
        """Return True when this stage was already completed before the resume point."""
        if resume_from is None:
            return False
        try:
            return STAGES.index(stage) < STAGES.index(resume_from)
        except ValueError:
            return False

    job_start_time = time.time()
    def check_job_timeout():
        if time.time() - job_start_time > MAX_JOB_DURATION_SECONDS:
            raise TimeoutError(f"Job exceeded the maximum execution limit of {MAX_JOB_DURATION_SECONDS} seconds.")

    segment_files = []
    audio_files_dict = {}
    srt_file = None
    current_stage = "init"
    job_paused = False

    try:
        check_job_timeout()
        topic = raw_data.get("topic", "Finanse osobiste")
        raw_data = apply_optimization_rules(raw_data, topic)
        aspect_ratio = raw_data.get("aspectRatio", "9:16")
        host = raw_data.get("host", "localhost:5000")
        custom_narration = raw_data.get("narration")
        hashtags = raw_data.get("hashtags")
        if not hashtags:
            hashtags = build_hashtags(topic, custom_narration, MAX_HASHTAGS)

        if resume_from:
            logger.info(f"▶️  RESUME Job {job_id} | Wznawianie od etapu: {resume_from}")
        else:
            logger.info(f"🚀 START renderowania Job ID: {job_id} | Temat: {topic}")

        if ENABLE_DRY_RUN:
            logger.info("🧪 DRY_RUN enabled: skipping external providers and returning simulated success")
            simulated_filename = f"dryrun_{job_id}.mp4"
            simulated_path = os.path.join(STORAGE_DIR, simulated_filename)
            with open(simulated_path, "wb") as f:
                f.write(b"DRY_RUN")
            video_url = f"https://{host}/videos/{simulated_filename}"
            save_render_to_db(job_id, topic, 'success', video_url, video_duration=0.0)
            if webhook_url:
                send_webhook(webhook_url, {
                    "event_type": "render.completed",
                    "job_id": job_id,
                    "status": "success",
                    "video_url": video_url,
                    # ... reszta payloadu dry_run bez zmian ...
                    "dry_run": True,
                })
            METRICS["jobs_success"] += 1
            return

        # Generate 3 prompt variants and select the best one
        narration_dict = {
            "hook": custom_narration.get("hook", "") if custom_narration else "",
            "problem": custom_narration.get("problem", "") if custom_narration else "",
            "rozwiązanie": custom_narration.get("rozwiązanie", "") if custom_narration else ""
        }
        
        # Generate variants for each scene
        prompts = {}
        for key in ["hook", "problem", "rozwiązanie"]:
            variants = generate_prompt_variants(topic, {key: narration_dict[key]})
            best_prompt, best_idx, best_score = select_best_prompt(variants)
            prompts[key] = best_prompt
            logger.info(f"  {key.upper()}: selected variant {best_idx} (score={best_score:.1f})")

        logger.info("📋 Szablon: HOOK → PROBLEM → ROZWIĄZANIE")

        stage_map = {
            "hook":       "hook_video",
            "problem":    "problem_video",
            "rozwiązanie": "solution_video",
        }

        for key, prompt_text in prompts.items():
            check_job_timeout()
            stage_name = stage_map[key]

            # POPRAWKA 1: Pliki dla wznawiania zadań MUSZĄ być w trwałym STORAGE_DIR, nie w ulotnym tempfile
            stable_path = os.path.join(STORAGE_DIR, f"seg_{job_id}_{key}.mp4")

            if _stage_done(stage_name):
                if os.path.exists(stable_path):
                    logger.info(f"⏩ Scena {key.upper()} już istnieje – pomijam HunyuanVideo.")
                    segment_files.append(stable_path)
                    continue
                logger.info(f"⚠️  Plik sceny {key} zaginął ze STORAGE_DIR – regeneruję...")

            current_stage = stage_name
            save_checkpoint(job_id, current_stage, data={"topic": topic, "key": key})
            logger.info(f"🎥 Generowanie sceny HunyuanVideo: {key.upper()}")

            generate_wan_video(prompt_text, stable_path)

            segment_files.append(stable_path)
            logger.info(f"✅ Scena {key} gotowa")

        logger.info(f"✅ Wszystkie 3 sceny gotowe")

        check_job_timeout()
        if not _stage_done("narration"):
            current_stage = "narration"
            save_checkpoint(job_id, current_stage, data={"topic": topic})
            logger.info("🎙️ Generowanie cichych ścieżek zastępczych...")

        narration_texts = custom_narration if custom_narration else NARRATION_TEMPLATES
        audio_files_dict = generate_silent_audio_narration(narration_texts, job_id)
        logger.info("✅ Ciche ścieżki zastępcze gotowe")

        check_job_timeout()
        srt_file = None

        # Generowanie napisów (pomijamy gdy SKIP_NARRATION=true – brak realnego audio do transkrypcji)
        if SKIP_NARRATION:
            logger.info("⏭️  SKIP_NARRATION=true – pomijam generowanie napisów (Whisper).")
        else:
            combined_for_transcription = os.path.join(tempfile.gettempdir(), f"combined_trans_{job_id}.mp3")
            audio_list_file = os.path.join(tempfile.gettempdir(), f"audio_list_trans_{job_id}.txt")

            try:
                with open(audio_list_file, "w") as f:
                    for scene_key in ["hook", "problem", "rozwiązanie"]:
                        if scene_key in audio_files_dict:
                            f.write(f"file '{audio_files_dict[scene_key]['path']}'\n")

                ffmpeg_concat = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', audio_list_file,
                    '-c:a', 'libmp3lame', '-q:a', '4',
                    combined_for_transcription
                ]

                # ZABEZPIECZENIE: Try-except dla łączenia audio
                try:
                    run_ffmpeg(ffmpeg_concat, timeout=120)
                    srt_file = generate_subtitles_from_audio(combined_for_transcription, job_id)
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Błąd FFmpeg przy łączeniu audio dla Whispera: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'Brak'}")
                    srt_file = None
                except subprocess.TimeoutExpired:
                    logger.error("❌ Błąd: Timeout FFmpeg przy łączeniu audio dla Whispera.")
                    srt_file = None

            except Exception as e:
                logger.warning(f"⚠️ Napisy niedostępne: {e}")
                srt_file = None
            finally:
                if os.path.exists(combined_for_transcription):
                    os.remove(combined_for_transcription)
                if os.path.exists(audio_list_file):
                    os.remove(audio_list_file)

        check_job_timeout()
        logger.info("⏱️ Etap automatycznego dopasowania długości...")

        target_duration = int(raw_data.get("targetDuration", 18))
        speed = calculate_video_speed(audio_files_dict, target_duration)

        if speed != 1.0:
            logger.info(f"⚡ Dopasowywanie prędkości wideo do {speed:.2f}x...")
            segment_files = generate_video_with_speed_adjustment(segment_files, speed)
            logger.info(f"✅ Segmenty dopasowane")

        check_job_timeout()
        current_stage = "assembly"
        save_checkpoint(job_id, current_stage, data={"topic": topic, "speed": speed})
        logger.info("🎬 Główny montaż (wideo + audio + napisy + watermark)...")

        final_filename = f"render_{job_id}.mp4"
        final_output_path = os.path.join(STORAGE_DIR, final_filename)

        final_output_path = concat_video_with_audio_and_subtitles(segment_files, audio_files_dict, srt_file, job_id, final_output_path, speed)

        if not final_output_path:
            raise RuntimeError("Nie udało się złożyć finalnego wideo (błąd w concat_video_with_audio_and_subtitles).")

        check_job_timeout()
        logger.info("🎨 Dodawanie planszy końcowej...")

        endscreen_path = os.path.join(tempfile.gettempdir(), f"endscreen_{job_id}.mp4")
        generate_end_screen(job_id, topic, endscreen_path)

        final_with_endscreen = os.path.join(tempfile.gettempdir(), f"final_with_endscreen_{job_id}.mp4")

        concat_list = os.path.join(tempfile.gettempdir(), f"final_concat_{job_id}.txt")
        with open(concat_list, "w") as f:
            f.write(f"file '{final_output_path}'\n")
            f.write(f"file '{endscreen_path}'\n")

        ffmpeg_final_concat = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
            final_with_endscreen
        ]
        
        # ZABEZPIECZENIE: Łapanie błędu krytycznego podczas finałowego sklejania
        try:
            run_ffmpeg(ffmpeg_final_concat, timeout=120)
            shutil.move(final_with_endscreen, final_output_path)
            logger.info(f"✅ Plansza końcowa dodana")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"❌ Nie udało się dodać planszy końcowej. Zostawiam wideo bez niej. Błąd: {e}")
            # Nie przerywamy zadania, po prostu wydamy wideo bez doklejonej planszy

        for f in [endscreen_path, concat_list]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        check_job_timeout()
        video_duration = get_video_duration(final_output_path)
        file_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
        if file_size_mb > MAX_OUTPUT_VIDEO_MB:
            raise ValueError(
                f"Output file too large: {file_size_mb:.1f} MB > {MAX_OUTPUT_VIDEO_MB} MB"
            )

        logger.info(f"✅ SUKCES! Film gotowy: {final_filename}")
        logger.info(f"  ⏱️ Czas trwania: {video_duration:.2f}s")
        logger.info(f"  📊 Rozmiar: {file_size_mb:.1f} MB")
        logger.info(f"  ⚡ Prędkość: {speed:.2f}x")

        video_url = f"https://{host}/videos/{final_filename}"
        logger.info(f"📺 URL: {video_url}")

        current_stage = "upload"
        save_checkpoint(job_id, current_stage, data={"video_url": video_url})
        save_render_to_db(job_id, topic, 'success', video_url, video_duration=video_duration)
        logger.info(f"💾 Historia zapisana")

        if webhook_url:
            logger.info(f"🔔 Wysyłanie webhook...")
            try:
                webhook_payload = {
                    "event_type": "render.completed",
                    "job_id": job_id,
                    "status": "success",
                    "video_url": video_url,
                    # ... reszta payloadu bez zmian ...
                    "timestamp": datetime.utcnow().isoformat()
                }
                response = send_webhook(webhook_url, webhook_payload)
                logger.info(f"✅ Webhook wysłany (status: {response.status_code})")
                METRICS["webhook_success"] += 1
            except requests.RequestException as e:
                logger.error(f"⚠️ Błąd webhook: {e}")
                METRICS["webhook_failed"] += 1
        METRICS["jobs_success"] += 1

    except Exception as e:
        logger.error(f"❌ BŁĄD KRYTYCZNY Job {job_id} na etapie '{current_stage}': {e}", exc_info=True)
        METRICS["jobs_failed"] += 1
        METRICS["last_error"] = str(e)

        job_paused = True
        save_checkpoint(job_id, current_stage, error=str(e))
        logger.error(f"⏸️ Job {job_id} PAUSED at '{current_stage}': {e}")

        if webhook_url:
            try:
                webhook_payload = {
                    "event_type": "render.paused",
                    "job_id": job_id,
                    "status": "paused",
                    "paused_at_stage": current_stage,
                    "error": str(e),
                    "resume_url": f"https://{raw_data.get('host', 'localhost:5000')}/resume/{job_id}",
                    "source": "cashmaker-veo-worker",
                    "timestamp": datetime.utcnow().isoformat()
                }
                send_webhook(webhook_url, webhook_payload)
                logger.info(f"🔔 Webhook pauzy wysłany")
            except Exception as webhook_error:
                logger.error(f"⚠️ Błąd webhook: {webhook_error}")

    finally:
        RENDER_SEMAPHORE.release()
        if job_paused:
            logger.info("⏸️ Job paused – zachowuję pliki w STORAGE_DIR dla wznowienia.")
        else:
            if not DISABLE_CLEANUP:
                logger.info("🧹 Czyszczenie plików...")
            for path in segment_files:
                # Nie usuwamy głównych segmentów ze STORAGE_DIR od razu, zostawiamy to funkcji cleanup_old_files() 
                # Zabezpiecza to pliki, gdyby API wznawiania ich wciąż potrzebowało w tle
                pass

            for scene_key, audio_info in audio_files_dict.items():
                audio_path = audio_info.get("path")
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception as e:
                        logger.warning(f"⚠️ Nie udało się usunąć {audio_path}: {e}")

            if srt_file and os.path.exists(srt_file):
                try:
                    os.remove(srt_file)
                except Exception as e:
                    logger.warning(f"⚠️ Nie udało się usunąć {srt_file}: {e}")


# ---------------------------------------------------------------------------
# CLEANUP STARYCH PLIKÓW
# ---------------------------------------------------------------------------

def cleanup_old_files(hours=24):
    """Czyszczenie plików starszych niż N godzin ze STORAGE_DIR"""
    cutoff_time = time.time() - (hours * 3600)
    cleaned_count = 0
    
    for filename in os.listdir(STORAGE_DIR):
        filepath = os.path.join(STORAGE_DIR, filename)
        
        if filename.endswith('.db'):
            continue
            
        if os.path.isfile(filepath):
            file_age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
            
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                    logger.info(f"🧹 Usunięty stary plik ({file_age_hours:.1f}h): {filename}")
                except Exception as e:
                    logger.error(f"❌ Błąd przy usuwaniu {filename}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"✅ Cleanup: Usunięto {cleaned_count} starych plików")

# ---------------------------------------------------------------------------
# ENDPOINTY FLASK
# ---------------------------------------------------------------------------

@app.route("/render-sequence", methods=["POST"])
def start_render_sequence():
    """
    POST /render-sequence
    """
    auth_error = require_api_key()
    if auth_error:
        return auth_error

    data = request.json or {}
    limits_error = validate_request_limits(data)
    if limits_error:
        return limits_error

    ok, retry_after = enforce_rate_limit(WORKER_API_KEY or "default")
    if not ok:
        return jsonify({
            "error": "Rate limit exceeded for free tier",
            "limit_per_hour": RATE_LIMIT_PER_HOUR,
            "retry_after_seconds": retry_after
        }), 429

    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if idempotency_key:
        cached = get_idempotency_response(idempotency_key)
        if cached:
            return jsonify(cached["body"]), cached["status"]

    topic = data.get("topic", "").strip()
    
    if not topic:
        return jsonify({"error": "Missing or empty 'topic'"}), 400
    
    webhook_url = data.get("webhookUrl")
    job_id = str(uuid.uuid4())
    METRICS["jobs_started"] += 1

    if not RENDER_SEMAPHORE.acquire(blocking=False):
        return jsonify({
            "error": "Too many concurrent renders",
            "max_concurrent_renders": MAX_CONCURRENT_RENDERS
        }), 429
    
    data['host'] = request.host
    save_render_to_db(job_id, topic, 'processing')
    logger.info(f"📥 Nowe zlecenie: Job {job_id} | Temat: {topic}")
    
    thread = threading.Thread(
        target=render_sequence_background,
        args=(job_id, data, webhook_url),
        daemon=True
    )
    thread.start()
    
    response_body = {
        "status": "queued",
        "job_id": job_id,
        "status_url": f"https://{request.host}/tasks/{job_id}"
    }
    if idempotency_key:
        remember_idempotency(idempotency_key, {"body": response_body, "status": 202})
    return jsonify(response_body), 202


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """GET /tasks/<job_id>"""
    auth_error = require_api_key()
    if auth_error:
        return auth_error

    render = get_render_from_db(task_id)

    if not render:
        return jsonify({"error": "Task not found"}), 404

    response = {
        "job_id": task_id,
        "state": render["status"],
        "created_at": render["created_at"],
        "completed_at": render["completed_at"]
    }

    if render["status"] == "processing":
        response["status"] = "⏳ Przetwarzanie..."
    elif render["status"] == "success":
        response["status"] = "✅ Zakończono sukcesem"
        response["video_url"] = render["video_url"]
        response["video_duration"] = render["video_duration"]
    elif render["status"] == "failed":
        response["status"] = "❌ Błąd wykonania"
        response["error"] = render["error"]
    elif render["status"] == "paused":
        response["status"] = "⏸️ Wstrzymano – błąd na etapie"
        response["paused_at_stage"] = render["current_stage"]
        response["paused_at"] = render["paused_at"]
        response["paused_reason"] = render["paused_reason"]
        response["resume_url"] = f"https://{request.host}/resume/{task_id}"
        response["retry_count"] = render.get("retry_count", 0)
        response["next_retry_at"] = render.get("next_retry_at")
        response["auto_retry_enabled"] = AUTO_RETRY_ENABLED

    return jsonify(response)


@app.route("/resume/<job_id>", methods=["POST"])
def resume_job(job_id):
    """
    POST /resume/<job_id>
    """
    auth_error = require_api_key()
    if auth_error:
        return auth_error

    checkpoint = get_checkpoint(job_id)

    if not checkpoint:
        return jsonify({"error": "Job not found"}), 404

    if checkpoint["status"] != "paused":
        return jsonify({
            "error": "Job is not paused",
            "current_status": checkpoint["status"],
            "job_id": job_id
        }), 409

    if not RENDER_SEMAPHORE.acquire(blocking=False):
        return jsonify({
            "error": "Too many concurrent renders",
            "max_concurrent_renders": MAX_CONCURRENT_RENDERS
        }), 429

    resume_from = checkpoint["current_stage"]
    topic = checkpoint["topic"]

    override_data = request.json or {}
    raw_data = {
        "topic": topic,
        "host": request.host,
        **checkpoint["checkpoint_data"],
        **override_data,
    }
    raw_data["host"] = request.host

    webhook_url = raw_data.get("webhookUrl")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE renders SET status = 'processing' WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()

    logger.info(f"▶️  Resuming Job {job_id} from stage '{resume_from}'")
    logger.info(f"   Reusing checkpoint data to skip already-completed API calls")

    thread = threading.Thread(
        target=render_sequence_background,
        args=(job_id, raw_data, webhook_url),
        kwargs={"resume_from": resume_from},
        daemon=True
    )
    thread.start()

    return jsonify({
        "status": "resumed",
        "job_id": job_id,
        "resuming_from_stage": resume_from,
        "status_url": f"https://{request.host}/tasks/{job_id}",
        "note": "Reusing checkpoint data — skipping already-completed API calls"
    }), 202


def auto_retry_worker():
    """Background thread that checks for paused jobs ready to retry."""
    while True:
        try:
            if not AUTO_RETRY_ENABLED:
                time.sleep(60)
                continue

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            now = datetime.utcnow()
            c.execute('''SELECT job_id, topic, checkpoint_data, current_stage, retry_count
                         FROM renders
                         WHERE status = 'paused'
                         AND next_retry_at IS NOT NULL
                         AND next_retry_at <= ?
                         AND retry_count <= ?''',
                      (now, AUTO_RETRY_MAX_ATTEMPTS))

            jobs_to_retry = c.fetchall()
            conn.close()

            for job_id, topic, checkpoint_json, stage, retry_count in jobs_to_retry:
                logger.info(f"🔄 Auto-retrying job {job_id} (attempt {retry_count}/{AUTO_RETRY_MAX_ATTEMPTS})")

                try:
                    checkpoint = json.loads(checkpoint_json) if checkpoint_json else {}

                    if not RENDER_SEMAPHORE.acquire(blocking=False):
                        logger.warning(f"⚠️  Cannot retry {job_id}: too many concurrent renders")
                        continue

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''UPDATE renders
                                 SET status = 'processing',
                                     paused_at = NULL,
                                     paused_reason = NULL,
                                     next_retry_at = NULL
                                 WHERE job_id = ?''', (job_id,))
                    conn.commit()
                    conn.close()

                    raw_data = {
                        "topic": topic,
                        **checkpoint
                    }

                    thread = threading.Thread(
                        target=render_sequence_background,
                        args=(job_id, raw_data, None),
                        kwargs={"resume_from": stage},
                        daemon=True
                    )
                    thread.start()

                except Exception as e:
                    logger.error(f"❌ Failed to auto-retry job {job_id}: {e}")
                    RENDER_SEMAPHORE.release()

            time.sleep(30)

        except Exception as e:
            logger.error(f"❌ Auto-retry worker error: {e}")
            time.sleep(60)


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """GET /videos/<filename>"""
    auth_error = require_api_key()
    if auth_error:
        return auth_error
    return send_from_directory(STORAGE_DIR, filename)


@app.route("/health", methods=["GET"])
def health_check():
    """GET /health"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "storage_dir": STORAGE_DIR,
        "metrics": METRICS,
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    """GET /metrics - proste metryki runtime."""
    auth_error = require_api_key()
    if auth_error:
        return auth_error
    return jsonify(METRICS), 200


@app.route("/nava-generate", methods=["POST"])
def nava_generate():
    """POST /nava-generate

    Generate a short video using the NAVA text-to-video model
    (prithivMLmods/NAVA-Text-to-Video) and return a public S3 URL.

    Request body (JSON):
        prompt       (str,   required) – text description of the video
        duration_sec (float, optional, default 6) – video length in seconds (4–6)
        aspect_ratio (str,   optional, default "1:1 (960×960)") – output aspect ratio
        steps        (float, optional, default 20) – number of inference steps
        webhookUrl   (str,   optional) – URL to POST a completion notification to

    Response 200:
        { "video_url": "<public S3 URL>" }

    Response 503:
        { "error": "NAVA is disabled …" }
    """
    auth_error = require_api_key()
    if auth_error:
        return auth_error

    if not NAVA_ENABLED:
        return jsonify({
            "error": "NAVA is disabled on this instance. Set NAVA_ENABLED=true to enable."
        }), 503

    data = request.json or {}

    # Validate webhook URL if provided
    webhook_url = data.get("webhookUrl")
    if webhook_url and not is_valid_public_url(webhook_url):
        return jsonify({"error": "Invalid or unsafe 'webhookUrl' (SSRF protection)"}), 400

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Missing or empty 'prompt'"}), 400

    try:
        duration_sec = float(data.get("duration_sec", 6))
    except (TypeError, ValueError):
        return jsonify({"error": "'duration_sec' must be a number"}), 400

    if not (4 <= duration_sec <= 6):
        return jsonify({"error": "'duration_sec' must be between 4 and 6"}), 400

    aspect_ratio = str(data.get("aspect_ratio", "1:1 (960×960)"))

    try:
        steps = float(data.get("steps", 20))
    except (TypeError, ValueError):
        return jsonify({"error": "'steps' must be a number"}), 400

    logger.info(
        f"📥 NAVA request | prompt={prompt[:60]!r} duration={duration_sec}s "
        f"aspect={aspect_ratio} steps={steps}"
    )

    try:
        video_url = generate_nava_video(
            prompt=prompt,
            duration_sec=duration_sec,
            aspect_ratio=aspect_ratio,
            steps=steps,
        )
    except RuntimeError as exc:
        logger.error(f"❌ NAVA generation failed: {exc}")
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        logger.error(f"❌ NAVA unexpected error: {exc}", exc_info=True)
        return jsonify({"error": "Internal server error during NAVA generation"}), 500

    logger.info(f"✅ NAVA video ready: {video_url}")

    if webhook_url:
        try:
            send_webhook(webhook_url, {
                "event_type": "nava.completed",
                "status": "success",
                "video_url": video_url,
                "prompt": prompt,
                "timestamp": datetime.utcnow().isoformat(),
            })
            logger.info("🔔 NAVA webhook sent")
        except Exception as exc:
            logger.warning(f"⚠️ NAVA webhook failed: {exc}")

    return jsonify({"video_url": video_url}), 200


@app.route("/", methods=["GET"])
def index():
    """API Info"""
    return jsonify({
        "name": "HunyuanVideo API",
        "version": "4.1.0",
        "features": {
            "hunyuan_generation": "3 sceny (HOOK/PROBLEM/ROZWIĄZANIE) via Hugging Face HunyuanVideo",
            "nava_generation": f"NAVA text-to-video via {NAVA_SPACE_ID} ({'✅ enabled' if NAVA_ENABLED else '❌ disabled'})",
            "audio_narration": "Silent placeholder (local generation)",
            "subtitles": "Whisper API (automatyczna transkrypcja)",
            "watermark": "raport-finansowy24.pl (dolny róg)",
            "endscreen": "Plansza końcowa (3s)",
            "auto_length_adjustment": "Dopasowanie prędkości do lektora",
            "checkpoint_resume": "Pause/resume – wznowienie od ostatniego etapu"
        },
        "endpoints": {
            "POST /render-sequence": "Uruchomienie renderowania (HunyuanVideo)",
            "POST /nava-generate": "Generowanie wideo NAVA (text-to-video)",
            "GET /tasks/<job_id>": "Status renderowania",
            "POST /resume/<job_id>": "Wznowienie wstrzymanego zadania od checkpointu",
            "GET /videos/<filename>": "Pobieranie wideo",
            "GET /health": "Health check"
        },
        "stages": [
            "hook_video",
            "problem_video",
            "solution_video",
            "narration",
            "assembly",
            "upload"
        ]
    }), 200

# ---------------------------------------------------------------------------
# STARTUP WALIDACJA (Gdy moduł jest importowany przez Gunicorn lub uruchamiany bezpośrednio)
# ---------------------------------------------------------------------------
try:
    validate_required_env()
    logger.info("✅ Startup validation passed successfully.")
except Exception as val_err:
    logger.error(f"⚠️ Uwaga: Błąd walidacji, ale startujemy dalej: {val_err}")
    # sys.exit(1)  # <--- COMMENTED OUT

# ---------------------------------------------------------------------------
# STARTUP INITIALIZATION (runs on import by Gunicorn or direct execution)
# ---------------------------------------------------------------------------
logger.info("🚀 Startup HunyuanVideo API v4.0 (Napisy + Watermark + Plansza + Checkpoint/Resume)")
logger.info(f"📁 Storage: {STORAGE_DIR}")
logger.info(f"🗄️  Database: {DB_PATH}")

cleanup_old_files(hours=24)

# Start auto-retry worker thread
retry_thread = threading.Thread(target=auto_retry_worker, daemon=True)
retry_thread.start()
logger.info("✅ Auto-retry worker started")
