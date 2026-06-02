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
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from elevenlabs.client import ElevenLabs

# ---------------------------------------------------------------------------
# KONFIGURACJA I INICJALIZACJA
# ---------------------------------------------------------------------------

STORAGE_DIR = os.getenv('STORAGE_DIR', '/app/data')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = "veo-3.1-fast-generate-preview"
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

# Auto-retry configuration for paused jobs
AUTO_RETRY_ENABLED = os.getenv("AUTO_RETRY_ENABLED", "true").lower() == "true"
AUTO_RETRY_MAX_ATTEMPTS = int(os.getenv("AUTO_RETRY_MAX_ATTEMPTS", "3"))
AUTO_RETRY_INITIAL_DELAY_SECONDS = int(os.getenv("AUTO_RETRY_INITIAL_DELAY_SECONDS", "30"))
AUTO_RETRY_MAX_DELAY_SECONDS = int(os.getenv("AUTO_RETRY_MAX_DELAY_SECONDS", "600"))  # 10 minutes

# Hard execution timeout configuration for background jobs
MAX_JOB_DURATION_SECONDS = int(os.getenv("MAX_JOB_DURATION_SECONDS", "1200"))  # 10 minutes
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"

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

# ElevenLabs API
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
if not ELEVENLABS_API_KEY:
    logger.warning("⚠️ ELEVENLABS_API_KEY not set!")

def is_valid_public_url(url):
    """
    Walidacja URL-a w celu ochrony przed SSRF.
    Zezwala tylko na publiczne adresy HTTP/HTTPS.
    """
    if not url:
        return False
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
    required = ["GEMINI_API_KEY", "HF_TOKEN", "ELEVENLABS_API_KEY", "OPENAI_API_KEY", "WORKER_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )

    # 3. Twarda walidacja kluczy przy starcie (tylko gdy nie jest to DRY_RUN / TESTING)
    if not ENABLE_DRY_RUN and not IS_TESTING:
        logger.info("🔒 Rozpoczynam twardą walidację kluczy API...")
        
        # Walidacja Gemini
        try:
            client = get_gemini_client()
            client.models.get(name=MODEL)  # <-- POPRAWIONE WCIĘCIE
            logger.info("✅ Klucz GEMINI_API_KEY zweryfikowany pomyślnie.")
        except Exception as e:
            raise RuntimeError(f"GEMINI_API_KEY validation failed: {e}")

        # Walidacja ElevenLabs
        try:
            el_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            el_client.voices.get_all()
            logger.info("✅ Klucz ELEVENLABS_API_KEY zweryfikowany pomyślnie.")
        except Exception as e:
            raise RuntimeError(f"ELEVENLABS_API_KEY validation failed: {e}")

       # Walidacja OpenAI
        try:
            from openai import OpenAI
            oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            oa_client.models.list()
            logger.info("✅ Klucz OPENAI_API_KEY zweryfikowany pomyślnie.")
        except Exception as e:
            raise RuntimeError(f"OPENAI_API_KEY validation failed: {e}")

        # Walidacja HF_TOKEN (wymagany do generowania wideo przez HunyuanVideo)
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN is empty")
            logger.info("✅ Klucz HF_TOKEN zweryfikowany pomyślnie.")
        except Exception as e:
            raise RuntimeError(f"HF_TOKEN validation failed: {e}")
    else:
        logger.info("🧪 DRY_RUN lub TESTING włączony - pomijam twardą walidację kluczy API.")

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
    """Retry helper z exponential backoff dla wywołań zewnętrznych API."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            retryable = _is_retryable_exception(exc)
            wait_s = base_delay ** attempt
            if attempt < max_retries - 1 and retryable:
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
        raw_data["targetDuration"] = 12
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

def get_gemini_client():
    """Inicjalizacja klienta Google Genai"""
    return genai.Client(
        http_options={"api_version": "v1beta"}, 
        api_key=os.getenv("GEMINI_API_KEY")
    )

def call_gemini_with_cache(prompt, cache_key_prefix, max_retries=2):
    """Call Gemini API with STABLE caching to minimize costs."""
    # Użycie MD5 zamiast losowego hash(), aby cache przetrwał restarty serwera
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
    cache_key = f"gemini:{cache_key_prefix}:{prompt_hash}"

    # Check cache first
    cached = get_cached_api_response(cache_key)
    if cached:
        return cached

    # Call API
    client = get_gemini_client()
    for attempt in range(max_retries):
        try:
            response = client.generate_content(prompt)
            result = response.text

            # Cache the result
            cache_api_response(cache_key, result)
            return result

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"⚠️  Gemini API attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    raise RuntimeError("Gemini API failed after retries")

def generate_audio_with_elevenlabs(text, voice_id="pNInz6obpgDQGcFmaJgB"):
    """Helper do generowania audio z ElevenLabs."""
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    response = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2"
    )
    audio_data = b""
    for chunk in response:
        if chunk:
            audio_data += chunk
    return audio_data

def call_elevenlabs_with_cache(text, voice_id, cache_key_prefix, max_retries=2):
    """Call ElevenLabs API with STABLE caching to minimize costs."""
    # Użycie MD5 zamiast losowego hash(), aby cache przetrwał restarty serwera
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    cache_key = f"elevenlabs:{cache_key_prefix}:{text_hash}"

    # Check cache first
    cached = get_cached_api_response(cache_key)
    if cached:
        return cached

    # Call API
    for attempt in range(max_retries):
        try:
            audio_data = generate_audio_with_elevenlabs(text, voice_id)

            # Cache the result
            cache_api_response(cache_key, audio_data)
            return audio_data

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"⚠️  ElevenLabs API attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError("ElevenLabs API failed after retries")
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
# POBIERANIE WIDEO Z RETRY/BACKOFF
# ---------------------------------------------------------------------------

def download_video_with_backoff(video_uri, temp_path, max_retries=5):
    """Pobieranie wideo z chmury z automatycznym retry/backoff"""
    api_key = os.getenv("GEMINI_API_KEY")
    headers = {"x-goog-api-key": api_key} if api_key else {}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(video_uri, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"✅ Wideo pobrane: {temp_path}")
            return True
            
        except Exception as exc:
            countdown = 2 ** attempt
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Błąd pobierania (próba {attempt+1}/{max_retries}). Ponowna próba za {countdown}s...")
                time.sleep(countdown)
            else:
                logger.error(f"❌ Nie udało się pobrać wideo po {max_retries} próbach: {exc}")
                raise exc

# ---------------------------------------------------------------------------
# GENEROWANIE SEGMENTÓW WIDEO
# ---------------------------------------------------------------------------

def generate_video_segment(prompt, aspect_ratio="9:16"):
    """Generuje pojedynczy klip wideo i zwraca lokalną ścieżkę tymczasową"""
    logger.info(f"🎬 Generowanie segmentu: {prompt[:50]}...")
    client = get_gemini_client()
    
    operation = retry_with_backoff(
        "Veo generate_videos",
        lambda: client.models.generate_videos(
            model=MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=max(4, min(8, int(os.getenv("VEO_DURATION_SECONDS", "4" if FREE_TIER_MODE else "8")))),
                resolution="1080p",
            ),
        )
    )
    
    attempt = 0
    total_wait_time = 0
    MAX_TOTAL_WAIT_SECONDS = 300  # 5 minute hard limit
    while not operation.done and attempt < 60:
        wait_time = min(10 * (2 ** attempt), 120)  # Exponential: 10s, 20s, 40s, 80s, 120s
        if total_wait_time + wait_time > MAX_TOTAL_WAIT_SECONDS:
            raise TimeoutError(f"❌ Veo API timeout after {total_wait_time}s and {attempt} attempts")
        logger.info(f"⏳ Veo API polling attempt {attempt+1}/60, waiting {wait_time}s (total: {total_wait_time}s)...")
        time.sleep(wait_time)
        operation = client.operations.get(operation)
        total_wait_time += wait_time
        attempt += 1
        
    if not operation.done:
        raise TimeoutError(f"❌ Veo API timeout after {total_wait_time}s and {attempt} attempts")

    result = operation.result
    if not result or not hasattr(result, 'generated_videos') or not result.generated_videos:
        raise ValueError(f"❌ Veo API returned invalid response: {result}")

    video_uri = result.generated_videos[0].video.uri
    
    temp_file = os.path.join(tempfile.gettempdir(), f"seg_{os.urandom(4).hex()}.mp4")
    download_video_with_backoff(video_uri, temp_file)
    return temp_file

# ---------------------------------------------------------------------------
# AUDIO: LEKTOR (ElevenLabs)
# ---------------------------------------------------------------------------

def generate_audio_narration(narration_texts, job_id):
    """Generowanie MP3 z lektorem dla każdej sceny z systemem checkpointów"""
    if not ELEVENLABS_API_KEY:
        logger.error("❌ ElevenLabs API key not configured!")
        raise ValueError("ELEVENLABS_API_KEY not set")

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_files = {}

    for scene_key, text in narration_texts.items():
        audio_file = os.path.join(tempfile.gettempdir(), f"narration_{scene_key}_{job_id}.mp3")

        if os.path.exists(audio_file):
            logger.info(f"⏩ Lektor {scene_key} już istnieje. Pomijam ElevenLabs.")
            # Zakładamy, że funkcja get_audio_duration jest zdefiniowana w dalszej części kodu
            duration = get_audio_duration(audio_file)
            audio_files[scene_key] = {
                "path": audio_file,
                "duration": duration,
                "text": text
            }
            continue

        logger.info(f"🎙️ Generowanie lektora: {scene_key} ({len(text)} znaków)")

        try:
            audio_stream = None
            try:
                logger.info(f"🎙️ Trying text_to_speech.convert()...")
                audio_stream = retry_with_backoff(
                    f"ElevenLabs text_to_speech.convert ({scene_key})",
                    lambda: client.text_to_speech.convert(
                        text=text,
                        voice_id="pNInz6obpgDQGcFmaJgB",
                        model_id="eleven_multilingual_v2"
                    )
                )
            except Exception as e1:
                logger.warning(f"⚠️ text_to_speech.convert() failed: {e1}. Trying generate()...")
                try:
                    audio_stream = retry_with_backoff(
                        f"ElevenLabs generate ({scene_key})",
                        lambda: client.generate(
                            text=text,
                            voice="Adam",
                            model="eleven_multilingual_v2"
                        )
                    )
                    logger.info(f"✅ Fallback to generate() worked!")
                except Exception as e2:
                    logger.error(f"❌ Both methods failed: convert={e1}, generate={e2}")
                    raise e2
            
            with open(audio_file, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Lektor dla {scene_key} zapisany pomyślnie.")
            duration = get_audio_duration(audio_file)
            
            audio_files[scene_key] = {
                "path": audio_file,
                "duration": duration,
                "text": text
            }

        except Exception as e:
            logger.error(f"❌ Błąd generowania lektora dla {scene_key}: {e}")
            raise 

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
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
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
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
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

def calculate_video_speed(audio_files, target_duration=15):
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
            '-af', f"atempo={speed}",
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2',
            '-c:a', 'aac',
            output_file
        ]
        
        logger.info(f"  ⏱️ Segment {i}: {speed:.2f}x")
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        
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
        '-c', 'copy',
        concat_output
    ]
    subprocess.run(ffmpeg_concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    logger.info(f"✅ Wideo połączone: {concat_output}")
    
    logger.info("🎙️ Etap 2: Miksowanie audio (lektory)...")
    
    combined_audio = os.path.join(tempfile.gettempdir(), f"combined_audio_{job_id}.mp3")
    
    audio_list_file = os.path.join(tempfile.gettempdir(), f"audio_list_{job_id}.txt")
    with open(audio_list_file, "w") as f:
        for scene_key in ["hook", "problem", "rozwiązanie"]:
            if scene_key in audio_files:
                f.write(f"file '{audio_files[scene_key]['path']}'\n")
    
    ffmpeg_audio_concat = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', audio_list_file,
        '-c:a', 'libmp3lame', '-q:a', '4',
        combined_audio
    ]
    subprocess.run(ffmpeg_audio_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    logger.info(f"✅ Lektory połączone: {combined_audio}")
    
    logger.info("🎨 Etap 3: Miksowanie wideo + audio + napisy...")
    
    # BŁĄD LOGICZNY USUNIĘTY: Skoro speed robimy w innej funkcji, tu dajemy zwykłe kopiowanie strumienia video (bez setpts)
    # Zostawiamy po prostu wejście wideo bez modyfikacji czasu, żeby nie podwoić przyspieszenia.
    video_filter = "[0:v]copy" if srt_file is None else "[0:v]format=yuv420p"
    
    # Budujemy łańcuch filtrów
    filters = []
    
    if srt_file and os.path.exists(srt_file):
        srt_path_escaped = srt_file.replace("\\", "\\\\").replace(":", "\\:")
        # Eleganckie, wyraźne napisy dopasowane do profesjonalnego brandingu
        filters.append(f"subtitles='{srt_path_escaped}':force_style='FontSize=28,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0'")
        logger.info(f"✅ Napisy będą wypalane")
    
    watermark_text = "raport-finansowy24.pl"
    filters.append(f"drawtext=text='{watermark_text}':x=w-text_w-20:y=h-text_h-20:fontsize=24:fontcolor=white@0.7:box=1:boxcolor=black@0.5")
    
    # Łączymy filtry wideo przecinkami
    final_video_filter = ",".join(filters)
    
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
    
    logger.info("🔄 Kodowanie finale (może potrwać trochę)...")
    subprocess.run(ffmpeg_final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    logger.info(f"✅ Finalne wideo: {output_path}")
    
    for file in [list_file_path, audio_list_file, concat_output, combined_audio]:
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

        prompts = {
            "hook": f"Dynamic cinematic shot, extreme close up, shock and stress, concept of {topic}, corporate finance style, 4k, professional",
            "problem": f"A person looking anxiously at bills and charts on a screen, dark moody lighting, financial stress, 4k, professional",
            "rozwiązanie": f"Bright clean studio lighting, a smartphone screen displaying green rising financial growth charts, relief, 4k, professional"
        }

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
                    logger.info(f"⏩ Scena {key.upper()} już istnieje – pomijam Veo.")
                    segment_files.append(stable_path)
                    continue
                logger.info(f"⚠️  Plik sceny {key} zaginął ze STORAGE_DIR – regeneruję...")

            current_stage = stage_name
            save_checkpoint(job_id, current_stage, data={"topic": topic, "key": key})
            logger.info(f"🎥 Generowanie sceny: {key.upper()}")
            
            file_path = generate_video_segment(prompt_text, aspect_ratio)

            import shutil
            shutil.copy2(file_path, stable_path)
            if os.path.exists(file_path):
                os.remove(file_path)

            segment_files.append(stable_path)
            logger.info(f"✅ Scena {key} gotowa")

        logger.info(f"✅ Wszystkie 3 sceny gotowe")

        check_job_timeout()
        if not _stage_done("narration"):
            current_stage = "narration"
            save_checkpoint(job_id, current_stage, data={"topic": topic})
            logger.info("🎙️ Generowanie lektora (ElevenLabs)...")

        narration_texts = custom_narration if custom_narration else NARRATION_TEMPLATES
        audio_files_dict = generate_audio_narration(narration_texts, job_id)
        logger.info(f"✅ Lektor wygenerowany")

        check_job_timeout()
        srt_file = None
        
        # Generowanie napisów
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
                subprocess.run(ffmpeg_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
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

        target_duration = int(raw_data.get("targetDuration", 15))
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
            '-c', 'copy',
            final_with_endscreen
        ]
        
        # ZABEZPIECZENIE: Łapanie błędu krytycznego podczas finałowego sklejania
        try:
            subprocess.run(ffmpeg_final_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            os.replace(final_with_endscreen, final_output_path)
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
# CLEANUP STARYCH PLIKÓW
# ---------------------------------------------------------------------------

def cleanup_old_files(hours=24):
    """Czyszczenie plików starszych niż N godzin"""
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
        "elevenlabs": "✅ configured" if ELEVENLABS_API_KEY else "❌ not configured",
        "metrics": METRICS
    }), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """GET /metrics - proste metryki runtime."""
    auth_error = require_api_key()
    if auth_error:
        return auth_error
    return jsonify(METRICS), 200


@app.route("/", methods=["GET"])
def index():
    """API Info"""
    return jsonify({
        "name": "VeoVideo API",
        "version": "3.1.0",
        "features": {
            "veo_generation": "3 sceny (HOOK/PROBLEM/ROZWIĄZANIE)",
            "audio_narration": "ElevenLabs (głos Adam)",
            "subtitles": "Whisper API (automatyczna transkrypcja)",
            "watermark": "raport-finansowy24.pl (dolny róg)",
            "endscreen": "Plansza końcowa (3s)",
            "auto_length_adjustment": "Dopasowanie prędkości do lektora",
            "checkpoint_resume": "Pause/resume – wznowienie od ostatniego etapu"
        },
        "endpoints": {
            "POST /render-sequence": "Uruchomienie renderowania",
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

if __name__ == "__main__":
    logger.info("🚀 Startup VeoVideo API v3.1 (Napisy + Watermark + Plansza + Checkpoint/Resume)")
    logger.info(f"📁 Storage: {STORAGE_DIR}")
    logger.info(f"🗄️  Database: {DB_PATH}")
    logger.info(f"🎙️ ElevenLabs: {'✅' if ELEVENLABS_API_KEY else '❌'}")

    cleanup_old_files(hours=24)

    # Start auto-retry worker thread
    retry_thread = threading.Thread(target=auto_retry_worker, daemon=True)
    retry_thread.start()
    logger.info("✅ Auto-retry worker started")

    app.run(host="0.0.0.0", port=5000, threaded=True)
