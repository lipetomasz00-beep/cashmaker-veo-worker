import os
import time
import json
import logging
import datetime
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.cloud import storage
from google.oauth2 import service_account
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = "veo-3.1-lite-generate-preview"
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "wiadrofilmy")
POLLING_INTERVAL = 10
MAX_POLLING_ATTEMPTS = 60  # Reduced to 10 minutes (more reasonable for HTTP)
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB max
REQUEST_TIMEOUT = 120  # 2 minutes timeout for HTTP requests

# Initialize GCS credentials once at startup
_gcs_credentials = None
_storage_client = None

def _initialize_gcs():
    """Initialize GCS client once"""
    global _gcs_credentials, _storage_client
    try:
        sa_json_string = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not sa_json_string:
            raise ValueError("Missing GOOGLE_CREDENTIALS_JSON environment variable")
        
        sa_info = json.loads(sa_json_string)
        _gcs_credentials = service_account.Credentials.from_service_account_info(sa_info)
        _storage_client = storage.Client(
            credentials=_gcs_credentials, 
            project=sa_info['project_id']
        )
        logger.info("GCS client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GCS: {str(e)}")
        raise

# Initialize on startup
try:
    client = genai.Client(
        http_options={"api_version": "v1beta"},
        api_key=os.getenv("GEMINI_API_KEY")
    )
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {str(e)}")
    raise

try:
    _initialize_gcs()
except Exception as e:
    logger.error(f"Failed to initialize at startup: {str(e)}")
    # Continue anyway, error will be caught during request

def download_video_from_uri(video_uri, temp_path):
    """Download video from Google Cloud URI with size validation"""
    logger.info(f"Starting video download from URI...")
    start_time = time.time()
    
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        headers = {"x-goog-api-key": api_key}
        response = requests.get(video_uri, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)       
        response.raise_for_status()
        
        # Check content length before downloading
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_VIDEO_SIZE:
            raise ValueError(f"Video size {content_length} exceeds maximum {MAX_VIDEO_SIZE}")
        
        bytes_downloaded = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > MAX_VIDEO_SIZE:
                        raise ValueError(f"Video download exceeded maximum size {MAX_VIDEO_SIZE}")
                    f.write(chunk)
        
        elapsed = time.time() - start_time
        logger.info(f"Video downloaded successfully: {bytes_downloaded} bytes in {elapsed:.2f}s")
        
    except requests.RequestException as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        raise

def upload_to_gcs(temp_path):
    """Upload video to Google Cloud Storage"""
    logger.info(f"Starting GCS upload...")
    start_time = time.time()
    
    try:
        if not _storage_client:
            raise ValueError("GCS client not initialized")
        
        # Verify file exists and is readable
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temporary file not found: {temp_path}")
        
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise ValueError("Downloaded video file is empty")
        
        logger.info(f"Uploading file of size {file_size} bytes...")
        
        bucket = _storage_client.bucket(BUCKET_NAME)
        blob_name = f"final_video_{int(time.time())}.mp4"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(temp_path, content_type="video/mp4")
        
        elapsed = time.time() - start_time
        logger.info(f"GCS upload completed in {elapsed:.2f}s: {blob_name}")
        
        return blob
        
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {str(e)}")
        raise

@app.route("/generate", methods=["POST"])
def generate():
    temp_path = None
    start_time = time.time()
    
    try:
        # Validate request data
        data = request.json
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "").strip()
        if not prompt:
            logger.warning("Missing or empty prompt in request")
            return jsonify({"error": "Missing or empty prompt"}), 400

        logger.info(f"Starting video generation for prompt: {prompt[:50]}...")

        # Generate video
        try:
            operation = client.models.generate_videos(
                model=MODEL,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio="9:16",
                    duration_seconds=8,
                    resolution="1080p",
                ),
            )
            logger.info(f"Video generation operation started: {operation.name}")
        except Exception as e:
            logger.error(f"Failed to start video generation: {str(e)}")
            raise

        # Poll for completion with timeout
        attempt = 0
        while not operation.done and attempt < MAX_POLLING_ATTEMPTS:
            logger.info(f"Polling video status (attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS})...")
            time.sleep(POLLING_INTERVAL)
            
            try:
                operation = client.operations.get(operation)
            except Exception as e:
                logger.error(f"Failed to get operation status: {str(e)}")
                raise
            
            attempt += 1

        if not operation.done:
            elapsed = time.time() - start_time
            logger.error(f"Video generation timeout after {elapsed:.2f}s")
            return jsonify({"error": "Video generation timeout"}), 504

        # Get result with proper error handling
        try:
            result = operation.result
        except Exception as e:
            logger.error(f"Failed to get operation result: {str(e)}")
            return jsonify({"error": f"Failed to retrieve generation result: {str(e)}"}), 500
        
        if not result:
            logger.error("Operation result is None")
            return jsonify({"error": "No result from video generation"}), 500
        
        if not hasattr(result, 'generated_videos') or not result.generated_videos:
            logger.error("No generated videos in result")
            return jsonify({"error": "No video generated"}), 500

        generated_video = result.generated_videos[0]
        if not generated_video or not hasattr(generated_video, 'video') or not generated_video.video:
            logger.error("Invalid video object in result")
            return jsonify({"error": "Invalid video object in result"}), 500
        
        if not hasattr(generated_video.video, 'uri') or not generated_video.video.uri:
            logger.error("No URI in generated video")
            return jsonify({"error": "Invalid video URI"}), 500

        # Download video
        try:
            temp_path = f"/tmp/v_{int(time.time())}_{os.urandom(4).hex()}.mp4"
            download_video_from_uri(generated_video.video.uri, temp_path)
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            return jsonify({"error": f"Download failed: {str(e)}"}), 502

        # Upload to GCS
        try:
            blob = upload_to_gcs(temp_path)
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}")
            return jsonify({"error": f"Upload failed: {str(e)}"}), 502

        # Generate signed URL
        try:
            video_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(days=7),
                method="GET"
            )
            logger.info(f"Signed URL generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {str(e)}")
            return jsonify({"error": f"Failed to generate URL: {str(e)}"}), 500

        elapsed = time.time() - start_time
        logger.info(f"Success! Video generated in {elapsed:.2f}s")
        return jsonify({"status": "success", "video_url": video_url}), 200

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {str(e)}")
        return jsonify({"error": "Invalid JSON format"}), 400
    except requests.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500
    finally:
        # Always cleanup temp files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
