import os
import time
import json
import logging
import shutil
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = "veo-3.1-lite-generate-preview"
STORAGE_DIR = '/app/data'
POLLING_INTERVAL = 10
MAX_POLLING_ATTEMPTS = 60  # Reduced to 10 minutes (more reasonable for HTTP)
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB max
REQUEST_TIMEOUT = 120  # 2 minutes timeout for HTTP requests

# Ensure local storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

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

def save_video_locally(temp_path, video_id):
    """Save video to local persistent storage"""
    logger.info(f"Saving video to local storage...")
    start_time = time.time()

    try:
        # Verify file exists and is not empty
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temporary file not found: {temp_path}")

        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise ValueError("Downloaded video file is empty")

        logger.info(f"Saving file of size {file_size} bytes...")

        filename = f"veo_render_{video_id}.mp4"
        dest_path = os.path.join(STORAGE_DIR, filename)
        shutil.copy2(temp_path, dest_path)

        elapsed = time.time() - start_time
        logger.info(f"Video saved locally in {elapsed:.2f}s: {filename}")

        return filename

    except Exception as e:
        logger.error(f"Failed to save video locally: {str(e)}")
        raise

@app.route("/render", methods=["POST"])
def render():
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

        # Handle aspectRatio parameter
        aspect_ratio = data.get("aspectRatio", "9:16")
        valid_aspect_ratios = {"9:16", "16:9", "1:1"}
        if aspect_ratio not in valid_aspect_ratios:
            logger.warning(f"Invalid aspectRatio '{aspect_ratio}', defaulting to '9:16'")
            aspect_ratio = "9:16"

        logger.info(f"Starting video generation for prompt: {prompt[:50]}...")

        # Generate video
        try:
            operation = client.models.generate_videos(
                model=MODEL,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
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
        video_id = f"{int(time.time())}_{os.urandom(4).hex()}"
        try:
            temp_path = f"/tmp/v_{video_id}.mp4"
            download_video_from_uri(generated_video.video.uri, temp_path)
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            return jsonify({"error": f"Download failed: {str(e)}"}), 502

        # Save to local storage
        try:
            filename = save_video_locally(temp_path, video_id)
        except Exception as e:
            logger.error(f"Failed to save video locally: {str(e)}")
            return jsonify({"error": f"Storage failed: {str(e)}"}), 502

        video_url = f"https://{request.host}/videos/{filename}"
        logger.info(f"Video URL generated: {video_url}")

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


@app.route("/videos/<filename>", methods=["GET"])
def serve_video(filename):
    """Serve a generated video file from local storage"""
    # Security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        logger.warning(f"Rejected potentially unsafe filename: {filename}")
        return jsonify({"error": "Invalid filename"}), 400

    try:
        return send_from_directory(STORAGE_DIR, filename, as_attachment=True, mimetype="video/mp4")
    except FileNotFoundError:
        logger.warning(f"Video file not found: {filename}")
        return jsonify({"error": "Video not found"}), 404
    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
