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
MAX_POLLING_ATTEMPTS = 60
MAX_VIDEO_SIZE = 500 * 1024 * 1024
REQUEST_TIMEOUT = 120
SEQUENCE_DELAY = 15

os.makedirs(STORAGE_DIR, exist_ok=True)

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

def generate_single_video(prompt, aspect_ratio="9:16"):
    """Generate a single video and return filename or None on error"""
    temp_path = None
    video_id = f"{int(time.time())}_{os.urandom(4).hex()}"
    
    try:
        logger.info(f"Generating video for prompt: {prompt[:50]}...")
        
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
            logger.error(f"Video generation timeout after {attempt * POLLING_INTERVAL}s")
            return None
        
        result = operation.result
        if not result or not hasattr(result, 'generated_videos') or not result.generated_videos:
            logger.error("No generated videos in result")
            return None
        
        generated_video = result.generated_videos[0]
        if not generated_video or not hasattr(generated_video, 'video') or not generated_video.video:
            logger.error("Invalid video object in result")
            return None
        
        if not hasattr(generated_video.video, 'uri') or not generated_video.video.uri:
            logger.error("No URI in generated video")
            return None
        
        temp_path = f"/tmp/v_{video_id}.mp4"
        download_video_from_uri(generated_video.video.uri, temp_path)
        
        filename = save_video_locally(temp_path, video_id)
        logger.info(f"Video generated successfully: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

@app.route("/render", methods=["POST"])
def render():
    temp_path = None
    start_time = time.time()

    try:
        data = request.json
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400

        prompt = data.get("prompt", "").strip()
        if not prompt:
            logger.warning("Missing or empty prompt in request")
            return jsonify({"error": "Missing or empty prompt"}), 400

        aspect_ratio = data.get("aspectRatio", "9:16")
        valid_aspect_ratios = {"9:16", "16:9", "1:1"}
        if aspect_ratio not in valid_aspect_ratios:
            logger.warning(f"Invalid aspectRatio '{aspect_ratio}', defaulting to '9:16'")
            aspect_ratio = "9:16"

        logger.info(f"Starting video generation for prompt: {prompt[:50]}...")

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

        video_id = f"{int(time.time())}_{os.urandom(4).hex()}"
        try:
            temp_path = f"/tmp/v_{video_id}.mp4"
            download_video_from_uri(generated_video.video.uri, temp_path)
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            return jsonify({"error": f"Download failed: {str(e)}"}), 502

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
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

@app.route("/render-sequence", methods=["POST"])
def render_sequence():
    """Generate multiple videos sequentially with rate limit protection."""
    start_time = time.time()
    
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompts = data.get("prompts", [])
        if not prompts or not isinstance(prompts, list):
            logger.warning("Missing or invalid prompts list")
            return jsonify({"error": "Missing or invalid prompts list"}), 400
        
        if len(prompts) == 0:
            return jsonify({"error": "Prompts list is empty"}), 400
        
        prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        if not prompts:
            return jsonify({"error": "All prompts are empty"}), 400
        
        aspect_ratio = data.get("aspectRatio", "9:16")
        valid_aspect_ratios = {"9:16", "16:9", "1:1"}
        if aspect_ratio not in valid_aspect_ratios:
            logger.warning(f"Invalid aspectRatio '{aspect_ratio}', defaulting to '9:16'")
            aspect_ratio = "9:16"
        
        logger.info(f"Starting sequential generation of {len(prompts)} videos...")
        
        results = []
        successful = 0
        failed = 0
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing video {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                filename = generate_single_video(prompt, aspect_ratio)
                
                if filename:
                    video_url = f"https://{request.host}/videos/{filename}"
                    results.append({
                        "prompt": prompt,
                        "filename": filename,
                        "url": video_url
                    })
                    successful += 1
                    logger.info(f"Video {i+1} generated successfully: {filename}")
                else:
                    results.append({
                        "prompt": prompt,
                        "error": "Video generation failed"
                    })
                    failed += 1
                    logger.warning(f"Video {i+1} generation failed")
                
                if i < len(prompts) - 1:
                    logger.info(f"Waiting {SEQUENCE_DELAY}s before next generation...")
                    time.sleep(SEQUENCE_DELAY)
            
            except Exception as e:
                logger.error(f"Error processing video {i+1}: {str(e)}")
                results.append({
                    "prompt": prompt,
                    "error": str(e)
                })
                failed += 1
                
                if i < len(prompts) - 1:
                    time.sleep(SEQUENCE_DELAY)
        
        elapsed = time.time() - start_time
        logger.info(f"Sequence generation complete: {successful} successful, {failed} failed in {elapsed:.2f}s")
        
        return jsonify({
            "status": "success",
            "total": len(prompts),
            "successful": successful,
            "failed": failed,
            "duration_seconds": round(elapsed, 2),
            "videos": results
        }), 200
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {str(e)}")
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        logger.error(f"Unexpected error in sequence generation: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route("/videos/<filename>", methods=["GET"])
def serve_video(filename):
    """Serve a generated video file from local storage"""
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
