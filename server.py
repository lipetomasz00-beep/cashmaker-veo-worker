import os
import time
import json
import datetime
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.cloud import storage
from google.oauth2 import service_account
import requests

app = Flask(__name__)

MODEL = "veo-3.1-fast-generate-preview"
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "wiadrofilmy")
POLLING_INTERVAL = 10
MAX_POLLING_ATTEMPTS = 360  # 1 hour max

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.getenv("GEMINI_API_KEY")
)

def download_video_from_uri(video_uri, temp_path):
    """Download video from Google Cloud URI"""
    response = requests.get(video_uri, timeout=300)
    response.raise_for_status()
    with open(temp_path, 'wb') as f:
        f.write(response.content)

def upload_to_gcs(temp_path):
    """Upload video to Google Cloud Storage"""
    sa_json_string = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not sa_json_string:
        raise ValueError("Missing GOOGLE_CREDENTIALS_JSON environment variable")
    
    sa_info = json.loads(sa_json_string)
    creds = service_account.Credentials.from_service_account_info(sa_info)
    storage_client = storage.Client(credentials=creds, project=sa_info['project_id'])
    
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f"final_video_{int(time.time())}.mp4"
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(temp_path, content_type="video/mp4")
    return blob

@app.route("/generate", methods=["POST"])
def generate():
    temp_path = None
    try:
        # Validate request data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Missing or empty prompt"}), 400

        print(f"Starting video generation for: {prompt}")

        # Generate video
        operation = client.models.generate_videos(
            model=MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                duration_seconds=8,
                resolution="1080p",
            ),
        )

        # Poll for completion with timeout
        attempt = 0
        while not operation.done and attempt < MAX_POLLING_ATTEMPTS:
            print(f"Video generating... checking again in {POLLING_INTERVAL}s (attempt {attempt + 1})")
            time.sleep(POLLING_INTERVAL)
            operation = client.operations.get(operation)
            attempt += 1

        if not operation.done:
            return jsonify({"error": "Video generation timeout"}), 504

        # Get result (with parentheses!)
        result = operation.result()
        if not result or not result.generated_videos:
            return jsonify({"error": "No video generated"}), 500

        generated_video = result.generated_videos[0]
        if not generated_video.video or not generated_video.video.uri:
            return jsonify({"error": "Invalid video URI"}), 500

        # Download video
        temp_path = f"/tmp/v_{int(time.time())}.mp4"
        download_video_from_uri(generated_video.video.uri, temp_path)
        print(f"Video saved to: {temp_path}")

        # Upload to GCS
        blob = upload_to_gcs(temp_path)
        video_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(days=7),
            method="GET"
        )

        print(f"Success! Video URL: {video_url}")
        return jsonify({"status": "success", "video_url": video_url}), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in credentials"}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Download failed: {str(e)}"}), 502
    except Exception as e:
        print(f"Critical error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Always cleanup temp files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
