import os
import time
import json
import datetime
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.cloud import storage
from google.oauth2 import service_account

app = Flask(__name__)

MODEL = "veo-3.1-fast-generate-preview"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

@app.route("/generate", methods=["POST"])
def generate_video():
    try:
        data = request.json
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "Brak promptu"}), 400

        video_config = types.GenerateVideosConfig(
            aspect_ratio="9:16",
            number_of_videos=1,
            duration_seconds=8,
            resolution="720p",
        )

        operation = client.models.generate_videos(
            model=MODEL,
            source=types.VideoGenerationSource(
                prompt=prompt,
            ),
            config=video_config,
        )

        while not operation.done:
            print("Renderowanie w toku...")
            time.sleep(10)
            operation = client.operations.get(operation)

        result = operation.result
        if not result or not result.generated_videos:
            return jsonify({"error": "Błąd podczas generowania wideo."}), 500

        generated_video = result.generated_videos[0]
        
        temp_filename = f"/tmp/veo_{int(time.time())}.mp4"
        client.files.download(file=generated_video.video)
        generated_video.video.save(temp_filename)

        sa_json_string = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not sa_json_string:
            return jsonify({"error": "Brak konfiguracji GOOGLE_CREDENTIALS_JSON na serwerze."}), 500
            
        sa_info = json.loads(sa_json_string)
        credentials = service_account.Credentials.from_service_account_info(sa_info)
        
        storage_client = storage.Client(credentials=credentials, project=sa_info.get("project_id"))
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        bucket = storage_client.bucket(bucket_name)
        
        blob_name = f"shorts_{int(time.time())}.mp4"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_filename, content_type="video/mp4")
        
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(days=7),
            method="GET"
        )

        os.remove(temp_filename)

        return jsonify({
            "status": "success",
            "video_url": signed_url
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
