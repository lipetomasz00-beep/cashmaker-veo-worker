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

# Konfiguracja z Twoich zasobów
MODEL = "veo-3.1-fast-generate-preview"
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "wiadrofilmy")

# Klient Google GenAI - inicjalizacja zgodnie z dokumentacją
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.getenv("GEMINI_API_KEY")
)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Pobranie danych z Make.com
        data = request.json
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Brak promptu w zapytaniu"}), 400

        print(f"Rozpoczynam proces generowania dla: {prompt}")

    operation = client.models.generate_videos(

            model=MODEL,

            prompt=prompt,

           config=types.GenerateVideosConfig(

                aspect_ratio="9:16",

                duration_seconds=8,

                resolution="1080p",

            ),

        )
        # 2. Czekanie na zakończenie operacji (Polling)
        while not operation.done:
            print("Wideo wciąż się generuje... sprawdzam za 10s")
            time.sleep(10)
            operation = client.operations.get(operation)

        # 3. Pobieranie wyniku (Metoda z Twojej dokumentacji)
        result = operation.result
        if not result or not result.generated_videos:
            return jsonify({"error": "Google nie zwrócił wygenerowanego wideo."}), 500

        generated_video = result.generated_videos[0]
        temp_path = f"/tmp/v_{int(time.time())}.mp4"

        # Oficjalne pobranie pliku na dysk serwera
        client.files.download(file=generated_video.video)
        generated_video.video.save(temp_path)
        print(f"Plik tymczasowy zapisany w: {temp_path}")

        # 4. Upload do Twojego koszyka 'wiadrofilmy' przy użyciu Omni-JSON
        sa_json_string = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not sa_json_string:
            return jsonify({"error": "Brak zmiennej GOOGLE_CREDENTIALS_JSON!"}), 500
            
        sa_info = json.loads(sa_json_string)
        creds = service_account.Credentials.from_service_account_info(sa_info)
        storage_client = storage.Client(credentials=creds, project=sa_info['project_id'])
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"final_video_{int(time.time())}.mp4"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(temp_path, content_type="video/mp4")

        # 5. Generowanie linku (Signed URL) dla Make.com na 7 dni
        video_url = blob.generate_signed_url(
            version="v4", 
            expiration=datetime.timedelta(days=7), 
            method="GET"
        )

        # Sprzątanie serwera
        os.remove(temp_path)
        
        print(f"Sukces! Link do wideo: {video_url}")
        return jsonify({"status": "success", "video_url": video_url}), 200

    except Exception as e:
        print(f"KRYTYCZNY BŁĄD: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start serwera na porcie Railway
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
