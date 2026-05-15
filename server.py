import os
import time
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

MODEL = "veo-3.1-fast-generate-preview"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

@app.route("/generate", methods=["POST"])
def generate_video():

    data = request.json
    prompt = data.get("prompt")

    video_config = types.GenerateVideosConfig(
        aspect_ratio="9:16",
        number_of_videos=1,
        duration_seconds=8,
        resolution="720p",
    )

    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        config=video_config,
    )

    while not operation.done:
        time.sleep(5)
        operation = client.operations.get(operation)

    result = operation.result

    generated_video = result.generated_videos[0]

    return jsonify({
    "status": "success",
    "video_url": generated_video.video.uri
})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
