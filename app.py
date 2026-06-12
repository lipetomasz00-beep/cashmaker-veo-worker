import gradio as gr
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
import os

model_id = "Wan-AI/Wan2.1-T2V-1.3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
pipe.to(device)

def generate_video(prompt, width=480, height=480, num_frames=81, steps=50):
    if not prompt or len(prompt) < 3:
        return None, "Prompt za krótki!"
    try:
        frames = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
        ).frames[0]
        path = "/tmp/output.mp4"
        export_to_video(frames, path, fps=24)
        return path, "OK"
    except Exception as e:
        return None, f"Błąd: {str(e)}"

demo = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A cat playing guitar..."),
        gr.Slider(320, 832, 480, 16, label="Width"),
        gr.Slider(320, 832, 480, 16, label="Height"),
        gr.Slider(16, 81, 81, 1, label="Frames (81 = ~3.4s)"),
        gr.Slider(20, 100, 50, 1, label="Steps"),
    ],
    outputs=[gr.Video(label="Video"), gr.Textbox(label="Status")],
    title="Wan2.1-T2V-1.3B"
)
demo.launch()
