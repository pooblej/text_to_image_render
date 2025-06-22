from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import os

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
auth_token = os.getenv("HF_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=auth_token,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

@app.get("/generate")
def generate(prompt: str = Query(..., description="Prompt for image generation")):
    image = pipe(prompt).images[0]
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
