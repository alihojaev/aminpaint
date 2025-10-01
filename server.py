import base64
import io
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

import torch
from diffusers import AutoPipelineForInpainting


class InpaintRequest(BaseModel):
    image: str
    mask: str


app = FastAPI(title="AnimeManga Inpainting (LaMa)")


_PIPELINE = None


def _decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")


def _decode_base64_mask(b64_string: str, target_size: Optional[tuple[int, int]] = None) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        mask_bytes = base64.b64decode(b64_string)
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        if target_size and mask.size != target_size:
            mask = mask.resize(target_size, resample=Image.NEAREST)
        return mask
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 mask: {exc}")


def _encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    model_path = os.environ.get("MODEL_PATH", "/models/AnimeMangaInpainting")
    model_id = os.environ.get("MODEL_ID", "dreMaz/AnimeMangaInpainting")
    fallback_model_id = os.environ.get("FALLBACK_MODEL_ID", "runwayml/stable-diffusion-inpainting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Если локальная папка с моделью существует — используем её, иначе скачиваем с HF по ID
    source = model_path if os.path.isdir(model_path) else model_id

    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            source,
            torch_dtype=torch_dtype
        )
        pipe = pipe.to(device)
    except Exception:
        # Фолбэк на стабильный диффузор инпейнтинг, если репозиторий не формата diffusers
        pipe = AutoPipelineForInpainting.from_pretrained(
            fallback_model_id,
            torch_dtype=torch_dtype
        ).to(device)

    _PIPELINE = pipe
    return _PIPELINE


@app.post("/inpaint")
def inpaint(req: InpaintRequest) -> dict:
    pipe = _get_pipeline()

    image = _decode_base64_image(req.image)
    mask = _decode_base64_mask(req.mask, target_size=image.size)

    device = pipe.device.type if isinstance(pipe.device, torch.device) else str(pipe.device)

    try:
        if device == "cuda":
            with torch.autocast("cuda"):
                result = pipe(
                    prompt="",
                    image=image,
                    mask_image=mask,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                ).images[0]
        else:
            result = pipe(
                prompt="",
                image=image,
                mask_image=mask,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    b64 = _encode_image_to_base64(result)
    return {"result": b64}


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


