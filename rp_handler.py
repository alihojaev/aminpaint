import base64
import io
import os
from typing import Optional, Dict, Any

import runpod
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting


_PIPELINE = None


def _decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _decode_base64_mask(b64_string: str, target_size: Optional[tuple[int, int]] = None) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    mask_bytes = base64.b64decode(b64_string)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if target_size and mask.size != target_size:
        mask = mask.resize(target_size, resample=Image.NEAREST)
    return mask


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    source = model_path if os.path.isdir(model_path) else model_id
    pipe = AutoPipelineForInpainting.from_pretrained(source, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    _PIPELINE = pipe
    return _PIPELINE


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless handler.
    Expects event["input"] with keys: image (base64), mask (base64),
    optional: prompt, guidance_scale, num_inference_steps, seed.
    """
    try:
        inputs = event.get("input") or {}
        image_b64 = inputs["image"]
        mask_b64 = inputs["mask"]

        prompt: str = inputs.get("prompt", "")
        guidance_scale: float = float(inputs.get("guidance_scale", 7.5))
        num_inference_steps: int = int(inputs.get("num_inference_steps", 30))
        seed = inputs.get("seed")

        image = _decode_base64_image(image_b64)
        mask = _decode_base64_mask(mask_b64, target_size=image.size)

        pipe = _get_pipeline()

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
            except Exception:
                generator = None

        device_type = pipe.device.type if isinstance(pipe.device, torch.device) else str(pipe.device)

        if device_type == "cuda":
            with torch.autocast("cuda"):
                result = pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
        else:
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

        return {"result": _encode_image_to_base64(result)}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


