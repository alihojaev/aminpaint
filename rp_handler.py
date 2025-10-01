import base64
import io
import os
from typing import Optional, Dict, Any

import runpod
from PIL import Image
import torch
import shutil
import subprocess
import uuid
from pathlib import Path


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
    fallback_model_id = os.environ.get("FALLBACK_MODEL_ID", "runwayml/stable-diffusion-inpainting")
    fallback_local_path = os.environ.get("FALLBACK_LOCAL_PATH", "/models/sd-inpainting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # порядок: локальная основная -> локальный фолбэк -> удалённая основная -> удалённый фолбэк
    if os.path.isdir(model_path):
        source = model_path
    elif os.path.isdir(fallback_local_path):
        source = fallback_local_path
    else:
        source = model_id
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(source, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
    except Exception:
        try:
            alt = fallback_local_path if os.path.isdir(fallback_local_path) else fallback_model_id
            pipe = AutoPipelineForInpainting.from_pretrained(alt, torch_dtype=torch_dtype).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load both primary and fallback models: {e}")
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

        workspace = Path("/workspace")
        lama_dir = workspace / "lama"
        model_dir = lama_dir / "big-lama"
        req_id = uuid.uuid4().hex[:12]
        input_root = workspace / "input" / req_id
        output_root = workspace / "output" / req_id
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        image = _decode_base64_image(image_b64).convert("RGB")
        mask = _decode_base64_mask(mask_b64, target_size=image.size)
        mask = mask.point(lambda x: 255 if x > 0 else 0)

        image_path = input_root / "image.png"
        mask_path = input_root / "image_mask.png"
        image.save(image_path)
        mask.save(mask_path)

        if not lama_dir.exists():
            return {"error": "LaMa code not found at /workspace/lama"}
        if not model_dir.exists():
            return {"error": "Big-Lama weights not found at /workspace/lama/big-lama"}

        cmd = [
            "python3",
            "bin/predict.py",
            f"model.path={model_dir}",
            f"indir={input_root}",
            f"outdir={output_root}",
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(lama_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            return {"error": f"LaMa failed: {e.stderr.decode(errors='ignore')[:1000]}"}

        result_path = None
        for p in output_root.rglob("*.png"):
            result_path = p
            break
        if not result_path:
            return {"error": "Result not found"}

        with open(result_path, "rb") as f:
            res_b64 = base64.b64encode(f.read()).decode()

        shutil.rmtree(input_root, ignore_errors=True)
        shutil.rmtree(output_root, ignore_errors=True)
        return {"result": res_b64}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


