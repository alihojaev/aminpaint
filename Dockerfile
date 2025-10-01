FROM python:3.10-slim

# System deps for opencv and runtime stability
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Best-effort: предзагрузим модели в образ (не фейлим билд при отсутствии сети)
RUN mkdir -p /models && python - <<'PY' || true
print("Preloading models (best-effort)...")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="dreMaz/AnimeMangaInpainting",
        local_dir="/models/AnimeMangaInpainting",
        local_dir_use_symlinks=False
    )
    print("Cached dreMaz/AnimeMangaInpainting -> /models/AnimeMangaInpainting")
except Exception as e:
    print("WARN: could not cache AnimeMangaInpainting:", e)
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="runwayml/stable-diffusion-inpainting",
        local_dir="/models/sd-inpainting",
        local_dir_use_symlinks=False
    )
    print("Cached runwayml/stable-diffusion-inpainting -> /models/sd-inpainting")
except Exception as e:
    print("WARN: could not cache sd-inpainting:", e)
PY

# Add application code
COPY server.py /app/server.py
COPY rp_handler.py /app/rp_handler.py
COPY runpod.yaml /app/runpod.yaml

# Для serverless воркера uvicorn не нужен, стартуем runpod handler
CMD ["python", "-u", "rp_handler.py"]


