FROM python:3.10-slim

# System deps for opencv and runtime stability
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
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

# Pre-download model to speed up cold starts
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="dreMaz/AnimeMangaInpainting",
    local_dir="/models/AnimeMangaInpainting",
    local_dir_use_symlinks=False
)
print("Model downloaded to /models/AnimeMangaInpainting")
PY

# Add application code
COPY server.py /app/server.py
COPY runpod.yaml /app/runpod.yaml

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]


