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

# Note: модель будет загружаться лениво при первом запросе (во время рантайма)

# Add application code
COPY server.py /app/server.py
COPY rp_handler.py /app/rp_handler.py
COPY runpod.yaml /app/runpod.yaml

# Для serverless воркера uvicorn не нужен, стартуем runpod handler
CMD ["python", "-u", "rp_handler.py"]


