FROM python:3.10-slim

# System deps for opencv and runtime stability
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        unzip \
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

# Клонируем LaMa (saicinpainting) и ставим её зависимости
ENV WORKSPACE=/workspace
RUN git clone https://github.com/advimman/lama.git ${WORKSPACE}/lama && \
    pip install --no-cache-dir -r ${WORKSPACE}/lama/requirements.txt || true
    
# Установка минимальных/совместимых пакетов для инференса (поверх, на случай несовместимостей)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Best-effort: скачиваем Big-Lama веса (не фейлим билд, если сеть недоступна)
RUN mkdir -p ${WORKSPACE}/lama && \
    (curl -L "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" -o ${WORKSPACE}/lama/big-lama.zip \
     && unzip -q ${WORKSPACE}/lama/big-lama.zip -d ${WORKSPACE}/lama/ \
     && rm -f ${WORKSPACE}/lama/big-lama.zip) || true

ENV PYTHONPATH=${WORKSPACE}/lama:${PYTHONPATH}

# Add application code
COPY server.py /app/server.py
COPY rp_handler.py /app/rp_handler.py
COPY runpod.yaml /app/runpod.yaml

# Для serverless воркера uvicorn не нужен, стартуем runpod handler
CMD ["python", "-u", "rp_handler.py"]


