FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
ENV HF_HOME=/models/hf_cache
ENV TRANSFORMERS_CACHE=/models/hf_cache

# Аргумент токена Hugging Face
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Устанавливаем Python, pip и git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN python3 -m pip install --upgrade pip

# PyTorch + torchvision с поддержкой CUDA 12.2
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Копируем зависимости проекта
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

CMD ["python3", "app.py"]
