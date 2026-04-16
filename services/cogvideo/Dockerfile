FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    diffusers accelerate transformers safetensors \
    fastapi uvicorn python-multipart

COPY server.py .

EXPOSE 8190

CMD ["python3", "server.py"]
