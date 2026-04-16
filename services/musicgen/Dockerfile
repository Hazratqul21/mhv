FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    audiocraft transformers \
    fastapi uvicorn python-multipart soundfile

COPY server.py .

EXPOSE 8191

CMD ["python3", "server.py"]
