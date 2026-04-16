FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        build-essential cmake git curl \
        ffmpeg libsm6 libxext6 \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies (layer cache)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
    && pip install -r requirements.txt

# Install playwright browsers
RUN python -m playwright install --with-deps chromium

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r miya && useradd -r -g miya -d /app -s /sbin/nologin miya \
    && mkdir -p /app/data /app/models \
    && chown -R miya:miya /app

USER miya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
