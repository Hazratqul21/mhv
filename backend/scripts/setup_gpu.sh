#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MIYA - GPU Setup Verification
# Checks NVIDIA drivers, CUDA, and Docker GPU support
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; }

echo "========================================"
echo " MIYA GPU Setup Check"
echo "========================================"
echo ""

# 1. NVIDIA Driver
echo "── NVIDIA Driver ──"
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    pass "NVIDIA driver $DRIVER_VER"
    pass "GPU: $GPU_NAME ($GPU_MEM)"
else
    fail "nvidia-smi not found. Install NVIDIA drivers first."
    echo "  → Ubuntu: sudo apt install nvidia-driver-535"
    echo "  → Or visit: https://www.nvidia.com/drivers"
    exit 1
fi
echo ""

# 2. CUDA
echo "── CUDA Toolkit ──"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    pass "CUDA $CUDA_VER"
else
    warn "nvcc not found. CUDA toolkit not installed (may still work via Docker)."
    echo "  → Ubuntu: sudo apt install nvidia-cuda-toolkit"
fi
echo ""

# 3. Docker
echo "── Docker ──"
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
    pass "Docker $DOCKER_VER"
else
    fail "Docker not installed."
    echo "  → https://docs.docker.com/engine/install/"
    exit 1
fi

# 4. Docker Compose
if command -v docker compose &>/dev/null; then
    COMPOSE_VER=$(docker compose version --short 2>/dev/null)
    pass "Docker Compose $COMPOSE_VER"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_VER=$(docker-compose --version | awk '{print $4}' | tr -d ',')
    pass "Docker Compose (standalone) $COMPOSE_VER"
else
    fail "Docker Compose not found."
fi
echo ""

# 5. NVIDIA Container Toolkit
echo "── NVIDIA Container Toolkit ──"
if docker info 2>/dev/null | grep -q "nvidia"; then
    pass "NVIDIA runtime detected in Docker"
elif command -v nvidia-container-toolkit &>/dev/null || \
     [ -f /usr/bin/nvidia-container-runtime ]; then
    pass "NVIDIA Container Toolkit installed"
else
    warn "NVIDIA Container Toolkit may not be installed."
    echo "  → Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "  → Ubuntu:"
    echo "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
fi
echo ""

# 6. Quick GPU test in Docker
echo "── Docker GPU Test ──"
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    pass "Docker GPU passthrough works!"
else
    warn "Docker GPU test failed. Ensure nvidia-container-toolkit is configured."
fi
echo ""

echo "========================================"
echo " Setup check complete."
echo "========================================"
