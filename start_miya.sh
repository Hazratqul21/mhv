#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MIYA — One-click startup & self-test
# Siz qaytganingizda shu scriptni ishga tushiring:
#   cd ~/Desktop/miya && ./start_miya.sh
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/.venv"
BACKEND="$PROJECT_DIR/backend"
MODELS="$BACKEND/models"

echo "========================================"
echo "  MIYA AI — Startup & Self-Test"
echo "========================================"
echo ""

# ── 1. Activate venv ─────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "❌ Virtual environment not found at $VENV"
    exit 1
fi
source "$VENV/bin/activate"
echo "✅ Virtual environment activated"

# ── 2. Check models ─────────────────────────────────────────────────────
echo ""
echo "📦 Checking models..."
REQUIRED_MODELS=(
    "mistral-7b-instruct-v0.3-q4_k_m.gguf"
    "qwen3.5-27b-opus-q4_k_m.gguf"
    "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    "llava-v1.6-mistral-7b-q4_k_m.gguf"
    "nomic-embed-text-v1.5.Q6_K.gguf"
)

ALL_OK=true
for model in "${REQUIRED_MODELS[@]}"; do
    filepath="$MODELS/$model"
    if [ -f "$filepath" ]; then
        size=$(du -h "$filepath" | cut -f1)
        echo "  ✅ $model ($size)"
    else
        echo "  ❌ $model — NOT FOUND"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "⚠️  Some models are missing. Run download first:"
    echo "  cd $BACKEND/scripts && ./download_models.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ── 3. Kill any existing MIYA processes ──────────────────────────────────
echo ""
echo "🔄 Checking for running instances..."
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "  Stopping existing backend on port 8000..."
    kill $(lsof -ti:8000) 2>/dev/null || true
    sleep 1
fi
if lsof -ti:7860 >/dev/null 2>&1; then
    echo "  Stopping existing frontend on port 7860..."
    kill $(lsof -ti:7860) 2>/dev/null || true
    sleep 1
fi

# ── 4. Quick self-test ───────────────────────────────────────────────────
echo ""
echo "🧪 Running quick self-test..."
cd "$BACKEND"
python3 -c "
from app.core.llm_engine import LLMEngine
from app.core.orchestrator import MiyaOrchestrator
from app.core.event_bus import EventBus
from app.memory import MemoryAdapter
from app.tools.registry import ToolRegistry
from app.tools.system.host_shell_tool import HostShellTool
from app.tools.search.duckduckgo_tool import DuckDuckGoTool
from app.agents.search_agent import SearchAgent
print('  ✅ All imports OK')
"

# ── 5. Start backend ────────────────────────────────────────────────────
echo ""
echo "🚀 Starting MIYA backend (port 8000)..."
cd "$BACKEND"
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    > "$PROJECT_DIR/logs/backend.log" 2>&1 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "  Waiting for backend..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "  ✅ Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ⚠️  Backend didn't start in 30s. Check logs/backend.log"
    fi
    sleep 1
done

# ── 6. Start frontend ───────────────────────────────────────────────────
echo ""
echo "🖥️  Starting MIYA desktop UI (port 7860)..."
cd "$PROJECT_DIR/frontend/desktop"
nohup python3 main.py --port 7860 --api-url http://localhost:8000 \
    > "$PROJECT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

sleep 3

# ── 7. Done ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  ✅ MIYA is running!"
echo ""
echo "  🌐 UI:      http://localhost:7860"
echo "  🔧 API:     http://localhost:8000"
echo "  📊 Health:  http://localhost:8000/health"
echo ""
echo "  Backend log:  $PROJECT_DIR/logs/backend.log"
echo "  Frontend log: $PROJECT_DIR/logs/frontend.log"
echo ""
echo "  To stop:  kill $BACKEND_PID $FRONTEND_PID"
echo "========================================"
