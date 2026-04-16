#!/usr/bin/env bash
# =============================================================================
# MIYA WATCHDOG — Yuklanishni kuzatadi, tugagach server ishga tushiradi
#
# Ishga tushirish:
#   cd ~/Desktop/miya && nohup ./watchdog.sh &
#
# Log:
#   tail -f ~/Desktop/miya/logs/watchdog.log
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$PROJECT_DIR/backend"
MODELS="$BACKEND/models"
VENV="$PROJECT_DIR/.venv"
LOG="$PROJECT_DIR/logs/watchdog.log"

mkdir -p "$PROJECT_DIR/logs"

# Load .env for API keys and watchdog config
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "========================================="
log "  MIYA WATCHDOG STARTED"
log "========================================="

# ── Required models ──────────────────────────────────────────────────────
REQUIRED=(
    "qwen3.5-27b-opus-q4_k_m.gguf"
    "mistral-7b-instruct-v0.3-q4_k_m.gguf"
    "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    "llava-v1.6-mistral-7b-q4_k_m.gguf"
    "nomic-embed-text-v1.5.Q6_K.gguf"
)

# ── Phase 1: Wait for downloads ─────────────────────────────────────────
log "Phase 1: Waiting for model downloads..."

while true; do
    ALL_DONE=true
    STATUS=""

    for model in "${REQUIRED[@]}"; do
        if [ -f "$MODELS/$model" ]; then
            size=$(du -h "$MODELS/$model" 2>/dev/null | cut -f1)
            STATUS="$STATUS\n  ✅ $model ($size)"
        elif [ -f "$MODELS/$model.tmp" ]; then
            size=$(du -h "$MODELS/$model.tmp" 2>/dev/null | cut -f1)
            STATUS="$STATUS\n  ⏳ $model — downloading ($size)"
            ALL_DONE=false
        else
            STATUS="$STATUS\n  ⏸️  $model — waiting"
            ALL_DONE=false
        fi
    done

    log "Status:$STATUS"

    if [ "$ALL_DONE" = true ]; then
        log "✅ All required models downloaded!"
        break
    fi

    sleep 120
done

# ── Phase 2: Validate models ────────────────────────────────────────────
log ""
log "Phase 2: Validating model files..."

VALID=true
for model in "${REQUIRED[@]}"; do
    filepath="$MODELS/$model"
    if [ ! -f "$filepath" ]; then
        log "❌ $model — MISSING"
        VALID=false
        continue
    fi

    size_bytes=$(stat --format="%s" "$filepath" 2>/dev/null || echo "0")
    size=$(du -h "$filepath" 2>/dev/null | cut -f1)

    if [ "$size_bytes" -lt 50000000 ]; then
        log "⚠️  $model ($size) — too small, may be corrupted"
    else
        log "✅ $model ($size)"
    fi
done

# ── Phase 3: Activate venv ──────────────────────────────────────────────
log ""
log "Phase 3: Activating virtual environment..."

if [ ! -d "$VENV" ]; then
    log "❌ venv not found at $VENV — cannot continue"
    log "WATCHDOG STOPPED: no venv"
    exit 1
fi

source "$VENV/bin/activate"
log "✅ venv activated"

# ── Phase 4: Self-test (xatoga chidamli) ────────────────────────────────
log ""
log "Phase 4: Running self-test..."

SELFTEST_OK=true

python3 -c "
import sys
errors = []

tests = [
    ('LLMEngine', 'from app.core.llm_engine import LLMEngine'),
    ('EventBus', 'from app.core.event_bus import EventBus'),
    ('Orchestrator', 'from app.core.orchestrator import MiyaOrchestrator'),
    ('ToolRegistry', 'from app.tools.registry import ToolRegistry'),
    ('HostShellTool', 'from app.tools.system.host_shell_tool import HostShellTool'),
    ('DuckDuckGoTool', 'from app.tools.search.duckduckgo_tool import DuckDuckGoTool'),
    ('ChatAgent', 'from app.agents.chat_agent import ChatAgent'),
    ('CodeAgent', 'from app.agents.code_agent import CodeAgent'),
    ('SearchAgent', 'from app.agents.search_agent import SearchAgent'),
    ('ShellAgent', 'from app.agents.shell_agent import ShellAgent'),
    ('MemoryAdapter', 'from app.memory import MemoryAdapter'),
    ('AutoPilot', 'from app.core.autonomous.autopilot import AutoPilot'),
]

for name, stmt in tests:
    try:
        exec(stmt)
        print(f'  ✅ {name}')
    except Exception as e:
        print(f'  ❌ {name}: {e}')
        errors.append(name)

if errors:
    print(f'SELFTEST_ERRORS: {len(errors)} failed: {\", \".join(errors)}')
    sys.exit(1)
else:
    print('SELFTEST_OK')
    sys.exit(0)
" >> "$LOG" 2>&1
# ^^^ working directory bilan
cd "$BACKEND"
python3 -c "
import sys
errors = []
tests = [
    ('LLMEngine', 'from app.core.llm_engine import LLMEngine'),
    ('EventBus', 'from app.core.event_bus import EventBus'),
    ('Orchestrator', 'from app.core.orchestrator import MiyaOrchestrator'),
    ('ToolRegistry', 'from app.tools.registry import ToolRegistry'),
    ('HostShellTool', 'from app.tools.system.host_shell_tool import HostShellTool'),
    ('DuckDuckGoTool', 'from app.tools.search.duckduckgo_tool import DuckDuckGoTool'),
    ('ChatAgent', 'from app.agents.chat_agent import ChatAgent'),
    ('CodeAgent', 'from app.agents.code_agent import CodeAgent'),
    ('SearchAgent', 'from app.agents.search_agent import SearchAgent'),
    ('ShellAgent', 'from app.agents.shell_agent import ShellAgent'),
    ('MemoryAdapter', 'from app.memory import MemoryAdapter'),
    ('AutoPilot', 'from app.core.autonomous.autopilot import AutoPilot'),
]
for name, stmt in tests:
    try:
        exec(stmt)
        print(f'  OK: {name}')
    except Exception as e:
        print(f'  FAIL: {name}: {e}')
        errors.append(f'{name}:{e}')
if errors:
    sys.exit(1)
else:
    sys.exit(0)
" >> "$LOG" 2>&1

if [ $? -ne 0 ]; then
    log "⚠️  Some imports failed — see details above"
    log "⚠️  Attempting to start anyway (chat may still work)..."
    SELFTEST_OK=false
else
    log "✅ Self-test passed — all imports OK"
fi

# ── Phase 5: Kill old processes ─────────────────────────────────────────
log ""
log "Phase 5: Cleaning up old processes..."

if lsof -ti:8000 >/dev/null 2>&1; then
    kill $(lsof -ti:8000) 2>/dev/null || true
    log "  Killed old process on :8000"
    sleep 2
fi
if lsof -ti:7860 >/dev/null 2>&1; then
    kill $(lsof -ti:7860) 2>/dev/null || true
    log "  Killed old process on :7860"
    sleep 2
fi

# ── Phase 6: Start backend (retry up to 3 times) ────────────────────────
log ""
log "Phase 6: Starting backend..."

BACKEND_STARTED=false

for attempt in 1 2 3; do
    log "  Attempt $attempt/3..."

    cd "$BACKEND"
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
        >> "$PROJECT_DIR/logs/backend.log" 2>&1 &
    BACKEND_PID=$!
    log "  Backend PID: $BACKEND_PID"

    # Wait for health (up to 90 seconds — model loading takes time)
    for i in $(seq 1 45); do
        # Check if process is still alive
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            log "  ❌ Backend crashed on attempt $attempt"
            break
        fi

        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            BACKEND_STARTED=true
            break
        fi
        sleep 2
    done

    if [ "$BACKEND_STARTED" = true ]; then
        HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "unknown")
        log "✅ Backend is HEALTHY: $HEALTH"
        break
    fi

    # If failed, kill and retry
    kill $BACKEND_PID 2>/dev/null || true
    sleep 3

    if [ $attempt -lt 3 ]; then
        log "  Retrying..."
    fi
done

if [ "$BACKEND_STARTED" = false ]; then
    log "❌ Backend failed to start after 3 attempts"
    log "  Check: cat $PROJECT_DIR/logs/backend.log"
    log ""
    log "WATCHDOG FINISHED WITH ERRORS"
    log "Qaytganingizda logni ko'ring: cat ~/Desktop/miya/logs/watchdog.log"
    log "Backend logni ko'ring: cat ~/Desktop/miya/logs/backend.log"
    exit 1
fi

# ── Phase 7: Start frontend ─────────────────────────────────────────────
log ""
log "Phase 7: Starting frontend..."

cd "$PROJECT_DIR/frontend/desktop"
nohup python3 main.py --port 7860 --api-url http://localhost:8000 \
    >> "$PROJECT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
log "  Frontend PID: $FRONTEND_PID"

sleep 5

# Check if frontend is alive
if kill -0 $FRONTEND_PID 2>/dev/null; then
    log "✅ Frontend started"
else
    log "⚠️  Frontend may have crashed — check logs/frontend.log"
    log "  Backend is still running, you can use API directly"
fi

# ── Phase 8: Launch AI Watchdog monitor ─────────────────────────────────
log ""

WATCHDOG_AI_ENABLED="${WATCHDOG_AI_ENABLED:-false}"

if [ "$WATCHDOG_AI_ENABLED" = "true" ]; then
    # Check if API keys are set
    if [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]; then
        log "Phase 8: Starting AI Watchdog monitor..."
        nohup "$VENV/bin/python3" "$PROJECT_DIR/watchdog_ai.py" --monitor \
            >> "$PROJECT_DIR/logs/watchdog_ai.log" 2>&1 &
        AI_PID=$!
        log "  AI Watchdog PID: $AI_PID"
        log "  AI will monitor system every ${WATCHDOG_CHECK_INTERVAL:-120}s"
        log "  AI log: $PROJECT_DIR/logs/watchdog_ai.log"
    else
        log "Phase 8: AI Watchdog SKIPPED — no API keys in .env"
        AI_PID=""
    fi
else
    log "Phase 8: AI Watchdog DISABLED (set WATCHDOG_AI_ENABLED=true in .env)"
    AI_PID=""
fi

# ── Phase 9: Final status ───────────────────────────────────────────────
log ""
log "========================================="
log "  ✅ MIYA WATCHDOG COMPLETE"
log "========================================="
log ""
log "  🌐 UI:      http://localhost:7860"
log "  🔧 API:     http://localhost:8000"
log "  📊 Health:  http://localhost:8000/health"
log ""
log "  Backend PID:  $BACKEND_PID"
log "  Frontend PID: $FRONTEND_PID"
if [ -n "$AI_PID" ]; then
log "  AI Watch PID: $AI_PID"
fi
log ""
log "  Logs:"
log "    watchdog:    $LOG"
log "    backend:     $PROJECT_DIR/logs/backend.log"
log "    frontend:    $PROJECT_DIR/logs/frontend.log"
if [ -n "$AI_PID" ]; then
log "    ai_watchdog: $PROJECT_DIR/logs/watchdog_ai.log"
fi
log ""
log "  Qaytganingizda brauzerda: http://localhost:7860"
log "========================================="
