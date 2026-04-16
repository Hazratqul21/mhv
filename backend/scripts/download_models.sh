#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MIYA - Model Downloader
# Downloads all required GGUF models from Hugging Face
# =============================================================================

MODELS_DIR="${1:-$(dirname "$0")/../models}"
mkdir -p "$MODELS_DIR"

HF_BASE="https://huggingface.co"

declare -A MODELS=(
    # Orchestrator: Qwopus 3.5-27B (Claude Opus reasoning distilled)
    ["qwen3.5-27b-opus-q4_k_m.gguf"]="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/resolve/main/Qwen3.5-27B.Q4_K_M.gguf"
    # Code: Qwen2.5-Coder-14B (bartowski single-file)
    ["qwen2.5-coder-14b-instruct-q4_k_m.gguf"]="bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
    # Chat: Mistral-7B-Instruct (bartowski — MistralAI repo needs auth)
    ["mistral-7b-instruct-v0.3-q4_k_m.gguf"]="bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    # Creative: Mythos Prime (myth, story, philosophy, worldbuilding)
    ["mythos-prime-q4_k_m.gguf"]="ronniealfaro/mythos-prime/resolve/main/mythos.Q4_K_M.gguf"
    # Vision: LLaVA-v1.6-Mistral-7B
    ["llava-v1.6-mistral-7b-q4_k_m.gguf"]="cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf"
    # Embeddings: nomic-embed-text-v1.5
    ["nomic-embed-text-v1.5.Q6_K.gguf"]="nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q6_K.gguf"
    # Uncensored Orchestrator: Qwopus abliterated (no refusal, no safety filters)
    ["qwen3.5-27b-opus-abliterated-q4_k_m.gguf"]="mradermacher/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-GGUF/resolve/main/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated.Q4_K_M.gguf"
    # Legacy Orchestrator (fallback, bartowski single-file)
    ["qwen2.5-14b-instruct-q4_k_m.gguf"]="bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)

echo "========================================"
echo " MIYA Model Downloader"
echo " Target: $MODELS_DIR"
echo "========================================"
echo ""

TOTAL=${#MODELS[@]}
CURRENT=0

for filename in "${!MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    filepath="$MODELS_DIR/$filename"
    url="$HF_BASE/${MODELS[$filename]}"

    echo "[$CURRENT/$TOTAL] $filename"

    if [ -f "$filepath" ]; then
        echo "  ✓ Already exists, skipping."
        continue
    fi

    echo "  ↓ Downloading from $url ..."
    curl -L --progress-bar -o "$filepath.tmp" "$url"
    mv "$filepath.tmp" "$filepath"
    echo "  ✓ Done."
    echo ""
done

echo "========================================"
echo " All models downloaded!"
echo ""
echo " Total disk usage:"
du -sh "$MODELS_DIR"
echo ""
echo " Files:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no .gguf files found)"
echo "========================================"
