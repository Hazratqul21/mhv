# MIYA - AI Models

Download all required GGUF models by running:

```bash
bash scripts/download_models.sh
```

## Required Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Qwen2.5-14B-Instruct | `qwen2.5-14b-instruct-q4_k_m.gguf` | ~9 GB | Orchestrator |
| Qwen2.5-Coder-14B | `qwen2.5-coder-14b-instruct-q4_k_m.gguf` | ~9 GB | Code tasks |
| Mistral-7B-Instruct v0.3 | `mistral-7b-instruct-v0.3-q4_k_m.gguf` | ~4.4 GB | Fast chat |
| LLaVA-v1.6-Mistral-7B | `llava-v1.6-mistral-7b-q4_k_m.gguf` | ~7.4 GB | Vision |
| nomic-embed-text-v1.5 | `nomic-embed-text-v1.5.Q6_K.gguf` | ~274 MB | Embeddings |

**Total disk space needed: ~30 GB**

## Manual Download

If the script doesn't work, download from Hugging Face directly:

```bash
# Example with curl
curl -L -o models/qwen2.5-14b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf
```

## Minimum Setup

For a quick start with limited GPU memory, download only:

1. `mistral-7b-instruct-v0.3-q4_k_m.gguf` (chat)
2. `nomic-embed-text-v1.5.Q6_K.gguf` (embeddings)

Then set `ORCHESTRATOR_MODEL=mistral-7b-instruct-v0.3-q4_k_m.gguf` in `.env`.
