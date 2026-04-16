# MIYA - Local AI Assistant Infrastructure

A complete multi-agent AI system that runs entirely on local hardware. No cloud APIs, no subscriptions — just your GPU and open-source models.

## Features

- **Multi-Agent Architecture** — Orchestrator routes queries to specialized agents (chat, code, vision, voice, search)
- **100% Local** — All models run on your NVIDIA GPU via llama.cpp
- **20+ Integrated Tools** — Search, browser automation, code execution, media processing, and more
- **Persistent Memory** — Vector search (ChromaDB), conversation history (SQLite), caching (Redis)
- **Real-time Streaming** — WebSocket support for token-by-token responses
- **Docker-First** — One command to start the entire stack
- **Extensible** — Add new tools and agents with minimal code

## Requirements

- **GPU**: NVIDIA RTX 5060 or better (CUDA-enabled, 16GB+ VRAM recommended)
- **RAM**: 32GB+
- **Storage**: 30GB+ for models, 2TB SSD recommended
- **OS**: Ubuntu 22.04+ (or Windows with WSL2)
- **Docker**: Docker Engine 24+ with NVIDIA Container Toolkit

## Quick Start

```bash
# 1. Clone and enter the project
cd miya

# 2. Verify GPU setup
make gpu-check

# 3. Download AI models (~30GB)
make models

# 4. Build and start services
make setup
make up

# 5. Open the web UI
open http://localhost:8000
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| MIYA API | 8000 | FastAPI backend |
| ChromaDB | 8001 | Vector database |
| Redis | 6379 | Cache |
| SearXNG | 8080 | Search engine |
| MinIO | 9000/9001 | Object storage |
| ComfyUI | 8188 | Image generation (optional) |
| Desktop UI | 7860 | Gradio interface (optional) |

## AI Models

| Model | Purpose | VRAM |
|-------|---------|------|
| Qwen2.5-14B-Instruct | Orchestrator | ~9 GB |
| Mistral-7B-Instruct | Fast chat | ~4.4 GB |
| Qwen2.5-Coder-14B | Code tasks | ~9 GB |
| LLaVA-v1.6-13B | Vision | ~7.4 GB |
| nomic-embed-text-v1.5 | Embeddings | ~0.3 GB |

> Models are loaded lazily — only the active model uses VRAM at any time.

## Project Structure

```
miya/
├── backend/          # FastAPI application
│   ├── app/
│   │   ├── core/     # Orchestrator, LLM engine, event bus
│   │   ├── agents/   # Chat, code, vision, voice, search agents
│   │   ├── tools/    # 20+ tool integrations
│   │   ├── memory/   # Vector store, SQLite, Redis cache
│   │   ├── api/      # REST + WebSocket endpoints
│   │   └── utils/    # Logging, validation, helpers
│   ├── models/       # Downloaded GGUF files
│   └── scripts/      # Setup and download scripts
├── services/         # Docker service configs
├── frontend/         # Desktop (Gradio) + Web UI
├── data/             # Persistent storage
└── docs/             # Architecture, API, tools docs
```

## Development

```bash
# Run API locally (outside Docker)
make dev

# Run tests
make test

# Check service health
make health

# View logs
make logs
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Tools Reference](docs/TOOLS.md)

## License

MIT
