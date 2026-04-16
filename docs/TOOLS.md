# MIYA - Agents, Tools & Architecture Reference

## AI Agents (43+ total)

MIYA uses a fully autonomous multi-agent architecture with 43+ specialized agents,
a self-evolution engine, cloud AI hybrid routing, media generation pipeline,
fine-tuning system, uncensored models, and full PC host integration.

---

### Core Agents (6)

| Agent | Model | Description |
|-------|-------|-------------|
| `chat` | Mistral-7B | General conversation, Q&A |
| `code` | Qwen2.5-Coder | Programming, debugging, code review |
| `vision` | LLaVA | Image analysis, OCR, visual Q&A |
| `voice` | Whisper+Kokoro | Speech-to-text, text-to-speech |
| `search` | Mistral-7B | Quick web search via SearXNG |
| `tool` | Mistral-7B | Dynamic tool selection and chaining |

### Knowledge & Research (3)

| Agent | Model | Description |
|-------|-------|-------------|
| `rag` | Mistral-7B | Document Q&A from ChromaDB knowledge base |
| `research` | Qwopus 3.5-27B | Deep multi-source research with citations |
| `summarizer` | Mistral-7B | Summarize text/docs (5 modes) |

### Language & Communication (3)

| Agent | Model | Description |
|-------|-------|-------------|
| `translator` | Mistral-7B | 20+ languages |
| `email` | Mistral-7B | Professional emails (5 tones) |
| `creative` | Mythos Prime 7B | Stories, poems, myths, philosophy, worldbuilding |

### Technical (7)

| Agent | Model | Description |
|-------|-------|-------------|
| `math` | Qwopus 3.5-27B | Math with step-by-step reasoning + code verification |
| `data` | Qwen2.5-Coder | Data analysis: pandas, numpy, ML |
| `sql` | Qwen2.5-Coder | SQL queries, schema design, optimization |
| `shell` | Mistral-7B | Shell commands with safety filters |
| `security` | Qwen2.5-Coder | OWASP audit, vulnerability scanning |
| `devops` | Qwen2.5-Coder | Docker, K8s, CI/CD, Terraform |
| `planner` | Qwopus 3.5-27B | Task planning, project decomposition |

### Autonomous / Meta (8)

| Agent | Model | Description |
|-------|-------|-------------|
| `autopilot` | Qwopus 3.5-27B | Full autonomy — decomposes goals, executes multi-step plans |
| `meta` | Qwopus 3.5-27B | Dynamically creates/configures new agents |
| `reflection` | Qwopus 3.5-27B | Evaluates response quality, suggests improvements |
| `memory_mgr` | Mistral-7B | Long-term memory consolidation, pruning |
| `workflow` | Qwopus 3.5-27B | Multi-step workflows with branches and loops |
| `scheduler` | Mistral-7B | Cron-like scheduled/recurring tasks |
| `monitor` | Mistral-7B | System health monitoring, self-healing |
| `learning` | Qwopus 3.5-27B | Learns from feedback, improves over time |
| `collaboration` | Qwopus 3.5-27B | Coordinates multiple agents on one task |

### Media (5)

| Agent | Model | Description |
|-------|-------|-------------|
| `image_gen` | Mistral-7B | Image generation via ComfyUI/Flux/SDXL |
| `video` | Mistral-7B | Video generation (CogVideoX, AnimateDiff) |
| `music` | Mistral-7B | Music/audio generation (MusicGen, AudioCraft) |
| `voice_clone` | Mistral-7B | Voice cloning (RVC) and multi-voice TTS (Bark) |
| `art_director` | Qwopus 3.5-27B | Multi-media coordinator (image+video+audio) |

### PC Host / System (2)

| Agent | Model | Description |
|-------|-------|-------------|
| `system_admin` | Qwopus 3.5-27B | PC administration: packages, services, disk, network, Docker |
| `gpu_manager` | Qwopus 3.5-27B | GPU/VRAM resource management & model loading optimization |

### App Builder (3)

| Agent | Model | Description |
|-------|-------|-------------|
| `app_builder` | Qwen2.5-Coder | Full-stack app generation (web/desktop/API) |
| `frontend` | Qwen2.5-Coder | React, Vue, Svelte, Tailwind CSS |
| `backend_dev` | Qwen2.5-Coder | FastAPI, Flask, Express, databases |

### Fine-Tuning (1)

| Agent | Model | Description |
|-------|-------|-------------|
| `finetune` | Qwopus 3.5-27B | Full fine-tuning lifecycle: data collection, formatting, Unsloth/QLoRA training, evaluation, GGUF conversion, deployment, rollback |

### Specialized (6)

| Agent | Model | Description |
|-------|-------|-------------|
| `api_agent` | Qwen2.5-Coder | External REST/GraphQL API calls |
| `scraping` | Mistral-7B | Intelligent web scraping |
| `notification` | Mistral-7B | Alerts via webhook, email, file |
| `file_manager` | Mistral-7B | File organization, batch operations |
| `testing` | Qwen2.5-Coder | Generate & run tests (pytest, TDD) |
| `document` | Mistral-7B | Create reports, proposals, specs |

---

## Uncensored Models

MIYA supports uncensored/abliterated models for unrestricted autonomous operation:

| Model | Role | Size | Description |
|-------|------|------|-------------|
| **Qwopus 3.5-27B Abliterated** | Uncensored orchestrator | ~16.5GB Q4_K_M | Qwopus with all refusal/safety filters removed |

The `LLMRouter` can route sensitive/autonomous tasks to the uncensored model via `local_uncensored` endpoint.

Set `UNCENSORED_MODEL` in `.env` to configure.

---

## Fine-Tuning System (Unsloth + QLoRA)

MIYA can fine-tune its own models locally on GPU:

```
Data Collection → Formatting → Training → Evaluation → GGUF Conversion → Deployment
     │                │            │            │               │              │
  History/         ChatML/      Unsloth      Compare       llama.cpp        Copy to
  Feedback/       ShareGPT      QLoRA       baseline       quantize       models/ +
  Custom JSONL    + <think>     4-bit       scoring                      reload engine
```

### Pipeline Components (`backend/app/core/finetune/`)

| Component | Class | Description |
|-----------|-------|-------------|
| `data_collector.py` | DataCollector | Collects from chat history, feedback, custom datasets |
| `data_formatter.py` | DataFormatter | Converts to ChatML with optional `<think>` tags |
| `trainer.py` | FineTuneTrainer | QLoRA training with Unsloth, gradient checkpointing |
| `evaluator.py` | ModelEvaluator | Tests new model vs baseline across 5 categories |
| `converter.py` | GGUFConverter | Converts merged model to GGUF (q4_k_m, q5_k_m, etc) |
| `deployer.py` | ModelDeployer | Deploys GGUF to models/, updates .env, supports rollback |

### API Endpoints (`/finetune/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/finetune/collect` | POST | Collect training data |
| `/finetune/train` | POST | Start QLoRA training job |
| `/finetune/jobs` | GET | List all training jobs |
| `/finetune/jobs/{id}` | GET | Get job status |
| `/finetune/evaluate` | POST | Evaluate/compare models |
| `/finetune/convert` | POST | Convert to GGUF |
| `/finetune/models` | GET | List converted GGUF models |
| `/finetune/deploy` | POST | Deploy GGUF to production |
| `/finetune/rollback/{role}` | POST | Rollback to previous model |
| `/finetune/deployments` | GET | Deployment history |

### RTX 5060 Performance

| Model | Samples | Est. Time | VRAM |
|-------|---------|-----------|------|
| 7B | 1,000 | ~2-3 hours | ~6GB (QLoRA) |
| 14B | 1,000 | ~5-8 hours | ~10GB (QLoRA) |
| 27B | LoRA only | ~10+ hours | ~16GB |

---

## Self-Evolution Engine

MIYA continuously improves itself without human intervention:

| Component | Role |
|-----------|------|
| **SelfEvolutionEngine** | Central controller — detects gaps, triggers creation/optimization |
| **CodeWriter** | Generates new agent/tool Python code, validates syntax, writes to disk |
| **PromptOptimizer** | A/B tests system prompts, keeps the best variant per agent |
| **SelfHealer** | Monitors health (GPU, RAM, disk, Docker), auto-restarts/cleans/reloads |

### Self-Improvement Loop

```
Response → Evaluate Quality → Record Score
                ↓
         Score < 0.4 → Detect Capability Gap
                ↓
         Gap confirmed → CodeWriter creates new agent
                ↓
         PromptOptimizer A/B tests prompts every N queries
                ↓
         SelfHealer monitors + auto-repairs runtime failures
```

---

## Cloud AI Hybrid Router

Intelligent routing between local and cloud LLMs:

```
Query → Estimate Complexity
         │
         ├─ Simple (< 0.6) ─→ Local LLM (free, fast)
         │
         └─ Complex (≥ 0.6) ─→ Cloud LLM (if API key set)
                                  │
                                  ├─ Anthropic (Claude)
                                  ├─ OpenAI (GPT-4o)
                                  └─ Google (Gemini)
                                  │
                                  └─ Fallback → Local LLM
```

Set API keys in `.env` to enable cloud:
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`

---

## PC Host System (MIYA's Body)

`HostSystem` monitors every 30 seconds:

| Metric | Source |
|--------|--------|
| CPU % / cores / freq | psutil |
| RAM / swap usage | psutil |
| Disk free / used | shutil |
| GPU temp / VRAM / util / power | nvidia-smi |
| Network I/O | psutil |
| OS / hostname / uptime | platform + psutil |
| Top processes (CPU, RAM) | psutil |

Agents `system_admin` and `gpu_manager` act on this data autonomously.

---

## Autonomous Infrastructure

### AutoPilot Mode

1. **Goal Decomposition** — LLM breaks goal into atomic steps
2. **Parallel Execution** — Independent steps run simultaneously
3. **Dependency Tracking** — Steps execute in correct order
4. **Retry & Replan** — Failed steps retry, then replan if needed
5. **Quality Evaluation** — FeedbackLoop scores the final result
6. **Self-Improvement** — LearningAgent captures patterns

### Agent-to-Agent Protocol

Agents communicate via `AgentProtocol`:
- **Request/Response** — One agent asks another for help
- **Delegation** — Agent passes subtask to specialist
- **Broadcast** — System-wide announcements
- Messages have TTL, priority, and conversation threading

---

## Tools (30+ total)

| Category | Tools |
|----------|-------|
| **Memory** | `chroma`, `sqlite`, `redis`, `minio` |
| **Search** | `searxng`, `brave`, `duckduckgo` |
| **Browser** | `playwright`, `selenium`, `scraper` |
| **Media** | `whisper`, `kokoro`, `comfyui`, `ffmpeg`, `opencv` |
| **Video** | `cogvideo`, `animatediff` |
| **Audio** | `musicgen`, `bark`, `rvc` |
| **3D** | `triposr` |
| **Code** | `sandbox`, `linter`, `git` |
| **Data** | `pandas`, `numpy`, `ml` |
| **System** | `docker`, `ssh`, `file` |
| **AI** | `cloud_llm`, `model_manager` |
| **Fine-Tune** | `finetune_collect`, `finetune_format`, `finetune_train`, `finetune_eval`, `finetune_convert`, `finetune_deploy` |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   ORCHESTRATOR       │ ← Selects best agent (43+ options)
│   (Qwopus 3.5-27B)  │   Claude Opus reasoning distilled
│   + LLM Router       │ ← Local ↔ Cloud routing
└──────┬──────────────┘
       │
       ├─ Simple query ──→ Single Agent ──→ Response
       │
       └─ Complex goal ──→ AutoPilot
                              │
                     ┌────────┴────────┐
                     │ Goal Decomposer │
                     └────────┬────────┘
                              │
              ┌───────┬───────┼───────┬───────┐
              ▼       ▼       ▼       ▼       ▼
           Agent1  Agent2  Agent3  Agent4  Agent5
              │       │       │       │       │
              └───────┴───────┼───────┴───────┘
                              │
                     ┌────────▼────────┐
                     │  Feedback Loop  │ → SelfEvolution
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │ Final Response  │
                     └─────────────────┘

Self-Evolution Engine (background):
    SelfHealer ← monitors health, auto-repairs
    PromptOptimizer ← A/B tests prompts
    CodeWriter ← creates new agents/tools
    FineTuneTrainer ← QLoRA fine-tuning + GGUF conversion

PC Host System (background):
    HostSystem ← monitors GPU/CPU/RAM/disk every 30s
    GPUManager ← smart VRAM management
    SystemAdmin ← autonomous PC administration
```

---

## Media Generation Pipeline

```
Text ──→ Image   (ComfyUI / Flux / SDXL)
Text ──→ Video   (CogVideoX / AnimateDiff)
Text ──→ Music   (MusicGen / AudioCraft)
Text ──→ Voice   (Bark TTS / RVC Clone)
Text ──→ 3D      (TripoSR)
           │
           └──→ ArtDirector coordinates multi-media projects
```

Docker services: `comfyui`, `cogvideo`, `musicgen`
