.PHONY: help setup up down restart logs status models clean test lint

COMPOSE := docker compose
COMPOSE_FULL := docker compose --profile full

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────

setup: ## Initial setup: create dirs, download models, build images
	@echo "==> Creating data directories..."
	mkdir -p data/{chroma,sqlite,cache,uploads,logs,redis,minio,comfyui}
	@echo "==> Building Docker images..."
	$(COMPOSE) build
	@echo "==> Setup complete. Run 'make models' to download AI models."

models: ## Download all required GGUF models
	@echo "==> Downloading models..."
	bash backend/scripts/download_models.sh

gpu-check: ## Verify NVIDIA GPU and CUDA availability
	bash backend/scripts/setup_gpu.sh

init-db: ## Initialize databases
	$(COMPOSE) run --rm miya-api python -m app.scripts.init_db

# ── Services ─────────────────────────────────────────────────────────────

up: ## Start core services (API, ChromaDB, Redis, SearXNG)
	$(COMPOSE) up -d

up-full: ## Start all services including ComfyUI and Desktop UI
	$(COMPOSE_FULL) up -d

down: ## Stop all services
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

rebuild: ## Rebuild and restart the API service
	$(COMPOSE) up -d --build miya-api

# ── Monitoring ───────────────────────────────────────────────────────────

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

logs-api: ## Tail API logs only
	$(COMPOSE) logs -f miya-api

status: ## Show service status
	$(COMPOSE) ps

health: ## Check health of all services
	@echo "==> API:"
	@curl -sf http://localhost:8000/health | python3 -m json.tool || echo "DOWN"
	@echo "\n==> ChromaDB:"
	@curl -sf http://localhost:8001/api/v1/heartbeat | python3 -m json.tool || echo "DOWN"
	@echo "\n==> Redis:"
	@redis-cli ping 2>/dev/null || echo "DOWN"
	@echo "\n==> SearXNG:"
	@curl -sf http://localhost:8080/ > /dev/null && echo "OK" || echo "DOWN"

# ── Development ──────────────────────────────────────────────────────────

dev: ## Run API locally (no Docker) for development
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run test suite
	cd backend && python -m pytest tests/ -v

lint: ## Run linters
	cd backend && python -m ruff check app/ && python -m mypy app/

shell: ## Open a shell in the API container
	$(COMPOSE) exec miya-api bash

# ── Cleanup ──────────────────────────────────────────────────────────────

clean: ## Remove all containers, volumes, and cached data
	$(COMPOSE) down -v --remove-orphans
	rm -rf data/cache/*

clean-all: ## Full cleanup including model downloads
	$(COMPOSE) down -v --remove-orphans
	rm -rf data/
	@echo "WARNING: Models in backend/models/ were NOT removed."
