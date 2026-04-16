from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

log = get_logger(__name__)


class ModelManagerTool(BaseTool):
    """Manages local GGUF model files — list, load, unload, download, delete."""

    name = "model_manager"
    description = "Manage local LLM models (list, load, unload, info, delete)"
    category = "ai"
    parameters: dict[str, Any] = {
        "action": {
            "type": "string",
            "required": True,
            "enum": ["list", "list_loaded", "load", "unload", "info", "delete", "disk_usage"],
        },
        "model_name": {"type": "string", "required": False, "description": "Model filename"},
        "n_ctx": {"type": "integer", "required": False, "default": 4096},
    }

    def __init__(self, llm_engine: Any | None = None) -> None:
        self._engine = llm_engine
        self._settings = get_settings()

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        action = input_data.get("action", "list")

        actions = {
            "list": self._list_models,
            "list_loaded": self._list_loaded,
            "load": self._load_model,
            "unload": self._unload_model,
            "info": self._model_info,
            "delete": self._delete_model,
            "disk_usage": self._disk_usage,
        }

        handler = actions.get(action)
        if not handler:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            return await handler(input_data)
        except Exception as exc:
            log.error("model_manager_failed", action=action, error=str(exc))
            return {"success": False, "error": str(exc)}

    async def _list_models(self, _: dict) -> dict[str, Any]:
        models_dir = self._settings.models_dir
        if not models_dir.exists():
            return {"success": True, "models": [], "count": 0}

        models = []
        for f in models_dir.iterdir():
            if f.suffix == ".gguf":
                size_gb = f.stat().st_size / (1024 ** 3)
                models.append({
                    "name": f.name,
                    "size_gb": round(size_gb, 2),
                    "path": str(f),
                })

        return {"success": True, "models": models, "count": len(models)}

    async def _list_loaded(self, _: dict) -> dict[str, Any]:
        if not self._engine:
            return {"success": False, "error": "Engine not available"}
        loaded = self._engine.list_loaded_models()
        return {"success": True, "loaded_models": loaded, "count": len(loaded)}

    async def _load_model(self, data: dict) -> dict[str, Any]:
        if not self._engine:
            return {"success": False, "error": "Engine not available"}
        model_name = data.get("model_name", "")
        if not model_name:
            return {"success": False, "error": "model_name required"}
        n_ctx = data.get("n_ctx", 4096)
        self._engine.load_model(model_name, n_ctx=n_ctx)
        return {"success": True, "message": f"Model {model_name} loaded (ctx={n_ctx})"}

    async def _unload_model(self, data: dict) -> dict[str, Any]:
        if not self._engine:
            return {"success": False, "error": "Engine not available"}
        model_name = data.get("model_name", "")
        if not model_name:
            return {"success": False, "error": "model_name required"}
        self._engine.unload_model(model_name)
        return {"success": True, "message": f"Model {model_name} unloaded"}

    async def _model_info(self, data: dict) -> dict[str, Any]:
        model_name = data.get("model_name", "")
        if not model_name:
            return {"success": False, "error": "model_name required"}

        model_path = self._settings.models_dir / model_name
        if not model_path.exists():
            return {"success": False, "error": f"Model not found: {model_name}"}

        stat = model_path.stat()
        return {
            "success": True,
            "name": model_name,
            "size_gb": round(stat.st_size / (1024 ** 3), 2),
            "modified": stat.st_mtime,
            "path": str(model_path),
        }

    async def _delete_model(self, data: dict) -> dict[str, Any]:
        model_name = data.get("model_name", "")
        if not model_name:
            return {"success": False, "error": "model_name required"}

        model_path = self._settings.models_dir / model_name
        if not model_path.exists():
            return {"success": False, "error": f"Model not found: {model_name}"}

        if self._engine:
            try:
                self._engine.unload_model(model_name)
            except Exception:
                pass

        size = model_path.stat().st_size
        model_path.unlink()
        return {
            "success": True,
            "message": f"Deleted {model_name}",
            "freed_gb": round(size / (1024 ** 3), 2),
        }

    async def _disk_usage(self, _: dict) -> dict[str, Any]:
        models_dir = self._settings.models_dir
        if not models_dir.exists():
            return {"success": True, "total_gb": 0, "models": 0}

        total = 0
        count = 0
        for f in models_dir.iterdir():
            if f.suffix == ".gguf":
                total += f.stat().st_size
                count += 1

        return {
            "success": True,
            "total_gb": round(total / (1024 ** 3), 2),
            "models": count,
        }
