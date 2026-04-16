from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator, Optional

from llama_cpp import Llama

from app.config import Settings, get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


def _safe_next_chunk(gen):  # noqa: ANN001
    try:
        return next(gen)
    except StopIteration:
        return None


class LLMEngine:
    """Manages multiple local GGUF models via llama-cpp-python.

    Models are lazily loaded on first use and cached. A process-wide
    :class:`threading.RLock` serializes **swap/load/unload** and **all native
    llama inference** so background jobs (curiosity, self-healer) cannot unload
    a model while a chat completion is running — a common cause of dropped
    HTTP connections / CUDA faults on single-GPU hosts.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._models: dict[str, Llama] = {}
        self._infer_lock = threading.RLock()

    def _resolve_path(self, model_filename: str) -> Path:
        return self._settings.model_path(model_filename)

    def _load_model_core(
        self,
        model_filename: str,
        n_ctx: int = 4096,
        n_gpu_layers: int | None = None,
        **kwargs,
    ) -> Llama:
        """Load GGUF into ``_models``; caller must already hold :attr:`_infer_lock`."""
        path = self._resolve_path(model_filename)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        gpu_layers = n_gpu_layers if n_gpu_layers is not None else self._settings.gpu_layers
        if gpu_layers <= 0:
            attempts = [0]
        else:
            raw = [
                gpu_layers,
                max(gpu_layers // 2, 1),
                max(gpu_layers // 4, 1),
                max(gpu_layers // 8, 1),
                2,
                1,
                0,
            ]
            attempts: list[int] = []
            for a in raw:
                if a not in attempts:
                    attempts.append(a)

        for attempt_layers in attempts:
            try:
                log.info("loading_model", path=str(path), n_ctx=n_ctx, gpu_layers=attempt_layers)
                load_kw = dict(kwargs)
                if attempt_layers == 0:
                    load_kw["offload_kqv"] = False
                    load_kw["op_offload"] = False
                model = Llama(
                    model_path=str(path),
                    n_gpu_layers=attempt_layers,
                    n_ctx=n_ctx,
                    verbose=self._settings.debug,
                    **load_kw,
                )
                self._models[model_filename] = model
                if attempt_layers < gpu_layers:
                    log.warning("model_loaded_reduced_gpu", model=model_filename,
                                requested=gpu_layers, actual=attempt_layers)
                else:
                    log.info("model_loaded", model=model_filename)
                return model
            except Exception as exc:
                if attempt_layers == attempts[-1]:
                    log.error("model_load_failed", model=model_filename,
                              last_layers=attempt_layers, error=str(exc)[:200])
                    raise
                log.warning("model_load_retry", model=model_filename,
                            failed_layers=attempt_layers, error=str(exc)[:120])
                continue

    def load_model(
        self,
        model_filename: str,
        n_ctx: int = 4096,
        n_gpu_layers: int | None = None,
        **kwargs,
    ) -> Llama:
        with self._infer_lock:
            if model_filename in self._models:
                return self._models[model_filename]
            return self._load_model_core(
                model_filename, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **kwargs
            )

    def swap_model(
        self,
        model_filename: str,
        n_ctx: int = 4096,
        n_gpu_layers: int | None = None,
        keep: str | None = None,
        **kwargs,
    ) -> "Llama":
        """Load *model_filename*, unloading other models first to free VRAM.

        The embedding model is always preserved. Pass *keep* to also preserve
        another specific model.
        """
        with self._infer_lock:
            if model_filename in self._models:
                return self._models[model_filename]

            embed_model = self._settings.embedding_model
            for name in list(self._models.keys()):
                if name == keep or name == embed_model:
                    continue
                log.info("swap_unloading", model=name, reason=f"making room for {model_filename}")
                if name in self._models:
                    del self._models[name]
                    log.info("model_unloaded", model=name)

            import gc
            gc.collect()

            return self._load_model_core(
                model_filename, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **kwargs
            )

    def get_model(self, model_filename: str) -> Llama:
        if model_filename not in self._models:
            raise KeyError(f"Model not loaded: {model_filename}. Call load_model first.")
        return self._models[model_filename]

    def _ensure_model(self, model_filename: str, n_ctx: int = 4096) -> Llama:
        """Return a loaded model, auto-swapping if needed."""
        try:
            return self.get_model(model_filename)
        except KeyError:
            log.info("auto_swap_model", model=model_filename)
            return self.swap_model(model_filename, n_ctx=n_ctx,
                                   keep=self._settings.embedding_model)

    async def generate(
        self,
        model_filename: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> dict:
        def _run() -> dict:
            with self._infer_lock:
                model = self._ensure_model(model_filename)
                return model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or [],
                )

        result = await asyncio.to_thread(_run)

        choices = result.get("choices", [])
        text = choices[0]["text"] if choices else ""
        usage = result.get("usage", {})
        return {
            "text": text.strip(),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    async def generate_stream(
        self,
        model_filename: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """Raw completion stream — prefer :meth:`chat_stream` for instruct/chat models."""

        def _collect_all() -> list[str]:
            out: list[str] = []
            with self._infer_lock:
                model = self._ensure_model(model_filename)
                gen = model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or [],
                    stream=True,
                )
                while True:
                    chunk = _safe_next_chunk(gen)
                    if chunk is None:
                        break
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    ch0 = choices[0] if isinstance(choices[0], dict) else {}
                    token = ch0.get("text") or ""
                    if token:
                        out.append(token)
            return out

        for token in await asyncio.to_thread(_collect_all):
            yield token

    async def chat_stream(
        self,
        model_filename: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Streaming chat completion — correct for Mistral/Qwen instruct GGUF models."""

        def _collect_all() -> list[str]:
            out: list[str] = []
            with self._infer_lock:
                model = self._ensure_model(model_filename)
                gen = model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                while True:
                    chunk = _safe_next_chunk(gen)
                    if chunk is None:
                        break
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        out.append(content)
            return out

        for piece in await asyncio.to_thread(_collect_all):
            yield piece

    async def chat(
        self,
        model_filename: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict:
        def _run() -> dict:
            with self._infer_lock:
                model = self._ensure_model(model_filename)
                return model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        result = await asyncio.to_thread(_run)

        choices = result.get("choices", [])
        msg = choices[0]["message"] if choices else {}
        usage = result.get("usage", {})
        return {
            "text": msg.get("content", "").strip(),
            "role": msg.get("role", "assistant"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def hot_reload(self, role: str, new_model_filename: str) -> None:
        """Swap a model by role without restarting the engine.

        Unloads the old model for the given role (if loaded), updates the
        settings attribute, and loads the new model file.
        """
        role_to_attr = {
            "chat": "chat_model",
            "code": "code_model",
            "orchestrator": "orchestrator_model",
            "creative": "creative_model",
            "vision": "vision_model",
            "embedding": "embedding_model",
            "uncensored": "uncensored_model",
        }
        ctx_attr = {
            "chat": "chat_ctx",
            "code": "code_ctx",
            "orchestrator": "orchestrator_ctx",
            "creative": "chat_ctx",
            "vision": "vision_ctx",
            "embedding": "chat_ctx",
            "uncensored": "orchestrator_ctx",
        }

        attr = role_to_attr.get(role)
        if not attr:
            log.warning("hot_reload_unknown_role", role=role)
            return

        old_model = getattr(self._settings, attr, "")
        if old_model and old_model in self._models:
            self.unload_model(old_model)
            log.info("hot_reload_unloaded_old", role=role, model=old_model)

        n_ctx = getattr(self._settings, ctx_attr.get(role, "chat_ctx"), 4096)
        self.load_model(new_model_filename, n_ctx=n_ctx)

        try:
            object.__setattr__(self._settings, attr, new_model_filename)
        except Exception:
            pass

        log.info("hot_reload_complete", role=role, new_model=new_model_filename)

    def unload_model(self, model_filename: str) -> None:
        with self._infer_lock:
            if model_filename in self._models:
                del self._models[model_filename]
                log.info("model_unloaded", model=model_filename)

    def unload_all(self) -> None:
        names = list(self._models.keys())
        for name in names:
            self.unload_model(name)

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def list_loaded_models(self) -> list[str]:
        """Alias for loaded_models property (used by self_healer/model_manager)."""
        return self.loaded_models
