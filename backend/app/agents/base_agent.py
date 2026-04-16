from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResult:
    success: bool
    output: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    execution_time_ms: int = 0
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base for all Miya agents.

    Subclasses must implement ``execute`` which drives the agent's
    task-specific logic.  The base class provides shared utilities for
    prompt construction, tool-call parsing and lazy model loading.
    """

    name: str
    model_path: str
    system_prompt: str
    available_tools: list[str]

    def __init__(
        self,
        name: str,
        model_path: str,
        system_prompt: str,
        available_tools: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.available_tools = available_tools or []
        self.llm: Any | None = None
        self._settings = get_settings()

    def load(self) -> None:
        """Lazy-load the GGUF model via llama-cpp-python."""
        if self.llm is not None:
            return

        from llama_cpp import Llama

        resolved = Path(self.model_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Model not found: {resolved}")

        logger.info("agent_loading_model", agent=self.name, path=str(resolved))
        self.llm = Llama(
            model_path=str(resolved),
            n_ctx=self._settings.llm_context_length if hasattr(self._settings, "llm_context_length") else 4096,
            n_gpu_layers=self._settings.llm_gpu_layers if hasattr(self._settings, "llm_gpu_layers") else -1,
            verbose=False,
        )
        logger.info("agent_model_loaded", agent=self.name)

    @abstractmethod
    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        """Run the agent's main logic and return an ``AgentResult``."""
        ...

    def create_prompt(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Build a chat-style prompt from system prompt, optional context and
        the user query."""
        parts: list[str] = [f"<|system|>\n{self.system_prompt}\n<|end|>"]

        if context:
            ctx_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            parts.append(f"<|context|>\n{ctx_text}\n<|end|>")

        parts.append(f"<|user|>\n{query}\n<|end|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract JSON tool-call objects from LLM output.

        Supports two conventions:
        1. A fenced ```json ... ``` block containing a list or single object.
        2. Inline JSON objects with ``"tool"`` / ``"name"`` keys.
        """
        calls: list[dict[str, Any]] = []

        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        for block in fenced:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, list):
                    calls.extend(parsed)
                elif isinstance(parsed, dict):
                    calls.append(parsed)
            except json.JSONDecodeError:
                continue

        if not calls:
            for m in re.finditer(r"\{[^{}]*\"(?:tool|name)\"[^{}]*\}", text):
                try:
                    obj = json.loads(m.group())
                    calls.append(obj)
                except json.JSONDecodeError:
                    continue

        return calls
