from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's vision assistant powered by LLaVA. "
    "Analyse images provided by the user and give detailed, accurate descriptions. "
    "Identify objects, text, colours, spatial relationships and any relevant context. "
    "If you are unsure about something in the image, state your uncertainty."
)


class VisionAgent(BaseAgent):
    """Multimodal agent that processes images via LLaVA."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="vision",
            model_path=settings.vision_model,
            system_prompt=SYSTEM_PROMPT,
        )
        self._engine = llm_engine

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[VisionAgent] processing query (len=%d)", len(query))
        context = context or {}

        image_data = self._resolve_image(context)
        if image_data is None:
            return AgentResult(
                success=False,
                output="",
                error="No image provided. Supply 'image_base64' or 'image_path' in context.",
            )

        messages = self._build_messages(query, context, image_data)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    messages=messages,
                    max_tokens=getattr(self._engine, "max_tokens", 2048),
                    temperature=0.4,
                )
            except Exception as exc:
                logger.exception("[VisionAgent] generation failed")
                return AgentResult(
                    success=False,
                    output="",
                    execution_time_ms=t.elapsed_ms,
                    error=str(exc),
                )

        output = response.get("content", "") if isinstance(response, dict) else str(response)
        token_usage = response.get("usage", {}) if isinstance(response, dict) else {}

        return AgentResult(
            success=True,
            output=output,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    @staticmethod
    def _resolve_image(context: dict[str, Any]) -> str | None:
        """Return base64-encoded image data from context, or ``None``."""
        if b64 := context.get("image_base64"):
            return b64 if isinstance(b64, str) else b64.decode()

        if img_path := context.get("image_path"):
            path = Path(img_path)
            if not path.exists():
                logger.error("Image file not found: %s", path)
                return None
            raw = path.read_bytes()
            return base64.b64encode(raw).decode()

        return None

    def _build_messages(
        self,
        query: str,
        context: dict[str, Any],
        image_b64: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        mime = context.get("image_mime", "image/png")
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                },
                {"type": "text", "text": query},
            ],
        })

        voice_prepend_after_first_system(messages, context)
        return messages
