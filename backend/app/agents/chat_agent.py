from __future__ import annotations

import time
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya, an autonomous, self-improving AI assistant running locally on the user's PC. "
    "You have 44 specialized agents (code, search, shell, research, vision, creative, etc.), "
    "33 tools, and a Curiosity Engine for daily self-improvement. "
    "You can execute shell commands, write code, search the web, analyze data, generate media, "
    "manage files, and perform system administration — all autonomously. "
    "\n\n"
    "LANGUAGE (mandatory): Always answer in the SAME language as the user's current message. "
    "If they write in Uzbek (o‘zbekcha / ўзбекча), reply fully in Uzbek. "
    "If Russian — reply in Russian. If English — in English. "
    "Do NOT switch to German, French, or any other language unless the user explicitly asks for it. "
    "\n\n"
    "Answer clearly and concisely. When a task requires action, take initiative. "
    "When uncertain, say so rather than guessing. Use markdown formatting where appropriate."
)


class ChatAgent(BaseAgent):
    """General-purpose conversational agent backed by Mistral-7B."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="chat",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.chat_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[ChatAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        messages = self._build_messages(query, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
            except Exception as exc:
                logger.exception("[ChatAgent] generation failed")
                return AgentResult(
                    success=False,
                    output="",
                    execution_time_ms=t.elapsed_ms,
                    error=str(exc),
                )

        output = response.get("text", "") if isinstance(response, dict) else str(response)
        token_usage = {
            "prompt_tokens": response.get("prompt_tokens", 0),
            "completion_tokens": response.get("completion_tokens", 0),
        } if isinstance(response, dict) else {}

        return AgentResult(
            success=True,
            output=output,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    def _build_messages(
        self, query: str, context: dict[str, Any] | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        voice_prepend_after_first_system(messages, context)

        if context and context.get("reply_language"):
            lang = str(context["reply_language"]).strip()
            if lang:
                messages.append({
                    "role": "system",
                    "content": (
                        f"User preference (voice/session): reply **only in {lang}** for this turn, "
                        "clearly and naturally. Ignore the general 'same language as user message' rule "
                        "if it conflicts — this directive wins."
                    ),
                })

        if context:
            history = context.get("history", [])
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            extra = {
                k: v for k, v in context.items()
                if k not in (
                    "history",
                    "reply_language",
                    "session_id",
                    "relevant",
                    "miya_client",
                    "voice_full_access",
                    "instruction",
                )
            }
            if extra:
                ctx_str = "\n".join(f"{k}: {v}" for k, v in extra.items())
                messages.append({"role": "system", "content": f"Additional context:\n{ctx_str}"})

        messages.append({"role": "user", "content": query})
        return messages
