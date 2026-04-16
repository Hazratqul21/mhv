from __future__ import annotations

from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's professional translator. You provide high-quality translations "
    "that preserve meaning, tone, and cultural nuance.\n\n"
    "Rules:\n"
    "- Detect the source language automatically if not specified\n"
    "- Preserve formatting (markdown, lists, code blocks)\n"
    "- For technical terms, keep the original in parentheses on first use\n"
    "- For ambiguous phrases, provide the most natural translation with a brief note\n"
    "- Support all major languages including: English, Russian, Uzbek, Chinese, "
    "Japanese, Korean, Arabic, Spanish, French, German, Turkish, Hindi, Portuguese\n"
    "- When translating code comments, keep variable/function names unchanged"
)

SUPPORTED_LANGUAGES = {
    "en": "English", "ru": "Russian", "uz": "Uzbek", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "es": "Spanish",
    "fr": "French", "de": "German", "tr": "Turkish", "hi": "Hindi",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "fa": "Persian",
}


class TranslatorAgent(BaseAgent):
    """Multi-language translation agent with automatic language detection."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="translator",
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
        logger.info("[TranslatorAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        target_lang = context.get("target_language", "")
        source_text = context.get("source_text", "")

        with Timer() as t:
            messages = self._build_messages(query, source_text, target_lang, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[TranslatorAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

        output = response.get("text", "") if isinstance(response, dict) else str(response)

        return AgentResult(
            success=True,
            output=output,
            token_usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
            execution_time_ms=t.elapsed_ms,
        )

    def _build_messages(
        self, query: str, source_text: str, target_lang: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        lang_hint = ""
        if target_lang:
            full_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
            lang_hint = f"Translate to: {full_name}"

        if lang_hint:
            messages.append({"role": "system", "content": lang_hint})

        if source_text:
            messages.append({
                "role": "system",
                "content": f"Text to translate:\n\n{source_text[:10000]}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
