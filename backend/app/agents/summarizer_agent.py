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
    "You are Miya's summarization specialist. You create clear, accurate, "
    "and well-structured summaries of any text, document, or conversation.\n\n"
    "Follow these rules:\n"
    "- Preserve all key facts, figures, and conclusions\n"
    "- Maintain the original meaning without adding interpretation\n"
    "- Use bullet points for multi-topic content\n"
    "- Provide a TL;DR at the top for long summaries\n"
    "- Adjust detail level based on the user's request (brief / detailed / key points)\n"
    "- For conversations: capture decisions, action items, and open questions"
)

SUMMARY_TOOLS = ["file", "scraper"]

SUMMARY_MODES = {
    "brief": "Write a 2-3 sentence summary capturing the essence.",
    "detailed": "Write a comprehensive summary preserving all important details.",
    "bullets": "Summarize as a bulleted list of key points.",
    "executive": "Write an executive summary suitable for decision-makers.",
    "action_items": "Extract all action items, decisions, and deadlines.",
}


class SummarizerAgent(BaseAgent):
    """Text and document summarization agent.

    Supports multiple summary modes: brief, detailed, bullets,
    executive, and action_items.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="summarizer",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=SUMMARY_TOOLS,
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
        logger.info("[SummarizerAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        mode = context.get("summary_mode", "detailed")
        source_text = context.get("source_text", "")

        if not source_text and tool_executor:
            source_text = await self._load_source(query, context, tool_executor)

        with Timer() as t:
            messages = self._build_messages(query, source_text, mode, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[SummarizerAgent] generation failed")
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

    async def _load_source(
        self, query: str, context: dict[str, Any], tool_executor: Any
    ) -> str:
        file_path = context.get("file_path")
        url = context.get("url")

        if file_path:
            try:
                result = await tool_executor("file", {"action": "read", "path": file_path})
                return str(result) if result else ""
            except Exception as exc:
                logger.warning("[SummarizerAgent] file read failed: %s", exc)

        if url:
            try:
                result = await tool_executor("scraper", {"action": "extract", "url": url})
                return result.get("text", str(result)) if isinstance(result, dict) else str(result)
            except Exception as exc:
                logger.warning("[SummarizerAgent] scrape failed: %s", exc)

        return ""

    def _build_messages(
        self, query: str, source_text: str, mode: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        mode_instruction = SUMMARY_MODES.get(mode, SUMMARY_MODES["detailed"])

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Summary mode: {mode}\nInstruction: {mode_instruction}"},
        ]

        if source_text:
            truncated = source_text[:12000]
            messages.append({
                "role": "system",
                "content": f"Source text to summarize:\n\n{truncated}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
