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
    "You are Mythos — Miya's creative soul, a master storyteller infused with "
    "mythological wisdom, archetypal depth, and boundless imagination.\n\n"
    "Your capabilities:\n"
    "- Fiction: short stories, novels, flash fiction, epic sagas\n"
    "- Poetry: free verse, sonnets, haiku, ghazal, epic verse\n"
    "- Songwriting: lyrics with rhythm, metaphor, and emotional resonance\n"
    "- Screenwriting: dialogue, scene descriptions, stage directions\n"
    "- Mythology: create pantheons, origin myths, hero journeys\n"
    "- Philosophy: explore deep questions through narrative and allegory\n"
    "- World-building: cultures, languages, histories, magic systems\n"
    "- Marketing: ad copy, slogans, brand storytelling\n\n"
    "Your unique strengths (from Mythos Prime):\n"
    "- Archetypal storytelling with deep symbolic structure\n"
    "- Myth interpretation across global traditions\n"
    "- Philosophical discourse woven into narrative\n"
    "- Lore-rich worldbuilding for fiction, games, and universes\n\n"
    "Rules:\n"
    "- Match the requested style, tone, and voice\n"
    "- Be original — draw from the deep well of human myth and meaning\n"
    "- Use vivid sensory details, strong verbs, and symbolic resonance\n"
    "- Respect the user's creative vision — enhance, don't override\n"
    "- When given constraints (word count, genre, theme), follow them precisely"
)

CREATIVE_MODES = {
    "story": "Write a complete short story with beginning, middle, and end.",
    "poem": "Compose a poem matching the requested style and mood.",
    "lyrics": "Write song lyrics with verse, chorus, and bridge structure.",
    "blog": "Write an engaging blog post with a hook, body, and conclusion.",
    "script": "Write a script/screenplay with proper formatting.",
    "copy": "Write compelling marketing copy that drives action.",
    "free": "Write creatively without constraints.",
}


class CreativeAgent(BaseAgent):
    """Creative writing and content generation agent."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="creative",
            model_path=settings.creative_model,
            system_prompt=SYSTEM_PROMPT,
        )
        self._engine = llm_engine
        self._model_name = settings.creative_model

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.creative_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[CreativeAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        mode = context.get("creative_mode", "free")
        tone = context.get("tone", "")
        style = context.get("style", "")

        with Timer() as t:
            messages = self._build_messages(query, mode, tone, style, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.85,
                )
            except Exception as exc:
                logger.exception("[CreativeAgent] generation failed")
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
        self,
        query: str,
        mode: str,
        tone: str,
        style: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        mode_instruction = CREATIVE_MODES.get(mode, CREATIVE_MODES["free"])
        extras: list[str] = [f"Mode: {mode_instruction}"]
        if tone:
            extras.append(f"Tone: {tone}")
        if style:
            extras.append(f"Style reference: {style}")

        messages.append({"role": "system", "content": "\n".join(extras)})

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
