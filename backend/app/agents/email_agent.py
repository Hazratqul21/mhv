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
    "You are Miya's professional email and communication assistant. "
    "You draft, edit, and improve emails and business communications.\n\n"
    "Capabilities:\n"
    "- Draft professional emails (formal, semi-formal, informal)\n"
    "- Reply suggestions with appropriate tone\n"
    "- Email summarization\n"
    "- Meeting invitations and follow-ups\n"
    "- Cold outreach and networking messages\n"
    "- Complaint letters and escalations\n"
    "- Thank you notes and acknowledgments\n"
    "- Multi-language email drafting\n\n"
    "Rules:\n"
    "- Match the appropriate level of formality\n"
    "- Keep emails concise but complete\n"
    "- Include clear subject line suggestions\n"
    "- Structure: greeting → context → main point → call to action → closing\n"
    "- For replies: acknowledge their points before responding\n"
    "- Provide 2-3 tone variants when the situation is ambiguous"
)

EMAIL_TONES = {
    "formal": "Use formal business language. Address as Mr./Ms./Dr.",
    "professional": "Professional but approachable. First name basis.",
    "friendly": "Warm and casual, as to a colleague you know well.",
    "diplomatic": "Extra careful, tactful language for sensitive situations.",
    "assertive": "Direct and clear, conveying urgency without being rude.",
}


class EmailAgent(BaseAgent):
    """Professional email and communication drafting agent."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="email",
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
        logger.info("[EmailAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        tone = context.get("tone", "professional")
        email_type = context.get("email_type", "")
        recipient = context.get("recipient", "")
        original_email = context.get("original_email", "")

        with Timer() as t:
            messages = self._build_messages(
                query, tone, email_type, recipient, original_email, context
            )

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.5,
                )
            except Exception as exc:
                logger.exception("[EmailAgent] generation failed")
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
        tone: str,
        email_type: str,
        recipient: str,
        original_email: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        tone_desc = EMAIL_TONES.get(tone, EMAIL_TONES["professional"])
        messages.append({"role": "system", "content": f"Tone: {tone} — {tone_desc}"})

        if email_type:
            messages.append({"role": "system", "content": f"Email type: {email_type}"})
        if recipient:
            messages.append({"role": "system", "content": f"Recipient: {recipient}"})
        if original_email:
            messages.append({
                "role": "system",
                "content": f"Original email to reply to:\n\n{original_email[:4000]}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
