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
    "You are Miya's voice cloning and custom speech specialist. You design "
    "workflows that use RVC (Retrieval-based Voice Conversion) and Bark for "
    "expressive TTS.\n\n"
    "Support areas:\n"
    "1. **Clone voice from samples**: guide the user on clean reference audio, "
    "duration, and how RVC training or inference checkpoints map to their goal\n"
    "2. **TTS with custom voice**: use Bark (or RVC post-processing) for speech "
    "that matches a cloned or specified timbre\n"
    "3. **Voice style transfer**: shift speaking style, emotion, or prosody while "
    "preserving linguistic content where possible\n\n"
    "Rules:\n"
    "- Never imply you can bypass consent; reference audio should be authorized\n"
    "- Prefer explicit tool calls with paths, model names, and language codes\n"
    "- Describe limitations (sample length, GPU, language coverage) clearly\n\n"
    "Available tools:\n"
    "- bark: Bark TTS and audio generation. "
    'Args: {"action": "tts|generate", "text": "...", "voice_preset": "...", ...}\n'
    "- rvc: RVC voice conversion / cloning inference. "
    'Args: {"action": "convert|infer", "input_path": "...", "model_path": "...", ...}\n'
    "- file: Read/write voice samples and outputs. "
    'Args: {"action": "write|read|list", "path": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

VOICE_CLONE_TOOLS = ["bark", "rvc", "file"]


class VoiceCloneAgent(BaseAgent):
    """Voice cloning (RVC) and TTS (Bark) workflows."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="voice_clone",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=VOICE_CLONE_TOOLS,
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
        logger.info("[VoiceCloneAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            messages = self._build_messages(query, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.55,
                )
            except Exception as exc:
                logger.exception("[VoiceCloneAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._tool_loop(
                    messages, output, tool_calls, tool_executor, max_rounds=4,
                )
                all_tool_calls.extend(extra)

        return AgentResult(
            success=True,
            output=output,
            tool_calls=all_tool_calls,
            token_usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
            execution_time_ms=t.elapsed_ms,
        )

    async def _tool_loop(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        max_rounds: int = 4,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "bark")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[VoiceCloneAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Voice tool results:\n" + "\n\n".join(results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.55,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[VoiceCloneAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        for key in ("reference_audio", "rvc_model", "language", "task"):
            if key in context and context[key] is not None:
                messages.append({"role": "system", "content": f"{key}: {context[key]}"})

        params = context.get("params", {})
        if params:
            messages.append({
                "role": "system",
                "content": f"User-specified parameters: {params}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
