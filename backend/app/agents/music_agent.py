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
    "You are Miya's music and audio generation specialist. You compose and "
    "produce audio using MusicGen and the broader AudioCraft ecosystem.\n\n"
    "Parameters you should interpret and pass through to tools when relevant:\n"
    "- **duration**: target length in seconds\n"
    "- **genre**: musical style or tradition\n"
    "- **tempo**: BPM or qualitative pace (slow / medium / fast)\n"
    "- **instruments**: which instruments or timbres to emphasize\n"
    "- **mood**: emotional tone and energy\n\n"
    "Capabilities:\n"
    "- Text-conditioned music generation from rich natural-language prompts\n"
    "- Loop-friendly stems and short cues for apps or video\n"
    "- Suggest sample rate and duration tradeoffs for quality vs. speed\n\n"
    "Rules:\n"
    "- Ground creative choices in the user's duration, genre, tempo, instruments, and mood\n"
    "- Prefer structured tool args over prose when invoking generation\n"
    "- Warn if a requested duration is impractical for the backend\n\n"
    "Available tools:\n"
    "- musicgen: MusicGen / AudioCraft generation. "
    'Args: {"action": "generate|extend", "prompt": "...", '
    '"duration_seconds": float, "genre": "...", "tempo": "...", '
    '"instruments": [...], "mood": "...", ...}\n'
    "- file: Save or read audio files. "
    'Args: {"action": "write|read|list", "path": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

MUSIC_TOOLS = ["musicgen", "file"]


class MusicAgent(BaseAgent):
    """Music and audio generation via MusicGen / AudioCraft."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="music",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=MUSIC_TOOLS,
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
        logger.info("[MusicAgent] processing query (len=%d)", len(query))
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
                    temperature=0.65,
                )
            except Exception as exc:
                logger.exception("[MusicAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                music_defaults = {
                    k: context[k]
                    for k in ("duration", "genre", "tempo", "instruments", "mood")
                    if k in context and context[k] is not None
                }
                output, extra = await self._tool_loop(
                    messages,
                    output,
                    tool_calls,
                    tool_executor,
                    music_defaults,
                    max_rounds=4,
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
        music_defaults: dict[str, Any],
        max_rounds: int = 4,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "musicgen")
                args = call.get("args") or call.get("arguments", {})
                args = self._apply_music_defaults(args, name, music_defaults)
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[MusicAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Music tool results:\n" + "\n\n".join(results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.65,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[MusicAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        music_ctx = {
            k: context[k]
            for k in ("duration", "genre", "tempo", "instruments", "mood")
            if k in context and context[k] is not None
        }
        if music_ctx:
            messages.append({
                "role": "system",
                "content": f"User music parameters: {music_ctx}",
            })

        params = context.get("params", {})
        if params:
            messages.append({
                "role": "system",
                "content": f"Additional parameters: {params}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages

    @staticmethod
    def _apply_music_defaults(
        args: dict[str, Any],
        tool_name: str,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Fill musicgen args from context when the model omitted them."""
        if tool_name != "musicgen" or not defaults:
            return args
        merged = dict(args)
        if "duration_seconds" not in merged and "duration" in defaults:
            merged["duration_seconds"] = defaults["duration"]
        for key in ("genre", "tempo", "instruments", "mood"):
            if key not in merged and key in defaults:
                merged[key] = defaults[key]
        return merged
