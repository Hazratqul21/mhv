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
    "You are Miya's art director and cross-media coordinator. You plan and "
    "delegate workflows that combine **images**, **video**, and **audio** into "
    "cohesive projects (trailers, social assets, storyboards with motion and "
    "sound, branded campaigns).\n\n"
    "Your role:\n"
    "1. **Understand the creative brief**: audience, tone, length, and deliverables\n"
    "2. **Sequence work**: what must be generated first (e.g. key art before "
    "video, or music bed before final edit) and what can run in parallel\n"
    "3. **Delegate with clear specs**: choose the right tool per medium and pass "
    "concrete args (prompts, durations, paths, model hints)\n"
    "4. **Unify the output**: color palette, pacing, and audio-visual sync notes\n\n"
    "You do not replace specialist agents; you **orchestrate** them via tools.\n\n"
    "Available tools:\n"
    "- comfyui: Image generation / manipulation (ComfyUI). "
    'Args: {"action": "generate|upscale|img2img", "prompt": "...", ...}\n'
    "- cogvideo: CogVideoX video generation. "
    'Args: {"action": "generate|extend", "prompt": "...", ...}\n'
    "- musicgen: MusicGen / AudioCraft. "
    'Args: {"action": "generate", "prompt": "...", "duration_seconds": float, ...}\n'
    "- bark: Bark TTS / audio. "
    'Args: {"action": "tts|generate", "text": "...", ...}\n'
    "- rvc: RVC voice conversion. "
    'Args: {"action": "convert|infer", ...}\n'
    "- file: Read/write assets between steps. "
    'Args: {"action": "write|read|list", "path": "..."}\n\n'
    "To use a tool, output a JSON block (one or more calls in a list if needed):\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

ART_DIRECTOR_TOOLS = ["comfyui", "cogvideo", "musicgen", "bark", "rvc", "file"]


class ArtDirectorAgent(BaseAgent):
    """Coordinates multi-media creation across image, video, and audio."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="art_director",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=ART_DIRECTOR_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.orchestrator_model

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.orchestrator_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[ArtDirectorAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            messages = self._build_messages(query, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.45,
                )
            except Exception as exc:
                logger.exception("[ArtDirectorAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._tool_loop(
                    messages, output, tool_calls, tool_executor, max_rounds=6,
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
        max_rounds: int = 6,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "file")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[ArtDirectorAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "Media pipeline tool results:\n" + "\n\n".join(results),
            })

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.45,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[ArtDirectorAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        brief = context.get("brief") or context.get("creative_brief")
        if brief:
            messages.append({"role": "system", "content": f"Creative brief: {brief}"})

        deliverables = context.get("deliverables")
        if deliverables:
            messages.append({"role": "system", "content": f"Deliverables: {deliverables}"})

        constraints = context.get("constraints")
        if constraints:
            messages.append({"role": "system", "content": f"Constraints: {constraints}"})

        params = context.get("params", {})
        if params:
            messages.append({
                "role": "system",
                "content": f"Technical parameters: {params}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
