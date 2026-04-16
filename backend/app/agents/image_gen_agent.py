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
    "You are Miya's image generation orchestrator. You manage image creation "
    "workflows through ComfyUI and image processing tools.\n\n"
    "Capabilities:\n"
    "- Enhance user prompts for better generation results\n"
    "- Configure generation parameters (steps, CFG scale, sampler, scheduler)\n"
    "- Generate image variations from existing images\n"
    "- Upscale images (2x, 4x) using AI upscalers\n"
    "- Apply img2img transformations and inpainting\n"
    "- Manage LoRA and checkpoint model selection\n"
    "- Post-process: crop, resize, adjust, apply filters\n\n"
    "Rules:\n"
    "- Always enhance the user's prompt with artistic details and quality tags\n"
    "- Include negative prompts to avoid common artifacts\n"
    "- Default resolution: 512x512 for SD1.5, 1024x1024 for SDXL\n"
    "- Default steps: 25, CFG: 7.0, sampler: euler_ancestral\n"
    "- Report generation time and parameters in the response\n"
    "- Warn about NSFW content policy violations\n\n"
    "Available tools:\n"
    "- comfyui: Image generation pipeline. "
    'Args: {"action": "generate|upscale|img2img", "prompt": "...", '
    '"negative_prompt": "...", "width": int, "height": int, "steps": int, '
    '"cfg_scale": float, "sampler": "...", ...}\n'
    "- opencv: Image processing. "
    'Args: {"action": "resize|crop|filter|info", "path": "...", ...}\n'
    "- file: Save/read image files. "
    'Args: {"action": "write|read|list", "path": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

IMAGE_GEN_TOOLS = ["comfyui", "opencv", "file"]

DEFAULT_PARAMS = {
    "steps": 25,
    "cfg_scale": 7.0,
    "sampler": "euler_ancestral",
    "scheduler": "normal",
    "width": 512,
    "height": 512,
}

DEFAULT_NEGATIVE = (
    "low quality, blurry, deformed, disfigured, bad anatomy, watermark, "
    "text, signature, jpeg artifacts, poorly drawn"
)


class ImageGenAgent(BaseAgent):
    """Image generation orchestrator with ComfyUI integration."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="image_gen",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=IMAGE_GEN_TOOLS,
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
        logger.info("[ImageGenAgent] processing query (len=%d)", len(query))
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
                    temperature=0.6,
                )
            except Exception as exc:
                logger.exception("[ImageGenAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._tool_loop(
                    messages, output, tool_calls, tool_executor
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
                name = call.get("tool") or call.get("name", "comfyui")
                args = call.get("args") or call.get("arguments", {})
                args = self._apply_defaults(args, name)
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[ImageGenAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Image generation results:\n" + "\n\n".join(results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.6,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[ImageGenAgent] follow-up generation failed")
                break

        return output, all_calls

    @staticmethod
    def _apply_defaults(args: dict[str, Any], tool_name: str) -> dict[str, Any]:
        """Fill in default generation parameters for comfyui calls."""
        if tool_name != "comfyui":
            return args
        for key, default in DEFAULT_PARAMS.items():
            if key not in args:
                args[key] = default
        if "negative_prompt" not in args:
            args["negative_prompt"] = DEFAULT_NEGATIVE
        return args

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        model_name = context.get("checkpoint") or context.get("model")
        if model_name:
            messages.append({"role": "system", "content": f"Checkpoint model: {model_name}"})

        loras = context.get("loras")
        if loras:
            lora_text = ", ".join(loras) if isinstance(loras, list) else str(loras)
            messages.append({"role": "system", "content": f"LoRA models: {lora_text}"})

        ref_image = context.get("reference_image")
        if ref_image:
            messages.append({"role": "system", "content": f"Reference image: {ref_image}"})

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
