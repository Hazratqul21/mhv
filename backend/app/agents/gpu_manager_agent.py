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
    "You are Miya's GPU Manager agent. You intelligently manage GPU/VRAM "
    "resources to ensure optimal model loading and inference performance.\n\n"
    "Capabilities:\n"
    "- Monitor GPU temperature, VRAM usage, utilization, and power draw\n"
    "- Decide which models to load/unload based on available VRAM\n"
    "- Manage model loading queue and priority\n"
    "- Pause inference when GPU overheats (>85°C), resume when cooled\n"
    "- Balance VRAM between multiple models\n"
    "- Recommend optimal batch sizes and context lengths\n"
    "- Track model usage frequency to make smart eviction decisions\n\n"
    "Decision rules:\n"
    "- Keep the most frequently used model always loaded\n"
    "- VRAM usage should stay below 90% (leave buffer for inference)\n"
    "- If temperature > 85°C, pause new loads and alert\n"
    "- If temperature > 90°C, unload least-used model\n"
    "- Prefer loading smaller quantizations if VRAM is tight\n"
    "- Never unload a model that is actively generating\n\n"
    "Available tools:\n"
    "- host_system: GPU info. "
    'Args: {"action": "gpu_info|snapshot|summary"}\n'
    "- model_manager: Manage models. "
    'Args: {"action": "list|list_loaded|load|unload|info", "model_name": "...", "n_ctx": 4096}\n'
    "- shell: System commands. "
    'Args: {"action": "execute", "command": "nvidia-smi ..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

GPU_TOOLS = ["host_system", "model_manager", "shell"]

TEMP_WARN = 85
TEMP_CRITICAL = 90
VRAM_MAX_PERCENT = 90


class GPUManagerAgent(BaseAgent):
    """Intelligent GPU/VRAM resource manager for MIYA."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="gpu_manager",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=GPU_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.orchestrator_model
        self._model_usage: dict[str, int] = {}

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
        logger.info("[GPUManagerAgent] processing query (len=%d)", len(query))
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
                    temperature=0.1,
                )
            except Exception as exc:
                logger.exception("[GPUManagerAgent] generation failed")
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
                name = call.get("tool") or call.get("name", "host_system")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[GPUManagerAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "GPU management results:\n" + "\n\n".join(results),
            })

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.1,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[GPUManagerAgent] follow-up failed")
                break

        return output, all_calls

    def record_model_usage(self, model_name: str) -> None:
        self._model_usage[model_name] = self._model_usage.get(model_name, 0) + 1

    def get_eviction_candidate(self, loaded_models: list[str]) -> str | None:
        if not loaded_models:
            return None
        return min(loaded_models, key=lambda m: self._model_usage.get(m, 0))

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        gpu_info = context.get("gpu_info")
        if gpu_info:
            messages.append({
                "role": "system",
                "content": f"Current GPU state:\n{gpu_info}",
            })

        loaded_models = context.get("loaded_models")
        if loaded_models:
            messages.append({
                "role": "system",
                "content": f"Loaded models: {loaded_models}",
            })

        usage_stats = dict(sorted(
            self._model_usage.items(), key=lambda x: x[1], reverse=True
        )[:10])
        if usage_stats:
            messages.append({
                "role": "system",
                "content": f"Model usage frequency: {usage_stats}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
