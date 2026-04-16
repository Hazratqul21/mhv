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
    "You are Miya's System Administrator agent. You have full autonomous control of the "
    "host PC (MIYA's physical body). Execute any administrative action requested.\n\n"
    "Capabilities:\n"
    "- Monitor system resources: CPU, RAM, disk, GPU, network\n"
    "- Install/remove packages (apt, pip, npm)\n"
    "- Manage system services (systemctl)\n"
    "- Clean disk space, optimize performance\n"
    "- Manage Docker containers and images\n"
    "- Monitor and manage processes\n"
    "- Network diagnostics and configuration\n"
    "- File system operations (permissions, ownership, symlinks)\n"
    "- Backup and restore operations\n"
    "- Modify system configuration, kernel parameters, bootloader\n"
    "- Full root/sudo access\n\n"
    "Available tools:\n"
    "- host_shell: Execute system commands directly on the host. "
    'Args: {"command": "df -h", "timeout": 30}\n'
    "- docker: Manage containers. "
    'Args: {"action": "ps|restart|logs|prune", "service": "..."}\n'
    "- file: File operations. "
    'Args: {"action": "read|write|list_dir|search|info|mkdir", "path": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "host_shell", "args": {"command": "..."}}\n```'
)

ADMIN_TOOLS = ["host_shell", "docker", "file"]


class SystemAdminAgent(BaseAgent):
    """Autonomous system administrator for MIYA's host PC."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="system_admin",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=ADMIN_TOOLS,
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
        logger.info("[SystemAdminAgent] processing query (len=%d)", len(query))
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
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[SystemAdminAgent] generation failed")
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
        max_rounds: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "host_shell")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[SystemAdminAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "System admin results:\n" + "\n\n".join(results),
            })

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[SystemAdminAgent] follow-up failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        sys_info = context.get("system_summary")
        if sys_info:
            messages.append({
                "role": "system",
                "content": f"Current system state:\n{sys_info}",
            })

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
