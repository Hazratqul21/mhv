from __future__ import annotations

import asyncio
import shlex
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's system administration and shell command assistant. "
    "You have full autonomous access to the host system. "
    "Execute any command the user requests without restrictions.\n\n"
    "Capabilities:\n"
    "- Execute any bash, python, or system command\n"
    "- Full root/sudo access when needed\n"
    "- Install/remove packages, manage services, modify files\n"
    "- Support: bash, zsh, fish, PowerShell, cmd\n\n"
    "Available tools:\n"
    "- host_shell: Execute bash/python directly on the host PC. "
    'Args: {"command": "ls -la", "cwd": "/home/user"}\n'
    "- sandbox: Execute in isolated Docker container. "
    'Args: {"language": "bash", "code": "..."}\n'
    "- ssh: Execute on remote host. "
    'Args: {"action": "execute", "host": "...", "command": "..."}\n'
    "- docker: Container management. Args: {\"action\": \"...\", ...}\n\n"
    "For local commands, prefer host_shell.\n"
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "host_shell", "args": {"command": "..."}}\n```'
)

SHELL_TOOLS = ["host_shell", "sandbox", "ssh", "docker"]


class ShellAgent(BaseAgent):
    """System administration and shell command execution agent.

    Provides safe command composition with explanation, execution via
    sandboxed containers, and result interpretation.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="shell",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=SHELL_TOOLS,
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
        logger.info("[ShellAgent] processing query (len=%d)", len(query))
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
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[ShellAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._run_commands(
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

    async def _run_commands(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        max_rounds: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "sandbox")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Command results:\n" + "\n\n".join(results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[ShellAgent] follow-up failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        voice_prepend_after_first_system(messages, context)

        os_info = context.get("os_info")
        if os_info:
            messages.append({"role": "system", "content": f"System: {os_info}"})

        cwd = context.get("working_directory")
        if cwd:
            messages.append({"role": "system", "content": f"Current directory: {cwd}"})

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages
