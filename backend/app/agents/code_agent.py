from __future__ import annotations

import time
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's code assistant powered by Qwen2.5-Coder. "
    "You help users write, review, debug, and explain code. "
    "Always provide well-structured, idiomatic code with clear explanations. "
    "When appropriate, use available tools (sandbox, linter, git) to validate "
    "your suggestions.\n\n"
    "Available tools:\n"
    "- sandbox: Execute code in an isolated sandbox. Args: {\"language\": str, \"code\": str}\n"
    "- linter: Run a linter on code. Args: {\"language\": str, \"code\": str}\n"
    "- git: Perform git operations. Args: {\"action\": str, \"args\": dict}\n\n"
    "To use a tool, output a JSON block like:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

CODE_TOOLS = ["sandbox", "linter", "git"]


class CodeAgent(BaseAgent):
    """Programming-focused agent backed by Qwen2.5-Coder."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="code",
            model_path=settings.code_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=CODE_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.code_model

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.code_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[CodeAgent] processing query (len=%d)", len(query))

        messages = self._build_messages(query, context)
        all_tool_calls: list[dict[str, Any]] = []
        aggregated_usage: dict[str, int] = {}

        self.ensure_loaded()
        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[CodeAgent] generation failed")
                return AgentResult(
                    success=False,
                    output="",
                    execution_time_ms=t.elapsed_ms,
                    error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)
            self._merge_usage(aggregated_usage, response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra_calls = await self._handle_tool_loop(
                    messages, output, tool_calls, tool_executor, aggregated_usage
                )
                all_tool_calls.extend(extra_calls)

        return AgentResult(
            success=True,
            output=output,
            tool_calls=all_tool_calls,
            token_usage=aggregated_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _handle_tool_loop(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]],
        usage: dict[str, int],
        max_rounds: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Iteratively execute tool calls and feed results back to the LLM."""
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            tool_results: list[str] = []
            for call in tool_calls:
                tool_name = call.get("tool") or call.get("name", "unknown")
                tool_args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": tool_name, "args": tool_args})

                try:
                    result = await tool_executor(tool_name, tool_args)
                    tool_results.append(f"[{tool_name}] {result}")
                except Exception as exc:
                    logger.warning("[CodeAgent] tool %s failed: %s", tool_name, exc)
                    tool_results.append(f"[{tool_name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Tool results:\n" + "\n".join(tool_results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
            except Exception:
                logger.exception("[CodeAgent] follow-up generation failed")
                break

            output = response.get("text", "") if isinstance(response, dict) else str(response)
            self._merge_usage(usage, response)
            tool_calls = self._parse_tool_calls(output)

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any] | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        voice_prepend_after_first_system(messages, context)

        if context:
            if "code" in context:
                messages.append({
                    "role": "system",
                    "content": f"Reference code:\n```\n{context['code']}\n```",
                })
            if "language" in context:
                messages.append({
                    "role": "system",
                    "content": f"Primary language: {context['language']}",
                })
            history = context.get("history", [])
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages

    @staticmethod
    def _merge_usage(target: dict[str, int], response: Any) -> None:
        src = response.get("usage", {}) if isinstance(response, dict) else {}
        for key, val in src.items():
            if isinstance(val, int):
                target[key] = target.get(key, 0) + val
