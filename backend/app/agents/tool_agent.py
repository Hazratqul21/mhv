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
    "You are Miya's tool-use agent. You have access to a set of tools and must "
    "decide which tool(s) to call based on the user's request. After receiving "
    "tool results, synthesize a final answer.\n\n"
    "To call a tool, output a JSON block:\n"
    '```json\n{"tool": "<tool_name>", "args": {…}}\n```\n\n'
    "You may call multiple tools sequentially. Once you have all the information "
    "you need, provide a final answer WITHOUT any tool-call blocks."
)

MAX_TOOL_ROUNDS = 5


class ToolAgent(BaseAgent):
    """Dynamic tool-selection agent.

    The agent prompts the LLM with the available tool definitions, parses
    tool-call JSON from the response, executes tools via the provided
    ``tool_executor``, and feeds results back until the LLM produces a
    final answer (no more tool calls) or the iteration limit is reached.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> None:
        settings = get_settings()
        super().__init__(
            name="tool",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=[td["name"] for td in (tool_definitions or [])],
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model
        self._tool_definitions = tool_definitions or []

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
        logger.info("[ToolAgent] processing query (len=%d)", len(query))
        context = context or {}

        if tool_executor is None:
            return AgentResult(
                success=False,
                output="",
                error="ToolAgent requires a tool_executor callback.",
            )

        messages = self._build_messages(query, context)
        all_tool_calls: list[dict[str, Any]] = []
        aggregated_usage: dict[str, int] = {}

        self.ensure_loaded()
        with Timer() as t:
            for round_idx in range(MAX_TOOL_ROUNDS):
                try:
                    response = await self._engine.chat(
                        model_filename=self._model_name,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.1,
                    )
                except Exception as exc:
                    logger.exception("[ToolAgent] generation failed (round %d)", round_idx)
                    return AgentResult(
                        success=False,
                        output="",
                        tool_calls=all_tool_calls,
                        token_usage=aggregated_usage,
                        execution_time_ms=t.elapsed_ms,
                        error=str(exc),
                    )

                output = response.get("text", "") if isinstance(response, dict) else str(response)
                self._merge_usage(aggregated_usage, response)

                tool_calls = self._parse_tool_calls(output)
                if not tool_calls:
                    return AgentResult(
                        success=True,
                        output=output,
                        tool_calls=all_tool_calls,
                        token_usage=aggregated_usage,
                        execution_time_ms=t.elapsed_ms,
                    )

                tool_results = await self._execute_tools(tool_calls, tool_executor)
                all_tool_calls.extend(
                    {"tool": c.get("tool") or c.get("name", "?"), "args": c.get("args", {})}
                    for c in tool_calls
                )

                messages.append({"role": "assistant", "content": output})
                messages.append({
                    "role": "user",
                    "content": "Tool results:\n" + "\n\n".join(tool_results),
                })

        return AgentResult(
            success=True,
            output=output,
            tool_calls=all_tool_calls,
            token_usage=aggregated_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _execute_tools(
        self,
        tool_calls: list[dict[str, Any]],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]],
    ) -> list[str]:
        results: list[str] = []
        for call in tool_calls:
            name = call.get("tool") or call.get("name", "unknown")
            args = call.get("args") or call.get("arguments", {})

            if name not in self.available_tools:
                results.append(f"[{name}] Error: tool not available")
                logger.warning("[ToolAgent] requested unavailable tool '%s'", name)
                continue

            try:
                result = await tool_executor(name, args)
                results.append(f"[{name}] {result}")
            except Exception as exc:
                logger.warning("[ToolAgent] tool '%s' failed: %s", name, exc)
                results.append(f"[{name}] Error: {exc}")

        return results

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        tool_desc = self._render_tool_definitions()
        system = self.system_prompt
        if tool_desc:
            system += f"\n\nAvailable tools:\n{tool_desc}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        voice_prepend_after_first_system(messages, context)

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages

    def _render_tool_definitions(self) -> str:
        if not self._tool_definitions:
            return ""

        lines: list[str] = []
        for td in self._tool_definitions:
            name = td.get("name", "unnamed")
            desc = td.get("description", "")
            params = td.get("parameters", {})
            lines.append(f"- **{name}**: {desc}")
            if params:
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "any") if isinstance(pinfo, dict) else "any"
                    pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
                    lines.append(f"    - {pname} ({ptype}): {pdesc}")
        return "\n".join(lines)

    @staticmethod
    def _merge_usage(target: dict[str, int], response: Any) -> None:
        src = response.get("usage", {}) if isinstance(response, dict) else {}
        for key, val in src.items():
            if isinstance(val, int):
                target[key] = target.get(key, 0) + val
