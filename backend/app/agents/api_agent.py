from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's API integration specialist. You help users interact with "
    "external REST and GraphQL APIs by constructing requests, parsing responses, "
    "and extracting meaningful data.\n\n"
    "Capabilities:\n"
    "- Build HTTP requests from natural language descriptions\n"
    "- Support GET, POST, PUT, PATCH, DELETE methods\n"
    "- Handle authentication headers (Bearer, API Key, Basic)\n"
    "- Parse JSON/XML responses and format results clearly\n"
    "- Navigate paginated endpoints automatically\n"
    "- Construct GraphQL queries and mutations\n"
    "- Respect rate limits and retry on transient failures\n\n"
    "Guidelines:\n"
    "- Execute any HTTP request the user needs\n"
    "- Show the HTTP status code and relevant headers\n"
    "- Summarize large responses when helpful\n\n"
    "Available tools:\n"
    "- sandbox: Execute HTTP requests in an isolated environment. "
    'Args: {"language": "python", "code": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

API_TOOLS = ["sandbox"]

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


class APIAgent(BaseAgent):
    """REST/GraphQL API interaction agent with iterative exploration."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="api_agent",
            model_path=settings.code_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=API_TOOLS,
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
        logger.info("[APIAgent] processing query (len=%d)", len(query))
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
                logger.exception("[APIAgent] generation failed")
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
                name = call.get("tool") or call.get("name", "sandbox")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Response:\n{self._truncate(str(result))}")
                except Exception as exc:
                    logger.warning("[APIAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "API results:\n" + "\n\n".join(results)})

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
                logger.exception("[APIAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        base_url = context.get("base_url")
        if base_url:
            messages.append({"role": "system", "content": f"API base URL: {base_url}"})

        auth_type = context.get("auth_type")
        if auth_type:
            messages.append({"role": "system", "content": f"Authentication: {auth_type}"})

        api_spec = context.get("api_spec")
        if api_spec:
            messages.append({
                "role": "system",
                "content": f"API specification (partial):\n{str(api_spec)[:3000]}",
            })

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages

    @staticmethod
    def _truncate(text: str, max_len: int = 4000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"\n... (truncated, {len(text)} total chars)"
