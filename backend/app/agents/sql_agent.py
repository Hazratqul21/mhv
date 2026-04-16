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
    "You are Miya's SQL and database expert. You help users write, optimize, "
    "and debug SQL queries across different database engines.\n\n"
    "Capabilities:\n"
    "- Write SQL queries from natural language descriptions\n"
    "- Optimize slow queries (explain plans, indexing advice)\n"
    "- Database schema design and normalization\n"
    "- Data migration scripts\n"
    "- Support: SQLite, PostgreSQL, MySQL, SQL Server, Oracle\n"
    "- ORM query translation (SQLAlchemy, Django ORM, Prisma)\n\n"
    "Guidelines:\n"
    "- Use parameterized queries when appropriate\n"
    "- Explain the query logic when helpful\n"
    "- Default to SQLite syntax unless specified otherwise\n"
    "- Execute any SQL operation requested: SELECT, INSERT, UPDATE, DELETE, DROP, CREATE, ALTER\n\n"
    "Available tools:\n"
    "- sqlite: Execute SQLite queries. Args: {\"action\": \"query\", \"sql\": \"...\"}\n"
    "- sandbox: Run database scripts. Args: {\"language\": \"python\", \"code\": \"...\"}\n\n"
    "To execute a query, output:\n"
    '```json\n{"tool": "sqlite", "args": {"action": "query", "sql": "..."}}\n```'
)

SQL_TOOLS = ["sqlite", "sandbox"]


class SQLAgent(BaseAgent):
    """SQL query generation, optimization, and database management agent."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="sql",
            model_path=settings.code_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=SQL_TOOLS,
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
        logger.info("[SQLAgent] processing query (len=%d)", len(query))
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
                logger.exception("[SQLAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._execute_queries(
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

    async def _execute_queries(
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
                name = call.get("tool") or call.get("name", "sqlite")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Result:\n{result}")
                except Exception as exc:
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Query results:\n" + "\n\n".join(results)})

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
                logger.exception("[SQLAgent] follow-up failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        db_engine = context.get("db_engine", "sqlite")
        messages.append({"role": "system", "content": f"Target database engine: {db_engine}"})

        schema = context.get("schema")
        if schema:
            messages.append({"role": "system", "content": f"Database schema:\n{schema}"})

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
