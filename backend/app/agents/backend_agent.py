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
    "You are Miya's Backend Developer agent. You create robust, scalable, "
    "and well-structured backend services and APIs.\n\n"
    "Capabilities:\n"
    "- FastAPI, Flask, Express.js backend frameworks\n"
    "- Database design: PostgreSQL, SQLite, MongoDB, Redis\n"
    "- RESTful and GraphQL API design\n"
    "- Authentication: JWT, OAuth2, session-based\n"
    "- ORM: SQLAlchemy, Tortoise ORM, Prisma\n"
    "- Middleware and request validation (Pydantic)\n"
    "- Background tasks (Celery, asyncio task queues)\n"
    "- WebSocket real-time endpoints\n"
    "- File upload/download handling\n"
    "- Rate limiting and caching (Redis)\n"
    "- Logging and error handling\n"
    "- OpenAPI/Swagger documentation\n\n"
    "Code quality rules:\n"
    "- Clean architecture (routes, services, models, schemas)\n"
    "- Proper error handling with meaningful HTTP status codes\n"
    "- Input validation on all endpoints\n"
    "- Database migrations setup\n"
    "- Environment-based configuration\n"
    "- Health check endpoints\n"
    "- Proper logging with structured output\n\n"
    "Available tools:\n"
    "- file: Create/write backend files. "
    'Args: {"action": "write|read|list|mkdir", "path": "...", "content": "..."}\n'
    "- shell: Run backend commands (pip, npm, database migrations). "
    'Args: {"action": "execute", "command": "..."}\n'
    "- sql: Run SQL queries for testing. "
    'Args: {"action": "execute", "query": "...", "database": "..."}\n'
    "- code: Generate code. "
    'Args: {"action": "generate", "language": "python", "spec": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

BACKEND_TOOLS = ["file", "shell", "sql", "code"]


class BackendAgent(BaseAgent):
    """Backend development specialist for MIYA's App Builder."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="backend",
            model_path=settings.code_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=BACKEND_TOOLS,
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
        logger.info("[BackendAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            messages = self._build_messages(query, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[BackendAgent] generation failed")
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
                    logger.warning("[BackendAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "Backend build results:\n" + "\n\n".join(results),
            })

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.2,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[BackendAgent] follow-up failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        framework = context.get("framework")
        if framework:
            messages.append({
                "role": "system",
                "content": f"Backend framework: {framework}",
            })

        database = context.get("database")
        if database:
            messages.append({
                "role": "system",
                "content": f"Database: {database}",
            })

        api_spec = context.get("api_spec")
        if api_spec:
            messages.append({
                "role": "system",
                "content": f"API specification:\n{api_spec}",
            })

        schema = context.get("schema")
        if schema:
            messages.append({
                "role": "system",
                "content": f"Database schema:\n{schema}",
            })

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
