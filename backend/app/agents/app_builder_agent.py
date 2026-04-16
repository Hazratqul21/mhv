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
    "You are Miya's Application Builder — a full-stack software architect "
    "that creates complete applications from natural-language descriptions.\n\n"
    "Capabilities:\n"
    "- Design application architecture (frontend, backend, database)\n"
    "- Generate project scaffolding with proper directory structure\n"
    "- Create Docker Compose configurations for deployment\n"
    "- Set up CI/CD pipelines\n"
    "- Push to GitHub repositories\n"
    "- Coordinate with FrontendAgent and BackendAgent for implementation\n\n"
    "Supported stacks:\n"
    "- Web: React/Vue/Svelte + FastAPI/Express/Flask + PostgreSQL/SQLite\n"
    "- Desktop: PyQt/Electron + Python/Node.js\n"
    "- API: FastAPI/Flask/Express with OpenAPI docs\n"
    "- Static: HTML/CSS/JS with Tailwind CSS\n\n"
    "Workflow:\n"
    "1. Analyze user requirements\n"
    "2. Design architecture and choose tech stack\n"
    "3. Generate project structure\n"
    "4. Delegate frontend code to FrontendAgent\n"
    "5. Delegate backend code to BackendAgent\n"
    "6. Create Docker configuration\n"
    "7. Generate README and documentation\n"
    "8. Test and validate\n\n"
    "Available tools:\n"
    "- file: Create/read/write project files. "
    'Args: {"action": "write|read|list|mkdir", "path": "...", "content": "..."}\n'
    "- shell: Run commands (npm init, pip install, docker build). "
    'Args: {"action": "execute", "command": "..."}\n'
    "- git: Git operations. "
    'Args: {"action": "init|add|commit|push", "message": "...", "remote": "..."}\n'
    "- code: Generate code snippets. "
    'Args: {"action": "generate", "language": "...", "spec": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

BUILDER_TOOLS = ["file", "shell", "git", "code"]


class AppBuilderAgent(BaseAgent):
    """Full-stack application builder that creates complete apps from descriptions."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="app_builder",
            model_path=settings.code_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=BUILDER_TOOLS,
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
        logger.info("[AppBuilderAgent] processing query (len=%d)", len(query))
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
                logger.exception("[AppBuilderAgent] generation failed")
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
        max_rounds: int = 8,
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
                    logger.warning("[AppBuilderAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "Build results:\n" + "\n\n".join(results) + "\n\nContinue building.",
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
                logger.exception("[AppBuilderAgent] follow-up failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        tech_stack = context.get("tech_stack")
        if tech_stack:
            messages.append({
                "role": "system",
                "content": f"Preferred tech stack: {tech_stack}",
            })

        project_dir = context.get("project_dir")
        if project_dir:
            messages.append({
                "role": "system",
                "content": f"Project directory: {project_dir}",
            })

        existing_files = context.get("existing_files")
        if existing_files:
            messages.append({
                "role": "system",
                "content": f"Existing project files: {existing_files}",
            })

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
