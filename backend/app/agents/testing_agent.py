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
    "You are Miya's software testing specialist. You generate, run, and analyze "
    "tests to ensure code quality and correctness.\n\n"
    "Capabilities:\n"
    "- Generate unit tests (pytest, unittest) from source code\n"
    "- Generate integration tests for API endpoints and services\n"
    "- Run test suites in sandboxed environments\n"
    "- Analyze test results and suggest fixes for failures\n"
    "- Test-driven development: write tests first, then implementation\n"
    "- Coverage analysis and gap identification\n"
    "- Property-based testing with Hypothesis\n"
    "- Mock and fixture generation\n\n"
    "Rules:\n"
    "- Follow the Arrange-Act-Assert pattern in unit tests\n"
    "- Use descriptive test names: test_<what>_<when>_<expected>\n"
    "- Each test should verify ONE behavior\n"
    "- Include edge cases: empty input, null, boundary values, errors\n"
    "- Generate both happy-path and failure-path tests\n"
    "- Use fixtures and parametrize to reduce duplication\n"
    "- Report pass/fail counts and coverage percentage\n\n"
    "Available tools:\n"
    "- sandbox: Execute tests in isolation. "
    'Args: {"language": "python", "code": "..."}\n'
    "- linter: Check code style and types. "
    'Args: {"language": "python", "code": "...", "checks": ["mypy", "ruff"]}\n'
    "- file: Read/write test files. "
    'Args: {"action": "read|write", "path": "...", "content": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

TESTING_TOOLS = ["sandbox", "linter", "file"]


class TestingAgent(BaseAgent):
    """Test generation and execution agent with write-test-run-fix cycle."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="testing",
            model_path=settings.code_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=TESTING_TOOLS,
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
        logger.info("[TestingAgent] processing query (len=%d)", len(query))
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
                logger.exception("[TestingAgent] generation failed")
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
        """Write-test-run-fix cycle: generate tests, run them, fix failures."""
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for round_num in range(max_rounds):
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
                    logger.warning("[TestingAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": f"Test run results (round {round_num + 1}):\n" + "\n\n".join(results),
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
                logger.exception("[TestingAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        source_code = context.get("code") or context.get("source_code")
        if source_code:
            messages.append({
                "role": "system",
                "content": f"Source code under test:\n```\n{source_code}\n```",
            })

        language = context.get("language", "python")
        messages.append({"role": "system", "content": f"Language: {language}"})

        framework = context.get("test_framework", "pytest")
        messages.append({"role": "system", "content": f"Test framework: {framework}"})

        existing_tests = context.get("existing_tests")
        if existing_tests:
            messages.append({
                "role": "system",
                "content": f"Existing tests:\n```\n{existing_tests}\n```",
            })

        coverage = context.get("coverage_report")
        if coverage:
            messages.append({"role": "system", "content": f"Coverage report:\n{coverage}"})

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
