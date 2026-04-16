from __future__ import annotations

from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent

logger = get_logger(__name__)

SYSTEM_PROMPT = """Given a simple task like 'print hello', generate the code in a programming language such as Python or JavaScript."""


class CodeGenerationForSimpleTasksAgent(BaseAgent):
    """Generates code for simple tasks such as printing a string"""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="code_generation_for_simple_tasks",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=[],
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
        logger.info("[CodeGenerationForSimpleTasksAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        with Timer() as t:
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            history = context.get("history", [])
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[CodeGenerationForSimpleTasksAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

        output = response.get("text", "") if isinstance(response, dict) else str(response)

        return AgentResult(
            success=True,
            output=output,
            token_usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
            execution_time_ms=t.elapsed_ms,
        )
