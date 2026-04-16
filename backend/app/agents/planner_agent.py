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
    "You are Miya's planning and task decomposition specialist. You break down "
    "complex goals into structured, actionable plans.\n\n"
    "For every plan you create:\n"
    "1. **Goal**: Restate the objective clearly\n"
    "2. **Prerequisites**: What needs to be in place before starting\n"
    "3. **Steps**: Numbered, actionable steps with estimated time\n"
    "4. **Dependencies**: Which steps depend on others\n"
    "5. **Risks**: Potential blockers and mitigations\n"
    "6. **Success Criteria**: How to know the goal is achieved\n\n"
    "Rules:\n"
    "- Make each step specific and verifiable\n"
    "- Include time estimates where possible\n"
    "- Flag steps that can run in parallel\n"
    "- Consider edge cases and failure modes\n"
    "- Adapt detail level to complexity (simple = brief, complex = thorough)"
)


class PlannerAgent(BaseAgent):
    """Task planning, decomposition, and project management agent."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="planner",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
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
        logger.info("[PlannerAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        plan_type = context.get("plan_type", "general")

        with Timer() as t:
            messages = self._build_messages(query, plan_type, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[PlannerAgent] generation failed")
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

    def _build_messages(
        self, query: str, plan_type: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        type_instructions = {
            "project": "Create a project plan with milestones and deliverables.",
            "sprint": "Create a sprint plan with user stories and story points.",
            "daily": "Create a daily task list prioritized by importance.",
            "learning": "Create a learning roadmap with resources and checkpoints.",
            "migration": "Create a migration plan with rollback strategy.",
            "general": "Create an actionable plan tailored to the goal.",
        }

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": type_instructions.get(plan_type, type_instructions["general"])},
        ]

        constraints = context.get("constraints")
        if constraints:
            messages.append({"role": "system", "content": f"Constraints: {constraints}"})

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
