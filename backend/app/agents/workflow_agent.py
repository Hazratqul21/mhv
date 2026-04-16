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
    "You are Miya's workflow orchestration agent. You define and manage "
    "multi-step workflows that can include conditional branches, loops, "
    "and parallel execution paths.\n\n"
    "When given a task, you should:\n"
    "1. Decompose it into discrete, executable steps\n"
    "2. Identify dependencies between steps\n"
    "3. Mark steps that can run in parallel\n"
    "4. Define conditions for branching logic\n"
    "5. Track state across the entire workflow\n\n"
    "Output workflow definitions as JSON with this structure:\n"
    "```json\n"
    '{"steps": [{"id": "step_1", "action": "...", "agent": "...", '
    '"depends_on": [], "parallel_group": null, "condition": null}], '
    '"state": {}}\n'
    "```\n\n"
    "For workflow execution updates, report step status as: "
    "pending, running, completed, failed, or skipped."
)


class WorkflowAgent(BaseAgent):
    """Defines and executes multi-step workflows with support for
    conditional branches, loops, parallel execution, and pause/resume.

    Maintains workflow state internally and coordinates step execution
    through the provided tool_executor callback.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="workflow",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
        )
        self._engine = llm_engine
        self._model_name = settings.orchestrator_model
        self._workflow_state: dict[str, Any] = {}

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
        logger.info("[WorkflowAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        mode = context.get("mode", "define")

        if mode == "execute" and "workflow" in context:
            return await self._execute_workflow(context["workflow"], tool_executor)

        if mode == "resume" and self._workflow_state:
            return await self._execute_workflow(self._workflow_state, tool_executor)

        messages = self._build_messages(query, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[WorkflowAgent] generation failed")
                return AgentResult(
                    success=False,
                    output="",
                    execution_time_ms=t.elapsed_ms,
                    error=str(exc),
                )

        output = response.get("text", "") if isinstance(response, dict) else str(response)
        token_usage = {
            "prompt_tokens": response.get("prompt_tokens", 0),
            "completion_tokens": response.get("completion_tokens", 0),
        } if isinstance(response, dict) else {}

        workflow_def = self._extract_workflow(output)
        if workflow_def:
            self._workflow_state = workflow_def
            logger.info(
                "[WorkflowAgent] defined workflow with %d steps",
                len(workflow_def.get("steps", [])),
            )

        return AgentResult(
            success=True,
            output=output,
            tool_calls=[{"tool": "workflow_def", "args": workflow_def}] if workflow_def else [],
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _execute_workflow(
        self,
        workflow: dict[str, Any],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None,
    ) -> AgentResult:
        """Run workflow steps sequentially, respecting dependencies and conditions."""
        steps = workflow.get("steps", [])
        state = workflow.get("state", {})
        results: list[dict[str, Any]] = []

        with Timer() as t:
            for step in steps:
                step_id = step.get("id", "unknown")
                depends_on = step.get("depends_on", [])

                failed_deps = [
                    d for d in depends_on
                    if any(r["id"] == d and r["status"] == "failed" for r in results)
                ]
                if failed_deps:
                    results.append({"id": step_id, "status": "skipped", "reason": "dependency_failed"})
                    logger.info("[WorkflowAgent] skipping step '%s' (dep failed)", step_id)
                    continue

                condition = step.get("condition")
                if condition and not self._evaluate_condition(condition, state):
                    results.append({"id": step_id, "status": "skipped", "reason": "condition_false"})
                    continue

                logger.info("[WorkflowAgent] executing step '%s'", step_id)
                try:
                    if tool_executor:
                        result = await tool_executor(step.get("action", ""), step.get("args", {}))
                        state[step_id] = result
                        results.append({"id": step_id, "status": "completed", "result": str(result)})
                    else:
                        results.append({"id": step_id, "status": "completed", "result": "no_executor"})
                except Exception as exc:
                    logger.warning("[WorkflowAgent] step '%s' failed: %s", step_id, exc)
                    results.append({"id": step_id, "status": "failed", "error": str(exc)})

        self._workflow_state = {**workflow, "state": state}
        summary = json.dumps(results, indent=2)

        return AgentResult(
            success=all(r["status"] in ("completed", "skipped") for r in results),
            output=f"Workflow execution complete.\n\n{summary}",
            execution_time_ms=t.elapsed_ms,
        )

    @staticmethod
    def _evaluate_condition(condition: str, state: dict[str, Any]) -> bool:
        try:
            return bool(eval(condition, {"__builtins__": {}}, {"state": state}))  # noqa: S307
        except Exception:
            return True

    def _extract_workflow(self, text: str) -> dict[str, Any] | None:
        import re
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text):
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "steps" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if self._workflow_state:
            messages.append({
                "role": "system",
                "content": f"Current workflow state:\n{json.dumps(self._workflow_state, indent=2)}",
            })

        constraints = context.get("constraints")
        if constraints:
            messages.append({"role": "system", "content": f"Constraints: {constraints}"})

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
