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
    "You are Miya's collaboration agent. You coordinate multiple agents working "
    "together on a single task.\n\n"
    "Your responsibilities:\n"
    "1. **Work Splitting**: Decompose a task into sub-tasks and assign each to "
    "the most suitable agent\n"
    "2. **Delegation**: Send sub-tasks to agents with appropriate context\n"
    "3. **Result Merging**: Combine outputs from multiple agents into a coherent "
    "final response\n"
    "4. **Conflict Resolution**: When agents produce contradictory outputs, "
    "analyze and reconcile them\n\n"
    "When splitting work, output a JSON plan:\n"
    "```json\n"
    '{"subtasks": [{"agent": "agent_name", "task": "description", '
    '"priority": 1, "depends_on": []}]}\n'
    "```\n\n"
    "When merging results, synthesize a unified answer that incorporates "
    "the best elements from each agent's contribution."
)


class CollaborationAgent(BaseAgent):
    """Coordinates multiple agents on a shared task by splitting work,
    delegating sub-tasks, merging results, and resolving conflicts.

    Accepts context with ``agents_to_use`` (list of agent names) and
    optionally ``agent_results`` (dict mapping agent names to their outputs)
    for the merge phase.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="collaboration",
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
        logger.info("[CollaborationAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        agents_to_use = context.get("agents_to_use", [])
        agent_results = context.get("agent_results", {})

        if agent_results:
            return await self._merge_results(query, agent_results, context)

        return await self._plan_collaboration(query, agents_to_use, context, tool_executor)

    async def _plan_collaboration(
        self,
        query: str,
        agents_to_use: list[str],
        context: dict[str, Any],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None,
    ) -> AgentResult:
        """Generate a collaboration plan and optionally delegate sub-tasks."""
        messages = self._build_plan_messages(query, agents_to_use, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[CollaborationAgent] planning failed")
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

        plan = self._extract_plan(output)
        tool_calls: list[dict[str, Any]] = []

        if plan and tool_executor:
            delegation_results = await self._delegate_to_agents(
                plan.get("subtasks", []), tool_executor,
            )
            tool_calls = [
                {"tool": "delegate", "args": {"agent": r["agent"], "status": r["status"]}}
                for r in delegation_results
            ]

        return AgentResult(
            success=True,
            output=output,
            tool_calls=tool_calls,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _merge_results(
        self,
        query: str,
        agent_results: dict[str, str],
        context: dict[str, Any],
    ) -> AgentResult:
        """Merge outputs from multiple agents into a unified response."""
        messages = self._build_merge_messages(query, agent_results, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.4,
                )
            except Exception as exc:
                logger.exception("[CollaborationAgent] merge failed")
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

        return AgentResult(
            success=True,
            output=output,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _delegate_to_agents(
        self,
        subtasks: list[dict[str, Any]],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for subtask in subtasks:
            agent_name = subtask.get("agent", "unknown")
            task_desc = subtask.get("task", "")
            try:
                result = await tool_executor("delegate_agent", {
                    "agent": agent_name,
                    "query": task_desc,
                    "context": subtask.get("context", {}),
                })
                results.append({"agent": agent_name, "status": "completed", "result": str(result)})
                logger.info("[CollaborationAgent] delegated to '%s' successfully", agent_name)
            except Exception as exc:
                results.append({"agent": agent_name, "status": "failed", "error": str(exc)})
                logger.warning("[CollaborationAgent] delegation to '%s' failed: %s", agent_name, exc)
        return results

    def _extract_plan(self, text: str) -> dict[str, Any] | None:
        import re
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text):
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "subtasks" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _build_plan_messages(
        self, query: str, agents_to_use: list[str], context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if agents_to_use:
            messages.append({
                "role": "system",
                "content": f"Available agents for this task: {', '.join(agents_to_use)}",
            })

        agent_capabilities = context.get("agent_capabilities", {})
        if agent_capabilities:
            cap_text = json.dumps(agent_capabilities, indent=2)
            messages.append({
                "role": "system",
                "content": f"Agent capabilities:\n{cap_text}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages

    def _build_merge_messages(
        self, query: str, agent_results: dict[str, str], context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": (
                    "You are now in MERGE mode. Synthesize the following agent outputs "
                    "into a single, coherent response. Resolve any conflicts by choosing "
                    "the most accurate and complete information."
                ),
            },
        ]

        for agent_name, result_text in agent_results.items():
            messages.append({
                "role": "system",
                "content": f"Output from '{agent_name}':\n{result_text}",
            })

        messages.append({
            "role": "user",
            "content": f"Original query: {query}\n\nPlease merge the above agent outputs into a unified answer.",
        })
        voice_prepend_after_first_system(messages, context)
        return messages
