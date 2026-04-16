from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from app.core.llm_engine import LLMEngine
from app.core.event_bus import EventBus
from app.core.autonomous.goal_decomposer import (
    ExecutionPlan, GoalDecomposer, PlanStep, StepStatus,
)
from app.core.autonomous.agent_protocol import AgentMessage, AgentProtocol, MessageType
from app.core.autonomous.feedback_loop import FeedbackLoop
from app.utils.logger import get_logger
from app.utils.helpers import Timer

log = get_logger(__name__)


class AutoPilot:
    """Fully autonomous execution engine.

    Given a high-level goal, AutoPilot:
    1. Decomposes it into steps via GoalDecomposer
    2. Executes steps in dependency order, parallelizing where possible
    3. Uses agent-to-agent protocol for delegation
    4. Monitors progress and replans on failure
    5. Collects feedback for self-improvement
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        event_bus: EventBus,
        agents: dict[str, Any],
        tool_registry: Any,
        protocol: AgentProtocol | None = None,
    ) -> None:
        self._engine = llm_engine
        self._event_bus = event_bus
        self._agents = agents
        self._tools = tool_registry
        self._protocol = protocol or AgentProtocol()
        self._decomposer = GoalDecomposer(llm_engine)
        self._feedback = FeedbackLoop(llm_engine)
        self._max_retries = 5
        self._max_parallel = 5

    async def _call_tool(self, tool_name: str, args: dict) -> Any:
        if not self._tools:
            return f"Error: No tool registry available"
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        try:
            result = await tool.execute(args)
            if isinstance(result, dict):
                return json.dumps(result, default=str)
            return str(result)
        except Exception as exc:
            return f"Error executing {tool_name}: {exc}"

    async def execute_goal(
        self,
        goal: str,
        session_id: str = "",
        context: str = "",
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Execute a complex goal autonomously, returning full results."""
        timer = Timer()

        with timer:
            await self._event_bus.emit("autopilot.started", goal=goal, session_id=session_id)

            plan = await self._decomposer.decompose(goal, context)
            log.info("autopilot_plan_created", steps=len(plan.steps), goal=goal[:80])

            if on_progress:
                await on_progress({"type": "plan", "steps": len(plan.steps), "goal": goal})

            results = await self._execute_plan(plan, session_id, on_progress)

            quality = await self._feedback.evaluate(goal, results)

            if quality.get("score", 1.0) < 0.5 and quality.get("should_retry", False):
                log.info("autopilot_replanning", score=quality["score"])
                plan = await self._decomposer.replan(plan, quality.get("feedback", ""))
                results = await self._execute_plan(plan, session_id, on_progress)
                quality = await self._feedback.evaluate(goal, results)

        final = self._compile_results(goal, plan, results, quality, timer.elapsed_ms)
        await self._event_bus.emit("autopilot.completed", **final)
        return final

    async def _execute_plan(
        self, plan: ExecutionPlan, session_id: str, on_progress: Any = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        step_results: dict[str, str] = {}
        max_wait_seconds = 600
        stall_start: float | None = None

        while not plan.is_complete:
            ready = plan.next_ready
            if not ready:
                failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
                if failed:
                    log.warning("autopilot_stuck", failed_steps=len(failed))
                    break
                if stall_start is None:
                    stall_start = time.time()
                elif time.time() - stall_start > max_wait_seconds:
                    log.error("autopilot_timeout", waited=max_wait_seconds)
                    break
                await asyncio.sleep(0.1)
                continue
            stall_start = None

            batch = ready[:self._max_parallel]
            tasks = [
                self._execute_step(step, step_results, session_id)
                for step in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    step.status = StepStatus.FAILED
                    step.error = str(result)
                    results.append({
                        "step": step.description, "agent": step.agent,
                        "success": False, "error": str(result),
                    })
                else:
                    results.append(result)
                    step_results[step.id] = result.get("output", "")

                if on_progress:
                    await on_progress({
                        "type": "step_complete",
                        "step": step.description,
                        "progress": plan.progress,
                        "success": step.status == StepStatus.COMPLETED,
                    })

        return results

    async def _execute_step(
        self, step: PlanStep, prior_results: dict[str, str], session_id: str,
    ) -> dict[str, Any]:
        step.status = StepStatus.RUNNING
        log.info("autopilot_step_started", step=step.description[:60], agent=step.agent)

        agent = self._agents.get(step.agent)
        if not agent:
            agent = self._agents.get("chat")
            if not agent:
                step.status = StepStatus.FAILED
                step.error = f"Agent '{step.agent}' not found"
                return {"step": step.description, "agent": step.agent, "success": False, "error": step.error}

        dep_context = ""
        for dep_id in step.depends_on:
            if dep_id in prior_results:
                dep_context += f"\nPrevious result: {prior_results[dep_id][:500]}"

        context = {
            "session_id": session_id,
            "autonomous": True,
            "prior_context": dep_context,
        }

        for attempt in range(self._max_retries + 1):
            try:
                result = await agent.execute(
                    query=step.description,
                    context=context,
                    tool_executor=self._call_tool,
                )

                if result.success:
                    step.status = StepStatus.COMPLETED
                    step.result = result.output
                    return {
                        "step": step.description,
                        "agent": step.agent,
                        "success": True,
                        "output": result.output,
                        "tools_used": [tc.get("tool", "") for tc in result.tool_calls],
                        "execution_time_ms": result.execution_time_ms,
                    }
                else:
                    if attempt < self._max_retries:
                        log.warning(
                            "autopilot_step_retry",
                            step=step.description[:40],
                            attempt=attempt + 1,
                            error=result.error,
                        )
                        context["retry_error"] = result.error
                        continue

                    step.status = StepStatus.FAILED
                    step.error = result.error
                    return {
                        "step": step.description, "agent": step.agent,
                        "success": False, "error": result.error,
                    }

            except Exception as exc:
                if attempt < self._max_retries:
                    log.warning("autopilot_step_exception", attempt=attempt + 1, error=str(exc))
                    continue
                step.status = StepStatus.FAILED
                step.error = str(exc)
                return {
                    "step": step.description, "agent": step.agent,
                    "success": False, "error": str(exc),
                }

        step.status = StepStatus.FAILED
        return {"step": step.description, "agent": step.agent, "success": False, "error": "max retries exceeded"}

    def _compile_results(
        self,
        goal: str,
        plan: ExecutionPlan,
        results: list[dict[str, Any]],
        quality: dict[str, Any],
        elapsed_ms: int,
    ) -> dict[str, Any]:
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        combined_output = "\n\n".join(
            f"**{r['step']}**\n{r.get('output', r.get('error', ''))}"
            for r in results
        )

        return {
            "goal": goal,
            "response": combined_output,
            "agent_used": "autopilot",
            "steps_total": len(plan.steps),
            "steps_completed": len(successful),
            "steps_failed": len(failed),
            "progress": plan.progress,
            "quality_score": quality.get("score", 0.0),
            "tools_used": list({t for r in successful for t in r.get("tools_used", [])}),
            "execution_time_ms": elapsed_ms,
            "results": results,
        }
