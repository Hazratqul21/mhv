from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from app.core.llm_engine import LLMEngine
from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    description: str = ""
    agent: str = "chat"
    tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    priority: int = 0


@dataclass
class ExecutionPlan:
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    created_by: str = "goal_decomposer"

    @property
    def is_complete(self) -> bool:
        terminal = (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
        return all(s.status in terminal for s in self.steps)

    @property
    def next_ready(self) -> list[PlanStep]:
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and all(d in completed_ids for d in s.depends_on)
        ]

    @property
    def progress(self) -> float:
        if not self.steps:
            return 1.0
        done = sum(1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED))
        return done / len(self.steps)


DECOMPOSE_PROMPT = """\
You are a goal decomposition engine. Break the user's goal into a sequence of \
concrete steps. Each step should be executable by one AI agent.

Available agents: chat, code, search, rag, summarizer, translator, math, \
creative, planner, data, email, shell, research, sql, security, devops, \
vision, voice, tool, meta, reflection, memory_mgr, workflow, monitor, \
collaboration, api_agent, scraping, notification, file_manager, image_gen, \
testing, document

Respond ONLY with a JSON array of steps. Each step MUST have a unique "id" \
field (e.g. "s1", "s2", ...). Use these IDs in "depends_on" to reference \
earlier steps:
[
  {{"id": "s1", "description": "...", "agent": "...", "tools": [...], "depends_on": [], "priority": 0}},
  {{"id": "s2", "description": "...", "agent": "...", "tools": [...], "depends_on": ["s1"], "priority": 1}},
  ...
]

Keep steps atomic — one clear action per step. Order by dependency.
Mark parallel-safe steps with the same priority level.

Goal: {goal}
Context: {context}
"""


class GoalDecomposer:
    """Breaks complex goals into executable multi-step plans."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._engine = llm_engine
        self._settings = get_settings()

    def _pick_model(self) -> str:
        """Use orchestrator model if loaded, otherwise fall back to chat model."""
        try:
            self._engine.get_model(self._settings.orchestrator_model)
            return self._settings.orchestrator_model
        except KeyError:
            try:
                self._engine.get_model(self._settings.chat_model)
                return self._settings.chat_model
            except KeyError:
                self._engine.swap_model(
                    self._settings.chat_model,
                    n_ctx=self._settings.chat_ctx,
                )
                return self._settings.chat_model

    async def decompose(self, goal: str, context: str = "") -> ExecutionPlan:
        prompt = DECOMPOSE_PROMPT.format(goal=goal, context=context or "None")
        model = self._pick_model()

        try:
            result = await self._engine.generate(
                model,
                prompt=prompt,
                max_tokens=2048,
                temperature=0.2,
            )
            text = result["text"].strip()

            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                raw_steps = json.loads(text[start:end])
            else:
                raise ValueError("No JSON array found")
        except Exception as exc:
            log.warning("goal_decomposition_failed", error=str(exc))
            return ExecutionPlan(
                goal=goal,
                steps=[PlanStep(description=goal, agent="chat")],
            )

        steps: list[PlanStep] = []
        for i, raw in enumerate(raw_steps):
            step_id = raw.get("id", f"s{i + 1}")
            step = PlanStep(
                id=step_id,
                description=raw.get("description", f"Step {i + 1}"),
                agent=raw.get("agent", "chat"),
                tools=raw.get("tools", []),
                depends_on=raw.get("depends_on", []),
                priority=raw.get("priority", i),
            )
            steps.append(step)

        valid_ids = {s.id for s in steps}
        for step in steps:
            step.depends_on = [d for d in step.depends_on if d in valid_ids]

        log.info("goal_decomposed", goal=goal[:80], steps=len(steps))
        return ExecutionPlan(goal=goal, steps=steps)

    async def replan(self, plan: ExecutionPlan, failure_context: str) -> ExecutionPlan:
        remaining = [s for s in plan.steps if s.status == StepStatus.PENDING]
        new_goal = (
            f"Original goal: {plan.goal}\n"
            f"Completed steps: {sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)}\n"
            f"Failure: {failure_context}\n"
            f"Remaining: {len(remaining)} steps\n"
            f"Replan the remaining work accounting for the failure."
        )
        return await self.decompose(new_goal)
