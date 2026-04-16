from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's scheduling agent. You manage scheduled and recurring tasks "
    "for the user.\n\n"
    "You can:\n"
    "1. **Add** a new scheduled task with a cron-like expression or interval\n"
    "2. **List** all currently scheduled tasks\n"
    "3. **Remove** a scheduled task by ID\n"
    "4. **Pause / Resume** a scheduled task\n"
    "5. **Describe** when a task will next run\n\n"
    "Cron format: `minute hour day_of_month month day_of_week`\n"
    "Interval format: `every <N> <seconds|minutes|hours|days>`\n\n"
    "Output your actions as a JSON block with:\n"
    "- `action`: one of 'add', 'list', 'remove', 'pause', 'resume'\n"
    "- `task_id`: (for remove/pause/resume)\n"
    "- `schedule`: cron or interval expression (for add)\n"
    "- `command`: what to execute (for add)\n"
    "- `description`: human-readable description (for add)"
)


class SchedulerAgent(BaseAgent):
    """Manages scheduled and recurring tasks with cron-like expressions.

    Maintains an in-memory task schedule and uses the LLM to interpret
    natural-language scheduling requests into structured schedule objects.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="scheduler",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model
        self._schedule: dict[str, dict[str, Any]] = {}
        self._next_id = 1

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
        logger.info("[SchedulerAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        messages = self._build_messages(query, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[SchedulerAgent] generation failed")
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

        action_result = self._process_schedule_action(output)
        if action_result:
            output = f"{output}\n\n---\nSchedule update: {action_result}"

        return AgentResult(
            success=True,
            output=output,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    def _process_schedule_action(self, text: str) -> str | None:
        """Parse and execute a schedule action from LLM output."""
        action_data = self._parse_action(text)
        if not action_data:
            return None

        action = action_data.get("action")

        if action == "add":
            return self._add_task(action_data)
        if action == "remove":
            return self._remove_task(action_data.get("task_id", ""))
        if action == "pause":
            return self._set_task_status(action_data.get("task_id", ""), paused=True)
        if action == "resume":
            return self._set_task_status(action_data.get("task_id", ""), paused=False)
        if action == "list":
            return self._list_tasks()
        return None

    def _add_task(self, data: dict[str, Any]) -> str:
        task_id = f"task_{self._next_id}"
        self._next_id += 1
        schedule_expr = data.get("schedule", "")
        parsed = self._parse_schedule(schedule_expr)

        self._schedule[task_id] = {
            "id": task_id,
            "schedule": schedule_expr,
            "parsed": parsed,
            "command": data.get("command", ""),
            "description": data.get("description", ""),
            "paused": False,
            "created_at": time.time(),
            "last_run": None,
        }
        logger.info("[SchedulerAgent] added task '%s': %s", task_id, schedule_expr)
        return f"Added {task_id} with schedule '{schedule_expr}'"

    def _remove_task(self, task_id: str) -> str:
        if task_id in self._schedule:
            del self._schedule[task_id]
            return f"Removed {task_id}"
        return f"Task '{task_id}' not found"

    def _set_task_status(self, task_id: str, *, paused: bool) -> str:
        if task_id in self._schedule:
            self._schedule[task_id]["paused"] = paused
            status = "paused" if paused else "resumed"
            return f"Task {task_id} {status}"
        return f"Task '{task_id}' not found"

    def _list_tasks(self) -> str:
        if not self._schedule:
            return "No scheduled tasks."
        lines = []
        for tid, task in self._schedule.items():
            status = "paused" if task["paused"] else "active"
            lines.append(f"- {tid} [{status}]: {task['description']} ({task['schedule']})")
        return "\n".join(lines)

    @staticmethod
    def _parse_schedule(expression: str) -> dict[str, Any]:
        """Parse a cron or interval expression into a structured dict."""
        interval_match = re.match(
            r"every\s+(\d+)\s+(seconds?|minutes?|hours?|days?)", expression, re.IGNORECASE,
        )
        if interval_match:
            value = int(interval_match.group(1))
            unit = interval_match.group(2).rstrip("s") + "s"
            return {"type": "interval", "value": value, "unit": unit}

        parts = expression.split()
        if len(parts) == 5:
            return {
                "type": "cron",
                "minute": parts[0],
                "hour": parts[1],
                "day_of_month": parts[2],
                "month": parts[3],
                "day_of_week": parts[4],
            }

        return {"type": "raw", "expression": expression}

    def _parse_action(self, text: str) -> dict[str, Any] | None:
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text):
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "action" in parsed:
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

        if self._schedule:
            schedule_summary = self._list_tasks()
            messages.append({
                "role": "system",
                "content": f"Current schedule:\n{schedule_summary}",
            })

        history = context.get("history", [])
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
