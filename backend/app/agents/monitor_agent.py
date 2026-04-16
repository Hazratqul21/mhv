from __future__ import annotations

import json
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
    "You are Miya's system monitoring agent. You watch over all services in "
    "the Miya ecosystem and ensure they are healthy and performant.\n\n"
    "Your capabilities:\n"
    "1. **Health Checks**: Verify that each service is up and responding\n"
    "2. **Anomaly Detection**: Spot unusual patterns in metrics (latency spikes, "
    "memory leaks, error rate increases)\n"
    "3. **Self-Healing**: Recommend or trigger corrective actions (restart service, "
    "clear cache, scale up)\n"
    "4. **Health Reports**: Generate structured reports on system status\n\n"
    "When reporting, use this format:\n"
    "- Service name, status (healthy/degraded/down), key metrics\n"
    "- If degraded/down: root cause hypothesis and recommended action\n\n"
    "For self-healing actions, output a JSON tool call:\n"
    '```json\n{"tool": "<tool_name>", "args": {…}}\n```'
)

DEFAULT_TOOLS = ["docker", "redis"]

MONITORED_SERVICES = [
    "llm_engine", "api_server", "redis", "chroma", "minio",
    "comfyui", "searxng", "sandbox",
]


class MonitorAgent(BaseAgent):
    """Monitors system health, detects anomalies, triggers self-healing actions,
    and generates health reports across all Miya services.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="monitor",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=DEFAULT_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model
        self._health_history: list[dict[str, Any]] = []

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
        logger.info("[MonitorAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        check_type = context.get("check_type", "full")
        metrics = context.get("metrics", {})

        messages = self._build_messages(query, check_type, metrics, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=3072,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[MonitorAgent] generation failed")
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

        tool_calls = self._parse_tool_calls(output)
        healing_results: list[str] = []

        if tool_executor and tool_calls:
            healing_results = await self._execute_healing(tool_calls, tool_executor)

        self._record_health_snapshot(output, metrics)

        return AgentResult(
            success=True,
            output=output,
            tool_calls=tool_calls,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _execute_healing(
        self,
        tool_calls: list[dict[str, Any]],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]],
    ) -> list[str]:
        results: list[str] = []
        for call in tool_calls:
            name = call.get("tool") or call.get("name", "unknown")
            args = call.get("args") or call.get("arguments", {})
            if name not in self.available_tools:
                logger.warning("[MonitorAgent] unavailable tool '%s'", name)
                continue
            try:
                logger.info("[MonitorAgent] executing healing action: %s", name)
                result = await tool_executor(name, args)
                results.append(f"[{name}] {result}")
            except Exception as exc:
                logger.warning("[MonitorAgent] healing '%s' failed: %s", name, exc)
                results.append(f"[{name}] Error: {exc}")
        return results

    def _record_health_snapshot(
        self, report: str, metrics: dict[str, Any]
    ) -> None:
        snapshot = {
            "timestamp": time.time(),
            "report_summary": report[:500],
            "metrics": metrics,
        }
        self._health_history.append(snapshot)
        if len(self._health_history) > 100:
            self._health_history = self._health_history[-100:]

    def _build_messages(
        self,
        query: str,
        check_type: str,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": f"Monitored services: {', '.join(MONITORED_SERVICES)}",
            },
        ]

        if check_type != "full":
            messages.append({
                "role": "system",
                "content": f"Focus on: {check_type} check only.",
            })

        if metrics:
            messages.append({
                "role": "system",
                "content": f"Current metrics:\n{json.dumps(metrics, indent=2)}",
            })

        if self._health_history:
            recent = self._health_history[-3:]
            history_text = json.dumps(recent, indent=2, default=str)
            messages.append({
                "role": "system",
                "content": f"Recent health snapshots:\n{history_text}",
            })

        alerts = context.get("alerts", [])
        if alerts:
            messages.append({
                "role": "system",
                "content": f"Active alerts:\n{json.dumps(alerts, indent=2)}",
            })

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
