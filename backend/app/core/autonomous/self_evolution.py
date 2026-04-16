from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.logger import get_logger

from .code_writer import CodeWriter
from .feedback_loop import FeedbackLoop
from .prompt_optimizer import PromptOptimizer
from .self_healer import SelfHealer

log = get_logger(__name__)


@dataclass
class EvolutionMetrics:
    agents_created: int = 0
    prompts_improved: int = 0
    errors_healed: int = 0
    evolution_cycles: int = 0
    last_cycle: float = 0.0
    quality_trend: list[float] = field(default_factory=list)


class SelfEvolutionEngine:
    """The brain behind MIYA's autonomous self-improvement.

    Continuously evaluates performance, creates new agents when capability
    gaps are detected, optimizes prompts via A/B testing, and heals runtime
    errors without human intervention.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._engine = llm_engine
        self._settings = get_settings()

        self.code_writer = CodeWriter(llm_engine)
        self.prompt_optimizer = PromptOptimizer(llm_engine)
        self.self_healer = SelfHealer(llm_engine)
        self.feedback_loop = FeedbackLoop(llm_engine)

        self._metrics = EvolutionMetrics()
        self._running = False
        self._cycle_interval = 300  # 5 minutes
        self._capability_gaps: list[dict[str, Any]] = []

    async def start(self) -> None:
        self._running = True
        await self.self_healer.start()
        asyncio.create_task(self._evolution_loop())
        log.info("self_evolution_started")

    async def stop(self) -> None:
        self._running = False
        await self.self_healer.stop()
        log.info("self_evolution_stopped")

    async def _evolution_loop(self) -> None:
        while self._running:
            try:
                await self._run_cycle()
            except Exception:
                log.exception("evolution_cycle_failed")
            await asyncio.sleep(self._cycle_interval)

    async def _run_cycle(self) -> None:
        self._metrics.evolution_cycles += 1
        self._metrics.last_cycle = time.time()

        try:
            gaps = await self._detect_capability_gaps()
        except Exception:
            log.exception("gap_detection_error_in_cycle")
            gaps = []

        for gap in gaps:
            try:
                created = await self._create_agent_for_gap(gap)
                if created:
                    self._metrics.agents_created += 1
            except Exception:
                log.exception("agent_creation_failed", gap=gap.get("name", "?"))

        try:
            opt_stats = self.prompt_optimizer.get_stats()
            for agent_name, stats in opt_stats.items():
                if stats.get("total_evaluations", 0) >= 5:
                    result = await self.prompt_optimizer.optimize_prompt(agent_name)
                    if result:
                        self._metrics.prompts_improved += 1
        except Exception:
            log.exception("prompt_optimization_failed_in_cycle")

        log.info(
            "evolution_cycle_complete",
            cycle=self._metrics.evolution_cycles,
            gaps=len(gaps),
            agents_created=self._metrics.agents_created,
        )

    async def _detect_capability_gaps(self) -> list[dict[str, Any]]:
        if not self._capability_gaps:
            return []

        gaps_text = json.dumps(self._capability_gaps[-20:], default=str)
        prompt = (
            "Analyze these capability gaps and determine which ones need a new specialized agent.\n"
            "A gap is valid if: (1) no existing agent handles it, (2) it has occurred 3+ times.\n\n"
            f"Gaps:\n{gaps_text}\n\n"
            "Respond with JSON array of gaps that need new agents:\n"
            '[{"name": "agent_name", "description": "what it does", "system_prompt": "the prompt"}]'
        )
        try:
            model = self._settings.chat_model
            try:
                self._engine.get_model(self._settings.orchestrator_model)
                model = self._settings.orchestrator_model
            except KeyError:
                pass
            result = await self._engine.generate(
                model,
                prompt=prompt,
                max_tokens=2048,
                temperature=0.3,
            )
            text = result["text"].strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            log.exception("gap_detection_failed")
        return []

    async def _create_agent_for_gap(self, gap: dict[str, Any]) -> bool:
        agent_name = gap.get("name", "").lower().replace(" ", "_")
        if not agent_name:
            return False

        class_name = "".join(w.capitalize() for w in agent_name.split("_")) + "Agent"

        result = await self.code_writer.create_agent(
            agent_name=agent_name,
            class_name=class_name,
            description=gap.get("description", "Auto-generated agent"),
            system_prompt=gap.get("system_prompt", f"You are {agent_name}, an AI assistant."),
        )
        return result.get("success", False)

    async def report_gap(self, query: str, agent_used: str, quality_score: float) -> None:
        if quality_score < 0.4:
            self._capability_gaps.append({
                "query": query,
                "agent": agent_used,
                "score": quality_score,
                "timestamp": time.time(),
            })
            if len(self._capability_gaps) > 100:
                self._capability_gaps = self._capability_gaps[-50:]

    async def evaluate_and_improve(
        self,
        agent_name: str,
        query: str,
        response: str,
    ) -> dict[str, Any]:
        score = await self.prompt_optimizer.evaluate_response(agent_name, query, response)
        await self.prompt_optimizer.record_result(agent_name, query, response, score)

        if score < 0.4:
            await self.report_gap(query, agent_name, score)

        avg_quality = score
        self._metrics.quality_trend.append(score)
        if len(self._metrics.quality_trend) > 100:
            self._metrics.quality_trend = self._metrics.quality_trend[-50:]
            avg_quality = sum(self._metrics.quality_trend) / len(self._metrics.quality_trend)

        return {
            "score": score,
            "avg_quality": avg_quality,
            "agent": agent_name,
            "trend": "improving" if len(self._metrics.quality_trend) > 5 and
                     sum(self._metrics.quality_trend[-5:]) / 5 >
                     sum(self._metrics.quality_trend[:5]) / max(len(self._metrics.quality_trend[:5]), 1)
                     else "stable",
        }

    async def handle_error(self, source: str, error: str, context: dict | None = None) -> dict:
        result = await self.self_healer.handle_error(source, error, context)
        self._metrics.errors_healed += 1
        return result

    def get_metrics(self) -> dict[str, Any]:
        return {
            "agents_created": self._metrics.agents_created,
            "prompts_improved": self._metrics.prompts_improved,
            "errors_healed": self._metrics.errors_healed,
            "evolution_cycles": self._metrics.evolution_cycles,
            "last_cycle": self._metrics.last_cycle,
            "avg_quality": (
                sum(self._metrics.quality_trend) / len(self._metrics.quality_trend)
                if self._metrics.quality_trend
                else 0.0
            ),
            "health": self.self_healer.get_health(),
        }
