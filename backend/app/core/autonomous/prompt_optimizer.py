from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PromptVariant:
    prompt: str
    score: float = 0.0
    usage_count: int = 0
    total_score: float = 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.usage_count, 1)


@dataclass
class AgentPromptRecord:
    agent_name: str
    current_prompt: str
    variants: list[PromptVariant] = field(default_factory=list)
    best_prompt: str = ""
    best_score: float = 0.0
    last_optimized: float = 0.0


class PromptOptimizer:
    """A/B tests and evolves system prompts for all agents.

    Uses the orchestrator model to evaluate prompt quality and generate
    improved variants.  Over time, agents converge on the best prompts
    discovered so far.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._engine = llm_engine
        self._settings = get_settings()
        self._records: dict[str, AgentPromptRecord] = {}
        self._min_samples = 5
        self._improvement_threshold = 0.1

    def register_agent(self, agent_name: str, system_prompt: str) -> None:
        if agent_name not in self._records:
            self._records[agent_name] = AgentPromptRecord(
                agent_name=agent_name,
                current_prompt=system_prompt,
                best_prompt=system_prompt,
                best_score=0.0,
            )

    async def record_result(
        self,
        agent_name: str,
        query: str,
        response: str,
        score: float,
    ) -> None:
        record = self._records.get(agent_name)
        if not record:
            return

        for variant in record.variants:
            if variant.prompt == record.current_prompt:
                variant.usage_count += 1
                variant.total_score += score
                break
        else:
            variant = PromptVariant(
                prompt=record.current_prompt,
                usage_count=1,
                total_score=score,
            )
            record.variants.append(variant)

        if score > record.best_score:
            record.best_score = score
            record.best_prompt = record.current_prompt

    async def evaluate_response(
        self, agent_name: str, query: str, response: str
    ) -> float:
        eval_prompt = (
            "Rate the following AI response on a scale of 0.0 to 1.0.\n"
            "Consider: accuracy, helpfulness, relevance, completeness.\n\n"
            f"Agent: {agent_name}\n"
            f"Query: {query}\n"
            f"Response: {response[:2000]}\n\n"
            "Respond ONLY with a JSON object: {\"score\": 0.0}"
        )
        try:
            try:
                model = self._settings.orchestrator_model
                self._engine.get_model(model)
            except KeyError:
                model = self._settings.chat_model
            result = await self._engine.generate(
                model,
                prompt=eval_prompt,
                max_tokens=64,
                temperature=0.1,
            )
            parsed = json.loads(result["text"].strip())
            return float(parsed.get("score", 0.5))
        except Exception:
            return 0.5

    async def optimize_prompt(self, agent_name: str) -> str | None:
        record = self._records.get(agent_name)
        if not record:
            return None

        total_usage = sum(v.usage_count for v in record.variants)
        if total_usage < self._min_samples:
            return None

        now = time.time()
        if now - record.last_optimized < 3600:
            return None

        improve_prompt = (
            "Improve the following AI agent system prompt to produce better responses.\n"
            "Keep the same purpose and capabilities, but make it clearer and more effective.\n"
            "Return ONLY the improved prompt text.\n\n"
            f"Agent: {agent_name}\n"
            f"Current prompt (score={record.best_score:.2f}):\n"
            f"{record.best_prompt}\n\n"
            "Known weak areas: "
            + ", ".join(
                f"variant avg={v.avg_score:.2f}"
                for v in sorted(record.variants, key=lambda v: v.avg_score)[:3]
            )
        )

        try:
            try:
                model = self._settings.orchestrator_model
                self._engine.get_model(model)
            except KeyError:
                model = self._settings.chat_model
            result = await self._engine.generate(
                model,
                prompt=improve_prompt,
                max_tokens=2048,
                temperature=0.4,
            )
            new_prompt = result["text"].strip()
            if len(new_prompt) < 20:
                return None

            record.variants.append(PromptVariant(prompt=new_prompt))
            record.current_prompt = new_prompt
            record.last_optimized = now

            log.info(
                "prompt_optimized",
                agent=agent_name,
                new_len=len(new_prompt),
            )
            return new_prompt
        except Exception:
            log.exception("prompt_optimization_failed", agent=agent_name)
            return None

    async def get_best_prompt(self, agent_name: str) -> str | None:
        record = self._records.get(agent_name)
        return record.best_prompt if record else None

    def get_stats(self) -> dict[str, Any]:
        stats = {}
        for name, record in self._records.items():
            stats[name] = {
                "best_score": record.best_score,
                "variants": len(record.variants),
                "total_evaluations": sum(v.usage_count for v in record.variants),
                "current_avg": (
                    next(
                        (
                            v.avg_score
                            for v in record.variants
                            if v.prompt == record.current_prompt
                        ),
                        0.0,
                    )
                ),
            }
        return stats
