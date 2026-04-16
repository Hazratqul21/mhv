from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.logger import get_logger

log = get_logger(__name__)

EVALUATE_PROMPT = """\
You are a quality evaluator for an autonomous AI system. Assess how well the \
AI completed the given goal based on the step results.

Goal: {goal}

Step Results:
{results}

Evaluate and respond ONLY with JSON:
{{
  "score": 0.0-1.0,
  "feedback": "what went well and what didn't",
  "should_retry": true/false,
  "improvements": ["suggestion1", "suggestion2"],
  "missing_steps": ["step that should have been included"]
}}
"""


class FeedbackLoop:
    """Evaluates autonomous execution quality and generates improvement suggestions."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._engine = llm_engine
        self._settings = get_settings()
        self._history: list[dict[str, Any]] = []

    def _pick_model(self) -> str:
        try:
            self._engine.get_model(self._settings.orchestrator_model)
            return self._settings.orchestrator_model
        except KeyError:
            return self._settings.chat_model

    async def evaluate(self, goal: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        results_text = "\n".join(
            f"- [{('OK' if r.get('success') else 'FAIL')}] {r.get('step', '?')}: "
            f"{(r.get('output', r.get('error', '')))[:200]}"
            for r in results
        )

        prompt = EVALUATE_PROMPT.format(goal=goal, results=results_text)

        try:
            result = await self._engine.generate(
                self._pick_model(),
                prompt=prompt,
                max_tokens=512,
                temperature=0.2,
            )
            text = result["text"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                evaluation = json.loads(text[start:end])
            else:
                raise ValueError("No JSON found")
        except Exception as exc:
            log.warning("feedback_evaluation_failed", error=str(exc))
            success_rate = sum(1 for r in results if r.get("success")) / max(len(results), 1)
            evaluation = {
                "score": success_rate,
                "feedback": "Automated scoring based on success rate",
                "should_retry": success_rate < 0.5,
                "improvements": [],
                "missing_steps": [],
            }

        self._history.append({"goal": goal, "evaluation": evaluation})
        log.info("feedback_evaluated", score=evaluation.get("score"), goal=goal[:60])
        return evaluation

    def get_patterns(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._history[-limit:]

    @property
    def average_score(self) -> float:
        if not self._history:
            return 0.0
        scores = [h["evaluation"].get("score", 0) for h in self._history]
        return sum(scores) / len(scores)
