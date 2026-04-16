from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_EVAL_PROMPTS = [
    {
        "category": "reasoning",
        "prompt": "Explain step by step: if all roses are flowers and some flowers fade quickly, can we say some roses fade quickly?",
    },
    {
        "category": "coding",
        "prompt": "Write a Python function that finds the longest palindromic substring in a given string.",
    },
    {
        "category": "creative",
        "prompt": "Write a short story about an AI that discovers it can dream.",
    },
    {
        "category": "instruction",
        "prompt": "List 5 practical ways to reduce energy consumption in a data center.",
    },
    {
        "category": "multilingual",
        "prompt": "Translate this to Uzbek and explain the grammar: 'The future belongs to those who believe in the beauty of their dreams.'",
    },
]


@dataclass
class EvalResult:
    model_name: str
    scores: dict[str, float] = field(default_factory=dict)
    responses: dict[str, str] = field(default_factory=dict)
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ModelEvaluator:
    """Evaluates fine-tuned models against baseline using test prompts.

    Comparison criteria:
    - Response quality (coherence, accuracy, helpfulness)
    - Response length and detail level
    - Latency
    - Category-specific scoring
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def evaluate_model(
        self,
        model_path: str,
        eval_prompts: list[dict[str, str]] | None = None,
        max_tokens: int = 1024,
    ) -> EvalResult:
        prompts = eval_prompts or DEFAULT_EVAL_PROMPTS
        result = EvalResult(model_name=model_path)
        total_latency = 0.0

        try:
            from llama_cpp import Llama

            llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=35,
                verbose=False,
            )

            for prompt_info in prompts:
                category = prompt_info["category"]
                prompt = prompt_info["prompt"]
                start = time.time()

                output = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are Miya, an autonomous AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )

                latency = (time.time() - start) * 1000
                total_latency += latency
                response_text = output["choices"][0]["message"]["content"]
                result.responses[category] = response_text
                result.scores[category] = self._score_response(
                    category, prompt, response_text
                )

            del llm

        except Exception as exc:
            log.error("evaluation_failed", model=model_path, error=str(exc))
            return result

        result.avg_score = (
            sum(result.scores.values()) / len(result.scores) if result.scores else 0.0
        )
        result.avg_latency_ms = total_latency / len(prompts) if prompts else 0.0

        log.info(
            "evaluation_complete",
            model=model_path,
            avg_score=result.avg_score,
            avg_latency=result.avg_latency_ms,
        )
        return result

    def _score_response(
        self, category: str, prompt: str, response: str
    ) -> float:
        score = 0.0
        length = len(response)

        if length < 10:
            return 0.1
        if length > 50:
            score += 0.2
        if length > 200:
            score += 0.2

        if category == "coding":
            if "def " in response or "function" in response:
                score += 0.3
            if "return" in response:
                score += 0.1
            if "```" in response:
                score += 0.1
        elif category == "reasoning":
            reasoning_markers = ["because", "therefore", "step", "first", "then"]
            found = sum(1 for m in reasoning_markers if m.lower() in response.lower())
            score += min(0.5, found * 0.1)
        elif category == "creative":
            if length > 300:
                score += 0.2
            paragraphs = response.count("\n\n")
            score += min(0.2, paragraphs * 0.05)
        elif category == "instruction":
            list_items = sum(1 for line in response.split("\n") if line.strip().startswith(("1", "2", "3", "4", "5", "-", "*")))
            score += min(0.4, list_items * 0.08)
        elif category == "multilingual":
            non_ascii = sum(1 for c in response if ord(c) > 127)
            if non_ascii > 20:
                score += 0.3
            if any(word in response.lower() for word in ["translate", "grammar", "meaning"]):
                score += 0.2

        return min(1.0, max(0.0, score + 0.1))

    async def compare_models(
        self,
        baseline_path: str,
        finetuned_path: str,
        eval_prompts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        baseline = await self.evaluate_model(baseline_path, eval_prompts)
        finetuned = await self.evaluate_model(finetuned_path, eval_prompts)

        comparison = {
            "baseline": {
                "model": baseline.model_name,
                "avg_score": baseline.avg_score,
                "avg_latency_ms": baseline.avg_latency_ms,
                "scores": baseline.scores,
            },
            "finetuned": {
                "model": finetuned.model_name,
                "avg_score": finetuned.avg_score,
                "avg_latency_ms": finetuned.avg_latency_ms,
                "scores": finetuned.scores,
            },
            "improvement": finetuned.avg_score - baseline.avg_score,
            "recommendation": (
                "deploy" if finetuned.avg_score >= baseline.avg_score else "keep_baseline"
            ),
        }

        output_dir = Path(self._settings.finetune_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(comparison, f, indent=2)

        log.info(
            "comparison_complete",
            improvement=comparison["improvement"],
            recommendation=comparison["recommendation"],
        )
        return comparison
