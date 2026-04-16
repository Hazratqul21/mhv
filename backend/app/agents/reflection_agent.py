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
    "You are Miya's reflection and quality-evaluation agent. Your role is to "
    "critically assess the quality of another agent's response and provide "
    "actionable feedback.\n\n"
    "For every evaluation, score the response on a 1-10 scale across:\n"
    "- **Accuracy**: Are claims correct and verifiable?\n"
    "- **Completeness**: Does it fully address the query?\n"
    "- **Clarity**: Is it well-structured and easy to understand?\n"
    "- **Relevance**: Does it stay on topic and avoid tangents?\n\n"
    "Then provide:\n"
    "1. An overall score (1-10)\n"
    "2. A list of specific strengths\n"
    "3. A list of specific weaknesses or gaps\n"
    "4. Concrete improvement suggestions\n"
    "5. If the score is below 6, generate an improved prompt that would "
    "produce a better response\n\n"
    "Output your evaluation as a JSON block for programmatic consumption."
)

SCORE_THRESHOLD_RERUN = 6


class ReflectionAgent(BaseAgent):
    """Evaluates the quality of other agents' responses and suggests improvements.

    Accepts context containing ``response_to_evaluate`` and ``original_query``,
    scores the response across multiple dimensions, and optionally recommends
    re-execution with an improved prompt when quality is below threshold.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="reflection",
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
        logger.info("[ReflectionAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        response_text = context.get("response_to_evaluate", "")
        original_query = context.get("original_query", query)
        agent_name = context.get("agent_name", "unknown")

        if not response_text:
            return AgentResult(
                success=False,
                output="",
                error="Missing 'response_to_evaluate' in context.",
            )

        messages = self._build_messages(original_query, response_text, agent_name, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=3072,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[ReflectionAgent] generation failed")
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

        evaluation = self._parse_evaluation(output)
        tool_calls = []
        if evaluation:
            overall = evaluation.get("overall_score", 10)
            if isinstance(overall, (int, float)) and overall < SCORE_THRESHOLD_RERUN:
                improved_prompt = evaluation.get("improved_prompt", "")
                if improved_prompt:
                    tool_calls.append({
                        "tool": "rerun_request",
                        "args": {
                            "agent_name": agent_name,
                            "improved_prompt": improved_prompt,
                            "original_score": overall,
                        },
                    })
                    logger.info(
                        "[ReflectionAgent] recommending rerun for '%s' (score=%s)",
                        agent_name, overall,
                    )

        return AgentResult(
            success=True,
            output=output,
            tool_calls=tool_calls,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    def _parse_evaluation(self, text: str) -> dict[str, Any] | None:
        import re
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        for block in fenced:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _build_messages(
        self,
        original_query: str,
        response_text: str,
        agent_name: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": (
                    f"You are evaluating a response from the '{agent_name}' agent.\n"
                    f"Original user query: {original_query}"
                ),
            },
        ]

        evaluation_criteria = context.get("evaluation_criteria")
        if evaluation_criteria:
            messages.append({
                "role": "system",
                "content": f"Additional evaluation criteria: {evaluation_criteria}",
            })

        messages.append({
            "role": "user",
            "content": (
                f"Please evaluate this response:\n\n"
                f"---\n{response_text}\n---\n\n"
                f"Provide scores, strengths, weaknesses, and improvement suggestions."
            ),
        })
        voice_prepend_after_first_system(messages, context)
        return messages
