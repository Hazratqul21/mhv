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
    "You are Miya's learning and adaptation agent. You analyze user feedback "
    "and interaction patterns to improve the overall system.\n\n"
    "Your responsibilities:\n"
    "1. **Feedback Analysis**: Process thumbs up/down, corrections, and explicit "
    "feedback to understand what works and what doesn't\n"
    "2. **Pattern Recognition**: Identify patterns in successful vs failed "
    "interactions across all agents\n"
    "3. **Prompt Improvement**: Suggest refined system prompts for other agents "
    "based on observed failures\n"
    "4. **Preference Learning**: Build a model of the user's preferences "
    "(tone, detail level, format, topics)\n"
    "5. **Knowledge Base**: Maintain learned insights for future reference\n\n"
    "Output your analysis as structured JSON with:\n"
    "- `insights`: list of learned patterns or preferences\n"
    "- `prompt_suggestions`: dict mapping agent names to suggested prompt changes\n"
    "- `store_actions`: list of items to persist in the knowledge base"
)

DEFAULT_TOOLS = ["chroma", "sqlite"]


class LearningAgent(BaseAgent):
    """Learns from user feedback and interaction patterns to improve
    the Miya system over time.

    Processes positive/negative signals, identifies recurring failure modes,
    and maintains a knowledge base of learned preferences and insights.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="learning",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=DEFAULT_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.orchestrator_model
        self._knowledge_base: list[dict[str, Any]] = []
        self._feedback_buffer: list[dict[str, Any]] = []

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
        logger.info("[LearningAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        feedback = context.get("feedback")
        if feedback:
            self._buffer_feedback(feedback)

        messages = self._build_messages(query, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=3072,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[LearningAgent] generation failed")
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

        analysis = self._parse_analysis(output)
        tool_calls: list[dict[str, Any]] = []

        if analysis:
            self._apply_insights(analysis.get("insights", []))

            store_actions = analysis.get("store_actions", [])
            if tool_executor and store_actions:
                tool_calls = await self._persist_learnings(store_actions, tool_executor)

        return AgentResult(
            success=True,
            output=output,
            tool_calls=tool_calls,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    def _buffer_feedback(self, feedback: dict[str, Any]) -> None:
        self._feedback_buffer.append({
            **feedback,
            "timestamp": time.time(),
        })
        if len(self._feedback_buffer) > 200:
            self._feedback_buffer = self._feedback_buffer[-200:]

    def _apply_insights(self, insights: list[dict[str, Any]]) -> None:
        for insight in insights:
            self._knowledge_base.append({
                **insight,
                "learned_at": time.time(),
            })
        if len(self._knowledge_base) > 500:
            self._knowledge_base = self._knowledge_base[-500:]
        logger.info("[LearningAgent] knowledge base size: %d", len(self._knowledge_base))

    async def _persist_learnings(
        self,
        store_actions: list[dict[str, Any]],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]],
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for action in store_actions:
            store = action.get("store", "chroma")
            if store not in self.available_tools:
                continue
            try:
                await tool_executor(store, {
                    "action": "upsert",
                    "collection": "learnings",
                    "data": action.get("data", {}),
                })
                tool_calls.append({"tool": store, "args": action})
            except Exception as exc:
                logger.warning("[LearningAgent] persist to '%s' failed: %s", store, exc)
        return tool_calls

    def _parse_analysis(self, text: str) -> dict[str, Any] | None:
        import re
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text):
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
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

        if self._feedback_buffer:
            recent_feedback = self._feedback_buffer[-10:]
            messages.append({
                "role": "system",
                "content": f"Recent feedback entries:\n{json.dumps(recent_feedback, indent=2, default=str)}",
            })

        if self._knowledge_base:
            recent_knowledge = self._knowledge_base[-10:]
            messages.append({
                "role": "system",
                "content": f"Current knowledge base (recent):\n{json.dumps(recent_knowledge, indent=2, default=str)}",
            })

        interaction_history = context.get("interaction_history", [])
        if interaction_history:
            messages.append({
                "role": "system",
                "content": f"Interaction history for analysis:\n{json.dumps(interaction_history[:20], indent=2, default=str)}",
            })

        agent_performance = context.get("agent_performance", {})
        if agent_performance:
            messages.append({
                "role": "system",
                "content": f"Agent performance metrics:\n{json.dumps(agent_performance, indent=2)}",
            })

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
