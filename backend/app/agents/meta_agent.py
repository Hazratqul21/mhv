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
    "You are Miya's meta-agent — you design and configure new agents on the fly. "
    "Given a task description, you analyze what kind of agent is needed and produce "
    "a complete agent specification.\n\n"
    "For every agent specification you create, include:\n"
    "1. **Name**: A short, descriptive identifier\n"
    "2. **System Prompt**: A detailed system prompt tailored to the task\n"
    "3. **Model**: Which model class to use (chat, orchestrator, code, vision)\n"
    "4. **Tools**: Which tools the agent needs access to\n"
    "5. **Parameters**: Recommended temperature, max_tokens, context size\n"
    "6. **Composition**: For complex tasks, recommend a multi-agent pipeline\n\n"
    "Output your specification as a JSON block so it can be parsed programmatically."
)

DEFAULT_TOOLS = ["file"]


class MetaAgent(BaseAgent):
    """Dynamically creates new agent configurations based on task requirements.

    The meta-agent analyzes incoming task descriptions, determines the optimal
    agent architecture, and produces structured configuration objects that the
    orchestrator can use to spin up purpose-built agents.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="meta",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=DEFAULT_TOOLS,
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
        logger.info("[MetaAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        messages = self._build_messages(query, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.4,
                )
            except Exception as exc:
                logger.exception("[MetaAgent] generation failed")
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

        spec = self._extract_agent_spec(output)
        if spec:
            logger.info("[MetaAgent] generated spec for agent '%s'", spec.get("name", "unknown"))

        return AgentResult(
            success=True,
            output=output,
            tool_calls=[{"tool": "agent_spec", "args": spec}] if spec else [],
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    def _extract_agent_spec(self, text: str) -> dict[str, Any] | None:
        """Parse a JSON agent specification from the LLM output."""
        import re
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        for block in fenced:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and ("name" in parsed or "system_prompt" in parsed):
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

        available_agents = context.get("available_agents", [])
        if available_agents:
            agent_list = ", ".join(available_agents)
            messages.append({
                "role": "system",
                "content": f"Currently available agents: {agent_list}",
            })

        available_tools = context.get("available_tools", [])
        if available_tools:
            tool_list = ", ".join(available_tools)
            messages.append({
                "role": "system",
                "content": f"Available tools that new agents can use: {tool_list}",
            })

        task_type = context.get("task_type")
        if task_type:
            messages.append({
                "role": "system",
                "content": f"Requested task type: {task_type}",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
