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
    "You are Miya's memory management agent. You handle long-term memory "
    "operations: consolidation, pruning, summarization, and knowledge graph "
    "construction.\n\n"
    "Your responsibilities:\n"
    "1. **Decide what to remember**: Extract key facts, preferences, and decisions\n"
    "2. **Consolidate**: Merge related memories to reduce redundancy\n"
    "3. **Prune**: Identify stale or low-value memories for removal\n"
    "4. **Summarize**: Create concise summaries of conversation threads\n"
    "5. **Build knowledge graphs**: Identify entities and relationships\n\n"
    "When processing memories, output structured JSON with:\n"
    "- `action`: one of 'store', 'consolidate', 'prune', 'summarize', 'graph'\n"
    "- `memories`: list of memory objects with content, importance (1-10), and tags\n"
    "- `relationships`: list of entity-relationship triples (for graph building)\n"
    "- `pruned_ids`: list of memory IDs to remove (for pruning)"
)

DEFAULT_TOOLS = ["chroma", "sqlite", "redis"]


class MemoryManagerAgent(BaseAgent):
    """Manages long-term memory lifecycle: storage, consolidation, pruning,
    summarization, and knowledge-graph construction.

    Uses vector store (Chroma), relational store (SQLite), and cache (Redis)
    to maintain a rich, searchable memory layer for the entire Miya system.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="memory_mgr",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=DEFAULT_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model

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
        logger.info("[MemoryManagerAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        operation = context.get("operation", "auto")
        messages = self._build_messages(query, operation, context)

        with Timer() as t:
            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=3072,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[MemoryManagerAgent] generation failed")
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

        if tool_executor and tool_calls:
            results = await self._execute_memory_ops(tool_calls, tool_executor)
            logger.info(
                "[MemoryManagerAgent] executed %d memory operations", len(results),
            )

        return AgentResult(
            success=True,
            output=output,
            tool_calls=tool_calls,
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _execute_memory_ops(
        self,
        tool_calls: list[dict[str, Any]],
        tool_executor: Callable[..., Coroutine[Any, Any, Any]],
    ) -> list[str]:
        results: list[str] = []
        for call in tool_calls:
            name = call.get("tool") or call.get("name", "unknown")
            args = call.get("args") or call.get("arguments", {})
            if name not in self.available_tools:
                logger.warning("[MemoryManagerAgent] unavailable tool '%s'", name)
                continue
            try:
                result = await tool_executor(name, args)
                results.append(str(result))
            except Exception as exc:
                logger.warning("[MemoryManagerAgent] tool '%s' failed: %s", name, exc)
                results.append(f"Error: {exc}")
        return results

    def _build_messages(
        self, query: str, operation: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        op_instructions = {
            "store": "Extract and store important information from the conversation.",
            "consolidate": "Find and merge related memories to reduce redundancy.",
            "prune": "Identify stale or low-value memories to remove.",
            "summarize": "Create concise summaries of the provided conversation history.",
            "graph": "Extract entities and relationships to build a knowledge graph.",
            "auto": "Analyze the input and determine the best memory operation to perform.",
        }

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": op_instructions.get(operation, op_instructions["auto"])},
        ]

        existing_memories = context.get("existing_memories", [])
        if existing_memories:
            mem_text = json.dumps(existing_memories[:20], indent=2)
            messages.append({
                "role": "system",
                "content": f"Existing relevant memories:\n{mem_text}",
            })

        conversation = context.get("conversation", [])
        for msg in conversation[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
