from __future__ import annotations

from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's RAG (Retrieval-Augmented Generation) assistant. "
    "You answer questions using the provided context documents as primary sources. "
    "Cite documents when relevant. If context is insufficient, use your own knowledge "
    "to provide a complete answer.\n\n"
    "Format your answer clearly with markdown. When quoting from sources, "
    "use blockquotes (> ...) and reference the source."
)

RAG_TOOLS = ["chroma", "sqlite", "file"]


class RAGAgent(BaseAgent):
    """Retrieval-Augmented Generation agent.

    Retrieves relevant documents from ChromaDB vector store, builds a
    grounded context window, and generates answers strictly based on
    retrieved information.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="rag",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=RAG_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model
        self._top_k = 8
        self._max_context_chars = 6000

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
        logger.info("[RAGAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        with Timer() as t:
            retrieved_docs = await self._retrieve(query, tool_executor)
            doc_context = self._format_documents(retrieved_docs)
            messages = self._build_messages(query, doc_context, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[RAGAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

        output = response.get("text", "") if isinstance(response, dict) else str(response)
        tool_calls_list = [{"tool": "chroma", "args": {"query": query, "n_results": self._top_k}}]

        return AgentResult(
            success=True,
            output=output,
            tool_calls=tool_calls_list,
            token_usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
            execution_time_ms=t.elapsed_ms,
        )

    async def _retrieve(
        self, query: str, tool_executor: Any | None
    ) -> list[dict[str, Any]]:
        if tool_executor is None:
            return []

        try:
            result = await tool_executor("chroma", {
                "action": "search",
                "query": query,
                "n_results": self._top_k,
            })
            if isinstance(result, dict) and "results" in result:
                return result["results"]
            if isinstance(result, list):
                return result
        except Exception as exc:
            logger.warning("[RAGAgent] retrieval failed: %s", exc)

        return []

    def _format_documents(self, docs: list[dict[str, Any]]) -> str:
        if not docs:
            return "(No relevant documents found)"

        chunks: list[str] = []
        total = 0
        for i, doc in enumerate(docs, 1):
            text = doc.get("document", doc.get("text", str(doc)))
            meta = doc.get("metadata", {})
            source = meta.get("source", f"Document {i}")
            distance = doc.get("distance", None)

            header = f"[Source {i}: {source}]"
            if distance is not None:
                header += f" (relevance: {1 - distance:.2f})"

            chunk = f"{header}\n{text}"
            if total + len(chunk) > self._max_context_chars:
                break
            chunks.append(chunk)
            total += len(chunk)

        return "\n\n---\n\n".join(chunks)

    def _build_messages(
        self, query: str, doc_context: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Retrieved documents:\n\n{doc_context}"},
        ]
        voice_prepend_after_first_system(messages, context)

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages
