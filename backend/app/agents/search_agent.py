from __future__ import annotations

import time
from typing import Any, Callable, Coroutine
from urllib.parse import quote_plus

import httpx

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's research assistant. You have access to web search results. "
    "Synthesize the search results into a clear, well-structured answer. "
    "Always cite sources by including relevant URLs. "
    "If the search results are insufficient, say so explicitly."
)

SEARCH_TOOLS = ["web_search"]


class SearchAgent(BaseAgent):
    """Web search agent using DuckDuckGo (with SearXNG fallback) for
    retrieval and an LLM for summarisation."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="search",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=SEARCH_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model
        self._searxng_url = settings.searxng_host
        self._max_results = 8
        self._use_duckduckgo = True

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
        logger.info("[SearchAgent] query: %s", query[:120])
        context = context or {}

        self.ensure_loaded()
        with Timer() as t:
            search_results = await self._search(query)

            if not search_results:
                logger.info("[SearchAgent] no web results, answering from LLM knowledge")
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": "Web search returned no results. Answer from your own knowledge."},
                    {"role": "user", "content": query},
                ]
                voice_prepend_after_first_system(messages, context)
                try:
                    response = await self._engine.chat(
                        model_filename=self._model_name,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.3,
                    )
                    fallback_text = response.get("text", "") if isinstance(response, dict) else str(response)
                    return AgentResult(
                        success=True,
                        output=fallback_text or "I couldn't find relevant information for your query.",
                        execution_time_ms=t.elapsed_ms,
                    )
                except Exception:
                    return AgentResult(
                        success=True,
                        output="Web search returned no results. Please try a different query.",
                        execution_time_ms=t.elapsed_ms,
                    )

            formatted = self._format_results(search_results)
            messages = self._build_messages(query, formatted, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[SearchAgent] LLM summarisation failed")
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

        return AgentResult(
            success=True,
            output=output,
            tool_calls=[{"tool": "web_search", "args": {"query": query, "num_results": len(search_results)}}],
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Search the web, trying DuckDuckGo first, SearXNG as fallback."""
        if self._use_duckduckgo:
            results = await self._search_duckduckgo(query)
            if results:
                return results
            logger.info("DuckDuckGo returned no results, trying SearXNG")

        return await self._search_searxng(query)

    async def _search_duckduckgo(self, query: str) -> list[dict[str, Any]]:
        """Search using the duckduckgo_search library (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            import asyncio

            def _do_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=self._max_results))

            raw = await asyncio.to_thread(_do_search)
            results = []
            for r in raw:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "content": r.get("body", r.get("snippet", "")),
                })
            return results
        except ImportError:
            logger.warning("duckduckgo_search not installed")
            self._use_duckduckgo = False
            return []
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

    async def _search_searxng(self, query: str) -> list[dict[str, Any]]:
        """Query SearXNG and return a list of result dicts."""
        url = f"{self._searxng_url}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "pageno": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("SearXNG HTTP error: %s", exc)
            return []
        except Exception:
            logger.exception("SearXNG request failed")
            return []

        results = data.get("results", [])
        return results[: self._max_results]

    @staticmethod
    def _format_results(results: list[dict[str, Any]]) -> str:
        """Render search results into a textual context block."""
        lines: list[str] = []
        for idx, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", r.get("snippet", ""))
            lines.append(f"[{idx}] {title}\n    URL: {url}\n    {snippet}")
        return "\n\n".join(lines)

    def _build_messages(
        self,
        query: str,
        search_context: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Search results:\n\n{search_context}"},
        ]
        voice_prepend_after_first_system(messages, context)

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages
