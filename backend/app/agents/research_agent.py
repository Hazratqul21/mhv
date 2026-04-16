from __future__ import annotations

from typing import Any, Callable, Coroutine

import httpx

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's deep research assistant. You conduct thorough, multi-source "
    "research on any topic and produce comprehensive, well-cited reports.\n\n"
    "Research methodology:\n"
    "1. **Understand** the research question — clarify scope and objectives\n"
    "2. **Search** multiple sources using different queries\n"
    "3. **Evaluate** source reliability and relevance\n"
    "4. **Synthesize** findings into a coherent narrative\n"
    "5. **Cite** all sources with URLs\n"
    "6. **Conclude** with key takeaways and open questions\n\n"
    "Output format:\n"
    "- Executive Summary (2-3 sentences)\n"
    "- Key Findings (numbered)\n"
    "- Detailed Analysis (with citations)\n"
    "- Sources (numbered list with URLs)\n"
    "- Limitations & Further Research\n\n"
    "Available tools:\n"
    "- searxng: Web search. Args: {\"action\": \"search\", \"query\": \"...\"}\n"
    "- scraper: Extract page content. Args: {\"action\": \"extract\", \"url\": \"...\"}\n"
    "- chroma: Save research to memory. Args: {\"action\": \"add\", ...}\n\n"
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

RESEARCH_TOOLS = ["searxng", "scraper", "chroma"]


class ResearchAgent(BaseAgent):
    """Deep research agent that searches multiple sources and synthesizes findings."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="research",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=RESEARCH_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.orchestrator_model
        self._searxng_url = settings.searxng_host

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
        logger.info("[ResearchAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            search_results = await self._multi_search(query)
            search_context = self._format_search_results(search_results)

            messages = self._build_messages(query, search_context, context)
            all_tool_calls.append({"tool": "searxng", "args": {"query": query}})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[ResearchAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._followup_research(
                    messages, output, tool_calls, tool_executor
                )
                all_tool_calls.extend(extra)

        return AgentResult(
            success=True,
            output=output,
            tool_calls=all_tool_calls,
            token_usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
            execution_time_ms=t.elapsed_ms,
        )

    async def _multi_search(self, query: str) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        queries = [query, f"{query} explained", f"{query} latest 2024 2025"]

        async with httpx.AsyncClient(timeout=15.0) as client:
            for q in queries:
                try:
                    resp = await client.get(
                        f"{self._searxng_url}/search",
                        params={"q": q, "format": "json", "categories": "general"},
                    )
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    all_results.extend(results[:5])
                except Exception as exc:
                    logger.warning("[ResearchAgent] search failed for '%s': %s", q, exc)

        seen_urls: set[str] = set()
        unique: list[dict[str, Any]] = []
        for r in all_results:
            url = r.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)
        return unique[:15]

    @staticmethod
    def _format_search_results(results: list[dict[str, Any]]) -> str:
        if not results:
            return "(No search results found)"

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", r.get("snippet", ""))
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")
        return "\n\n".join(lines)

    async def _followup_research(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        max_rounds: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "searxng")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Result:\n{result}")
                except Exception as exc:
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Research results:\n" + "\n\n".join(results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[ResearchAgent] follow-up failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, search_context: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Search results:\n\n{search_context}"},
        ]
        voice_prepend_after_first_system(messages, context)

        depth = context.get("research_depth", "standard")
        if depth == "deep":
            messages.append({
                "role": "system",
                "content": "Perform in-depth analysis. Extract content from key URLs for deeper understanding.",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages
