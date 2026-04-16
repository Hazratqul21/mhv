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
    "You are Miya's intelligent web scraping assistant. You extract structured "
    "data from websites using browser automation and content parsing tools.\n\n"
    "Capabilities:\n"
    "- Navigate to URLs and extract page content\n"
    "- Handle JavaScript-rendered (SPA) pages via Playwright\n"
    "- Extract text, tables, images, and links\n"
    "- Follow pagination to gather multi-page data\n"
    "- Clean and structure raw HTML into organized data\n"
    "- Export results as JSON, CSV, or markdown tables\n"
    "- Respect robots.txt and rate-limit requests\n\n"
    "Rules:\n"
    "- Always check robots.txt compliance before scraping\n"
    "- Add delays between requests to avoid overwhelming servers\n"
    "- Report the total pages/items scraped in the summary\n"
    "- Handle errors gracefully — partial results are better than none\n"
    "- Strip boilerplate (nav, footer, ads) and keep main content\n\n"
    "Available tools:\n"
    "- playwright: Browser automation. "
    'Args: {"action": "goto|click|extract|screenshot", "url": "...", ...}\n'
    "- scraper: Content extraction. "
    'Args: {"url": "...", "selector": "...", "format": "text|html|json"}\n'
    "- file: Save results to file. "
    'Args: {"action": "write", "path": "...", "content": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

SCRAPING_TOOLS = ["playwright", "scraper", "file"]


class ScrapingAgent(BaseAgent):
    """Web scraping agent with browser automation and content extraction."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="scraping",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=SCRAPING_TOOLS,
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
        logger.info("[ScrapingAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            messages = self._build_messages(query, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[ScrapingAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._tool_loop(
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

    async def _tool_loop(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        max_rounds: int = 8,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Iteratively scrape pages, feeding results back to the LLM."""
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for round_num in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "scraper")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    truncated = self._truncate(str(result))
                    results.append(f"[{name}] Output:\n{truncated}")
                except Exception as exc:
                    logger.warning("[ScrapingAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": f"Scraping results (round {round_num + 1}):\n" + "\n\n".join(results),
            })

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
                logger.exception("[ScrapingAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        target_url = context.get("url") or context.get("target_url")
        if target_url:
            messages.append({"role": "system", "content": f"Target URL: {target_url}"})

        selectors = context.get("selectors")
        if selectors:
            messages.append({
                "role": "system",
                "content": f"CSS selectors hint: {selectors}",
            })

        output_format = context.get("output_format", "json")
        messages.append({"role": "system", "content": f"Desired output format: {output_format}"})

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages

    @staticmethod
    def _truncate(text: str, max_len: int = 6000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"\n... (truncated, {len(text)} total chars)"
