from __future__ import annotations

import asyncio
from typing import Any

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DuckDuckGoTool(BaseTool):
    name = "duckduckgo"
    description = "Fallback web search using DuckDuckGo (no API key needed)"
    category = "search"
    parameters = {
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results",
            "default": 10,
        },
        "region": {
            "type": "string",
            "description": "Region code (e.g. us-en, uk-en)",
            "default": "wt-wt",
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        query = input_data.get("query", "")
        if not query:
            return {"error": "'query' is required"}

        max_results = int(input_data.get("max_results", 10))
        region = input_data.get("region", "wt-wt")

        try:
            from duckduckgo_search import DDGS

            def _do_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, region=region, max_results=max_results))

            raw = await asyncio.to_thread(_do_search)

            results = []
            for item in raw:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "content": item.get("body", ""),
                })

            return {
                "query": query,
                "results": results,
                "count": len(results),
            }

        except Exception as exc:
            logger.error("duckduckgo_error", error=str(exc))
            return {"error": str(exc)}
