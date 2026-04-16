from __future__ import annotations

import os
from typing import Any

import httpx

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveTool(BaseTool):
    name = "brave_search"
    description = "Web search via Brave Search API (requires API key)"
    category = "search"
    parameters = {
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "count": {
            "type": "integer",
            "description": "Number of results",
            "default": 10,
        },
        "country": {
            "type": "string",
            "description": "Two-letter country code filter",
            "default": "",
        },
    }

    def __init__(self) -> None:
        self._api_key = os.environ.get("BRAVE_API_KEY", "")

    async def execute(self, input_data: dict[str, Any]) -> Any:
        if not self._api_key:
            return {"error": "BRAVE_API_KEY not configured"}

        query = input_data.get("query", "")
        if not query:
            return {"error": "'query' is required"}

        count = int(input_data.get("count", 10))
        country = input_data.get("country", "")

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params: dict[str, Any] = {"q": query, "count": count}
        if country:
            params["country"] = country

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(_BRAVE_API_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                })

            return {
                "query": query,
                "results": results,
                "count": len(results),
            }

        except httpx.HTTPStatusError as exc:
            logger.error("brave_http_error", status=exc.response.status_code)
            return {"error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.error("brave_error", error=str(exc))
            return {"error": str(exc)}
