from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SearXNGTool(BaseTool):
    name = "searxng"
    description = "Web search via self-hosted SearXNG meta-search engine"
    category = "search"
    parameters = {
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "categories": {
            "type": "string",
            "description": "Comma-separated categories (general, images, news, videos, files, science)",
            "default": "general",
        },
        "language": {
            "type": "string",
            "description": "Search language code",
            "default": "en",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results",
            "default": 10,
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.searxng_host.rstrip("/")

    async def execute(self, input_data: dict[str, Any]) -> Any:
        query = input_data.get("query", "")
        if not query:
            return {"error": "'query' is required"}

        categories = input_data.get("categories", "general")
        language = input_data.get("language", "en")
        max_results = int(input_data.get("max_results", 10))

        params = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{self._base_url}/search", params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "engine": item.get("engine", ""),
                })

            return {
                "query": query,
                "results": results,
                "count": len(results),
            }

        except httpx.HTTPStatusError as exc:
            logger.error("searxng_http_error", status=exc.response.status_code)
            return {"error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.error("searxng_error", error=str(exc))
            return {"error": str(exc)}
