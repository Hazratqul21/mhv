from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ScraperTool(BaseTool):
    name = "scraper"
    description = "Lightweight HTML scraper: fetch a URL and extract structured text with BeautifulSoup"
    category = "browser"
    parameters = {
        "url": {
            "type": "string",
            "description": "URL to scrape",
        },
        "selector": {
            "type": "string",
            "description": "CSS selector to narrow extraction",
            "default": "body",
        },
        "extract": {
            "type": "string",
            "enum": ["text", "links", "images", "html"],
            "description": "What to extract from the page",
            "default": "text",
        },
        "max_length": {
            "type": "integer",
            "description": "Max characters to return for text extraction",
            "default": 8000,
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        url = input_data.get("url", "")
        if not url:
            return {"error": "'url' is required"}

        selector = input_data.get("selector", "body")
        extract = input_data.get("extract", "text")
        max_length = int(input_data.get("max_length", 8000))

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            target = soup.select_one(selector)
            if target is None:
                return {"error": f"Selector '{selector}' not found"}

            if extract == "text":
                text = target.get_text(separator="\n", strip=True)
                return {"url": url, "text": text[:max_length], "truncated": len(text) > max_length}

            if extract == "links":
                links = []
                for a in target.find_all("a", href=True):
                    links.append({"text": a.get_text(strip=True), "href": a["href"]})
                return {"url": url, "links": links, "count": len(links)}

            if extract == "images":
                images = []
                for img in target.find_all("img"):
                    images.append({
                        "src": img.get("src", ""),
                        "alt": img.get("alt", ""),
                    })
                return {"url": url, "images": images, "count": len(images)}

            if extract == "html":
                html = str(target)
                return {"url": url, "html": html[:max_length], "truncated": len(html) > max_length}

            return {"error": f"Unknown extract mode '{extract}'"}

        except httpx.HTTPStatusError as exc:
            logger.error("scraper_http_error", url=url, status=exc.response.status_code)
            return {"error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.error("scraper_error", url=url, error=str(exc))
            return {"error": str(exc)}
