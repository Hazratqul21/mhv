from __future__ import annotations

import base64
from typing import Any

from playwright.async_api import async_playwright, Browser

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PlaywrightTool(BaseTool):
    name = "playwright"
    description = "Browser automation: navigate pages, take screenshots, and extract text via Playwright"
    category = "browser"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["navigate", "screenshot", "extract_text"],
            "description": "Browser action to perform",
        },
        "url": {
            "type": "string",
            "description": "URL to navigate to",
        },
        "selector": {
            "type": "string",
            "description": "CSS selector to target (for extract_text)",
        },
        "wait_ms": {
            "type": "integer",
            "description": "Milliseconds to wait after navigation",
            "default": 2000,
        },
        "full_page": {
            "type": "boolean",
            "description": "Capture full-page screenshot",
            "default": True,
        },
    }

    def __init__(self) -> None:
        self._browser: Browser | None = None

    async def _get_browser(self) -> Browser:
        if self._browser is None or not self._browser.is_connected():
            pw = await async_playwright().start()
            self._browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
        return self._browser

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        url = input_data.get("url", "")

        if not url and action in ("navigate", "screenshot", "extract_text"):
            return {"error": "'url' is required"}

        try:
            browser = await self._get_browser()
            page = await browser.new_page()
            wait_ms = int(input_data.get("wait_ms", 2000))

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if wait_ms > 0:
                    await page.wait_for_timeout(wait_ms)

                if action == "navigate":
                    title = await page.title()
                    return {"status": "ok", "url": url, "title": title}

                if action == "screenshot":
                    full_page = input_data.get("full_page", True)
                    img_bytes = await page.screenshot(full_page=full_page)
                    return {
                        "status": "ok",
                        "url": url,
                        "screenshot_b64": base64.b64encode(img_bytes).decode("ascii"),
                        "size": len(img_bytes),
                    }

                if action == "extract_text":
                    selector = input_data.get("selector", "body")
                    element = await page.query_selector(selector)
                    if element is None:
                        return {"error": f"Selector '{selector}' not found on page"}
                    text = await element.inner_text()
                    return {"url": url, "selector": selector, "text": text}

                return {"error": f"Unknown action '{action}'"}

            finally:
                await page.close()

        except Exception as exc:
            logger.error("playwright_error", action=action, error=str(exc))
            return {"error": str(exc)}
