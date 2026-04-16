from __future__ import annotations

import base64
from typing import Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SeleniumTool(BaseTool):
    name = "selenium"
    description = "Fallback browser automation using Selenium with headless Chrome"
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
            "description": "CSS selector to target",
        },
        "wait_seconds": {
            "type": "integer",
            "description": "Seconds to wait for elements",
            "default": 10,
        },
    }

    def _make_driver(self) -> webdriver.Chrome:
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        return webdriver.Chrome(options=opts)

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        url = input_data.get("url", "")

        if not url:
            return {"error": "'url' is required"}

        driver = None
        try:
            driver = self._make_driver()
            wait_seconds = int(input_data.get("wait_seconds", 10))
            driver.get(url)

            if action == "navigate":
                return {"status": "ok", "url": url, "title": driver.title}

            if action == "screenshot":
                png = driver.get_screenshot_as_png()
                return {
                    "status": "ok",
                    "url": url,
                    "screenshot_b64": base64.b64encode(png).decode("ascii"),
                    "size": len(png),
                }

            if action == "extract_text":
                selector = input_data.get("selector", "body")
                wait = WebDriverWait(driver, wait_seconds)
                element = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                return {"url": url, "selector": selector, "text": element.text}

            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("selenium_error", action=action, error=str(exc))
            return {"error": str(exc)}
        finally:
            if driver:
                driver.quit()
