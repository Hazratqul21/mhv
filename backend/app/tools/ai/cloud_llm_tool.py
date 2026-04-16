from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

log = get_logger(__name__)


class CloudLLMTool(BaseTool):
    """Provides tool-level access to cloud LLM APIs.

    Allows agents to explicitly request cloud model inference
    when local models are insufficient.
    """

    name = "cloud_llm"
    description = "Send a prompt to a cloud LLM (Anthropic/OpenAI/Google)"
    category = "ai"
    parameters: dict[str, Any] = {
        "prompt": {"type": "string", "required": True, "description": "Text prompt"},
        "provider": {
            "type": "string",
            "required": False,
            "description": "Provider: anthropic, openai, google. Default auto-selects.",
        },
        "max_tokens": {"type": "integer", "required": False, "default": 4096},
        "temperature": {"type": "number", "required": False, "default": 0.3},
        "system": {"type": "string", "required": False, "description": "System prompt"},
    }

    def __init__(self, llm_router: Any | None = None) -> None:
        self._router = llm_router

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if not self._router:
            return {"success": False, "error": "LLM Router not initialized"}

        prompt = input_data.get("prompt", "")
        if not prompt:
            return {"success": False, "error": "prompt is required"}

        provider = input_data.get("provider")
        system = input_data.get("system", "")
        max_tokens = input_data.get("max_tokens", 4096)
        temperature = input_data.get("temperature", 0.3)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model_hint = None
        if provider == "anthropic":
            model_hint = "claude"
        elif provider == "openai":
            model_hint = "gpt4"
        elif provider == "google":
            model_hint = "gemini"

        try:
            result = await self._router.route(
                messages,
                prefer_cloud=True,
                max_tokens=max_tokens,
                temperature=temperature,
                model_hint=model_hint,
            )
            return {
                "success": result.get("success", False),
                "text": result.get("text", ""),
                "provider": result.get("provider", "unknown"),
                "model": result.get("model", ""),
                "tokens": {
                    "prompt": result.get("prompt_tokens", 0),
                    "completion": result.get("completion_tokens", 0),
                },
            }
        except Exception as exc:
            log.error("cloud_llm_tool_failed", error=str(exc))
            return {"success": False, "error": str(exc)}
