from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.logger import get_logger

log = get_logger(__name__)


class ModelProvider(str, Enum):
    LOCAL = "local"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class ModelEndpoint:
    provider: ModelProvider
    model_name: str
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    enabled: bool = True
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_calls: int = 0
    failed_calls: int = 0


class LLMRouter:
    """Intelligently routes LLM requests between local and cloud providers.

    Routing strategy:
    - Simple/fast queries -> local models (free, low latency)
    - Complex reasoning / long context -> cloud (if API key available)
    - Fallback: cloud unavailable -> always local
    - API keys absent -> cloud disabled automatically
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._local_engine = llm_engine
        self._settings = get_settings()
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._cloud_clients: dict[ModelProvider, Any] = {}
        self._complexity_threshold = 0.6
        self._setup_endpoints()

    def _setup_endpoints(self) -> None:
        s = self._settings

        self._endpoints["local_orchestrator"] = ModelEndpoint(
            provider=ModelProvider.LOCAL,
            model_name=s.orchestrator_model,
            enabled=True,
        )
        self._endpoints["local_chat"] = ModelEndpoint(
            provider=ModelProvider.LOCAL,
            model_name=s.chat_model,
            enabled=True,
        )
        self._endpoints["local_code"] = ModelEndpoint(
            provider=ModelProvider.LOCAL,
            model_name=s.code_model,
            enabled=True,
        )
        self._endpoints["local_uncensored"] = ModelEndpoint(
            provider=ModelProvider.LOCAL,
            model_name=s.uncensored_model,
            enabled=True,
        )
        self._endpoints["local_creative"] = ModelEndpoint(
            provider=ModelProvider.LOCAL,
            model_name=s.creative_model,
            enabled=True,
        )

        anthropic_key = getattr(s, "anthropic_api_key", "")
        if anthropic_key:
            self._endpoints["claude"] = ModelEndpoint(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-sonnet-4-20250514",
                api_key=anthropic_key,
                base_url="https://api.anthropic.com",
                max_tokens=8192,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
            )

        openai_key = getattr(s, "openai_api_key", "")
        if openai_key:
            self._endpoints["gpt4"] = ModelEndpoint(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                api_key=openai_key,
                base_url="https://api.openai.com/v1",
                max_tokens=8192,
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
            )

        google_key = getattr(s, "google_api_key", "")
        if google_key:
            self._endpoints["gemini"] = ModelEndpoint(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-2.0-flash",
                api_key=google_key,
                base_url="https://generativelanguage.googleapis.com",
                max_tokens=8192,
                cost_per_1k_input=0.00035,
                cost_per_1k_output=0.0014,
            )

        cloud_count = sum(
            1 for ep in self._endpoints.values()
            if ep.provider != ModelProvider.LOCAL and ep.enabled
        )
        log.info("llm_router_init", local=True, cloud_endpoints=cloud_count)

    async def route(
        self,
        messages: list[dict[str, str]],
        *,
        prefer_cloud: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        model_hint: str | None = None,
    ) -> dict[str, Any]:
        if model_hint and model_hint in self._endpoints:
            ep = self._endpoints[model_hint]
            if ep.enabled:
                return await self._call_endpoint(ep, messages, max_tokens, temperature)

        complexity = self._estimate_complexity(messages)
        cloud_available = self._get_best_cloud_endpoint()

        use_cloud = (
            cloud_available is not None
            and (prefer_cloud or complexity > self._complexity_threshold)
        )

        if use_cloud and cloud_available:
            try:
                result = await self._call_endpoint(
                    cloud_available, messages, max_tokens, temperature
                )
                if result.get("success"):
                    return result
            except Exception as exc:
                log.warning("cloud_fallback_to_local", error=str(exc))

        return await self._call_local(messages, max_tokens, temperature)

    async def _call_local(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        model = model_name or self._settings.orchestrator_model
        start = time.monotonic()
        try:
            result = await self._local_engine.chat(
                model_filename=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency = (time.monotonic() - start) * 1000
            return {
                "success": True,
                "text": result.get("text", ""),
                "provider": "local",
                "model": model,
                "latency_ms": latency,
                "cost": 0.0,
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "provider": "local"}

    async def _call_endpoint(
        self,
        endpoint: ModelEndpoint,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        if endpoint.provider == ModelProvider.LOCAL:
            return await self._call_local(messages, max_tokens, temperature, model_name=endpoint.model_name)

        start = time.monotonic()
        endpoint.total_calls += 1

        try:
            if endpoint.provider == ModelProvider.ANTHROPIC:
                result = await self._call_anthropic(endpoint, messages, max_tokens, temperature)
            elif endpoint.provider == ModelProvider.OPENAI:
                result = await self._call_openai(endpoint, messages, max_tokens, temperature)
            elif endpoint.provider == ModelProvider.GOOGLE:
                result = await self._call_google(endpoint, messages, max_tokens, temperature)
            else:
                raise ValueError(f"Unknown provider: {endpoint.provider}")

            latency = (time.monotonic() - start) * 1000
            endpoint.avg_latency_ms = (
                endpoint.avg_latency_ms * 0.9 + latency * 0.1
            )
            endpoint.success_rate = 1 - (endpoint.failed_calls / max(endpoint.total_calls, 1))

            result["latency_ms"] = latency
            result["provider"] = endpoint.provider.value
            result["model"] = endpoint.model_name
            return result

        except Exception as exc:
            endpoint.failed_calls += 1
            endpoint.success_rate = 1 - (endpoint.failed_calls / max(endpoint.total_calls, 1))
            log.error("cloud_call_failed", provider=endpoint.provider.value, error=str(exc))
            return {"success": False, "error": str(exc), "provider": endpoint.provider.value}

    async def _call_anthropic(
        self, ep: ModelEndpoint, messages: list, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        system_msg = ""
        api_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                api_msgs.append({"role": m["role"], "content": m["content"]})

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ep.base_url}/v1/messages",
                headers={
                    "x-api-key": ep.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ep.model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_msg,
                    "messages": api_msgs,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"] if data.get("content") else ""
            usage = data.get("usage", {})
            return {
                "success": True,
                "text": text,
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            }

    async def _call_openai(
        self, ep: ModelEndpoint, messages: list, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ep.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {ep.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ep.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0] if data.get("choices") else {}
            usage = data.get("usage", {})
            return {
                "success": True,
                "text": choice.get("message", {}).get("content", ""),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

    async def _call_google(
        self, ep: ModelEndpoint, messages: list, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ep.base_url}/v1beta/models/{ep.model_name}:generateContent?key={ep.api_key}",
                json={
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [{}])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = parts[0].get("text", "") if parts else ""
            usage = data.get("usageMetadata", {})
            return {
                "success": True,
                "text": text,
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
            }

    def _estimate_complexity(self, messages: list[dict[str, str]]) -> float:
        total_len = sum(len(m.get("content", "")) for m in messages)
        score = min(total_len / 5000, 0.5)

        last_msg = messages[-1].get("content", "").lower() if messages else ""
        complex_indicators = [
            "analyze", "compare", "explain in detail", "step by step",
            "architecture", "design", "implement", "refactor",
            "debug", "optimize", "review",
        ]
        for indicator in complex_indicators:
            if indicator in last_msg:
                score += 0.15
                break

        return min(score, 1.0)

    def _get_best_cloud_endpoint(self) -> ModelEndpoint | None:
        cloud_eps = [
            ep for ep in self._endpoints.values()
            if ep.provider != ModelProvider.LOCAL and ep.enabled and ep.api_key
        ]
        if not cloud_eps:
            return None
        return max(cloud_eps, key=lambda e: e.success_rate)

    def add_endpoint(self, name: str, endpoint: ModelEndpoint) -> None:
        self._endpoints[name] = endpoint
        log.info("endpoint_added", name=name, provider=endpoint.provider.value)

    def remove_endpoint(self, name: str) -> bool:
        if name in self._endpoints:
            del self._endpoints[name]
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        return {
            name: {
                "provider": ep.provider.value,
                "model": ep.model_name,
                "enabled": ep.enabled,
                "has_key": bool(ep.api_key),
                "total_calls": ep.total_calls,
                "success_rate": ep.success_rate,
                "avg_latency_ms": ep.avg_latency_ms,
            }
            for name, ep in self._endpoints.items()
        }
