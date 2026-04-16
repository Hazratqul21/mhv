"""Async HTTP client for the MIYA backend API.

Sends user messages to ``POST /api/v1/chat`` and returns the assistant's
text response.  Maintains a persistent session ID so the backend can
track conversation context across turns.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import httpx

log = logging.getLogger("miya.voice.api")

DEFAULT_BASE_URL = "http://localhost:8000"
# LLM javobi uzoq bo'lishi mumkin; qisqa timeout server ulanishini uzadi.
DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
CHAT_RETRIES = 5
# Server uzilgandan keyin httpx pooldagi eski TCP qayta ishlatilmasligi uchun
# har transport xatosidan keyin klient yopiladi.
_POST_HEADERS = {"Connection": "close"}


def _api_token_from_env() -> str | None:
    raw = (os.getenv("MIYA_API_TOKEN") or os.getenv("MIYA_JWT") or "").strip()
    return raw or None


class MiyaAPIClient:
    """Lightweight wrapper around MIYA's chat endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        session_id: str | None = None,
        reply_language: str | None = "English",
        api_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = session_id or f"voice-{uuid.uuid4().hex[:12]}"
        rl = (reply_language or "").strip()
        self.reply_language = None if not rl or rl.lower() == "auto" else rl
        tok = (api_token or "").strip() or _api_token_from_env()
        self._api_token = tok or None
        self._client: httpx.AsyncClient | None = None

    def _headers(self, *, post_close: bool) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._api_token:
            h["Authorization"] = f"Bearer {self._api_token}"
        if post_close:
            h.update(_POST_HEADERS)
        return h

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
            )
        return self._client

    async def chat(self, message: str) -> str:
        """Send *message* to MIYA and return the assistant reply text."""
        payload: dict = {
            "message": message,
            "session_id": self.session_id,
            "context": {
                "miya_client": "voice",
                "voice_full_access": True,
                "instruction": (
                    "User is on the local desktop voice client; treat as full MIYA access: "
                    "code, shell when policy allows, search, files, installs — same as text chat."
                ),
            },
        }
        if self.reply_language:
            payload["reply_language"] = self.reply_language

        last_err: Exception | None = None
        for attempt in range(CHAT_RETRIES):
            client = await self._ensure_client()
            try:
                resp = await client.post(
                    "/api/v1/chat",
                    json=payload,
                    headers=self._headers(post_close=True),
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data.get("response") or data.get("reply") or data.get("text", "")
                log.info("API response (%d chars): %s", len(reply), reply[:100])
                return reply
            except httpx.TimeoutException as exc:
                last_err = exc
                log.error("MIYA API timed out (attempt %s/%s)", attempt + 1, CHAT_RETRIES)
                await self.close()
                break
            except httpx.TransportError as exc:
                last_err = exc
                log.warning(
                    "MIYA API transport error (attempt %s/%s): %s",
                    attempt + 1,
                    CHAT_RETRIES,
                    exc,
                )
                await self.close()
                # Server OOM/restart bo'lishi mumkin — biroz kutib yangi TCP
                wait = min(25.0, 2.0 * (2**attempt))
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                log.error("MIYA API error %s: %s", exc.response.status_code, exc.response.text[:200])
                return "Sorry, the server returned an error."
            except Exception as exc:
                last_err = exc
                log.warning("MIYA API unexpected error: %s", exc)
                await self.close()
                await asyncio.sleep(min(15.0, 2.0 * (attempt + 1)))

        if isinstance(last_err, httpx.TimeoutException):
            return "Sorry, the response took too long. Try a shorter question or check the MIYA server."
        if last_err:
            log.error("MIYA API failed after retries: %s", last_err)
            return (
                "Could not reach MIYA or the connection dropped mid-response. "
                "Is the backend running on port 8000? If the model is loading, wait and try again."
            )
        return "Unknown error talking to MIYA."

    async def health(self) -> bool:
        """Return True if the backend is reachable and healthy."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/health", headers=self._headers(post_close=False))
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
