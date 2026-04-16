from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api.auth import verify_token
from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.public_errors import safe_error_detail

log = get_logger(__name__)

ws_router = APIRouter()


class ConnectionManager:
    """Track active WebSocket connections per session."""

    def __init__(self) -> None:
        self._active: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.setdefault(session_id, []).append(ws)
        log.info("ws_connected", session_id=session_id)

    async def disconnect(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            conns = self._active.get(session_id, [])
            if ws in conns:
                conns.remove(ws)
            if not conns:
                self._active.pop(session_id, None)
        log.info("ws_disconnected", session_id=session_id)

    async def send_json(self, session_id: str, data: dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self._active.get(session_id, ()))
        stale: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            await self.disconnect(session_id, ws)

    async def broadcast(self, data: dict[str, Any]) -> None:
        for session_id in list(self._active):
            await self.send_json(session_id, data)

    @property
    def active_count(self) -> int:
        return sum(len(v) for v in self._active.values())

    @property
    def active_sessions(self) -> list[str]:
        return list(self._active.keys())


manager = ConnectionManager()

HEARTBEAT_INTERVAL = 30  # seconds
# Receive idle limit (slowloris / hung clients)
WS_RECEIVE_TIMEOUT = 300.0


@ws_router.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str, token: str = Query(default="")):
    settings = get_settings()
    if settings.env != "development":
        if not token:
            await ws.close(code=4001, reason="Token required")
            return
        try:
            verify_token(token)
        except Exception:
            await ws.close(code=4003, reason="Invalid token")
            return

    await manager.connect(session_id, ws)

    heartbeat_task: asyncio.Task | None = None

    async def _heartbeat():
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await ws.send_json({"type": "ping", "ts": time.time()})
        except Exception:
            pass

    try:
        heartbeat_task = asyncio.create_task(_heartbeat())

        while True:
            try:
                raw = await asyncio.wait_for(
                    ws.receive_text(), timeout=WS_RECEIVE_TIMEOUT
                )
            except asyncio.TimeoutError:
                await ws.send_json(
                    {"type": "error", "detail": "Receive idle timeout; send a message or reconnect."}
                )
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "detail": "Invalid JSON"})
                continue

            msg_type = data.get("type", "chat")

            if msg_type == "pong":
                continue

            if msg_type == "chat":
                await _handle_chat(ws, session_id, data)
            elif msg_type == "ping":
                await ws.send_json({"type": "pong", "ts": time.time()})
            elif msg_type == "status":
                await ws.send_json({
                    "type": "status_response",
                    "session_id": session_id,
                    "connections": manager.active_count,
                    "sessions": manager.active_sessions,
                    "ts": time.time(),
                })
            elif msg_type == "agents":
                orch = ws.app.state.orchestrator
                agents = list(orch._agents.keys()) if orch else []
                await ws.send_json({"type": "agents_response", "agents": agents})
            else:
                await ws.send_json({"type": "error", "detail": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("ws_error", session_id=session_id, error=str(exc))
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        await manager.disconnect(session_id, ws)


def _chat_extra_context(data: dict[str, Any]) -> dict[str, Any] | None:
    """Mirror HTTP ChatRequest extras (reply_language, context) for WebSocket."""
    extra: dict[str, Any] = {}
    rl = data.get("reply_language")
    if isinstance(rl, str) and rl.strip():
        extra["reply_language"] = rl.strip()
    ctx = data.get("context")
    if isinstance(ctx, dict):
        for k, v in ctx.items():
            if v is not None:
                extra[str(k)] = v
    return extra or None


async def _handle_chat(ws: WebSocket, session_id: str, data: dict[str, Any]):
    message = data.get("message", "")
    if not message:
        await ws.send_json({"type": "error", "detail": "Empty message"})
        return

    orchestrator = ws.app.state.orchestrator
    if orchestrator is None:
        await ws.send_json({"type": "error", "detail": "Orchestrator not ready"})
        return

    stream = data.get("stream", False)
    extra_context = _chat_extra_context(data)

    await ws.send_json({"type": "status", "status": "processing", "session_id": session_id})

    try:
        if stream and hasattr(orchestrator, "process_stream"):
            await ws.send_json({"type": "stream_start", "session_id": session_id})
            async for token in orchestrator.process_stream(
                query=message,
                session_id=session_id,
                extra_context=extra_context,
            ):
                await ws.send_json({"type": "token", "content": token})
            await ws.send_json({"type": "stream_end", "session_id": session_id})
        else:
            result = await asyncio.wait_for(
                orchestrator.process(
                    query=message,
                    session_id=session_id,
                    extra_context=extra_context,
                ),
                timeout=360,
            )
            await ws.send_json({
                "type": "response",
                "session_id": session_id,
                **result,
            })
    except asyncio.TimeoutError:
        log.error("ws_chat_timeout", session_id=session_id)
        await ws.send_json({"type": "error", "detail": "Request timed out (6 min). Try a simpler query."})
    except Exception as exc:
        log.error("ws_chat_error", session_id=session_id, error=str(exc))
        settings = get_settings()
        await ws.send_json(
            {"type": "error", "detail": safe_error_detail(settings.env, exc)}
        )
