from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[misc, assignment]

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[misc, assignment]

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MusicGenTool(BaseTool):
    name = "musicgen"
    description = "Generate or continue music clips via an AudioCraft / MusicGen HTTP service."
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "description": "generate | continue_music",
            "enum": ["generate", "continue_music"],
        },
        "prompt": {"type": "string", "description": "Text prompt describing the music"},
        "duration_seconds": {
            "type": "number",
            "description": "Target duration in seconds",
            "default": 10,
        },
        "temperature": {"type": "number", "description": "Sampling temperature", "default": 1.0},
        "top_k": {"type": "integer", "description": "Top-k sampling", "default": 250},
        "model_size": {
            "type": "string",
            "description": "Model size preset",
            "enum": ["small", "medium", "large"],
            "default": "small",
        },
        "audio_path": {
            "type": "string",
            "description": "Existing audio to continue from (continue_music)",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.musicgen_host.rstrip("/")

    async def _post_json(self, path: str, body: dict[str, Any], timeout: float = 600.0) -> Any:
        url = f"{self._base}{path}"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "application/json" in ct:
                    return resp.json()
                return {"_raw_bytes": resp.content, "_content_type": ct}
        if aiohttp is not None:
            to = aiohttp.ClientTimeout(total=int(timeout))
            async with aiohttp.ClientSession(timeout=to) as session:
                async with session.post(url, json=body) as resp:
                    resp.raise_for_status()
                    if resp.content_type == "application/json":
                        return await resp.json()
                    return {"_raw_bytes": await resp.read(), "_content_type": resp.content_type}
        raise RuntimeError("Install httpx or aiohttp for MusicGenTool")

    def _save_audio(self, data: bytes, out_dir: Path, suffix: str = ".wav") -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"musicgen_{uuid.uuid4().hex}{suffix}"
        path.write_bytes(data)
        return path

    def _handle_response_dict(self, data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        for key in ("audio_base64", "audio_wav_base64", "audio", "wav_base64"):
            if key in data and data[key]:
                raw = base64.b64decode(str(data[key]))
                path = self._save_audio(raw, out_dir)
                return {"success": True, "output_path": str(path)}
        if "_raw_bytes" in data and data["_raw_bytes"]:
            ct = str(data.get("_content_type", ""))
            suf = ".wav" if "wav" in ct else ".mp3" if "mpeg" in ct else ".bin"
            path = self._save_audio(data["_raw_bytes"], out_dir, suffix=suf)
            return {"success": True, "output_path": str(path)}
        if data.get("output_path") or data.get("path"):
            p = data.get("output_path") or data.get("path")
            return {"success": True, "output_path": str(p), "detail": data}
        if data.get("job_id"):
            return {"success": True, "job_id": data["job_id"], "detail": data}
        return {"success": False, "error": "Unexpected MusicGen response", "detail": data}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        action = input_data.get("action", "generate")
        out_dir = Path(settings.output_dir) / "music"

        try:
            if action == "generate":
                prompt = input_data.get("prompt", "")
                if not prompt:
                    return {"success": False, "error": "prompt is required"}

                body = {
                    "prompt": prompt,
                    "duration_seconds": float(input_data.get("duration_seconds", 10)),
                    "temperature": float(input_data.get("temperature", 1.0)),
                    "top_k": int(input_data.get("top_k", 250)),
                    "model_size": input_data.get("model_size", "small"),
                }
                data = await self._post_json("/generate", body, timeout=600.0)
                if isinstance(data, dict):
                    handled = self._handle_response_dict(data, out_dir)
                    handled["action"] = "generate"
                    return handled
                return {"success": False, "error": "Invalid response type", "detail": str(data)}

            if action == "continue_music":
                audio_path = input_data.get("audio_path", "")
                if not audio_path:
                    return {"success": False, "error": "audio_path is required for continue_music"}
                p = Path(audio_path).expanduser()
                if not p.is_file():
                    return {"success": False, "error": f"audio file not found: {p}"}

                body = {
                    "prompt": input_data.get("prompt", ""),
                    "duration_seconds": float(input_data.get("duration_seconds", 10)),
                    "temperature": float(input_data.get("temperature", 1.0)),
                    "top_k": int(input_data.get("top_k", 250)),
                    "model_size": input_data.get("model_size", "small"),
                }

                if httpx is not None:
                    async with httpx.AsyncClient(timeout=600.0) as client:
                        with p.open("rb") as fh:
                            files = {"audio": (p.name, fh, "application/octet-stream")}
                            resp = await client.post(
                                f"{self._base}/continue",
                                data={k: str(v) for k, v in body.items()},
                                files=files,
                            )
                        resp.raise_for_status()
                        ct = resp.headers.get("content-type", "")
                        if "application/json" in ct:
                            data = resp.json()
                        else:
                            data = {"_raw_bytes": resp.content, "_content_type": ct}
                elif aiohttp is not None:
                    to = aiohttp.ClientTimeout(total=600)
                    form = aiohttp.FormData()
                    for k, v in body.items():
                        form.add_field(k, str(v))
                    audio_bytes = p.read_bytes()
                    form.add_field(
                        "audio",
                        io.BytesIO(audio_bytes),
                        filename=p.name,
                        content_type="application/octet-stream",
                    )
                    async with aiohttp.ClientSession(timeout=to) as session:
                        async with session.post(f"{self._base}/continue", data=form) as resp:
                            resp.raise_for_status()
                            if resp.content_type == "application/json":
                                data = await resp.json()
                            else:
                                data = {
                                    "_raw_bytes": await resp.read(),
                                    "_content_type": resp.content_type,
                                }
                else:
                    return {"success": False, "error": "httpx or aiohttp required for continue_music"}

                if isinstance(data, dict):
                    handled = self._handle_response_dict(data, out_dir)
                    handled["action"] = "continue_music"
                    return handled
                return {"success": False, "error": "Invalid response"}

            return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as exc:
            logger.error("musicgen_error", action=action, error=str(exc))
            return {"success": False, "error": str(exc), "action": action}
