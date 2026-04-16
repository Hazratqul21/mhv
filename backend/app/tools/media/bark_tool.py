from __future__ import annotations

import asyncio
import base64
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


class BarkTool(BaseTool):
    name = "bark"
    description = "Bark text-to-speech over HTTP, with optional local transformers fallback."
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "description": "tts | list_voices",
            "enum": ["tts", "list_voices"],
        },
        "text": {"type": "string", "description": "Text to speak (tts)"},
        "voice_preset": {
            "type": "string",
            "description": "Bark voice preset id",
            "default": "v2/en_speaker_6",
        },
        "sample_rate": {
            "type": "integer",
            "description": "Output sample rate hint",
            "default": 24000,
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base = (settings.bark_host or "").rstrip("/")

    async def _get_json(self, path: str) -> Any:
        url = f"{self._base}{path}"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        if aiohttp is not None:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        raise RuntimeError("Install httpx or aiohttp for BarkTool")

    async def _post_tts_http(self, text: str, voice_preset: str, sample_rate: int) -> dict[str, Any]:
        url = f"{self._base}/tts"
        body = {"text": text, "voice_preset": voice_preset, "sample_rate": sample_rate}
        if httpx is not None:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "application/json" in ct:
                    return resp.json()
                return {"_raw_audio": resp.content, "_content_type": ct}
        if aiohttp is not None:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                async with session.post(url, json=body) as resp:
                    resp.raise_for_status()
                    if resp.content_type == "application/json":
                        return await resp.json()
                    return {"_raw_audio": await resp.read(), "_content_type": resp.content_type}
        raise RuntimeError("Install httpx or aiohttp for BarkTool")

    def _save_wav(self, audio_bytes: bytes, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"bark_{uuid.uuid4().hex}.wav"
        path.write_bytes(audio_bytes)
        return path

    def _tts_local_sync(self, text: str, voice_preset: str, sample_rate: int, out_path: Path) -> None:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoProcessor, BarkModel

        model_id = "suno/bark-small"
        processor = AutoProcessor.from_pretrained(model_id)
        model = BarkModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        inputs = processor(text, voice_preset=voice_preset).to(device)
        with torch.inference_mode():
            audio_arr = model.generate(**inputs)

        arr = audio_arr.cpu().numpy().squeeze()
        if arr.ndim > 1:
            arr = arr[0]
        arr = np.clip(arr, -1.0, 1.0)
        int16 = (arr * 32767).astype(np.int16)
        sf.write(str(out_path), int16, sample_rate, subtype="PCM_16")

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        action = input_data.get("action", "tts")
        out_dir = Path(settings.output_dir) / "voice"

        try:
            if action == "list_voices":
                if not self._base:
                    return {
                        "success": False,
                        "error": "bark_host is empty; set BARK_HOST or use local presets only",
                    }
                data = await self._get_json("/voices")
                return {"success": True, "action": "list_voices", "voices": data}

            if action != "tts":
                return {"success": False, "error": f"Unknown action: {action}"}

            text = input_data.get("text", "")
            if not text:
                return {"success": False, "error": "text is required"}

            voice_preset = input_data.get("voice_preset", "v2/en_speaker_6")
            sample_rate = int(input_data.get("sample_rate", 24000))
            out_dir.mkdir(parents=True, exist_ok=True)

            if self._base:
                try:
                    data = await self._post_tts_http(text, voice_preset, sample_rate)
                    if isinstance(data, dict):
                        if "audio_base64" in data and data["audio_base64"]:
                            raw = base64.b64decode(str(data["audio_base64"]))
                            path = self._save_wav(raw, out_dir)
                            return {
                                "success": True,
                                "action": "tts",
                                "output_path": str(path),
                                "voice_preset": voice_preset,
                            }
                        if "_raw_audio" in data:
                            path = self._save_wav(data["_raw_audio"], out_dir)
                            return {
                                "success": True,
                                "action": "tts",
                                "output_path": str(path),
                                "voice_preset": voice_preset,
                            }
                        if data.get("output_path"):
                            return {
                                "success": True,
                                "action": "tts",
                                "output_path": str(data["output_path"]),
                                "voice_preset": voice_preset,
                            }
                    return {"success": False, "error": "Unexpected Bark HTTP response", "detail": data}
                except Exception as http_exc:
                    logger.warning("bark_http_failed", error=str(http_exc))

            out_path = out_dir / f"bark_{uuid.uuid4().hex}.wav"
            await asyncio.to_thread(
                self._tts_local_sync, text, voice_preset, sample_rate, out_path
            )
            return {
                "success": True,
                "action": "tts",
                "output_path": str(out_path),
                "voice_preset": voice_preset,
                "mode": "transformers_local",
            }

        except Exception as exc:
            logger.error("bark_error", action=action, error=str(exc))
            return {"success": False, "error": str(exc), "action": action}
