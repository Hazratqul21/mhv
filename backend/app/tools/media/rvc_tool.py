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


class RVCTool(BaseTool):
    name = "rvc"
    description = "Retrieval-based voice conversion and training via an RVC HTTP service."
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "description": "convert | train | list_models",
            "enum": ["convert", "train", "list_models"],
        },
        "audio_path": {"type": "string", "description": "Input wav path (convert / train)"},
        "model_name": {"type": "string", "description": "RVC model name or id"},
        "pitch_shift": {"type": "integer", "description": "Semitone pitch shift", "default": 0},
        "index_rate": {"type": "number", "description": "Feature retrieval index rate", "default": 0.75},
        "dataset_path": {
            "type": "string",
            "description": "Optional dataset directory for train",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.rvc_host.rstrip("/")

    async def _get_json(self, path: str, timeout: float = 60.0) -> Any:
        url = f"{self._base}{path}"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        if aiohttp is not None:
            to = aiohttp.ClientTimeout(total=int(timeout))
            async with aiohttp.ClientSession(timeout=to) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        raise RuntimeError("Install httpx or aiohttp for RVCTool")

    async def _post_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        file_path: Path,
        file_field: str = "audio",
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        url = f"{self._base}{path}"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                with file_path.open("rb") as fh:
                    files = {file_field: (file_path.name, fh, "application/octet-stream")}
                    resp = await client.post(url, data=fields, files=files)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "application/json" in ct:
                    return resp.json()
                return {"_raw": resp.content, "_content_type": ct}
        if aiohttp is not None:
            to = aiohttp.ClientTimeout(total=int(timeout))
            form = aiohttp.FormData()
            for k, v in fields.items():
                form.add_field(k, str(v))
            raw = file_path.read_bytes()
            form.add_field(
                file_field,
                io.BytesIO(raw),
                filename=file_path.name,
                content_type="application/octet-stream",
            )
            async with aiohttp.ClientSession(timeout=to) as session:
                async with session.post(url, data=form) as resp:
                    resp.raise_for_status()
                    if resp.content_type == "application/json":
                        return await resp.json()
                    return {"_raw": await resp.read(), "_content_type": resp.content_type}
        raise RuntimeError("Install httpx or aiohttp for RVCTool")

    def _save_output(self, data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if data.get("audio_base64"):
            raw = base64.b64decode(str(data["audio_base64"]))
            path = out_dir / f"rvc_{uuid.uuid4().hex}.wav"
            path.write_bytes(raw)
            return {"success": True, "output_path": str(path)}
        if data.get("_raw"):
            suf = ".wav" if "wav" in str(data.get("_content_type", "")) else ".bin"
            path = out_dir / f"rvc_{uuid.uuid4().hex}{suf}"
            path.write_bytes(data["_raw"])
            return {"success": True, "output_path": str(path)}
        if data.get("output_path") or data.get("path"):
            p = data.get("output_path") or data.get("path")
            return {"success": True, "output_path": str(p)}
        if data.get("job_id"):
            return {"success": True, "job_id": data["job_id"], "detail": data}
        return {"success": False, "error": "Unexpected RVC response", "detail": data}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        action = input_data.get("action", "convert")
        out_dir = Path(settings.output_dir) / "voice"

        try:
            if action == "list_models":
                data = await self._get_json("/models")
                return {"success": True, "action": "list_models", "models": data}

            if action == "convert":
                audio_path = input_data.get("audio_path", "")
                model_name = input_data.get("model_name", "")
                if not audio_path or not model_name:
                    return {"success": False, "error": "audio_path and model_name are required"}
                src = Path(audio_path).expanduser()
                if not src.is_file():
                    return {"success": False, "error": f"audio file not found: {src}"}

                fields = {
                    "model_name": model_name,
                    "pitch_shift": int(input_data.get("pitch_shift", 0)),
                    "index_rate": float(input_data.get("index_rate", 0.75)),
                }
                data = await self._post_multipart("/convert", fields, src, timeout=600.0)
                result = self._save_output(data, out_dir)
                result["action"] = "convert"
                return result

            if action == "train":
                model_name = input_data.get("model_name", "")
                if not model_name:
                    return {"success": False, "error": "model_name is required for train"}
                audio_path = input_data.get("audio_path", "")
                dataset_path = input_data.get("dataset_path", "")

                if audio_path:
                    src = Path(audio_path).expanduser()
                    if not src.is_file():
                        return {"success": False, "error": f"audio file not found: {src}"}
                    fields = {
                        "model_name": model_name,
                        "pitch_shift": int(input_data.get("pitch_shift", 0)),
                        "index_rate": float(input_data.get("index_rate", 0.75)),
                    }
                    if dataset_path:
                        fields["dataset_path"] = dataset_path
                    data = await self._post_multipart("/train", fields, src, file_field="audio", timeout=3600.0)
                else:
                    if not dataset_path:
                        return {"success": False, "error": "audio_path or dataset_path required for train"}
                    fields = {
                        "model_name": model_name,
                        "dataset_path": dataset_path,
                        "index_rate": float(input_data.get("index_rate", 0.75)),
                    }
                    if httpx is not None:
                        async with httpx.AsyncClient(timeout=3600.0) as client:
                            resp = await client.post(f"{self._base}/train", json=fields)
                            resp.raise_for_status()
                            data = resp.json()
                    elif aiohttp is not None:
                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=3600)
                        ) as session:
                            async with session.post(f"{self._base}/train", json=fields) as resp:
                                resp.raise_for_status()
                                data = await resp.json()
                    else:
                        return {"success": False, "error": "httpx or aiohttp required"}

                if isinstance(data, dict) and (data.get("job_id") or data.get("status")):
                    return {"success": True, "action": "train", **data}
                return {"success": True, "action": "train", "detail": data}

            return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as exc:
            logger.error("rvc_error", action=action, error=str(exc))
            return {"success": False, "error": str(exc), "action": action}
