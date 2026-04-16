from __future__ import annotations

import asyncio
import base64
import json
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


class CogVideoTool(BaseTool):
    name = "cogvideo"
    description = (
        "CogVideoX text-to-video: generate clips, poll job status, or list models "
        "via the cogvideo HTTP service (or optional local CLI)."
    )
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "description": "generate | status | list_models",
            "enum": ["generate", "status", "list_models"],
        },
        "prompt": {"type": "string", "description": "Positive prompt (generate)"},
        "negative_prompt": {
            "type": "string",
            "description": "Negative prompt",
            "default": "",
        },
        "num_frames": {"type": "integer", "description": "Frame count", "default": 49},
        "fps": {"type": "integer", "description": "Frames per second", "default": 8},
        "width": {"type": "integer", "description": "Video width", "default": 720},
        "height": {"type": "integer", "description": "Video height", "default": 480},
        "guidance_scale": {
            "type": "number",
            "description": "Classifier-free guidance",
            "default": 6.0,
        },
        "num_inference_steps": {
            "type": "integer",
            "description": "Denoising steps",
            "default": 50,
        },
        "seed": {"type": "integer", "description": "Random seed (optional)"},
        "job_id": {"type": "string", "description": "Job / task id (status)"},
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.cogvideo_host.rstrip("/")

    async def _post_json(self, path: str, payload: dict[str, Any], timeout: float = 600.0) -> Any:
        url = f"{self._base}{path}"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "application/json" in ct:
                    return resp.json()
                return {"raw": resp.text}
        if aiohttp is not None:
            to = aiohttp.ClientTimeout(total=int(timeout))
            async with aiohttp.ClientSession(timeout=to) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    if resp.content_type == "application/json":
                        return await resp.json()
                    return {"raw": await resp.text()}
        raise RuntimeError("Install httpx or aiohttp for CogVideoTool HTTP calls")

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
        raise RuntimeError("Install httpx or aiohttp for CogVideoTool HTTP calls")

    def _save_video_bytes(self, data: bytes, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"cogvideo_{uuid.uuid4().hex}.mp4"
        path = out_dir / name
        path.write_bytes(data)
        return path

    async def _generate_via_subprocess(
        self,
        payload: dict[str, Any],
        out_dir: Path,
    ) -> dict[str, Any]:
        import os
        import shlex

        cmd = os.environ.get("COGVIDEO_GENERATE_CMD", "").strip()
        if not cmd:
            return {"success": False, "error": "COGVIDEO_GENERATE_CMD is not set for subprocess mode"}

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cogvideo_{uuid.uuid4().hex}.mp4"
        env_payload = json.dumps(payload)
        parts = shlex.split(cmd) + [env_payload, str(out_path)]

        proc = await asyncio.create_subprocess_exec(
            *parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {
                "success": False,
                "error": (stderr or stdout or b"subprocess failed").decode(errors="replace"),
                "returncode": proc.returncode,
            }
        if not out_path.exists():
            return {"success": False, "error": "Subprocess did not produce output file", "stderr": stderr.decode(errors="replace")}
        return {"success": True, "action": "generate", "output_path": str(out_path)}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        action = input_data.get("action", "generate")
        out_dir = Path(settings.output_dir) / "videos"

        try:
            if action == "list_models":
                data = await self._get_json("/models")
                return {"success": True, "action": "list_models", "models": data}

            if action == "status":
                job_id = input_data.get("job_id") or input_data.get("task_id")
                if not job_id:
                    return {"success": False, "error": "job_id (or task_id) is required for status"}
                data = await self._get_json(f"/status/{job_id}")
                return {"success": True, "action": "status", "status": data}

            if action != "generate":
                return {"success": False, "error": f"Unknown action: {action}"}

            prompt = input_data.get("prompt", "")
            if not prompt:
                return {"success": False, "error": "prompt is required for generate"}

            body = {
                "prompt": prompt,
                "negative_prompt": input_data.get("negative_prompt", ""),
                "num_frames": int(input_data.get("num_frames", 49)),
                "fps": int(input_data.get("fps", 8)),
                "width": int(input_data.get("width", 720)),
                "height": int(input_data.get("height", 480)),
                "guidance_scale": float(input_data.get("guidance_scale", 6.0)),
                "num_inference_steps": int(input_data.get("num_inference_steps", 50)),
            }
            if input_data.get("seed") is not None:
                body["seed"] = int(input_data["seed"])

            try:
                data = await self._post_json("/generate", body, timeout=900.0)
            except Exception as http_exc:
                logger.warning("cogvideo_http_generate_failed", error=str(http_exc))
                sub = await self._generate_via_subprocess(body, out_dir)
                if sub.get("success"):
                    return sub
                return {"success": False, "error": str(http_exc), "subprocess": sub}

            out_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(data, dict):
                if data.get("job_id") or data.get("task_id"):
                    jid = data.get("job_id") or data.get("task_id")
                    return {"success": True, "action": "generate", "job_id": jid, "detail": data}
                for key in ("video_base64", "video", "output_base64"):
                    if key in data and data[key]:
                        raw = base64.b64decode(str(data[key]))
                        path = self._save_video_bytes(raw, out_dir)
                        return {"success": True, "action": "generate", "output_path": str(path)}
                if data.get("output_path") or data.get("path"):
                    p = data.get("output_path") or data.get("path")
                    return {"success": True, "action": "generate", "output_path": str(p), "detail": data}

            return {"success": False, "error": "Unexpected response from cogvideo service", "detail": data}

        except Exception as exc:
            logger.error("cogvideo_error", action=action, error=str(exc))
            return {"success": False, "error": str(exc), "action": action}
