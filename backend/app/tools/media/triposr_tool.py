from __future__ import annotations

import base64
import io
import mimetypes
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


class TripoSRTool(BaseTool):
    name = "triposr"
    description = "Single-image to 3D mesh reconstruction via a TripoSR-compatible HTTP service."
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "description": "generate_3d | list_models",
            "enum": ["generate_3d", "list_models"],
        },
        "image_path": {"type": "string", "description": "Path to input RGB image"},
        "output_format": {
            "type": "string",
            "description": "Mesh format",
            "enum": ["obj", "glb"],
            "default": "glb",
        },
        "resolution": {
            "type": "integer",
            "description": "Marching cubes / grid resolution",
            "default": 256,
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.triposr_host.rstrip("/")

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
        raise RuntimeError("Install httpx or aiohttp for TripoSRTool")

    async def _post_generate(
        self,
        image_path: Path,
        output_format: str,
        resolution: int,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        url = f"{self._base}/generate_3d"
        fields = {"output_format": output_format, "resolution": str(resolution)}
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                with image_path.open("rb") as fh:
                    files = {"image": (image_path.name, fh, mime)}
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
                form.add_field(k, v)
            img_bytes = image_path.read_bytes()
            form.add_field(
                "image",
                io.BytesIO(img_bytes),
                filename=image_path.name,
                content_type=mime,
            )
            async with aiohttp.ClientSession(timeout=to) as session:
                async with session.post(url, data=form) as resp:
                    resp.raise_for_status()
                    if resp.content_type == "application/json":
                        return await resp.json()
                    return {"_raw": await resp.read(), "_content_type": resp.content_type}
        raise RuntimeError("Install httpx or aiohttp for TripoSRTool")

    def _save_mesh(self, data: dict[str, Any], out_dir: Path, fmt: str) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = ".glb" if fmt == "glb" else ".obj"
        if data.get("mesh_base64") or data.get("model_base64"):
            raw = base64.b64decode(str(data.get("mesh_base64") or data.get("model_base64")))
            path = out_dir / f"triposr_{uuid.uuid4().hex}{ext}"
            path.write_bytes(raw)
            return {"success": True, "output_path": str(path)}
        if data.get("_raw"):
            path = out_dir / f"triposr_{uuid.uuid4().hex}{ext}"
            path.write_bytes(data["_raw"])
            return {"success": True, "output_path": str(path)}
        if data.get("output_path") or data.get("path"):
            p = data.get("output_path") or data.get("path")
            return {"success": True, "output_path": str(p)}
        if data.get("job_id"):
            return {"success": True, "job_id": data["job_id"], "detail": data}
        return {"success": False, "error": "Unexpected TripoSR response", "detail": data}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        action = input_data.get("action", "generate_3d")
        out_dir = Path(settings.output_dir) / "3d"

        try:
            if action == "list_models":
                data = await self._get_json("/models")
                return {"success": True, "action": "list_models", "models": data}

            if action != "generate_3d":
                return {"success": False, "error": f"Unknown action: {action}"}

            image_path = input_data.get("image_path", "")
            if not image_path:
                return {"success": False, "error": "image_path is required"}
            img = Path(image_path).expanduser()
            if not img.is_file():
                return {"success": False, "error": f"image not found: {img}"}

            fmt = input_data.get("output_format", "glb")
            if fmt not in ("obj", "glb"):
                return {"success": False, "error": "output_format must be obj or glb"}
            resolution = int(input_data.get("resolution", 256))

            data = await self._post_generate(img, fmt, resolution, timeout=600.0)
            result = self._save_mesh(data, out_dir, fmt)
            result["action"] = "generate_3d"
            return result

        except Exception as exc:
            logger.error("triposr_error", action=action, error=str(exc))
            return {"success": False, "error": str(exc), "action": action}
