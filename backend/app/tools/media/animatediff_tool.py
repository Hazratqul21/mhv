from __future__ import annotations

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


class AnimateDiffTool(BaseTool):
    name = "animatediff"
    description = (
        "AnimateDiff motion generation or frame interpolation via ComfyUI HTTP API "
        "(/prompt and optional /animatediff/* extension routes)."
    )
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "description": "generate | interpolate",
            "enum": ["generate", "interpolate"],
        },
        "prompt": {"type": "string", "description": "Positive prompt"},
        "negative_prompt": {"type": "string", "description": "Negative prompt", "default": ""},
        "num_frames": {"type": "integer", "description": "Number of frames", "default": 16},
        "fps": {"type": "integer", "description": "FPS", "default": 8},
        "width": {"type": "integer", "description": "Width", "default": 512},
        "height": {"type": "integer", "description": "Height", "default": 512},
        "steps": {"type": "integer", "description": "Sampling steps", "default": 25},
        "guidance_scale": {"type": "number", "description": "CFG scale", "default": 7.5},
        "video_path": {
            "type": "string",
            "description": "Source video path for interpolate",
        },
        "interpolation_multiplier": {
            "type": "integer",
            "description": "Frame multiplier for interpolate",
            "default": 2,
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.comfyui_host.rstrip("/")

    def _build_animatediff_workflow(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Minimal ComfyUI API workflow; replace checkpoint / AD nodes in deployment."""
        seed = input_data.get("seed")
        if seed is None:
            seed = uuid.uuid4().int % (2**32 - 1)

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(seed),
                    "steps": int(input_data.get("steps", 25)),
                    "cfg": float(input_data.get("guidance_scale", 7.5)),
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": int(input_data.get("width", 512)),
                    "height": int(input_data.get("height", 512)),
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": input_data.get("prompt", ""), "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": input_data.get("negative_prompt", ""),
                    "clip": ["4", 1],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": f"miya_ad_{uuid.uuid4().hex[:8]}",
                },
            },
        }

    async def _post_comfy(self, path: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
        url = f"{self._base}{path}"
        if httpx is not None:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        if aiohttp is not None:
            to = aiohttp.ClientTimeout(total=int(timeout))
            async with aiohttp.ClientSession(timeout=to) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        raise RuntimeError("Install httpx or aiohttp for AnimateDiffTool")

    async def _try_extension_generate(self, body: dict[str, Any]) -> dict[str, Any] | None:
        for path in ("/animatediff/generate", "/api/animatediff/generate"):
            try:
                return await self._post_comfy(path, body, timeout=600.0)
            except Exception:
                continue
        return None

    async def _try_extension_interpolate(self, body: dict[str, Any]) -> dict[str, Any] | None:
        for path in ("/animatediff/interpolate", "/api/animatediff/interpolate"):
            try:
                return await self._post_comfy(path, body, timeout=600.0)
            except Exception:
                continue
        return None

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        action = input_data.get("action", "generate")
        out_dir = Path(settings.output_dir) / "videos"

        try:
            if action == "generate":
                prompt = input_data.get("prompt", "")
                if not prompt:
                    return {"success": False, "error": "prompt is required"}

                ext_body = {
                    "prompt": prompt,
                    "negative_prompt": input_data.get("negative_prompt", ""),
                    "num_frames": int(input_data.get("num_frames", 16)),
                    "fps": int(input_data.get("fps", 8)),
                    "width": int(input_data.get("width", 512)),
                    "height": int(input_data.get("height", 512)),
                    "steps": int(input_data.get("steps", 25)),
                    "guidance_scale": float(input_data.get("guidance_scale", 7.5)),
                }
                ext = await self._try_extension_generate(ext_body)
                if ext is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    return {"success": True, "action": "generate", "comfyui_response": ext}

                client_id = str(uuid.uuid4())
                workflow = self._build_animatediff_workflow(input_data)
                data = await self._post_comfy(
                    "/prompt",
                    {"prompt": workflow, "client_id": client_id},
                    timeout=300.0,
                )
                return {
                    "success": True,
                    "action": "generate",
                    "mode": "comfyui_prompt",
                    "note": "Extension route unavailable; queued default workflow (customize nodes for AnimateDiff).",
                    "prompt_id": data.get("prompt_id", ""),
                    "client_id": client_id,
                    "meta": {"num_frames_hint": ext_body["num_frames"], "fps": ext_body["fps"]},
                }

            if action == "interpolate":
                video_path = input_data.get("video_path", "")
                if not video_path:
                    return {"success": False, "error": "video_path is required for interpolate"}

                body = {
                    "video_path": video_path,
                    "multiplier": int(input_data.get("interpolation_multiplier", 2)),
                }
                ext = await self._try_extension_interpolate(body)
                if ext is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    return {"success": True, "action": "interpolate", "comfyui_response": ext}

                return {
                    "success": False,
                    "error": "No AnimateDiff interpolate endpoint found on ComfyUI host; "
                    "expose POST /animatediff/interpolate or /api/animatediff/interpolate.",
                }

            return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as exc:
            logger.error("animatediff_error", action=action, error=str(exc))
            return {"success": False, "error": str(exc), "action": action}
