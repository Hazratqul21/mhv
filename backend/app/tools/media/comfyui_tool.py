from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ComfyUITool(BaseTool):
    name = "comfyui"
    description = "Generate images by posting workflows to a ComfyUI server"
    category = "media"
    parameters = {
        "prompt": {
            "type": "string",
            "description": "Positive text prompt for image generation",
        },
        "negative_prompt": {
            "type": "string",
            "description": "Negative prompt",
            "default": "",
        },
        "width": {
            "type": "integer",
            "description": "Image width",
            "default": 512,
        },
        "height": {
            "type": "integer",
            "description": "Image height",
            "default": 512,
        },
        "steps": {
            "type": "integer",
            "description": "Sampling steps",
            "default": 20,
        },
        "cfg_scale": {
            "type": "number",
            "description": "CFG scale",
            "default": 7.0,
        },
        "seed": {
            "type": "integer",
            "description": "Random seed (-1 for random)",
            "default": -1,
        },
        "workflow": {
            "type": "object",
            "description": "Full ComfyUI workflow JSON (overrides simple params)",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.comfyui_host.rstrip("/")

    def _build_default_workflow(self, input_data: dict[str, Any]) -> dict:
        """Build a minimal txt2img workflow in ComfyUI API format."""
        seed = input_data.get("seed", -1)
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": input_data.get("steps", 20),
                    "cfg": input_data.get("cfg_scale", 7.0),
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
                    "width": input_data.get("width", 512),
                    "height": input_data.get("height", 512),
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": input_data.get("prompt", ""),
                    "clip": ["4", 1],
                },
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
                "inputs": {"images": ["8", 0], "filename_prefix": "miya"},
            },
        }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        prompt_text = input_data.get("prompt", "")
        if not prompt_text and "workflow" not in input_data:
            return {"error": "'prompt' or 'workflow' is required"}

        workflow = input_data.get("workflow") or self._build_default_workflow(input_data)
        client_id = str(uuid.uuid4())

        try:
            payload = {"prompt": workflow, "client_id": client_id}
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self._base_url}/prompt",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                prompt_id = data.get("prompt_id", "")

            return {
                "status": "queued",
                "prompt_id": prompt_id,
                "client_id": client_id,
                "comfyui_url": self._base_url,
            }

        except httpx.HTTPStatusError as exc:
            logger.error("comfyui_http_error", status=exc.response.status_code)
            return {"error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
        except Exception as exc:
            logger.error("comfyui_error", error=str(exc))
            return {"error": str(exc)}
