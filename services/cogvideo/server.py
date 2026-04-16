"""CogVideoX inference server — exposes a minimal REST API for video generation."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="CogVideoX Server")

OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = None


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_frames: int = Field(49, ge=1, le=200)
    fps: int = Field(8, ge=1, le=60)
    width: int = Field(720, ge=256, le=1920)
    height: int = Field(480, ge=256, le=1080)
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, ge=10, le=100)
    seed: int = -1


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    try:
        from diffusers import CogVideoXPipeline
        import torch

        pipeline = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.float16,
        ).to("cuda")
    except Exception as exc:
        print(f"[CogVideoX] Pipeline load failed (will run in stub mode): {exc}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "pipeline": "loaded" if pipeline else "stub"}


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return {"models": ["CogVideoX-2b"], "loaded": pipeline is not None}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    job_id = str(uuid.uuid4())

    if pipeline is None:
        return {
            "success": False,
            "error": "Pipeline not loaded. Check GPU/model availability.",
            "job_id": job_id,
        }

    import torch

    generator = None
    if req.seed >= 0:
        generator = torch.Generator("cuda").manual_seed(req.seed)

    try:
        result = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            generator=generator,
        )

        output_path = OUTPUT_DIR / f"{job_id}.mp4"

        from diffusers.utils import export_to_video
        export_to_video(result.frames[0], str(output_path), fps=req.fps)

        return {
            "success": True,
            "job_id": job_id,
            "path": str(output_path),
            "frames": req.num_frames,
            "fps": req.fps,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "job_id": job_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
