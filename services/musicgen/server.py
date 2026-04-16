"""MusicGen / AudioCraft inference server."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

app = FastAPI(title="MusicGen Server")

OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = None


class GenerateRequest(BaseModel):
    prompt: str
    duration_seconds: float = Field(10.0, ge=1.0, le=120.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(250, ge=1, le=1000)
    model_size: str = Field("small", pattern="^(small|medium|large)$")


@app.on_event("startup")
async def startup() -> None:
    global model
    try:
        from audiocraft.models import MusicGen

        model = MusicGen.get_pretrained("facebook/musicgen-small")
        model.set_generation_params(duration=10)
    except Exception as exc:
        print(f"[MusicGen] Model load failed (stub mode): {exc}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if model else "stub"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    job_id = str(uuid.uuid4())

    if model is None:
        return {"success": False, "error": "Model not loaded", "job_id": job_id}

    try:
        import soundfile as sf

        model.set_generation_params(
            duration=req.duration_seconds,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        wav = model.generate([req.prompt])
        audio = wav[0].cpu().numpy()

        output_path = OUTPUT_DIR / f"{job_id}.wav"
        sf.write(str(output_path), audio.T, samplerate=32000)

        return {
            "success": True,
            "job_id": job_id,
            "path": str(output_path),
            "duration_seconds": req.duration_seconds,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "job_id": job_id}


@app.post("/continue")
async def continue_music(
    prompt: str = "continue this melody",
    duration_seconds: float = 10.0,
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())

    if model is None:
        return {"success": False, "error": "Model not loaded", "job_id": job_id}

    try:
        import soundfile as sf
        import torch
        import io

        audio_data, sr = sf.read(io.BytesIO(await audio.read()))
        melody = torch.tensor(audio_data).unsqueeze(0).float()

        model.set_generation_params(duration=duration_seconds)
        wav = model.generate_with_chroma([prompt], melody[..., :sr * 30], sr)
        result_audio = wav[0].cpu().numpy()

        output_path = OUTPUT_DIR / f"{job_id}_cont.wav"
        sf.write(str(output_path), result_audio.T, samplerate=32000)

        return {
            "success": True,
            "job_id": job_id,
            "path": str(output_path),
            "duration_seconds": duration_seconds,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "job_id": job_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8191)
