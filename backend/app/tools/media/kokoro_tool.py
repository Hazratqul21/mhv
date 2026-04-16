from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class KokoroTool(BaseTool):
    name = "kokoro_tts"
    description = "Text-to-speech synthesis using Kokoro TTS engine"
    category = "media"
    parameters = {
        "text": {
            "type": "string",
            "description": "Text to synthesize into speech",
        },
        "voice": {
            "type": "string",
            "description": "Voice ID to use",
        },
        "output_path": {
            "type": "string",
            "description": "Output file path (defaults to temp file)",
        },
        "speed": {
            "type": "number",
            "description": "Speech speed multiplier",
            "default": 1.0,
        },
    }

    def __init__(self) -> None:
        self._pipeline = None
        settings = get_settings()
        self._default_voice = settings.tts_voice

    def _load_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
        return self._pipeline

    async def execute(self, input_data: dict[str, Any]) -> Any:
        text = input_data.get("text", "")
        if not text:
            return {"error": "'text' is required"}

        voice = input_data.get("voice", self._default_voice)
        output_path = input_data.get("output_path", "")
        speed = float(input_data.get("speed", 1.0))

        try:
            pipeline = self._load_pipeline()

            if not output_path:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_path = tmp.name
                tmp.close()

            import soundfile as sf
            all_audio = []

            for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
                if audio is not None:
                    all_audio.append(audio)

            if not all_audio:
                return {"error": "No audio generated"}

            import numpy as np
            combined = np.concatenate(all_audio)
            sf.write(output_path, combined, 24000)

            return {
                "status": "ok",
                "output_path": output_path,
                "duration_seconds": round(len(combined) / 24000, 2),
                "voice": voice,
            }

        except Exception as exc:
            logger.error("kokoro_error", error=str(exc))
            return {"error": str(exc)}
