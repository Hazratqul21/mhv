from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperTool(BaseTool):
    name = "whisper"
    description = "Transcribe audio files to text using faster-whisper"
    category = "media"
    parameters = {
        "audio_path": {
            "type": "string",
            "description": "Path to the audio file",
        },
        "language": {
            "type": "string",
            "description": "Language code (e.g. en, ja). Auto-detect if omitted",
        },
        "task": {
            "type": "string",
            "enum": ["transcribe", "translate"],
            "description": "Whisper task",
            "default": "transcribe",
        },
    }

    def __init__(self) -> None:
        self._model = None
        settings = get_settings()
        self._model_size = settings.whisper_model
        self._device = settings.whisper_device

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type="float16" if self._device == "cuda" else "int8",
            )
        return self._model

    async def execute(self, input_data: dict[str, Any]) -> Any:
        audio_path = input_data.get("audio_path", "")
        if not audio_path:
            return {"error": "'audio_path' is required"}

        path = Path(audio_path)
        if not path.exists():
            return {"error": f"File not found: {audio_path}"}

        language = input_data.get("language")
        task = input_data.get("task", "transcribe")

        try:
            model = self._load_model()
            segments, info = model.transcribe(
                str(path),
                language=language,
                task=task,
                beam_size=5,
            )

            transcript_parts = []
            segment_data = []
            for seg in segments:
                transcript_parts.append(seg.text)
                segment_data.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })

            return {
                "text": " ".join(part.strip() for part in transcript_parts),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": segment_data,
            }

        except Exception as exc:
            logger.error("whisper_error", error=str(exc))
            return {"error": str(exc)}
