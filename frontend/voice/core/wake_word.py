"""Wake-word detection using openWakeWord.

Listens to the continuous mic stream and fires when it detects
the wake word.  Uses "hey jarvis" model by default (closest
available openWakeWord model).  Pass a custom ONNX model via
``custom_model_path`` for a true "hey miya" trigger.
Runs entirely on CPU so the GPU stays free for LLM / Whisper.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("miya.voice.wake")

# openWakeWord expects 16kHz int16 in 1280-sample chunks
OWW_CHUNK = 1280
DETECTION_THRESHOLD = 0.5


class WakeWordDetector:
    """Wraps openwakeword for async usage."""

    def __init__(
        self,
        model_names: list[str] | None = None,
        threshold: float = DETECTION_THRESHOLD,
        custom_model_path: str | None = None,
    ) -> None:
        self.model_names = model_names or ["hey_jarvis"]
        self.threshold = threshold
        self.custom_model_path = custom_model_path
        self._model: object | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import openwakeword
        from openwakeword.model import Model

        openwakeword.utils.download_models()

        kwargs: dict = {"inference_framework": "onnx"}
        if self.custom_model_path and Path(self.custom_model_path).exists():
            kwargs["wakeword_models"] = [self.custom_model_path]
        else:
            kwargs["wakeword_models"] = self.model_names

        self._model = Model(**kwargs)
        log.info("openWakeWord loaded: %s", self.model_names)

    def detect(self, audio_chunk: np.ndarray) -> str | None:
        """Feed a 1280-sample float32 chunk; return model name or None."""
        self._load()

        int16 = (audio_chunk * 32767).astype(np.int16)
        self._model.predict(int16)  # type: ignore[union-attr]

        for mdl_name in self._model.prediction_buffer.keys():  # type: ignore[union-attr]
            scores = self._model.prediction_buffer[mdl_name]  # type: ignore[union-attr]
            if scores and scores[-1] > self.threshold:
                self._model.reset()  # type: ignore[union-attr]
                log.info("wake word detected: %s (%.2f)", mdl_name, scores[-1])
                return mdl_name
        return None

    async def listen(self, audio_stream, cancel_event: asyncio.Event | None = None) -> str:
        """Block until a wake word is detected on *audio_stream*.

        *audio_stream* must be an async iterable yielding float32 numpy
        arrays of >=1280 samples at 16kHz.

        If *cancel_event* is set, the method checks it each iteration
        and raises ``asyncio.CancelledError`` for graceful shutdown.
        """
        self._load()
        async for chunk in audio_stream:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError("shutdown requested")

            if len(chunk) < OWW_CHUNK:
                continue

            for offset in range(0, len(chunk) - OWW_CHUNK + 1, OWW_CHUNK):
                sub = chunk[offset : offset + OWW_CHUNK]
                detected = await asyncio.to_thread(self.detect, sub)
                if detected:
                    return detected
        raise RuntimeError("audio stream ended unexpectedly")
