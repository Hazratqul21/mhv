"""Voice Activity Detection using silero-vad.

After the wake word fires, VAD determines when the user has finished
speaking.  It watches for a configurable silence duration before
yielding the collected speech audio.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import torch

log = logging.getLogger("miya.voice.vad")

SILENCE_THRESHOLD_S = 0.6  # seconds of silence before "end of speech"
MAX_SPEECH_S = 15.0  # hard cap to avoid infinite recording
VAD_SAMPLE_RATE = 16_000
VAD_WINDOW = 512  # silero-vad expects 512-sample windows at 16kHz


class VoiceActivityDetector:
    """Async silero-vad wrapper."""

    def __init__(
        self,
        silence_duration: float = SILENCE_THRESHOLD_S,
        max_speech: float = MAX_SPEECH_S,
    ) -> None:
        self.silence_duration = silence_duration
        self.max_speech = max_speech
        self._model: torch.jit.ScriptModule | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torchaudio  # noqa: F401 — silero-vad (torch.hub) imports this
        except ModuleNotFoundError as exc:
            log.error(
                "Silero VAD uchun torchaudio kerak. O'rnating: pip install torchaudio "
                "(CUDA bo'lsa torch va torchaudio bir xil manbadan)."
            )
            raise RuntimeError(
                "Missing torchaudio (required by Silero VAD). Run: pip install torchaudio"
            ) from exc

        self._model, _ = torch.hub.load(
            "snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self._model.eval()
        log.info("silero-vad loaded")

    def _is_speech(self, audio_window: np.ndarray) -> bool:
        self._load()
        tensor = torch.from_numpy(audio_window).float()
        prob = self._model(tensor, VAD_SAMPLE_RATE).item()  # type: ignore[union-attr]
        return prob > 0.5

    async def capture_speech(self, audio_stream) -> np.ndarray:
        """Collect audio until the user stops speaking.

        Returns the concatenated speech audio as a float32 numpy array.
        """
        self._load()

        speech_chunks: list[np.ndarray] = []
        last_speech_time = time.monotonic()
        start_time = last_speech_time
        collecting = False

        async for chunk in audio_stream:
            now = time.monotonic()

            if now - start_time > self.max_speech:
                log.warning("max speech duration reached (%.0fs)", self.max_speech)
                break

            for offset in range(0, len(chunk) - VAD_WINDOW + 1, VAD_WINDOW):
                window = chunk[offset : offset + VAD_WINDOW]
                is_speech = await asyncio.to_thread(self._is_speech, window)

                if is_speech:
                    collecting = True
                    last_speech_time = now

                if collecting:
                    speech_chunks.append(window)

            if collecting and (now - last_speech_time) > self.silence_duration:
                log.debug(
                    "silence detected after %.1fs of speech",
                    now - start_time,
                )
                break

        if not speech_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(speech_chunks)
