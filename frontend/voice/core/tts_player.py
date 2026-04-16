"""Text-to-Speech playback using Kokoro + sounddevice.

Generates speech audio from text via the Kokoro TTS library and
streams it to the default speaker in near-real-time.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

import numpy as np
import sounddevice as sd

log = logging.getLogger("miya.voice.tts")

DEFAULT_VOICE = "af_heart"
TTS_SAMPLE_RATE = 24_000


def _tts_gpu_allowed() -> bool:
    return os.getenv("MIYA_TTS_ALLOW_GPU", "").strip().lower() in ("1", "true", "yes", "on")


def _normalize_kokoro_device(raw: str | None) -> str:
    """Resolve Kokoro / torch device: default cpu; GPU only if MIYA_TTS_ALLOW_GPU=1."""
    d = (raw or os.getenv("MIYA_TTS_DEVICE", "cpu") or "cpu").strip().lower()
    if not d or d in ("default", "auto"):
        d = "cuda" if _tts_gpu_allowed() else "cpu"
    if d in ("gpu",):
        d = "cuda"
    if not _tts_gpu_allowed() and d in ("cuda", "mps"):
        log.warning(
            "MIYA_TTS_DEVICE=%s -> cpu (VRAM). GPU uchun MIYA_TTS_ALLOW_GPU=1",
            d,
        )
        d = "cpu"
    if d not in ("cpu", "cuda", "mps"):
        d = "cpu"
    return d


@lru_cache(maxsize=8)
def _load_pipeline(voice: str, kokoro_device: str):
    try:
        from kokoro import KPipeline

        try:
            pipeline = KPipeline(lang_code="a", device=kokoro_device)
        except TypeError:
            pipeline = KPipeline(lang_code="a")
        log.info("Kokoro TTS loaded (voice=%s, device=%s)", voice, kokoro_device)
        return pipeline, voice
    except ImportError:
        log.warning("kokoro not installed — TTS will be unavailable")
        return None, voice


class TTSPlayer:
    """Generates and plays speech audio."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = TTS_SAMPLE_RATE,
        device: int | str | None = None,
        *,
        kokoro_device: str | None = None,
    ) -> None:
        self.voice = voice
        self.sample_rate = sample_rate
        self.device = device
        self._kokoro_device = _normalize_kokoro_device(kokoro_device)
        self._muted = False

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    @property
    def is_muted(self) -> bool:
        return self._muted

    def _generate_sync(self, text: str) -> np.ndarray | None:
        pipeline, voice = _load_pipeline(self.voice, self._kokoro_device)
        if pipeline is None:
            return None

        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(text, voice=voice):
            if audio is not None:
                chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))

        if not chunks:
            return None
        return np.concatenate(chunks)

    def _play_sync(self, audio: np.ndarray) -> None:
        sd.play(audio, samplerate=self.sample_rate, device=self.device)
        sd.wait()

    async def speak(self, text: str) -> None:
        """Generate TTS audio and play it through speakers."""
        if self._muted or not text.strip():
            return

        log.info("TTS generating: %s", text[:80])
        audio = await asyncio.to_thread(self._generate_sync, text)
        if audio is None:
            log.warning("TTS produced no audio")
            return

        log.debug("TTS playing %d samples (%.1fs)", len(audio), len(audio) / self.sample_rate)
        await asyncio.to_thread(self._play_sync, audio)

    async def speak_streamed(self, text: str) -> None:
        """Generate and play TTS chunk-by-chunk for lower latency."""
        if self._muted or not text.strip():
            return

        pipeline, voice = _load_pipeline(self.voice, self._kokoro_device)
        if pipeline is None:
            log.warning(
                "TTS o‘chiq: `pip install 'kokoro>=0.9'` — ovoz chiqmaydi, faqat matn terminalda."
            )
            return

        log.info("TTS streaming: %s", text[:80])

        for _, _, audio in pipeline(text, voice=voice):
            if audio is None:
                continue
            arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            await asyncio.to_thread(self._play_sync, arr)

    async def play_file(self, path: str) -> None:
        """Play a .wav file through speakers."""
        import soundfile as sf

        data, sr = await asyncio.to_thread(sf.read, path)
        if data.ndim > 1:
            data = data[:, 0]
        await asyncio.to_thread(sd.play, data, sr, self.device)
        await asyncio.to_thread(sd.wait)

    async def warmup(self) -> None:
        """Pre-load the TTS model."""
        await asyncio.to_thread(_load_pipeline, self.voice, self._kokoro_device)
