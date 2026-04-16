"""Streaming Speech-to-Text using faster-whisper.

Accepts a numpy float32 audio buffer (16kHz mono) and returns the
transcribed text.  The Whisper model is loaded once on first use and
kept resident in memory for fast subsequent calls.

Default device is **cpu** so STT does not compete for VRAM with MIYA
backend (llama.cpp) + Kokoro TTS on one GPU. Override with
``MIYA_STT_DEVICE=cuda`` if you have spare VRAM.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os

import numpy as np

log = logging.getLogger("miya.voice.stt")

DEFAULT_MODEL = "base"
# Bir GPUda backend + TTS bilan CUDA STT tez-tez OOM beradi.
DEFAULT_DEVICE = os.getenv("MIYA_STT_DEVICE", "cpu")


def _stt_gpu_explicitly_allowed() -> bool:
    """GPU STT faqat `MIYA_STT_ALLOW_GPU=1` bo‘lsa — aks holda CPU (OOM oldini olish)."""
    v = os.getenv("MIYA_STT_ALLOW_GPU", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _default_compute_type(device: str, explicit: str | None) -> str:
    env = (os.getenv("MIYA_STT_COMPUTE") or "").strip()
    if env:
        return env
    if explicit and explicit != "default":
        return explicit
    return "int8" if device.lower() == "cpu" else "default"


class StreamingSTT:
    """Async wrapper around faster-whisper for single-shot transcription."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        raw = (device or DEFAULT_DEVICE).strip().lower() or "cpu"
        if not _stt_gpu_explicitly_allowed() and raw in ("auto", "cuda", "gpu"):
            log.warning(
                "MIYA_STT_DEVICE=%s — GPU STT o‘chirildi (VRAM: MIYA backend + Kokoro). "
                "CPU ishlatiladi. GPU STT uchun: MIYA_STT_ALLOW_GPU=1 va bo‘sh VRAM.",
                raw,
            )
            raw = "cpu"
        self.device = raw
        self.compute_type = _default_compute_type(self.device, compute_type)
        self.language = language
        self._model = None
        self._model_key: tuple[str, str, str] | None = None

    def _drop_model(self) -> None:
        self._model = None
        self._model_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_model(self):
        from faster_whisper import WhisperModel

        key = (self.model_size, self.device, self.compute_type)
        if self._model is not None and self._model_key == key:
            return self._model

        log.info(
            "loading faster-whisper model=%s device=%s compute=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._model_key = key
        log.info("faster-whisper ready")
        return self._model

    def _fallback_cpu(self, reason: str) -> None:
        log.warning("STT %s — CPU (int8) ga o'tkazilmoqda", reason)
        self.device = "cpu"
        self.compute_type = "int8"
        self._drop_model()

    def _transcribe_once(self, audio: np.ndarray) -> str:
        model = self._ensure_model()
        kwargs: dict = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 300},
        }
        if self.language:
            kwargs["language"] = self.language

        segments, info = model.transcribe(audio, **kwargs)
        text_parts = [seg.text for seg in segments]
        text = " ".join(text_parts).strip()

        log.info(
            "STT [%s prob=%.2f]: %s",
            info.language,
            info.language_probability,
            text[:120],
        )
        return text

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        try:
            return self._transcribe_once(audio)
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_cuda_oom = "out of memory" in msg or "cuda" in msg
            if not is_cuda_oom:
                raise
            log.warning("STT CUDA/OOM: %s", exc)
            # device=cpu bo‘lsa ham ba’zi buildlar GPU chaqirishi mumkin — modelni CPU+int8 qayta yuklash
            self._fallback_cpu("cuda/oom recovery")
            try:
                return self._transcribe_once(audio)
            except RuntimeError as exc2:
                log.error("STT CPU qayta urinish ham xato: %s", exc2)
                return (
                    "[Nutqni matnga aylantirishda xotira yetmadi. "
                    "MIYA_STT_MODEL=tiny qilib qayta ishga tushiring yoki boshqa GPU dasturlarni yoping.]"
                )
        except Exception:
            log.exception("STT unexpected error")
            return (
                "[Nutqni tanishda xatolik. faster-whisper o‘rnatilganini, "
                "MIYA_STT_MODEL o‘lchamini va loglarni tekshiring.]"
            )

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe *audio* (float32, 16kHz, mono) and return text."""
        if audio.size == 0:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    async def warmup(self) -> None:
        """Pre-load the model so the first real call is fast."""
        await asyncio.to_thread(self._ensure_model)
