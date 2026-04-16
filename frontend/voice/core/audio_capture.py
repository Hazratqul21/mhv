"""Microphone capture using sounddevice.

Provides a continuous audio stream at 16kHz mono (the format expected by
Whisper and openWakeWord).  The stream runs in a background thread managed
by sounddevice; frames are pushed into an asyncio queue so the rest of the
pipeline can ``await`` them.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
BLOCK_SIZE = 1280  # 80ms at 16kHz — good balance for wake-word + VAD


class AudioCapture:
    """Async wrapper around a sounddevice InputStream."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        block_size: int = BLOCK_SIZE,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device
        # maxsize<=0 => cheksiz; aks holda put_nowait event loopda QueueFull beradi
        # (sounddevice callback thread-safe chaqiruvda try/except ishlamaydi).
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=0)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # TTS/dinamikdan qaytgan ovozni mikrofon navbatiga yozilmasin (akustik loop).
        self._suppress_input = threading.Event()

    def set_suppress_input(self, on: bool) -> None:
        """Yoniq bo‘lsa, mikrofon bloklari navbatga qo‘yilmaydi (TTS/STT paytida)."""
        if on:
            self._suppress_input.set()
        else:
            self._suppress_input.clear()

    async def drain_queue(self) -> int:
        """Navbatdagi eski chunklarni tashlash (TTS tugagach chaqiring)."""
        n = 0
        while True:
            try:
                self._queue.get_nowait()
                n += 1
            except asyncio.QueueEmpty:
                break
        return n

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            pass  # drop-outs are non-fatal
        if self._suppress_input.is_set():
            return
        try:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait, indata[:, 0].copy()
                )
        except asyncio.QueueFull:
            pass  # drop oldest — real-time is more important

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read(self) -> np.ndarray:
        """Return the next audio block (float32, mono, 16kHz)."""
        return await self._queue.get()

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Infinite async iterator of audio blocks."""
        while True:
            yield await self._queue.get()

    async def read_seconds(self, seconds: float) -> np.ndarray:
        """Accumulate *seconds* worth of audio and return a single array."""
        needed = int(seconds * self.sample_rate)
        chunks: list[np.ndarray] = []
        collected = 0
        while collected < needed:
            chunk = await self.read()
            chunks.append(chunk)
            collected += len(chunk)
        audio = np.concatenate(chunks)
        return audio[:needed]
