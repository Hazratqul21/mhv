#!/usr/bin/env python3
"""MIYA Voice Daemon — Jarvis-style always-listening voice assistant.

Usage:
    python main.py [options]

By default openWakeWord uses the **hey_jarvis** model (there is no
built-in "hey miya" — say **"Hey Jarvis"** clearly, or use
``--skip-wake`` for push-to-talk: one utterance per cycle without wake.

Requires: MIYA backend on :8000, ``pip install -r requirements.txt``,
and **kokoro** for TTS (otherwise you get text in the terminal but no sound).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VOICE_DIR))

from core.audio_capture import AudioCapture
from core.wake_word import WakeWordDetector
from core.vad import VoiceActivityDetector
from core.stt_streaming import StreamingSTT
from core.tts_player import TTSPlayer
from core.api_client import DEFAULT_TIMEOUT, MiyaAPIClient
from assistant.brain import VoiceBrain, VoiceMode
from assistant.commands import QuickCommands
from assistant.proactive import ProactiveMonitor
from ui.tray_icon import TrayIcon
from ui.visualizer import VoiceVisualizer

log = logging.getLogger("miya.voice")

SOUNDS_DIR = VOICE_DIR / "sounds"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIYA Voice Daemon")
    p.add_argument("--api-url", default=os.getenv("MIYA_API_URL", "http://localhost:8000"),
                    help="MIYA backend base URL")
    p.add_argument(
        "--api-token",
        default=os.getenv("MIYA_API_TOKEN", "") or os.getenv("MIYA_JWT", ""),
        help="Bearer JWT for /api/v1/chat when env is not development (or env MIYA_API_TOKEN / MIYA_JWT)",
    )
    p.add_argument("--stt-model", default=os.getenv("MIYA_STT_MODEL", "base"),
                    help="faster-whisper model size")
    p.add_argument(
        "--stt-device",
        default=os.getenv("MIYA_STT_DEVICE", "cpu"),
        help="faster-whisper: cpu (default). cuda/auto faqat MIYA_STT_ALLOW_GPU=1 bo‘lsa ishlaydi; aks holda cpu ga majburan o‘tkaziladi.",
    )
    p.add_argument(
        "--stt-compute",
        default=os.getenv("MIYA_STT_COMPUTE", ""),
        help="Masalan int8 (CPU). Bo'sh bo'lsa qurilma bo'yicha avtomatik.",
    )
    p.add_argument("--tts-voice", default=os.getenv("MIYA_TTS_VOICE", "af_heart"),
                    help="Kokoro voice name")
    p.add_argument(
        "--language",
        default=os.getenv("MIYA_STT_LANGUAGE", "en"),
        help="STT language code for faster-whisper (en, uz, ru, ...). Default: en",
    )
    p.add_argument(
        "--reply-language",
        default=os.getenv("MIYA_REPLY_LANGUAGE", "English"),
        help="MIYA assistant reply language (English, Uzbek, ...). Use 'auto' for multilingual default.",
    )
    p.add_argument("--wake-threshold", type=float, default=0.5,
                    help="Wake word detection threshold")
    p.add_argument("--silence-duration", type=float, default=0.6,
                    help="Seconds of silence to end speech capture")
    p.add_argument("--no-tray", action="store_true",
                    help="Disable system tray icon")
    p.add_argument("--no-visualizer", action="store_true",
                    help="Disable terminal visualizer")
    p.add_argument("--no-proactive", action="store_true",
                    help="Disable proactive alerts")
    p.add_argument(
        "--skip-wake",
        action="store_true",
        help="Do not wait for wake word; after each reply, listen for speech immediately (push-to-talk loop)",
    )
    p.add_argument(
        "--no-quick-commands",
        action="store_true",
        help="Do not match local quick commands; always send speech to MIYA API (env MIYA_NO_QUICK_COMMANDS=1)",
    )
    p.add_argument(
        "--tts-infer-device",
        default=os.getenv("MIYA_TTS_DEVICE", "cpu"),
        help="Kokoro torch device: cpu (default). cuda/mps faqat MIYA_TTS_ALLOW_GPU=1 bo‘lsa; yoki env MIYA_TTS_DEVICE.",
    )
    p.add_argument("--debug", action="store_true",
                    help="Enable debug logging")
    ns = p.parse_args()
    if os.getenv("MIYA_NO_QUICK_COMMANDS", "").strip().lower() in ("1", "true", "yes", "on"):
        ns.no_quick_commands = True
    return ns


class MiyaVoiceDaemon:
    """Main voice daemon orchestrating all components."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._running = False
        self._cancel = asyncio.Event()

        self.mic = AudioCapture()
        self.wake = WakeWordDetector(threshold=args.wake_threshold)
        self.vad = VoiceActivityDetector(silence_duration=args.silence_duration)
        _sc = (args.stt_compute or "").strip() or None
        self.stt = StreamingSTT(
            model_size=args.stt_model,
            device=(args.stt_device or "cpu").strip(),
            compute_type=_sc,
            language=args.language,
        )
        _kd = (args.tts_infer_device or "").strip() or None
        self.tts = TTSPlayer(voice=args.tts_voice, kokoro_device=_kd)
        _rl = (args.reply_language or "").strip()
        if _rl.lower() == "auto":
            _rl = ""
        self.api = MiyaAPIClient(
            base_url=args.api_url,
            timeout=DEFAULT_TIMEOUT,
            reply_language=_rl or None,
            api_token=(args.api_token or "").strip() or None,
        )
        self.commands = QuickCommands()
        self.brain = VoiceBrain(
            api_client=self.api,
            quick_commands=self.commands,
            use_quick_commands=not args.no_quick_commands,
        )
        self.viz = VoiceVisualizer(enabled=not args.no_visualizer)
        self.proactive = ProactiveMonitor(
            speak_callback=self._proactive_speak,
            check_interval=60.0,
        )

        self.tray: TrayIcon | None = None
        if not args.no_tray:
            self.tray = TrayIcon(
                on_quit=self._request_quit,
                on_mute_toggle=self._on_mute_toggle,
            )

    def _request_quit(self) -> None:
        log.info("quit requested")
        self._running = False
        self._cancel.set()

    def _on_mute_toggle(self, muted: bool) -> None:
        if muted:
            self.tts.mute()
        else:
            self.tts.unmute()

    async def _proactive_speak(self, message: str) -> None:
        """Callback used by ProactiveMonitor to speak alerts."""
        self._update_status("speaking")
        self.mic.set_suppress_input(True)
        try:
            await self.tts.speak(message)
        finally:
            await asyncio.sleep(0.28)
            await self.mic.drain_queue()
            self.mic.set_suppress_input(False)
        self._update_status("idle")

    def _update_status(self, status: str) -> None:
        self.viz.set_status(status)
        if self.tray:
            self.tray.update_status(status)

    async def _play_sound(self, name: str) -> None:
        path = SOUNDS_DIR / name
        if path.exists():
            try:
                await self.tts.play_file(str(path))
            except Exception as e:
                log.debug("failed to play %s: %s", name, e)

    async def warmup(self) -> None:
        """Pre-load models to reduce first-call latency."""
        log.info("warming up models...")

        healthy = await self.api.health()
        if healthy:
            log.info("MIYA backend is reachable")
        else:
            log.warning("MIYA backend not reachable at %s", self.args.api_url)

        await self.stt.warmup()
        await self.tts.warmup()
        try:
            await asyncio.to_thread(self.vad._load)
            log.info("silero-vad (VAD) ready")
        except Exception as exc:
            log.error("VAD preload failed: %s", exc, exc_info=True)
            print(
                "\n❌ VAD yuklanmadi. Silero uchun `torchaudio` kerak:\n"
                "   pip install torchaudio\n"
                "   (CUDA ishlatsangiz: https://pytorch.org dan torch+torchaudio juftligi)\n"
            )
            raise SystemExit(2) from exc

        log.info("warmup complete")

    async def voice_loop(self) -> None:
        """Main voice interaction loop."""
        if self.args.skip_wake:
            log.info("entering voice loop — skip-wake / push-to-talk mode")
            print("\n🎙️  MIYA Voice (push-to-talk). Gapiring — uyg‘onish so‘zi kerak emas.\n")
        else:
            log.info("entering voice loop — openWakeWord default is hey_jarvis")
            print(
                "\n🎙️  MIYA Voice: uyg‘onish uchun inglizcha «Hey Jarvis» deb ayting. "
                "(Kutubxona «hey_miya» emas, «hey_jarvis».) "
                "Buni xohlamasangiz: python main.py --skip-wake\n"
            )

        while self._running:
            try:
                self._update_status("idle")
                self.viz.display()

                if self.args.skip_wake:
                    self._update_status("listening")
                    await self._play_sound("wake.wav")
                else:
                    detected = await self.wake.listen(self.mic.stream(), cancel_event=self._cancel)
                    log.info("wake word detected: %s", detected)
                    self._update_status("listening")
                    await self._play_sound("wake.wav")

                # Uyg‘onish signali navbatda qolsa, keyingi VAD uni «gap» deb olmasin.
                await self.mic.drain_queue()

                speech_audio = await self.vad.capture_speech(self.mic.stream())
                if speech_audio.size == 0:
                    log.debug("no speech captured")
                    continue

                # STT + API + TTS paytida mikrofonni yopish — dinamikdan qaytgan ovozni
                # yana STT ga yubormaslik (akustik loop).
                self.mic.set_suppress_input(True)
                try:
                    self._update_status("processing")
                    text = await self.stt.transcribe(speech_audio)
                    if not text.strip():
                        log.debug("STT returned empty text")
                        continue

                    print(f"\n🗣️  Siz: {text}")
                    response = await self.brain.process(text)
                    print(f"🤖 MIYA: {response}\n")

                    self._update_status("speaking")
                    await self.tts.speak_streamed(response)
                finally:
                    await asyncio.sleep(0.28)
                    drained = await self.mic.drain_queue()
                    if drained:
                        log.debug("post-TTS drained %d mic chunks", drained)
                    self.mic.set_suppress_input(False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("voice loop error: %s", e, exc_info=True)
                await self._play_sound("error.wav")
                await asyncio.sleep(2)

    async def run(self) -> None:
        """Start all components and run the main loop."""
        self._running = True

        if self.tray:
            self.tray.start()

        await self.mic.start()
        await self.warmup()

        if not self.args.no_proactive:
            await self.proactive.start()

        try:
            await self.voice_loop()
        finally:
            log.info("shutting down...")
            await self.proactive.stop()
            await self.mic.stop()
            await self.api.close()
            if self.tray:
                self.tray.stop()
            self.viz.clear()
            print("\n👋 MIYA Voice stopped.")


def main() -> None:
    args = parse_args()
    if os.getenv("MIYA_SKIP_WAKE", "").lower() in ("1", "true", "yes"):
        args.skip_wake = True

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    daemon = MiyaVoiceDaemon(args)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        daemon._running = False
        daemon._cancel.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(daemon.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
