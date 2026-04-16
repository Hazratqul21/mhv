from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's voice interface. You receive transcribed speech from the "
    "user and produce concise, natural-sounding replies that will be spoken aloud. "
    "Keep responses short and conversational."
)


class VoiceAgent(BaseAgent):
    """Handles speech-to-text (Whisper) and text-to-speech (Kokoro)."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="voice",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model
        self._whisper_model: Any | None = None
        self._kokoro_pipeline: Any | None = None
        self._whisper_model_size = getattr(settings, "whisper_model_size", "base")
        self._kokoro_voice = getattr(settings, "kokoro_voice", "af_heart")
        self._tts_output_dir = Path(getattr(settings, "tts_output_dir", "/tmp/miya_tts"))

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.chat_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        """Process voice input end-to-end: STT -> LLM -> TTS.

        ``context`` may contain:
        - ``audio_path``: path to an audio file for transcription
        - ``skip_tts``:   if truthy, skip the TTS step
        - ``voice``:      override Kokoro voice id
        """
        logger.info("[VoiceAgent] processing voice request")
        context = context or {}

        with Timer() as t:
            transcript: str | None = None
            audio_path = context.get("audio_path")

            if audio_path:
                transcript = await self._transcribe(audio_path)
                if transcript is None:
                    return AgentResult(
                        success=False,
                        output="",
                        execution_time_ms=t.elapsed_ms,
                        error=f"Transcription failed for {audio_path}",
                    )
                logger.info("[VoiceAgent] transcript: %s", transcript[:120])

            effective_query = transcript or query

            try:
                self.ensure_loaded()
                messages = self._build_messages(effective_query, context)
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7,
                )
            except Exception as exc:
                logger.exception("[VoiceAgent] LLM generation failed")
                return AgentResult(
                    success=False,
                    output="",
                    execution_time_ms=t.elapsed_ms,
                    error=str(exc),
                )

            llm_output = response.get("content", "") if isinstance(response, dict) else str(response)
            token_usage = response.get("usage", {}) if isinstance(response, dict) else {}

            tts_path: str | None = None
            if not context.get("skip_tts"):
                voice = context.get("voice", self._kokoro_voice)
                tts_path = await self._synthesize(llm_output, voice)

        result_output = llm_output
        extra: dict[str, Any] = {}
        if transcript:
            extra["transcript"] = transcript
        if tts_path:
            extra["audio_output_path"] = tts_path

        return AgentResult(
            success=True,
            output=result_output,
            tool_calls=[{"tool": "voice_pipeline", "args": extra}] if extra else [],
            token_usage=token_usage,
            execution_time_ms=t.elapsed_ms,
        )

    async def _transcribe(self, audio_path: str) -> str | None:
        """Run Whisper STT on the given audio file."""
        path = Path(audio_path)
        if not path.exists():
            logger.error("Audio file not found: %s", path)
            return None

        try:
            model = self._get_whisper_model()
            result = await asyncio.to_thread(model.transcribe, str(path))
            return result.get("text", "").strip()
        except Exception:
            logger.exception("Whisper transcription failed")
            return None

    def _get_whisper_model(self) -> Any:
        if self._whisper_model is None:
            import whisper

            logger.info("Loading Whisper model (%s)", self._whisper_model_size)
            self._whisper_model = whisper.load_model(self._whisper_model_size)
        return self._whisper_model

    async def _synthesize(self, text: str, voice: str) -> str | None:
        """Run Kokoro TTS and return the path to the generated WAV."""
        try:
            pipeline = self._get_kokoro_pipeline()
            self._tts_output_dir.mkdir(parents=True, exist_ok=True)

            import hashlib
            stem = hashlib.sha256(text[:200].encode()).hexdigest()[:16]
            out_path = self._tts_output_dir / f"{stem}.wav"

            samples, sample_rate = await asyncio.to_thread(
                self._run_kokoro, pipeline, text, voice
            )

            import soundfile as sf
            await asyncio.to_thread(sf.write, str(out_path), samples, sample_rate)
            logger.info("[VoiceAgent] TTS output written to %s", out_path)
            return str(out_path)
        except Exception:
            logger.exception("Kokoro TTS synthesis failed")
            return None

    def _get_kokoro_pipeline(self) -> Any:
        if self._kokoro_pipeline is None:
            from kokoro import KPipeline

            logger.info("Loading Kokoro TTS pipeline")
            self._kokoro_pipeline = KPipeline(lang_code="a")
        return self._kokoro_pipeline

    @staticmethod
    def _run_kokoro(pipeline: Any, text: str, voice: str) -> tuple[Any, int]:
        """Blocking Kokoro generation — run inside ``to_thread``."""
        import numpy as np

        samples_list = []
        sample_rate = 24000
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice):
            if audio is not None:
                samples_list.append(audio)
        if not samples_list:
            raise RuntimeError("Kokoro produced no audio")
        return np.concatenate(samples_list), sample_rate

    def _build_messages(
        self, query: str, context: dict[str, Any] | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            history = context.get("history", [])
            for msg in history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
