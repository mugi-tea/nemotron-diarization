"""OpenAI whisper via mlx-whisper (Apple Silicon)."""
from __future__ import annotations

import os
import sys
import tempfile
import wave

from ..audio import SAMPLE_RATE
from ..textfilters import clean_segment, finalize_text
from .base import ASRBackend

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class WhisperMLX(ASRBackend):
    """Per-turn whisper with hallucination guards.

    Decoding uses temperature 0 without conditioning on previous text (mlx-whisper has no beam
    search). Whisper's no_speech / avg_logprob scores are per 30 s window, so they gate whole
    windows; runaway repetition is judged per segment with our own compression ratio.
    The last ``context_chars`` characters of earlier turns are passed as the prompt to keep
    terminology consistent across turns (0 disables).
    """

    name = "whisper"

    def __init__(self, model: str = DEFAULT_MODEL, language: str = "ja", prompt: str | None = None, context_chars: int = 120):
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise RuntimeError("mlx-whisper is required for the whisper backend: pip install 'livediar[whisper]'") from e
        self.model, self.language, self.prompt = model, language, prompt
        self.context_chars = context_chars
        self.recent = ""
        print(f"asr: loading {model} ...", file=sys.stderr, flush=True)
        warm = os.path.join(tempfile.gettempdir(), "livediar_warmup.wav")
        with wave.open(warm, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * SAMPLE_RATE)
        self.transcribe(warm)
        self.recent = ""

    def transcribe(self, wav_path: str) -> str:
        import mlx_whisper

        audio = self.load_audio(wav_path)
        context = self.recent[-self.context_chars :] if self.context_chars else ""
        prompt = " ".join(x for x in (self.prompt, context) if x) or None
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.model, language=self.language, fp16=True, temperature=0.0,
            condition_on_previous_text=False, initial_prompt=prompt, no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )
        kept = []
        for seg in result.get("segments", []):
            if seg.get("no_speech_prob", 0) >= 0.6 or seg.get("avg_logprob", 0) < -1.3:
                continue
            text = clean_segment(seg["text"])
            if text:
                kept.append(text)
        text = finalize_text("".join(kept), len(audio) / SAMPLE_RATE)
        if text:
            self.recent = (self.recent + text)[-400:]
        return text
