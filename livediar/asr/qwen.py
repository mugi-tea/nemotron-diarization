"""Qwen3-ASR via mlx-audio (Apple Silicon). Default backend for conversational Japanese."""
from __future__ import annotations

import sys

from ..audio import SAMPLE_RATE
from .base import ASRBackend

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"


def parse_hotwords(prompt: str | None) -> list[str] | None:
    words = [w.strip("。、 ") for w in (prompt or "").replace("、", ",").replace(" ", ",").split(",")]
    return [w for w in words if w] or None


class QwenASR(ASRBackend):
    name = "qwen"

    def __init__(self, model: str = DEFAULT_MODEL, language: str = "ja", prompt: str | None = None):
        try:
            from mlx_audio.stt.utils import load_model
        except ImportError as e:
            raise RuntimeError("mlx-audio is required for the qwen backend: pip install 'livediar[qwen]'") from e
        self.model_name, self.language = model, language
        self.hotwords = parse_hotwords(prompt)
        print(f"asr: loading {model} ...", file=sys.stderr, flush=True)
        self.model = load_model(model)
        import numpy as np

        self._generate(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm up Metal kernels

    def _generate(self, audio) -> str:
        import mlx.core as mx

        r = self.model.generate(mx.array(audio), language=self.language, hotwords=self.hotwords)
        text = getattr(r, "text", None) or (r[0].text if isinstance(r, list) else str(r))
        return text.strip()

    def transcribe(self, wav_path: str) -> str:
        audio = self.load_audio(wav_path)
        return self.gate(self._generate(audio), len(audio) / SAMPLE_RATE)
