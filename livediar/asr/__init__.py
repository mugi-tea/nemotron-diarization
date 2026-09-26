"""ASR backends. ``create_backend("auto")`` picks the best one that is installed."""
from __future__ import annotations

import importlib.util

from .base import ASRBackend, NoASR

BACKENDS = ("qwen", "whisper", "nemo", "whisper-cli", "none")
ALIASES = {"mlx": "whisper"}


def auto_backend_name() -> str:
    """qwen (mlx-audio) > whisper (mlx-whisper) > built-in nemo, whichever is importable."""
    if importlib.util.find_spec("mlx_audio"):
        return "qwen"
    if importlib.util.find_spec("mlx_whisper"):
        return "whisper"
    return "nemo"


def create_backend(
    name: str,
    *,
    language: str | None = None,
    prompt: str | None = None,
    qwen_model: str | None = None,
    whisper_model: str | None = None,
    whisper_context_chars: int = 120,
    whisper_cli_model: str | None = None,
    nemo_url: str | None = None,
    nemo_port: int = 8090,
) -> ASRBackend:
    name = ALIASES.get(name, name)
    if name == "auto":
        name = auto_backend_name()
    if name == "qwen":
        from .qwen import DEFAULT_MODEL, QwenASR

        return QwenASR(model=qwen_model or DEFAULT_MODEL, language=language or "ja", prompt=prompt)
    if name == "whisper":
        from .whisper_mlx import DEFAULT_MODEL, WhisperMLX

        return WhisperMLX(model=whisper_model or DEFAULT_MODEL, language=language or "ja", prompt=prompt,
                          context_chars=whisper_context_chars)
    if name == "nemo":
        from .nemo_http import NemoHTTP

        return NemoHTTP(url=nemo_url, port=nemo_port, language=language or "ja-JP")
    if name == "whisper-cli":
        from .whisper_cli import WhisperCLI

        return WhisperCLI(model=whisper_cli_model or "", language=language or "ja")
    if name == "none":
        return NoASR()
    raise ValueError(f"unknown ASR backend: {name}")


__all__ = ["ASRBackend", "NoASR", "BACKENDS", "create_backend", "auto_backend_name"]
