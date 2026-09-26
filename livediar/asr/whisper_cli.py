"""whisper.cpp's ``whisper-cli`` invoked per turn (any platform, no Python ML dependencies)."""
from __future__ import annotations

import shutil
import subprocess

from .base import ASRBackend


class WhisperCLI(ASRBackend):
    name = "whisper-cli"

    def __init__(self, model: str, language: str = "ja"):
        self.exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
        if not self.exe or not model:
            raise RuntimeError("whisper-cli (brew install whisper-cpp) and --whisper-model <ggml-*.bin> are required")
        self.model, self.language = model, language

    def transcribe(self, wav_path: str) -> str:
        out = subprocess.run(
            [self.exe, "-m", self.model, "-l", self.language, "-nt", "-np", "-f", wav_path],
            capture_output=True, text=True,
        ).stdout
        return " ".join(line.strip() for line in out.splitlines() if line.strip())
