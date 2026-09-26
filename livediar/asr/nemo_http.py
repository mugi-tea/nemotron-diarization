"""Built-in NeMo-Speech.cpp ASR (Nemotron 3.5) through its local HTTP server.

Fallback only: the engine drops utterance-initial phrases in Japanese.
Prepending 3 s of silence to each turn recovers most of them, hence ``lead_silence``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid

from .base import ASRBackend


class NemoHTTP(ASRBackend):
    name = "nemo"
    lead_silence = 3.0

    def __init__(self, url: str | None = None, port: int = 8090, language: str = "ja-JP", asr_model: str = "nemotron-3.5"):
        self.language = language
        self.proc: subprocess.Popen | None = None
        if url:
            self.base = url.rstrip("/")
        else:
            self.base = f"http://127.0.0.1:{port}"
            exe = shutil.which("nemo-speech") or os.path.expanduser("~/.local/bin/nemo-speech")
            self.log = open(os.path.join(tempfile.gettempdir(), "livediar-nemo-serve.log"), "w")
            self.proc = subprocess.Popen(
                [exe, "serve", "--asr-model", asr_model, "--no-ui", "--port", str(port)],
                stdout=self.log, stderr=subprocess.STDOUT,
            )
        self._wait_ready()

    def _wait_ready(self, timeout_sec: float = 60.0) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self.base + "/ready", timeout=2).read()
                return
            except Exception:
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError(f"nemo-speech serve exited; see {self.log.name}")
                time.sleep(0.5)
        raise RuntimeError("nemo-speech serve did not become ready")

    def transcribe(self, wav_path: str) -> str:
        boundary = uuid.uuid4().hex
        fields = {"model": "nemotron-3.5", "language": self.language, "response_format": "json"}
        body = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()
        )
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="turn.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        with open(wav_path, "rb") as f:
            body += f.read()
        body += f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.base + "/v1/audio/transcriptions", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=120).read()).get("text", "").strip()

    def close(self) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc = None
