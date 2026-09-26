"""Audio I/O and frame sources. Everything is 16 kHz mono float32 in [-1, 1]."""
from __future__ import annotations

import array
import os
import subprocess
import time
import wave
from collections.abc import Iterable, Iterator

SAMPLE_RATE = 16000
CHUNK_SEC = 0.16  # push cadence into the diarizer (two 80 ms encoder frames)

Samples = array.array  # typecode "f"


def read_wav_f32(path: str) -> array.array:
    """Read a 16 kHz mono 16-bit PCM WAV file into float32 samples."""
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz mono 16-bit PCM WAV")
        pcm = array.array("h")
        pcm.frombytes(w.readframes(w.getnframes()))
    return array.array("f", (x / 32768.0 for x in pcm))


def f32_to_pcm16(samples: Iterable[float]) -> bytes:
    """Convert float samples to little-endian 16-bit PCM bytes, clipping to range."""
    return array.array("h", (max(-32768, min(32767, int(x * 32767))) for x in samples)).tobytes()


class WavWriter:
    """Incrementally write float32 samples to a 16 kHz mono 16-bit WAV file."""

    def __init__(self, path: str):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.path = path
        self._wav = wave.open(path, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)

    def write(self, samples: Iterable[float]) -> None:
        self._wav.writeframes(f32_to_pcm16(samples))

    def close(self) -> None:
        self._wav.close()


def write_wav_f32(path: str, samples: Iterable[float], lead_silence_sec: float = 0.0) -> None:
    """Write samples to a WAV file, optionally prefixed with digital silence."""
    writer = WavWriter(path)
    try:
        if lead_silence_sec > 0:
            writer._wav.writeframes(b"\x00\x00" * int(lead_silence_sec * SAMPLE_RATE))
        writer.write(samples)
    finally:
        writer.close()


class FileSource:
    """Yield fixed-size chunks from a WAV file, optionally paced at real time."""

    def __init__(self, path: str, chunk_sec: float = CHUNK_SEC, realtime: bool = False):
        self.path = path
        self.chunk = int(SAMPLE_RATE * chunk_sec)
        self.realtime = realtime

    def __iter__(self) -> Iterator[array.array]:
        audio = read_wav_f32(self.path)
        t0 = time.time()
        for i in range(0, len(audio), self.chunk):
            if self.realtime:
                time.sleep(max(0.0, t0 + i / SAMPLE_RATE - time.time()))
            yield audio[i : i + self.chunk]

    def close(self) -> None:  # symmetry with MicSource
        pass


def default_micpipe_path() -> str:
    """Location of the `micpipe` helper: $LIVEDIAR_MICPIPE, else <repo>/tools/micpipe."""
    env = os.environ.get("LIVEDIAR_MICPIPE")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "tools", "micpipe")


class MicSource:
    """Yield chunks from the default microphone via the `micpipe` helper.

    `micpipe` (tools/micpipe.swift) captures with AVFoundation, resamples to 16 kHz mono
    PCM16 and streams it to stdout, so no Python audio dependencies are needed. Build it with
    `make micpipe` or `swiftc -O -o tools/micpipe tools/micpipe.swift`.
    """

    def __init__(self, micpipe: str | None = None, chunk_sec: float = CHUNK_SEC):
        self.exe = micpipe or default_micpipe_path()
        if not os.path.exists(self.exe):
            raise FileNotFoundError(f"micpipe not found at {self.exe}; build it with `make micpipe`")
        self.chunk_bytes = int(SAMPLE_RATE * chunk_sec) * 2
        self._proc: subprocess.Popen | None = None

    def __iter__(self) -> Iterator[array.array]:
        self._proc = subprocess.Popen([self.exe], stdout=subprocess.PIPE)
        assert self._proc.stdout is not None
        try:
            while True:
                buf = self._proc.stdout.read(self.chunk_bytes)
                if not buf:
                    return
                pcm = array.array("h")
                pcm.frombytes(buf)
                yield array.array("f", (x / 32768.0 for x in pcm))
        finally:
            self.close()

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
