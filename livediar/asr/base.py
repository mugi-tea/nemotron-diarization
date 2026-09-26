"""ASR backend interface and shared audio/text post-processing."""
from __future__ import annotations

import wave

from ..audio import SAMPLE_RATE
from ..textfilters import clean_segment, finalize_text


class ASRBackend:
    """Transcribe one speaker turn stored as a 16 kHz mono WAV file.

    ``lead_silence`` seconds of digital silence are prepended to each turn before it is
    written; backends that lose the first words of an utterance can ask for some.
    """

    name = "base"
    lead_silence = 0.0

    def transcribe(self, wav_path: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- helpers for numpy-based backends ----
    @staticmethod
    def load_audio(wav_path: str, normalize_gain: bool = True):
        """WAV -> float32 ndarray. Quiet recordings (peak < -10 dBFS) are raised to -3 dBFS."""
        import numpy as np

        with wave.open(wav_path, "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if normalize_gain and audio.size:
            peak = float(np.max(np.abs(audio)))
            if 0.0 < peak < 0.3:
                audio = audio * (0.7 / peak)
        return audio

    @staticmethod
    def gate(raw: str, duration_sec: float) -> str:
        """Apply the repetition / hallucination / length guards to a raw transcript."""
        return finalize_text(clean_segment(raw), duration_sec)


class NoASR(ASRBackend):
    """Speaker turns only, no transcription."""

    name = "none"

    def transcribe(self, wav_path: str) -> str:
        return ""
