"""ctypes binding for the standalone streaming diarizer in NeMo-Speech.cpp (``diar.h``).

Audio in, per-frame speaker probabilities and speaker segments out, no ASR involved.
The library and model are the ones installed by the ``nemo-speech`` CLI; override the
locations with ``NEMO_SPEECH_HOME`` and ``NEMO_SPEECH_MODEL_CACHE`` if needed.
"""
from __future__ import annotations

import array
import ctypes
import glob
import os
from typing import NamedTuple

from .audio import SAMPLE_RATE

SDK_HOME = os.environ.get("NEMO_SPEECH_HOME") or os.path.expanduser("~/Library/Application Support/NeMoSpeech")
MODEL_CACHE = os.environ.get("NEMO_SPEECH_MODEL_CACHE") or os.path.expanduser("~/Library/Caches/NeMoSpeech/models")
LIBRARY = os.path.join(SDK_HOME, "lib", "libnemo_speech_asr_c.dylib")
DEFAULT_MODEL = "Nemotron-3-Diarization"


class DiarizerUnavailable(RuntimeError):
    """Raised when the NeMo-Speech.cpp library or the diarization model is not installed."""


def find_model(name: str = DEFAULT_MODEL) -> str:
    """Return the newest cached GGUF for ``name`` (as downloaded by ``nemo-speech pull``)."""
    hits = sorted(glob.glob(os.path.join(MODEL_CACHE, "nvidia", name, "*", "*.gguf")))
    if not hits:
        raise DiarizerUnavailable(
            f"no GGUF for {name} under {MODEL_CACHE}; run `nemo-speech pull nemotron-3-diarization` first"
        )
    return hits[-1]


class Segment(NamedTuple):
    start: float
    end: float
    speaker: int  # 1-based


class FrameProbs(NamedTuple):
    """Per-frame speaker probabilities, frame-major: ``buf[(frame - first) * n + speaker_index]``."""

    first: int
    buf: array.array
    num_speakers: int
    seconds_per_frame: float

    @property
    def n_frames(self) -> int:
        return len(self.buf) // self.num_speakers

    def at(self, frame: int, speaker: int) -> float | None:
        """Probability of ``speaker`` (1-based) at absolute ``frame``, or None if not retained."""
        i = frame - self.first
        if i < 0 or i >= self.n_frames:
            return None
        return self.buf[i * self.num_speakers + speaker - 1]

    def mean(self, start_sec: float, end_sec: float) -> list[float]:
        """Mean probability per speaker over ``[start_sec, end_sec)`` (index 0 = speaker 1)."""
        k = self.num_speakers
        f0 = max(int(start_sec / self.seconds_per_frame) - self.first, 0)
        f1 = min(int(end_sec / self.seconds_per_frame) - self.first, self.n_frames)
        if f1 <= f0:
            return [0.0] * k
        acc = [0.0] * k
        for f in range(f0, f1):
            base = f * k
            for j in range(k):
                acc[j] += self.buf[base + j]
        return [a / (f1 - f0) for a in acc]


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("gpu", ctypes.c_int32),
        ("preset", ctypes.c_char_p),
        ("chunk_frames", ctypes.c_int32),
        ("right_context_frames", ctypes.c_int32),
        ("left_context_frames", ctypes.c_int32),
        ("fifo_frames", ctypes.c_int32),
        ("spkcache_frames", ctypes.c_int32),
        ("update_period_frames", ctypes.c_int32),
    ]


class _SegmentationConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("onset", ctypes.c_float),
        ("offset", ctypes.c_float),
        ("pad_onset_sec", ctypes.c_double),
        ("pad_offset_sec", ctypes.c_double),
        ("min_gap_sec", ctypes.c_double),
        ("min_duration_sec", ctypes.c_double),
    ]


class _Segment(ctypes.Structure):
    _fields_ = [("start_time", ctypes.c_double), ("end_time", ctypes.c_double), ("speaker", ctypes.c_int32)]


def _bind(lib: ctypes.CDLL) -> None:
    p = ctypes.POINTER
    lib.nemo_speech_asr_last_error.restype = ctypes.c_char_p
    lib.nemo_speech_diar_create.argtypes = [p(_ModelConfig), p(ctypes.c_void_p)]
    lib.nemo_speech_diar_destroy.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_num_speakers.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_seconds_per_frame.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_seconds_per_frame.restype = ctypes.c_double
    lib.nemo_speech_diar_stream_open.argtypes = [ctypes.c_void_p, p(ctypes.c_void_p)]
    lib.nemo_speech_diar_stream_push_f32.argtypes = [ctypes.c_void_p, p(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32]
    lib.nemo_speech_diar_stream_finish.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_stream_close.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_frame_count.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_frame_count.restype = ctypes.c_int64
    lib.nemo_speech_diar_frame_probs_start.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_diar_frame_probs_start.restype = ctypes.c_int64
    lib.nemo_speech_diar_frame_probs.argtypes = [ctypes.c_void_p, p(ctypes.c_float), ctypes.c_size_t]
    lib.nemo_speech_diar_segments.argtypes = [ctypes.c_void_p, p(_SegmentationConfig), p(_Segment), ctypes.c_size_t, p(ctypes.c_size_t)]


class NemoDiarizer:
    """Streaming speaker diarization (Sortformer V2/V3) over the NeMo-Speech.cpp C ABI.

    Typical use::

        d = NemoDiarizer()                     # Nemotron 3, Metal, model default geometry
        d.set_segmentation(onset=0.55, offset=0.45)
        d.push(samples)                        # 16 kHz mono float32, any chunk size
        d.segments()                           # -> [Segment(start, end, speaker), ...]
        d.finish(); d.close()

    ``chunk_frames`` / ``right_context_frames`` are in 80 ms encoder frames; larger values
    trade latency for accuracy (0 keeps the preset value).
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        gpu: int = 0,
        preset: str | None = None,
        chunk_frames: int = 0,
        right_context_frames: int = 0,
        library: str = LIBRARY,
    ):
        if not os.path.exists(library):
            raise DiarizerUnavailable(f"NeMo-Speech.cpp library not found at {library}")
        self.lib = ctypes.CDLL(library)
        _bind(self.lib)
        self.model_path = model_path or find_model()
        self._cfg = _ModelConfig(
            size=ctypes.sizeof(_ModelConfig),
            model_path=self.model_path.encode(),
            gpu=gpu,
            preset=(preset or "").encode() or None,
            chunk_frames=chunk_frames,
            right_context_frames=right_context_frames,
            left_context_frames=-1,
        )
        self.model = ctypes.c_void_p()
        self._check(self.lib.nemo_speech_diar_create(ctypes.byref(self._cfg), ctypes.byref(self.model)), "diar_create")
        self.num_speakers: int = self.lib.nemo_speech_diar_num_speakers(self.model)
        self.seconds_per_frame: float = self.lib.nemo_speech_diar_seconds_per_frame(self.model)
        self.stream = ctypes.c_void_p()
        self._check(self.lib.nemo_speech_diar_stream_open(self.model, ctypes.byref(self.stream)), "stream_open")
        self._seg_cfg: _SegmentationConfig | None = None

    def _check(self, status: int, what: str) -> None:
        if status != 0:
            err = self.lib.nemo_speech_asr_last_error().decode(errors="replace")
            raise RuntimeError(f"{what} failed (status {status}): {err}")

    def set_segmentation(
        self,
        onset: float = 0.0,
        offset: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        min_gap: float = 0.0,
        min_duration: float = 0.0,
    ) -> None:
        """Configure probability-to-segment post-processing (0 keeps the library default)."""
        self._seg_cfg = _SegmentationConfig(
            size=ctypes.sizeof(_SegmentationConfig), onset=onset, offset=offset, pad_onset_sec=pad_onset,
            pad_offset_sec=pad_offset, min_gap_sec=min_gap, min_duration_sec=min_duration,
        )

    def push(self, samples, sample_rate: int = SAMPLE_RATE) -> None:
        """Buffer mono float32 audio; diarization advances as whole chunks become available."""
        if not isinstance(samples, array.array) or samples.typecode != "f":
            samples = array.array("f", samples)
        buf = (ctypes.c_float * len(samples)).from_buffer(samples)
        self._check(self.lib.nemo_speech_diar_stream_push_f32(self.stream, buf, len(samples), sample_rate), "push")

    def seconds(self) -> float:
        """Seconds of audio labeled so far (lags the pushed audio by chunk + right context)."""
        return self.lib.nemo_speech_diar_frame_count(self.stream) * self.seconds_per_frame

    def segments(self) -> list[Segment]:
        cfg = ctypes.byref(self._seg_cfg) if self._seg_cfg else None
        n = ctypes.c_size_t()
        self._check(self.lib.nemo_speech_diar_segments(self.stream, cfg, None, 0, ctypes.byref(n)), "segments")
        if n.value == 0:
            return []
        out = (_Segment * n.value)()
        self._check(self.lib.nemo_speech_diar_segments(self.stream, cfg, out, n.value, ctypes.byref(n)), "segments")
        return [Segment(s.start_time, s.end_time, s.speaker) for s in out[: n.value]]

    def frame_probs(self) -> FrameProbs:
        """Retained per-frame speaker probabilities (about the last 20 minutes on long streams)."""
        n = self.lib.nemo_speech_diar_frame_count(self.stream)
        first = self.lib.nemo_speech_diar_frame_probs_start(self.stream)
        count = (n - first) * self.num_speakers
        buf = array.array("f", bytes(4 * count)) if count > 0 else array.array("f")
        if count > 0:
            cbuf = (ctypes.c_float * count).from_buffer(buf)
            self._check(self.lib.nemo_speech_diar_frame_probs(self.stream, cbuf, count), "frame_probs")
        return FrameProbs(first, buf, self.num_speakers, self.seconds_per_frame)

    def finish(self) -> None:
        """Flush: label the remaining tail. Pushes after this are ignored."""
        self._check(self.lib.nemo_speech_diar_stream_finish(self.stream), "finish")

    def close(self) -> None:
        if self.stream:
            self.lib.nemo_speech_diar_stream_close(self.stream)
            self.stream = None
        if self.model:
            self.lib.nemo_speech_diar_destroy(self.model)
            self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
