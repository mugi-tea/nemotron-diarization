"""Orchestration: audio source -> diarizer -> turn tracker -> per-turn ASR -> ordered results."""
from __future__ import annotations

import array
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable

from .asr.base import ASRBackend, NoASR
from .audio import SAMPLE_RATE, WavWriter, write_wav_f32
from .diarizer import NemoDiarizer
from .output import TurnResult
from .textfilters import trigram_similarity
from .turns import TurnRequest, TurnTracker

ResultSink = Callable[[TurnResult], None]


class Pipeline:
    """Run diarization and per-turn transcription over a stream of audio chunks.

    Results are delivered to ``sink`` in start-time order: a result is held until every
    segment that started earlier has been finalized (the tracker's ``frontier``). Near-duplicate
    lines from overlapping detections are dropped.
    """

    def __init__(self, diarizer: NemoDiarizer, asr: ASRBackend, tracker: TurnTracker, sink: ResultSink,
                 *, min_turn: float = 0.8, dedupe: bool = True):
        self.diarizer, self.asr, self.tracker, self.sink = diarizer, asr, tracker, sink
        self.min_turn, self.dedupe = min_turn, dedupe
        self._audio = array.array("f")
        self._queue: queue.Queue[tuple[TurnRequest, str] | None] = queue.Queue()
        self._done: list[TurnResult] = []
        self._done_lock = threading.Lock()
        self._printed: list[TurnResult] = []
        self._tmpdir = tempfile.mkdtemp(prefix="livediar_")

    # ---- public ----
    def run(self, source: Iterable[array.array], *, save_path: str | None = None, stop: threading.Event | None = None) -> None:
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        saver = WavWriter(save_path) if save_path else None
        try:
            for chunk in source:
                if stop is not None and stop.is_set():
                    break
                self._audio.extend(chunk)
                self.diarizer.push(chunk)
                if saver:
                    saver.write(chunk)
                self._step()
        finally:
            self.diarizer.finish()
            self._step(final=True)
            self._queue.put(None)
            worker.join(timeout=300)
            self._emit_ready(float("inf"))
            if saver:
                saver.close()
            self.asr.close()
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ---- internals ----
    def _step(self, final: bool = False) -> None:
        requests, frontier = self.tracker.update(final=final)
        for req in requests:
            self._enqueue(req)
        self._emit_ready(frontier)

    def _enqueue(self, req: TurnRequest) -> None:
        a = int(max(req.cut_start, 0.0) * SAMPLE_RATE)
        b = min(len(self._audio), int(req.cut_end * SAMPLE_RATE))
        if b - a < int(self.min_turn * SAMPLE_RATE):
            return
        path = os.path.join(self._tmpdir, f"{uuid.uuid4().hex}.wav")
        write_wav_f32(path, self._audio[a:b], lead_silence_sec=self.asr.lead_silence)
        self._queue.put((req, path))

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            req, path = item
            t0 = time.time()
            try:
                text = self.asr.transcribe(path)
            except Exception as ex:  # keep the stream alive; surface the error in the transcript
                text = f"<ASR error: {ex}>"
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            with self._done_lock:
                self._done.append(TurnResult(req.start, req.end, req.speaker, text, time.time() - t0))

    def _emit_ready(self, frontier: float) -> None:
        with self._done_lock:
            ready = sorted((r for r in self._done if r.start < frontier), key=lambda r: r.start)
            self._done = [r for r in self._done if r.start >= frontier]
        for r in ready:
            if not r.text and not isinstance(self.asr, NoASR):
                continue
            if self.dedupe and self._is_duplicate(r):
                continue
            self._printed.append(r)
            del self._printed[:-12]
            self.sink(r)

    def _is_duplicate(self, r: TurnResult) -> bool:
        for p in self._printed[-6:]:
            overlap = min(r.end, p.end) - max(r.start, p.start)
            same_time = overlap > 0.5 * min(r.end - r.start, p.end - p.start)
            if same_time and trigram_similarity(r.text, p.text) >= 0.6:
                return True  # simultaneous-speech duplicate
            if abs(r.start - p.end) < 4.0 and trigram_similarity(r.text, p.text) >= 0.75:
                return True  # window bled into the neighboring turn
        return False
