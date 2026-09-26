"""Turn the diarizer's growing segment list into finalized speaker turns with audio cut bounds."""
from __future__ import annotations

from dataclasses import dataclass, field

from .diarizer import FrameProbs, NemoDiarizer, Segment


@dataclass
class TurnConfig:
    hold: float = 1.2          # seconds a segment must stop growing before it is final
    min_turn: float = 0.8      # shorter segments are dropped (backchannels, fragments)
    max_turn: float = 15.0     # longer segments are cut into pieces so text appears early
    pre_extend: float = 1.5    # max seconds of audio added before the segment for ASR
    post_extend: float = 2.0   # max seconds added after (diarizer misses word endings)
    other_threshold: float = 0.35  # another speaker "dominates" a frame above this probability
    cover_ratio: float = 0.6   # a segment covered this much by a stronger speaker is a duplicate
    match_tolerance: float = 0.25  # seconds; segment starts drift slightly between updates
    stale_after: float = 5.0   # never wait longer than hold + this for a segment to settle


@dataclass
class TurnRequest:
    """A finalized piece of a speaker's segment and the audio window to transcribe for it."""

    start: float
    end: float
    speaker: int
    cut_start: float
    cut_end: float


@dataclass
class _Record:
    speaker: int
    start: float
    emitted_until: float


@dataclass
class TurnTracker:
    """Track segments across diarizer updates and decide when each becomes a turn.

    A segment is final once its end has not moved for ``hold`` seconds. Segments that are
    mostly covered by another speaker with higher frame probability are treated as duplicate
    detections of that speaker's speech and skipped. The returned ``frontier`` is the earliest
    start among segments still open; results starting before it can be printed in order.
    """

    diarizer: NemoDiarizer
    config: TurnConfig = field(default_factory=TurnConfig)
    _records: list[_Record] = field(default_factory=list)

    def update(self, final: bool = False) -> tuple[list[TurnRequest], float]:
        cfg = self.config
        now = self.diarizer.seconds()
        segs = self.diarizer.segments()
        probs = self.diarizer.frame_probs()
        requests: list[TurnRequest] = []
        for seg in segs:
            rec = self._record_for(seg)
            start = rec.emitted_until
            while seg.end - start >= cfg.max_turn:
                requests.append(self._request(start, start + cfg.max_turn, seg.speaker, probs))
                start += cfg.max_turn
                rec.emitted_until = start
            if final or seg.end < now - cfg.hold:
                # Mark short or dominated segments as consumed too, or they would block the frontier.
                if seg.end - start >= cfg.min_turn and not self._dominated(start, seg.end, seg.speaker, segs, probs):
                    requests.append(self._request(start, seg.end, seg.speaker, probs))
                rec.emitted_until = seg.end
        pending = [
            rec.emitted_until
            for rec in self._records
            for seg in segs
            if rec.speaker == seg.speaker
            and abs(rec.start - seg.start) < cfg.match_tolerance
            and rec.emitted_until < seg.end - 0.05
            and seg.end > now - cfg.hold - cfg.stale_after
        ]
        frontier = min(pending) if pending and not final else float("inf")
        if len(self._records) > 500:
            del self._records[:100]
        return requests, frontier

    def _record_for(self, seg: Segment) -> _Record:
        for rec in self._records:
            if rec.speaker == seg.speaker and abs(rec.start - seg.start) < self.config.match_tolerance:
                return rec
        rec = _Record(seg.speaker, seg.start, seg.start)
        self._records.append(rec)
        return rec

    def _request(self, start: float, end: float, speaker: int, probs: FrameProbs) -> TurnRequest:
        lo, hi = self.extend_bounds(start, end, speaker, probs)
        return TurnRequest(start, end, speaker, lo, hi)

    def extend_bounds(self, start: float, end: float, speaker: int, probs: FrameProbs) -> tuple[float, float]:
        """Widen the audio window using frame probabilities.

        The diarizer clips a few hundred milliseconds off utterance edges, so grow the window
        outward until another speaker becomes more likely than this one (and exceeds
        ``other_threshold``), bounded by ``pre_extend`` / ``post_extend``.
        """
        cfg = self.config
        spf = probs.seconds_per_frame
        n = probs.num_speakers

        def other_dominates(frame: int) -> bool:
            pk = probs.at(frame, speaker)
            if pk is None:
                return False
            return any(
                (pj := probs.at(frame, j)) is not None and pj > cfg.other_threshold and pj > pk
                for j in range(1, n + 1) if j != speaker
            )

        f_lo, f_hi = int(start / spf), int(end / spf)
        stop = max(int((start - cfg.pre_extend) / spf), 0)
        while f_lo - 1 >= stop and not other_dominates(f_lo - 1):
            f_lo -= 1
        stop = int((end + cfg.post_extend) / spf)
        while f_hi + 1 <= stop and not other_dominates(f_hi + 1):
            f_hi += 1
        return f_lo * spf, f_hi * spf

    def _dominated(self, start: float, end: float, speaker: int, segs: list[Segment], probs: FrameProbs) -> bool:
        covered: dict[int, float] = {}
        for other in segs:
            if other.speaker != speaker:
                overlap = max(0.0, min(end, other.end) - max(start, other.start))
                covered[other.speaker] = covered.get(other.speaker, 0.0) + overlap
        if not covered:
            return False
        mean = probs.mean(start, end)
        length = max(end - start, 1e-6)
        return any(cov / length >= self.config.cover_ratio and mean[j - 1] >= mean[speaker - 1] for j, cov in covered.items())
