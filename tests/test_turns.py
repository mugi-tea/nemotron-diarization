"""Turn tracking logic against a scripted fake diarizer (no models needed)."""
import array

from livediar.diarizer import FrameProbs, Segment
from livediar.turns import TurnConfig, TurnTracker

SPF = 0.01


class FakeDiarizer:
    """Speaker probabilities are 1.0 inside each scripted segment and 0.0 elsewhere."""

    num_speakers = 4
    seconds_per_frame = SPF

    def __init__(self, segments, now):
        self._segments = segments
        self._now = now

    def seconds(self):
        return self._now

    def segments(self):
        return [Segment(s, e, k) for s, e, k in self._segments]

    def frame_probs(self):
        n = int(self._now / SPF) + 1
        buf = array.array("f", bytes(4 * n * self.num_speakers))
        for s, e, k in self._segments:
            for f in range(int(s / SPF), min(n, int(e / SPF))):
                buf[f * self.num_speakers + k - 1] = 1.0
        return FrameProbs(0, buf, self.num_speakers, SPF)


def test_segment_becomes_a_turn_after_hold():
    d = FakeDiarizer([(0.0, 3.0, 1)], now=3.5)
    tracker = TurnTracker(d, TurnConfig(hold=1.0, min_turn=0.5, post_extend=0.0, pre_extend=0.0))
    assert tracker.update() == ([], 0.0)  # still within hold: nothing final, frontier at its start
    d._now = 4.5
    requests, frontier = tracker.update()
    assert [(r.start, r.end, r.speaker) for r in requests] == [(0.0, 3.0, 1)]
    assert frontier == float("inf")


def test_short_segment_does_not_block_the_frontier():
    # Regression: a segment shorter than min_turn must still be marked consumed, otherwise every
    # later result waits for it and nothing is printed until the stream ends.
    d = FakeDiarizer([(0.0, 0.4, 2), (1.0, 4.0, 1)], now=6.0)
    tracker = TurnTracker(d, TurnConfig(hold=1.0, min_turn=0.8, post_extend=0.0, pre_extend=0.0))
    requests, frontier = tracker.update()
    assert [(r.start, r.speaker) for r in requests] == [(1.0, 1)]
    assert frontier == float("inf")


def test_long_segment_is_cut_into_pieces_before_it_ends():
    d = FakeDiarizer([(0.0, 21.0, 1)], now=21.0)  # still growing
    tracker = TurnTracker(d, TurnConfig(hold=1.0, max_turn=10.0, post_extend=0.0, pre_extend=0.0))
    requests, frontier = tracker.update()
    assert [(r.start, r.end) for r in requests] == [(0.0, 10.0), (10.0, 20.0)]
    assert frontier == 20.0


def test_dominated_overlap_is_skipped():
    # Speaker 2 is detected for 2 s entirely inside speaker 1's segment; probabilities favour 1.
    segs = [(0.0, 6.0, 1), (2.0, 4.0, 2)]
    d = FakeDiarizer(segs, now=8.0)
    probs = d.frame_probs()
    for f in range(200, 400):  # make speaker 1 stronger than speaker 2 over the overlap
        probs.buf[f * 4 + 1] = 0.5
    d.frame_probs = lambda: probs
    tracker = TurnTracker(d, TurnConfig(hold=1.0, post_extend=0.0, pre_extend=0.0))
    requests, _ = tracker.update()
    assert [r.speaker for r in requests] == [1]


def test_extend_bounds_stops_where_another_speaker_dominates():
    d = FakeDiarizer([(0.0, 2.0, 1), (2.5, 5.0, 2)], now=6.0)
    tracker = TurnTracker(d, TurnConfig(pre_extend=1.5, post_extend=2.0))
    lo, hi = tracker.extend_bounds(0.5, 1.5, 1, d.frame_probs())
    assert lo == 0.0
    assert 2.4 <= hi <= 2.5  # grows through the gap but not into speaker 2
