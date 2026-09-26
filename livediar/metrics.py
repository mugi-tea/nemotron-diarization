"""Evaluation helpers: character error rate and a simple diarization error rate."""
from __future__ import annotations

import re
import unicodedata
from itertools import permutations

from .diarizer import Segment

_PUNCT = re.compile(r"[\s、。，．,.!?！？「」()（）:：;；\-ー～~・…'\"]")


def normalize_text(s: str) -> str:
    """NFKC, lowercase, strip punctuation and whitespace so only content characters count."""
    return _PUNCT.sub("", unicodedata.normalize("NFKC", s).lower())


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate in [0, inf) after normalization."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    return levenshtein(ref, hyp) / max(len(ref), 1)


def load_rttm(path: str) -> list[Segment]:
    segs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            segs.append(Segment(start, start + dur, p[7]))  # speaker label kept as string
    return segs


def write_rttm(path: str, segments: list[Segment], recording_id: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"SPEAKER {recording_id} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {s.speaker} <NA> <NA>\n")


def der(reference: list[Segment], hypothesis: list[Segment], step: float = 0.01) -> tuple[float, dict]:
    """Diarization error rate (missed + false alarm + confusion) / reference speech, with the
    best speaker-label mapping found by brute force. Suitable for a handful of speakers."""
    end = max([s.end for s in reference + hypothesis] + [0.0])
    n = int(end / step) + 1

    def frames(segs):
        out = [set() for _ in range(n)]
        for s in segs:
            for i in range(int(s.start / step), min(n, int(s.end / step) + 1)):
                out[i].add(s.speaker)
        return out

    rf, hf = frames(reference), frames(hypothesis)
    ref_spk = sorted({s.speaker for s in reference}, key=str)
    hyp_spk = sorted({s.speaker for s in hypothesis}, key=str)
    k = max(len(ref_spk), len(hyp_spk))
    padded = ref_spk + [f"_unmapped{i}" for i in range(k - len(ref_spk))]
    best = None
    for perm in permutations(padded, len(hyp_spk)):
        mapping = dict(zip(hyp_spk, perm))
        miss = fa = conf = total = 0
        for r, h in zip(rf, hf):
            hm = {mapping.get(x, x) for x in h}
            total += len(r)
            miss += max(0, len(r) - len(hm))
            fa += max(0, len(hm) - len(r))
            conf += min(len(r), len(hm)) - len(r & hm)
        rate = (miss + fa + conf) / max(total, 1)
        if best is None or rate < best[0]:
            best = (rate, {"mapping": mapping, "miss": miss * step, "false_alarm": fa * step,
                           "confusion": conf * step, "speech": total * step})
    return best if best else (0.0, {})
