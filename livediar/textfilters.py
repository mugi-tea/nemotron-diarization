"""Text-level guards against ASR failure modes (runaway repetition, hallucinated boilerplate)."""
from __future__ import annotations

import re
import zlib

# Boilerplate whisper tends to hallucinate on silence or noise in Japanese content.
HALLUCINATIONS = ("ご視聴ありがとうございました", "チャンネル登録", "字幕", "おやすみなさい。", "by H.", "Thank you.")

_REPEAT = re.compile(r"((.{2,10}?))\2{3,}")


def collapse_repeats(text: str) -> str:
    """Fold a 2-10 character pattern repeated 4+ times in a row down to two repetitions."""
    return _REPEAT.sub(lambda m: m.group(1) * 2, text)


def compression_ratio(text: str) -> float:
    """UTF-8 length divided by zlib-compressed length; high values mean repetitive text."""
    b = text.encode("utf-8")
    return len(b) / max(len(zlib.compress(b)), 1)


def is_hallucination(text: str) -> bool:
    t = text.strip()
    return len(t) <= 20 and any(h in t for h in HALLUCINATIONS)


def plausible_length(text: str, duration_sec: float, max_chars_per_sec: float = 12.0, slack: int = 12) -> bool:
    """Japanese speech is at most ~9 characters per second; far more means the decoder ran away."""
    return len(text) <= max_chars_per_sec * duration_sec + slack


def clean_segment(raw: str, max_compression_ratio: float = 2.4) -> str:
    """Collapse repeats and reject a segment that was mostly repetition or is still repetitive."""
    text = collapse_repeats(raw.strip())
    if len(raw.strip()) > 2.5 * len(text) + 5:
        return ""
    if text and compression_ratio(text) > max_compression_ratio:
        return ""
    return text


def finalize_text(text: str, duration_sec: float) -> str:
    """Final sanity gate applied to a whole turn's transcript."""
    if not text or is_hallucination(text) or not plausible_length(text, duration_sec):
        return ""
    return text


def trigram_similarity(a: str, b: str) -> float:
    """Fraction of the shorter string's character trigrams that occur in the longer one."""
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    grams = {short[i : i + 3] for i in range(max(len(short) - 2, 1))}
    return sum(1 for g in grams if g in long_) / max(len(grams), 1)
