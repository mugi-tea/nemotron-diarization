"""End-to-end checks on the synthetic 3-speaker meeting (needs NeMo-Speech.cpp + cached models)."""
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import pytest

from livediar.diarizer import Segment
from livediar.metrics import cer, der, load_rttm

from .conftest import requires_diarizer, requires_qwen, requires_whisper

pytestmark = [pytest.mark.integration, requires_diarizer]


def run_cli(wav: str, *extra: str) -> list[dict]:
    out = os.path.join(tempfile.mkdtemp(), "out.jsonl")
    subprocess.run([sys.executable, "-m", "livediar", "--file", wav, "--jsonl", out, "--quiet", *extra],
                   check=True, capture_output=True)
    return [json.loads(line) for line in open(out, encoding="utf-8")]


def reference_text(fixtures_dir: str) -> str:
    lines = open(os.path.join(fixtures_dir, "meeting_ja_3spk.ref.txt"), encoding="utf-8").read().splitlines()[1:]
    return "".join(line.split(None, 3)[3] for line in lines if line.strip())


def test_turns_and_speakers(meeting_wav, fixtures_dir):
    turns = run_cli(meeting_wav, "--asr", "none")
    assert 8 <= len(turns) <= 16
    assert {t["speaker"] for t in turns} == {1, 2, 3}
    hyp = [Segment(t["start"], t["end"], t["speaker"]) for t in turns]
    rate, _ = der(load_rttm(os.path.join(fixtures_dir, "meeting_ja_3spk.ref.rttm")), hyp)
    assert rate < 0.25


@requires_qwen
def test_qwen_transcript(meeting_wav, fixtures_dir):
    turns = run_cli(meeting_wav, "--asr", "qwen")
    text = "".join(t["text"] for t in turns)
    assert cer(reference_text(fixtures_dir), text) < 0.10
    for phrase in ("認証", "結合テスト", "木曜日", "承知しました"):
        assert phrase in text


@requires_whisper
def test_whisper_transcript(meeting_wav, fixtures_dir):
    turns = run_cli(meeting_wav, "--asr", "whisper")
    text = "".join(t["text"] for t in turns)
    assert cer(reference_text(fixtures_dir), text) < 0.08
    for phrase in ("進捗", "認証", "フロントエンド", "木曜日"):
        assert phrase in text


def test_streaming_output_is_not_stalled(meeting_wav):
    """Lines must appear while audio is still playing (regression for the frontier stall)."""
    cut = os.path.join(tempfile.mkdtemp(), "cut.wav")
    with wave.open(meeting_wav, "rb") as r, wave.open(cut, "wb") as w:
        w.setparams(r.getparams())
        w.writeframes(r.readframes(16000 * 30))
    proc = subprocess.Popen([sys.executable, "-m", "livediar", "--file", cut, "--realtime", "--asr", "none", "--quiet"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t0 = time.time()
    stamps = [time.time() - t0 for line in proc.stdout if line.startswith("[")]
    proc.wait()
    assert len(stamps) >= 4
    assert sum(1 for t in stamps if t < 24.0) >= 2, stamps
