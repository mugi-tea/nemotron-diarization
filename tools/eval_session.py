#!/usr/bin/env python3
"""Compare ASR backends turn by turn on a recorded session.

    .venv/bin/python tools/eval_session.py sessions/20260926-225911.wav [--prompt "固有名詞、..."] [--all]

Diarization runs once (same settings as `livediar`), then each turn's audio window is transcribed
with every configuration so you can judge on your own recordings which model fits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from livediar.asr.base import ASRBackend  # noqa: E402
from livediar.audio import SAMPLE_RATE  # noqa: E402
from livediar.textfilters import clean_segment  # noqa: E402

CONFIGS = [
    ("qwen3-1.7B", dict(kind="qwen", model="mlx-community/Qwen3-ASR-1.7B-8bit")),
    ("turbo", dict(kind="whisper", model="mlx-community/whisper-large-v3-turbo", context=True)),
]
EXTRA = [
    ("qwen3-0.6B", dict(kind="qwen", model="mlx-community/Qwen3-ASR-0.6B-8bit")),
    ("turbo+no-ctx", dict(kind="whisper", model="mlx-community/whisper-large-v3-turbo", context=False)),
    ("large-v3", dict(kind="whisper", model="mlx-community/whisper-large-v3-mlx", context=True)),
]


def diarize_turns(wav: str) -> list[dict]:
    out = os.path.join(tempfile.mkdtemp(), "turns.jsonl")
    subprocess.run([sys.executable, "-m", "livediar", "--file", wav, "--asr", "none", "--jsonl", out, "--quiet"],
                   check=True, capture_output=True, cwd=ROOT)
    return [json.loads(line) for line in open(out, encoding="utf-8")]


def cut(audio: np.ndarray, turns: list[dict], i: int) -> np.ndarray:
    t = turns[i]
    prev_end = max([u["end"] for u in turns[:i] if u["end"] <= t["start"] + 0.05] + [-1.0])
    next_start = min([u["start"] for u in turns[i + 1 :] if u["start"] >= t["end"] - 0.05] + [1e9])
    lo = max(t["start"] - 0.5, prev_end + 0.15, 0.0)
    hi = min(t["end"] + 0.6, next_start - 0.3)
    if hi <= lo + 0.2:
        lo, hi = max(t["start"] - 0.15, 0.0), t["end"] + 0.15
    return audio[int(lo * SAMPLE_RATE) : min(len(audio), int(hi * SAMPLE_RATE))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--prompt", help="vocabulary hint (hotwords for Qwen, initial prompt for whisper)")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--all", action="store_true", help="also compare Qwen 0.6B, whisper without context and large-v3")
    ap.add_argument("--only", help="comma-separated configuration names")
    args = ap.parse_args()

    configs = CONFIGS + (EXTRA if args.all else [])
    if args.only:
        configs = [c for c in configs if c[0] in args.only.split(",")]
    turns = diarize_turns(args.wav)[: args.max_turns]
    audio = ASRBackend.load_audio(args.wav, normalize_gain=False)
    hotwords = [w.strip("。、 ") for w in (args.prompt or "").replace("、", ",").split(",") if w.strip("。、 ")] or None

    models = {}
    for name, cfg in configs:
        if cfg["kind"] == "qwen":
            from mlx_audio.stt.utils import load_model

            t0 = time.time()
            models[name] = load_model(cfg["model"])
            print(f"loaded {cfg['model']} in {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"{len(turns)} turns, speakers {sorted({t['speaker'] for t in turns})}\n")
    results = {name: [] for name, _ in configs}
    times = {name: 0.0 for name, _ in configs}
    for i, t in enumerate(turns):
        seg = cut(audio, turns, i)
        peak = float(np.max(np.abs(seg))) if seg.size else 0.0
        if 0.0 < peak < 0.3:
            seg = seg * (0.7 / peak)
        for name, cfg in configs:
            t0 = time.time()
            if cfg["kind"] == "qwen":
                import mlx.core as mx

                r = models[name].generate(mx.array(seg), language="ja", hotwords=hotwords)
                text = (getattr(r, "text", None) or (r[0].text if isinstance(r, list) else str(r))).strip()
            else:
                import mlx_whisper

                recent = "".join(results[name][-3:])[-120:] if cfg["context"] else ""
                prompt = " ".join(x for x in (args.prompt, recent) if x) or None
                r = mlx_whisper.transcribe(seg, path_or_hf_repo=cfg["model"], language="ja", fp16=True, temperature=0.0,
                                           condition_on_previous_text=False, initial_prompt=prompt)
                text = "".join(clean_segment(x["text"]) for x in r["segments"] if x.get("no_speech_prob", 0) < 0.6)
            times[name] += time.time() - t0
            results[name].append(text)
        print(f"[{t['start']:6.1f}-{t['end']:6.1f}] Speaker {t['speaker']}")
        for name, _ in configs:
            print(f"    {name:14s} {results[name][-1][:90]}")
    print("\ntotal ASR time:", "  ".join(f"{n}={times[n]:.1f}s" for n, _ in configs))


if __name__ == "__main__":
    main()
