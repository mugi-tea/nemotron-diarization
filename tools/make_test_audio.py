#!/usr/bin/env python3
"""Synthesize a 3-speaker Japanese meeting with macOS `say` and write it plus reference labels
to tests/fixtures/. No ffmpeg needed; standard library only."""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tests", "fixtures")
SR = 16000
GAP_SEC = 0.6

VOICES = {"A": "Kyoko", "B": "Reed (Japanese (Japan))", "C": "Sandy (Japanese (Japan))"}
TURNS = [
    ("A", "お疲れ様です。今日は来週のリリース計画について話しましょう。"),
    ("B", "はい、お願いします。まずバックエンドの進捗から共有しますね。"),
    ("A", "ありがとうございます。APIの変更点は仕様書に反映済みですか。"),
    ("B", "反映済みです。ただ、認証まわりのテストがまだ残っています。"),
    ("C", "すみません、遅れました。フロントエンド側は画面の実装が終わっています。"),
    ("A", "了解です。では、金曜日までに結合テストを終わらせましょう。"),
    ("C", "承知しました。テスト環境の準備は私がやります。"),
    ("B", "では、木曜日の午後にレビューの時間を取りましょう。"),
    ("A", "いいですね。他に共有事項はありますか。"),
    ("C", "特にありません。ありがとうございました。"),
]


def synth(voice: str, text: str, work: str, idx: int) -> bytes:
    aiff = os.path.join(work, f"turn_{idx:02d}.aiff")
    wav = os.path.join(work, f"turn_{idx:02d}.wav")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", aiff, wav], check=True)
    with wave.open(wav, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)
        return w.readframes(w.getnframes())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    work = tempfile.mkdtemp(prefix="make_test_audio_")
    frames = bytearray()
    silence = b"\x00\x00" * int(SR * GAP_SEC)
    segments = []
    t = 0.0
    for i, (spk, text) in enumerate(TURNS):
        data = synth(VOICES[spk], text, work, i)
        dur = len(data) / 2 / SR
        segments.append((spk, t, t + dur, text))
        frames += data + silence
        t += dur + GAP_SEC
    name = "meeting_ja_3spk"
    with wave.open(os.path.join(OUT_DIR, f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    with open(os.path.join(OUT_DIR, f"{name}.ref.rttm"), "w", encoding="utf-8") as f:
        for spk, s, e, _ in segments:
            f.write(f"SPEAKER {name} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    with open(os.path.join(OUT_DIR, f"{name}.ref.txt"), "w", encoding="utf-8") as f:
        f.write(f"{'start':>7} {'end':>7}  spk  text\n")
        for spk, s, e, text in segments:
            f.write(f"{s:7.2f} {e:7.2f}  {spk}    {text}\n")
    print(f"wrote {OUT_DIR}/{name}.wav ({t:.1f}s, {len(VOICES)} speakers, {len(TURNS)} turns)")


if __name__ == "__main__":
    main()
