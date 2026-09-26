"""Command-line entry point: ``livediar --mic`` or ``livediar --file recording.wav``."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading

from . import __version__
from .asr import BACKENDS, create_backend
from .audio import FileSource, MicSource
from .diarizer import DiarizerUnavailable, NemoDiarizer
from .output import ConsoleWriter, JsonlWriter
from .pipeline import Pipeline
from .turns import TurnConfig, TurnTracker


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="livediar",
        description="Real-time 'who said what' for Japanese: Nemotron 3 diarization + per-turn ASR (Apple Silicon).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"livediar {__version__}")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="capture the default microphone; Ctrl-C to stop")
    src.add_argument("--file", metavar="WAV", help="read a 16 kHz mono WAV instead of the microphone")
    p.add_argument("--realtime", action="store_true", help="pace --file at real time, for latency checks")
    p.add_argument("--save-audio", metavar="WAV", help="also record the captured audio to this file")
    p.add_argument("--jsonl", metavar="PATH", help="append results as JSON Lines")
    p.add_argument("--quiet", action="store_true", help="omit per-turn ASR latency from console output")

    a = p.add_argument_group("transcription")
    a.add_argument("--asr", default="auto", choices=("auto", *BACKENDS, "mlx"),
                   help="auto = qwen > whisper > nemo, whichever is installed")
    a.add_argument("--language", help="language code: ja for qwen and whisper, ja-JP for nemo")
    a.add_argument("--prompt", help="comma-separated vocabulary hint: hotwords for qwen, initial prompt for whisper")
    a.add_argument("--qwen-model", default="mlx-community/Qwen3-ASR-1.7B-8bit", help="or mlx-community/Qwen3-ASR-0.6B-8bit")
    a.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-turbo", help="or mlx-community/whisper-large-v3-mlx")
    a.add_argument("--no-context", action="store_true", help="whisper: do not feed previous turns as context")
    a.add_argument("--whisper-cli-model", help="ggml model path for --asr whisper-cli")
    a.add_argument("--nemo-url", help="URL of an already running `nemo-speech serve`")
    a.add_argument("--nemo-port", type=int, default=8090, help="port for the auto-started nemo-speech server")

    d = p.add_argument_group("diarization")
    d.add_argument("--diar-preset", help="v3-streaming or v3-offline; default is the model low-latency preset")
    d.add_argument("--diar-chunk", type=int, default=24, help="chunk length in 80 ms frames; 24 = 1.9 s; 0 keeps the preset")
    d.add_argument("--diar-rc", type=int, default=8, help="right context in 80 ms frames; 0 keeps the preset")
    d.add_argument("--onset", type=float, default=0.55, help="speaker activation threshold")
    d.add_argument("--offset", type=float, default=0.45, help="speaker deactivation threshold")
    d.add_argument("--pad", type=float, default=0.1, help="segment padding in seconds")
    d.add_argument("--min-gap", type=float, default=0.6, help="merge same-speaker gaps shorter than this")
    d.add_argument("--fast", action="store_true", help="low-latency mode: preset chunking and hold 0.8 s; slightly worse at turn changes")

    t = p.add_argument_group("turns")
    t.add_argument("--hold", type=float, default=1.2,
                   help="seconds a segment must stop growing to be final; raised to at least min-gap + 0.4 and post-extend + 0.2, so 2.2 s by default")
    t.add_argument("--min-turn", type=float, default=0.8, help="drop segments shorter than this")
    t.add_argument("--max-turn", type=float, default=15.0, help="cut long monologues into pieces of this length")
    t.add_argument("--pre-extend", type=float, default=1.5, help="max audio added before a turn for ASR")
    t.add_argument("--post-extend", type=float, default=2.0, help="max audio added after a turn for ASR")
    t.add_argument("--no-dedupe", action="store_true", help="keep near-duplicate lines from overlapping detections")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fast:
        args.diar_chunk = args.diar_rc = 0
        args.hold = min(args.hold, 0.8)
        args.min_gap = min(args.min_gap, 0.5)
    hold = max(args.hold, args.min_gap + 0.4, args.post_extend + 0.2)

    try:
        diarizer = NemoDiarizer(preset=args.diar_preset, chunk_frames=args.diar_chunk, right_context_frames=args.diar_rc)
    except DiarizerUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    diarizer.set_segmentation(onset=args.onset, offset=args.offset, pad_onset=args.pad, pad_offset=args.pad, min_gap=args.min_gap)
    try:
        asr = create_backend(
            args.asr, language=args.language, prompt=args.prompt, qwen_model=args.qwen_model,
            whisper_model=args.whisper_model, whisper_context_chars=0 if args.no_context else 120,
            whisper_cli_model=args.whisper_cli_model, nemo_url=args.nemo_url, nemo_port=args.nemo_port,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        diarizer.close()
        return 2
    tracker = TurnTracker(diarizer, TurnConfig(hold=hold, min_turn=args.min_turn, max_turn=args.max_turn,
                                               pre_extend=args.pre_extend, post_extend=args.post_extend))
    console = ConsoleWriter(show_latency=not args.quiet)
    jsonl = JsonlWriter(args.jsonl) if args.jsonl else None

    def sink(r):
        console(r)
        if jsonl:
            jsonl(r)

    pipeline = Pipeline(diarizer, asr, tracker, sink, min_turn=args.min_turn, dedupe=not args.no_dedupe)
    source = MicSource() if args.mic else FileSource(args.file, realtime=args.realtime)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print(f"diarizer: {os.path.basename(diarizer.model_path)} ({diarizer.num_speakers} spk, "
          f"{diarizer.seconds_per_frame * 1000:.0f} ms)  asr: {asr.name}" + ("  [mic] speak; Ctrl-C to stop" if args.mic else ""),
          file=sys.stderr, flush=True)
    try:
        pipeline.run(source, save_path=args.save_audio, stop=stop)
    finally:
        diarizer.close()
        if jsonl:
            jsonl.close()
        if args.save_audio:
            print(f"audio saved: {args.save_audio}", file=sys.stderr)
    return 0
