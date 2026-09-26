#!/bin/sh
# マイクからのライブ文字起こし。使い方は README。Terminal.app / iTerm から直接実行すること。
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
[ -x tools/micpipe ] || swiftc -O -o tools/micpipe tools/micpipe.swift
[ -t 2 ] || echo "[warn] not attached to a terminal; run from Terminal.app / iTerm directly" >&2

SAVE=""
case " $* " in
  *" --no-save "*) set -- $(printf '%s\n' "$@" | grep -vx -- --no-save) ;;
  *) mkdir -p sessions; TS=$(date +%Y%m%d-%H%M%S)
     SAVE="--save-audio sessions/$TS.wav --jsonl sessions/$TS.jsonl"
     echo "[info] recording to sessions/$TS.{wav,jsonl} (local only; --no-save to disable)" >&2 ;;
esac
exec "$PY" -m livediar --mic $SAVE "$@"
