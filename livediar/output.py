"""Result sinks: colored console lines and JSON Lines."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

COLORS = ["\x1b[1;36m", "\x1b[1;32m", "\x1b[1;35m", "\x1b[1;33m", "\x1b[1;34m", "\x1b[1;31m", "\x1b[1;37m", "\x1b[1;90m"]
RESET = "\x1b[0m"


@dataclass
class TurnResult:
    start: float
    end: float
    speaker: int
    text: str
    latency: float = 0.0  # ASR wall-clock seconds


def format_time(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


class ConsoleWriter:
    def __init__(self, show_latency: bool = True, stream=None):
        self.stream = stream or sys.stdout
        self.show_latency = show_latency
        self.color = self.stream.isatty() and not os.environ.get("NO_COLOR")

    def __call__(self, r: TurnResult) -> None:
        c = COLORS[(r.speaker - 1) % len(COLORS)] if self.color else ""
        reset = RESET if self.color else ""
        tail = f"   ({r.latency * 1000:.0f}ms)" if self.show_latency else ""
        print(f"[{format_time(r.start)}-{format_time(r.end)}] {c}Speaker {r.speaker}{reset}  {r.text}{tail}",
              file=self.stream, flush=True)


class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", encoding="utf-8")

    def __call__(self, r: TurnResult) -> None:
        d = asdict(r)
        d.pop("latency", None)
        d["start"], d["end"] = round(r.start, 2), round(r.end, 2)
        self._f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
