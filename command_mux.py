from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional


@dataclass
class CommandSample:
    cmd: str
    source: str
    ts: float


class CommandMux:
    def __init__(self, key_timeout_s: float = 0.35) -> None:
        self._lock = threading.Lock()
        self._key_timeout_s = key_timeout_s
        self._keyboard: Optional[CommandSample] = None
        self._cv: Optional[CommandSample] = None

    def update_keyboard(self, cmd: str) -> None:
        now = time.time()
        with self._lock:
            self._keyboard = CommandSample(cmd=cmd, source="keyboard", ts=now)

    def update_cv(self, cmd: str, ts: Optional[float] = None) -> None:
        now = ts if ts is not None else time.time()
        with self._lock:
            self._cv = CommandSample(cmd=cmd, source="cv", ts=now)

    def sample(self) -> CommandSample:
        now = time.time()
        with self._lock:
            if self._keyboard and (now - self._keyboard.ts) <= self._key_timeout_s:
                return self._keyboard
            if self._cv:
                return self._cv
        return CommandSample(cmd="stop", source="none", ts=now)

