from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

from .cmd_vel import CmdVel


@dataclass
class CommandSample:
    cmd: CmdVel


class CommandMux:
    def __init__(self, key_timeout_s: float = 0.35) -> None:
        self._lock = threading.Lock()
        self._key_timeout_s = key_timeout_s
        self._keyboard: Optional[CmdVel] = None
        self._cv: Optional[CmdVel] = None

    def update_keyboard(self, cmd: CmdVel) -> None:
        with self._lock:
            self._keyboard = cmd

    def update_cv(self, cmd: CmdVel) -> None:
        with self._lock:
            self._cv = cmd

    def sample(self) -> CmdVel:
        now = time.time()
        with self._lock:
            if self._keyboard and (now - self._keyboard.ts) <= self._key_timeout_s:
                return self._keyboard
            if self._cv:
                return self._cv
        return CmdVel(0.0, 0.0, 0.0, ts=now, source="none")

