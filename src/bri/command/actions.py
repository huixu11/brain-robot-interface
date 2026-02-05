from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STOP = "STOP"

    @classmethod
    def from_str(cls, s: str) -> "Action":
        return cls(s.strip().upper())

