from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class CmdVel:
    vx: float
    vy: float
    yaw_rate: float
    ts: float
    source: str

    def clamp(self, max_vx: float, max_vy: float, max_yaw: float) -> "CmdVel":
        return CmdVel(
            vx=max(-max_vx, min(max_vx, self.vx)),
            vy=max(-max_vy, min(max_vy, self.vy)),
            yaw_rate=max(-max_yaw, min(max_yaw, self.yaw_rate)),
            ts=self.ts,
            source=self.source,
        )


class CmdVelSmoother:
    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = max(0.0, min(1.0, alpha))
        self._last = CmdVel(0.0, 0.0, 0.0, ts=time.time(), source="none")

    def update(self, target: CmdVel) -> CmdVel:
        a = self._alpha
        smoothed = CmdVel(
            vx=(1.0 - a) * self._last.vx + a * target.vx,
            vy=(1.0 - a) * self._last.vy + a * target.vy,
            yaw_rate=(1.0 - a) * self._last.yaw_rate + a * target.yaw_rate,
            ts=target.ts,
            source=target.source,
        )
        self._last = smoothed
        return smoothed

