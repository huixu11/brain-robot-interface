from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .actions import Action
from .cmd_vel import CmdVel, CmdVelSmoother


@dataclass
class ActionConfig:
    forward_speed: float = 0.6
    yaw_rate: float = 1.5
    hold_s: float = 0.3
    smooth_alpha: float = 0.2


class ActionController:
    def __init__(self, cfg: ActionConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._last_action = Action.STOP
        self._last_ts = time.time()
        self._smoother = CmdVelSmoother(alpha=cfg.smooth_alpha)
        self._last_cmd = CmdVel(0.0, 0.0, 0.0, ts=self._last_ts, source="policy")

    def set_action(self, action: Action, *, ts: float | None = None) -> None:
        with self._lock:
            self._last_action = action
            self._last_ts = ts if ts is not None else time.time()

    def get_cmd_vel(self, now: float | None = None) -> CmdVel:
        now = now if now is not None else time.time()
        with self._lock:
            action = self._last_action
            age = now - self._last_ts

        if age > self._cfg.hold_s:
            action = Action.STOP

        if action == Action.FORWARD:
            cmd = CmdVel(self._cfg.forward_speed, 0.0, 0.0, ts=now, source="policy")
        elif action == Action.LEFT:
            cmd = CmdVel(0.0, 0.0, self._cfg.yaw_rate, ts=now, source="policy")
        elif action == Action.RIGHT:
            cmd = CmdVel(0.0, 0.0, -self._cfg.yaw_rate, ts=now, source="policy")
        else:
            cmd = CmdVel(0.0, 0.0, 0.0, ts=now, source="policy")

        cmd = self._smoother.update(cmd)
        with self._lock:
            self._last_cmd = cmd
        return cmd

    def latest_cmd(self) -> CmdVel:
        with self._lock:
            return self._last_cmd

    @property
    def hold_s(self) -> float:
        return self._cfg.hold_s

