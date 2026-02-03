from __future__ import annotations

from dataclasses import dataclass
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from ..command.cmd_vel import CmdVel


@dataclass
class SportRpcConfig:
    domain_id: int = 0
    interface: str = "en0"
    cmd_hz: float = 10.0
    cmd_timeout_s: float = 0.6


class UnitreeSportRpc:
    def __init__(self, cfg: SportRpcConfig) -> None:
        self.cfg = cfg
        self._client = LocoClient()
        self._last_cmd = CmdVel(0.0, 0.0, 0.0, ts=time.time(), source="none")
        self._last_sent = 0.0

    def start(self) -> None:
        ChannelFactoryInitialize(self.cfg.domain_id, self.cfg.interface)
        self._client.Init()
        self._client.Start()

    def set_cmd(self, cmd: CmdVel) -> None:
        self._last_cmd = cmd

    def step(self) -> None:
        now = time.time()
        interval = 0.0 if self.cfg.cmd_hz <= 0 else 1.0 / self.cfg.cmd_hz
        if now - self._last_sent < interval:
            return
        self._last_sent = now

        if now - self._last_cmd.ts > self.cfg.cmd_timeout_s:
            self._client.StopMove()
            return

        self._client.Move(
            self._last_cmd.vx,
            self._last_cmd.vy,
            self._last_cmd.yaw_rate,
            continous_move=True,
        )

