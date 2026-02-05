from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from ..command.cmd_vel import CmdVel
from .mirror import MirrorBackend


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


class RobotBackend:
    def __init__(
        self,
        *,
        interface: str = "en0",
        domain_id: int = 0,
        cmd_hz: float = 10.0,
        hold_s: float = 0.3,
        mirror: bool = True,
        mirror_bundle_dir: str | None = None,
        robot_type: str = "g1",
        ctrl_hz: float = 50.0,
    ) -> None:
        self._robot = UnitreeSportRpc(
            SportRpcConfig(
                domain_id=domain_id,
                interface=interface,
                cmd_hz=cmd_hz,
                cmd_timeout_s=hold_s,
            )
        )
        self._mirror_enabled = mirror
        self._mirror = None
        if mirror:
            bundle = mirror_bundle_dir or str(ROOT / "bundles" / "g1_mjlab")
            self._mirror = MirrorBackend(
                bundle_dir=bundle,
                interface=interface,
                domain_id=domain_id,
                robot_type=robot_type,
            )
        self._ctrl_hz = ctrl_hz

    def start(self) -> None:
        self._robot.start()
        if self._mirror:
            self._mirror.start()

    def is_running(self) -> bool:
        if self._mirror_enabled and self._mirror:
            return self._mirror.is_running()
        return True

    def step(self, cmd: CmdVel) -> None:
        self._robot.set_cmd(cmd)
        self._robot.step()
        if self._mirror:
            self._mirror.step()

    def stop(self) -> None:
        if self._mirror:
            self._mirror.stop()

