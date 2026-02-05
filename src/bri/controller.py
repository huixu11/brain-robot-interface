from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .command.action_controller import ActionConfig, ActionController
from .command.actions import Action


@dataclass
class Controller:
    backend: str = "sim"
    hold_s: float = 0.3
    forward_speed: float = 0.6
    yaw_rate: float = 1.5
    smooth_alpha: float = 0.2
    ctrl_hz: float = 50.0
    bundle_dir: str | None = None
    interface: str = "en0"
    domain_id: int = 0
    cmd_hz: float = 10.0
    mirror: bool = True
    robot_type: str = "g1"

    def __post_init__(self) -> None:
        cfg = ActionConfig(
            forward_speed=self.forward_speed,
            yaw_rate=self.yaw_rate,
            hold_s=self.hold_s,
            smooth_alpha=self.smooth_alpha,
        )
        self._action = ActionController(cfg)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend = self._build_backend()

    def _build_backend(self):
        backend = self.backend.lower().strip()
        if backend == "sim":
            from .backends.sim import SimBackend

            return SimBackend(
                bundle_dir=self.bundle_dir,
                ctrl_hz=self.ctrl_hz,
            )
        if backend == "robot":
            from .backends.robot import RobotBackend

            return RobotBackend(
                interface=self.interface,
                domain_id=self.domain_id,
                cmd_hz=self.cmd_hz,
                hold_s=self.hold_s,
                mirror=self.mirror,
                mirror_bundle_dir=self.bundle_dir,
                robot_type=self.robot_type,
                ctrl_hz=self.ctrl_hz,
            )
        raise ValueError("backend must be 'sim' or 'robot'")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._backend.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = 0.0 if self.ctrl_hz <= 0 else 1.0 / self.ctrl_hz
        while not self._stop.is_set() and self._backend.is_running():
            start = time.perf_counter()
            cmd = self._action.get_cmd_vel()
            self._backend.step(cmd)
            elapsed = time.perf_counter() - start
            sleep_s = interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._backend.stop()

    def set_action(self, action: Action) -> None:
        self._action.set_action(action)

