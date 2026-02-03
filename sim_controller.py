from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ForceConfig:
    magnitude: float = 150.0
    smoothing: float = 0.2
    forward: tuple[float, float, float] = (1.0, 0.0, 0.0)
    left: tuple[float, float, float] = (0.0, 1.0, 0.0)


class SimController:
    def __init__(self, body_id: int, cfg: ForceConfig) -> None:
        self._body_id = body_id
        self._cfg = cfg
        self._force = np.zeros(3, dtype=np.float64)

    def update(self, cmd: str) -> np.ndarray:
        target = np.zeros(3, dtype=np.float64)
        if cmd == "forward":
            target = np.array(self._cfg.forward, dtype=np.float64)
        elif cmd == "left":
            target = np.array(self._cfg.left, dtype=np.float64)
        elif cmd == "right":
            target = -np.array(self._cfg.left, dtype=np.float64)

        if np.linalg.norm(target) > 0:
            target = target / (np.linalg.norm(target) + 1e-6)
            target = target * self._cfg.magnitude

        alpha = np.clip(self._cfg.smoothing, 0.0, 1.0)
        self._force = (1.0 - alpha) * self._force + alpha * target
        return self._force

    def apply(self, mj_data, force: np.ndarray) -> None:
        mj_data.xfrc_applied[:, :] = 0.0
        mj_data.xfrc_applied[self._body_id, :3] = force

