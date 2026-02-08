from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bri import Action


@dataclass(frozen=True)
class StabilityConfig:
    """Explicit temporal stability controls (per Requirement_doc.md).

    This is intentionally simple and fast: EWMA smoothing + hysteresis thresholds + debouncing.
    """

    ewma_alpha: float = 0.2
    p_move_on: float = 0.6
    p_move_off: float = 0.4
    move_on_k: int = 3
    move_off_k: int = 3
    p_dir: float = 0.5
    dir_k: int = 3


class IntentStabilizer:
    """Convert per-tick model probabilities into stable discrete Actions."""

    def __init__(self, cfg: StabilityConfig | None = None) -> None:
        self.cfg = cfg or StabilityConfig()

        self._p_move_ema: float | None = None
        self._p_dir_ema: np.ndarray | None = None  # (4,)

        self._in_move: bool = False
        self._on_count: int = 0
        self._off_count: int = 0

        self._dir_candidate: int | None = None
        self._dir_count: int = 0
        self._dir_current: int | None = None

    @staticmethod
    def _idx_to_action(idx: int) -> Action:
        if idx == 0:
            return Action.LEFT
        if idx == 1:
            return Action.RIGHT
        if idx == 2:
            return Action.FORWARD
        if idx == 3:
            return Action.BACKWARD
        raise ValueError(f"Bad direction idx: {idx}")

    def step(self, *, p_move: float, p_dir: np.ndarray | None = None) -> Action:
        """Advance the stabilizer by one tick.

        p_move: scalar probability in [0,1]
        p_dir: (4,) probability over directions (LEFT, RIGHT, FORWARD, BACKWARD)
        """

        # Sanitize.
        if not np.isfinite(p_move):
            p_move = 0.0
        p_move = float(np.clip(p_move, 0.0, 1.0))

        if p_dir is not None:
            p_dir = np.asarray(p_dir, dtype=np.float32).reshape(-1)
            if p_dir.shape[0] != 4:
                raise ValueError(f"Expected p_dir shape (4,), got {p_dir.shape}")
            if not np.all(np.isfinite(p_dir)):
                p_dir = np.zeros((4,), dtype=np.float32)
            s = float(p_dir.sum())
            p_dir = p_dir / (s if s > 0 else 1.0)

        # EWMA smoothing.
        a = float(self.cfg.ewma_alpha)
        if self._p_move_ema is None:
            self._p_move_ema = p_move
        else:
            self._p_move_ema = a * p_move + (1.0 - a) * float(self._p_move_ema)

        if p_dir is not None:
            if self._p_dir_ema is None:
                self._p_dir_ema = p_dir.astype(np.float32, copy=True)
            else:
                self._p_dir_ema = (a * p_dir + (1.0 - a) * self._p_dir_ema).astype(np.float32, copy=False)

        # Hysteresis + debounce for move/rest state.
        if not self._in_move:
            self._off_count = 0
            if float(self._p_move_ema) >= float(self.cfg.p_move_on):
                self._on_count += 1
            else:
                self._on_count = 0
            if self._on_count >= int(self.cfg.move_on_k):
                self._in_move = True
                self._on_count = 0
        else:
            self._on_count = 0
            if float(self._p_move_ema) <= float(self.cfg.p_move_off):
                self._off_count += 1
            else:
                self._off_count = 0
            if self._off_count >= int(self.cfg.move_off_k):
                self._in_move = False
                self._off_count = 0
                # Reset direction debounce when leaving MOVE.
                self._dir_candidate = None
                self._dir_count = 0
                self._dir_current = None

        if not self._in_move:
            return Action.STOP

        # Direction debounce (only when in MOVE and p_dir available).
        if self._p_dir_ema is not None:
            idx = int(np.argmax(self._p_dir_ema))
            conf = float(self._p_dir_ema[idx])
            if conf >= float(self.cfg.p_dir):
                if self._dir_candidate == idx:
                    self._dir_count += 1
                else:
                    self._dir_candidate = idx
                    self._dir_count = 1
                if self._dir_count >= int(self.cfg.dir_k):
                    self._dir_current = idx
            else:
                # Low confidence: do not switch.
                self._dir_candidate = None
                self._dir_count = 0

        # If direction isn't stabilized yet, be conservative and STOP.
        if self._dir_current is None:
            return Action.STOP
        return self._idx_to_action(self._dir_current)

