from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thoughtlink.stability import IntentStabilizer, StabilityConfig


def main() -> None:
    cfg = StabilityConfig(
        ewma_alpha=0.25,
        p_move_on=0.65,
        p_move_off=0.45,
        move_on_k=3,
        move_off_k=3,
        p_dir=0.55,
        dir_k=3,
    )
    stab = IntentStabilizer(cfg)

    # Synthetic sequence: REST -> noisy MOVE-LEFT -> brief drop -> MOVE-RIGHT -> REST
    seq = []
    # 20 ticks rest
    for _ in range(20):
        seq.append((0.1, np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)))
    # 30 ticks left with noise
    rng = np.random.default_rng(0)
    for _ in range(30):
        p_move = 0.85 + 0.05 * float(rng.standard_normal())
        p_dir = np.array([0.75, 0.10, 0.10, 0.05], dtype=np.float32)
        p_dir = p_dir + 0.05 * rng.random(4).astype(np.float32)
        seq.append((p_move, p_dir))
    # 5 ticks brief dip (should not immediately drop due to hysteresis)
    for _ in range(5):
        seq.append((0.50, np.array([0.60, 0.15, 0.15, 0.10], dtype=np.float32)))
    # 30 ticks right
    for _ in range(30):
        p_move = 0.85 + 0.05 * float(rng.standard_normal())
        p_dir = np.array([0.10, 0.75, 0.10, 0.05], dtype=np.float32)
        p_dir = p_dir + 0.05 * rng.random(4).astype(np.float32)
        seq.append((p_move, p_dir))
    # 20 ticks rest
    for _ in range(20):
        seq.append((0.1, np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)))

    last = None
    for i, (pm, pd) in enumerate(seq):
        a = stab.step(p_move=float(pm), p_dir=pd)
        if a != last:
            print(f"[stability_smoke] i={i:03d} p_move={pm:.3f} action={a}")
            last = a


if __name__ == "__main__":
    main()

