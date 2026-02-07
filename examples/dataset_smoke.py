from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thoughtlink.data import WindowConfig, iter_eeg_windows, load_chunk
from thoughtlink.features import eeg_window_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test: load dataset chunk -> window -> features.")
    ap.add_argument("--npz", type=str, default=None, help="Path to a dataset .npz chunk.")
    ap.add_argument("--max-windows", type=int, default=5, help="How many windows to print.")
    args = ap.parse_args()

    if args.npz:
        path = Path(args.npz)
    else:
        data_dir = ROOT / "robot_control_data" / "data"
        path = sorted(data_dir.glob("*.npz"))[0]

    chunk = load_chunk(path)
    cfg = WindowConfig()
    print(f"[smoke] npz={chunk.path.name} cue={chunk.meta.cue.value} duration_s={chunk.meta.duration_s:.3f}")
    print(f"[smoke] eeg_shape={chunk.eeg.shape} fs_hz={cfg.fs_hz} window_s={cfg.window_s} hop_s={cfg.hop_s}")

    for i, w in enumerate(iter_eeg_windows(chunk, cfg)):
        feat = eeg_window_features(w.x, fs_hz=cfg.fs_hz)
        print(
            f"[smoke] i={i} t=[{w.t0_s:.3f},{w.t1_s:.3f}] is_move={w.is_move} "
            f"feat_shape={feat.shape} feat_mean={float(np.mean(feat)):.6f}"
        )
        if i + 1 >= args.max_windows:
            break


if __name__ == "__main__":
    main()

