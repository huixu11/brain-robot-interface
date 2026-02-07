from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bri import Action, Controller


def _normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())

    # Common variants/typos observed in the dataset README.
    if "tongue" in s:
        return "TONGUE_TAPPING"
    if "relax" in s:
        return "RELAX"
    if "both" in s and ("fist" in s or "first" in s):
        return "BOTH_FISTS"
    if "left" in s and ("fist" in s or "first" in s):
        return "LEFT_FIST"
    if "right" in s and ("fist" in s or "first" in s):
        return "RIGHT_FIST"

    raise ValueError(f"Unknown label string: {raw!r}")


def _label_to_action(canon_label: str) -> Action:
    # Requirement_doc.md expects left/right/forward/backward intent decoding.
    # The provided dataset has 5 cues; we use Tongue Tapping as a backward proxy.
    if canon_label == "LEFT_FIST":
        return Action.LEFT
    if canon_label == "RIGHT_FIST":
        return Action.RIGHT
    if canon_label == "BOTH_FISTS":
        return Action.FORWARD
    if canon_label == "TONGUE_TAPPING":
        return Action.BACKWARD
    if canon_label == "RELAX":
        return Action.STOP
    raise ValueError(f"Unhandled canonical label: {canon_label}")


def _default_npz_path() -> Path:
    data_dir = ROOT / "robot_control_data" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {data_dir}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="ThoughtLink oracle intent policy -> bri.Controller (sim).")
    ap.add_argument("--npz", type=str, default=None, help="Path to a dataset .npz chunk (15s).")
    ap.add_argument("--backend", type=str, default="sim", choices=["sim", "robot"], help="bri backend")
    ap.add_argument("--update-hz", type=float, default=10.0, help="How often to send Action updates")
    ap.add_argument("--hold-s", type=float, default=0.6, help="Controller hold_s (must exceed update interval)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")
    args = ap.parse_args()

    path = Path(args.npz) if args.npz else _default_npz_path()
    arr = np.load(str(path), allow_pickle=True)
    meta = arr["label"].item()

    raw_label = str(meta.get("label", ""))
    duration_s = float(meta.get("duration", 0.0))
    subject_id = str(meta.get("subject_id", ""))
    session_id = str(meta.get("session_id", ""))

    canon = _normalize_label(raw_label)
    move_action = _label_to_action(canon)

    cue_start_s = 3.0
    cue_end_s = min(15.0, cue_start_s + max(0.0, duration_s))

    print(f"[intent_policy] npz={path}")
    print(f"[intent_policy] subject_id={subject_id} session_id={session_id}")
    print(f"[intent_policy] raw_label={raw_label!r} canon={canon} action_during_cue={move_action}")
    print(f"[intent_policy] cue_window_s=[{cue_start_s:.3f}, {cue_end_s:.3f}] duration={duration_s:.3f}")

    ctrl = Controller(backend=args.backend, hold_s=float(args.hold_s))
    ctrl.start()
    try:
        dt = 0.0 if args.update_hz <= 0 else 1.0 / float(args.update_hz)
        if dt > 0 and args.hold_s <= dt:
            print(
                f"[intent_policy] WARNING: hold_s={args.hold_s} <= update_dt={dt:.3f}. "
                "This may auto-STOP between updates."
            )

        start = time.perf_counter()
        next_tick = start
        last_action: Action | None = None

        while True:
            now = time.perf_counter()
            t_sim = (now - start) * float(args.speed)
            if t_sim >= 15.0:
                break

            if cue_start_s <= t_sim <= cue_end_s and canon != "RELAX":
                action = move_action
            else:
                action = Action.STOP

            if action != last_action:
                print(f"[intent_policy] t={t_sim:6.3f}s action={action}")
                last_action = action

            ctrl.set_action(action)

            if dt <= 0:
                continue
            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

        ctrl.set_action(Action.STOP)
        time.sleep(0.5)
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()

