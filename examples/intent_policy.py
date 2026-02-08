from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bri import Action, Controller
from thoughtlink.features import eeg_window_features
from thoughtlink.intent_model import IntentModel
from thoughtlink.labels import cue_is_rest, cue_to_action, normalize_cue_label
from thoughtlink.stability import IntentStabilizer, StabilityConfig


def _default_npz_path() -> Path:
    data_dir = ROOT / "robot_control_data" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {data_dir}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="ThoughtLink intent policy (oracle/model) -> bri.Controller (sim).")
    ap.add_argument("--npz", type=str, default=None, help="Path to a dataset .npz chunk (15s).")
    ap.add_argument("--backend", type=str, default="sim", choices=["sim", "robot"], help="bri backend")
    ap.add_argument("--mode", type=str, default="oracle", choices=["oracle", "model"], help="oracle uses labels; model uses a trained decoder")
    ap.add_argument("--model", type=str, default=str(ROOT / "artifacts" / "intent_baseline.npz"), help="Path to IntentModel .npz")
    ap.add_argument("--update-hz", type=float, default=10.0, help="How often to send Action updates")
    ap.add_argument("--hold-s", type=float, default=0.6, help="Controller hold_s (must exceed update interval)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")

    # Stability knobs (defaults match StabilityConfig).
    ap.add_argument("--ewma-alpha", type=float, default=0.2)
    ap.add_argument("--p-move-on", type=float, default=0.6)
    ap.add_argument("--p-move-off", type=float, default=0.4)
    ap.add_argument("--move-on-k", type=int, default=3)
    ap.add_argument("--move-off-k", type=int, default=3)
    ap.add_argument("--p-dir", type=float, default=0.4)
    ap.add_argument("--dir-k", type=int, default=3)
    args = ap.parse_args()

    path = Path(args.npz) if args.npz else _default_npz_path()
    arr = np.load(str(path), allow_pickle=True)
    meta = arr["label"].item()

    raw_label = str(meta.get("label", ""))
    duration_s = float(meta.get("duration", 0.0))
    subject_id = str(meta.get("subject_id", ""))
    session_id = str(meta.get("session_id", ""))

    cue = normalize_cue_label(raw_label)
    move_action = cue_to_action(cue)

    cue_start_s = 3.0
    cue_end_s = min(15.0, cue_start_s + max(0.0, duration_s))

    print(f"[intent_policy] npz={path}")
    print(f"[intent_policy] subject_id={subject_id} session_id={session_id}")
    print(f"[intent_policy] raw_label={raw_label!r} canon={cue.value} action_during_cue={move_action}")
    print(f"[intent_policy] cue_window_s=[{cue_start_s:.3f}, {cue_end_s:.3f}] duration={duration_s:.3f}")
    print(f"[intent_policy] mode={args.mode}")

    model: IntentModel | None = None
    stabilizer: IntentStabilizer | None = None
    include_fft = True
    base_feat: np.ndarray | None = None
    if str(args.mode) == "model":
        model = IntentModel.load_npz(Path(args.model))
        d = int(model.stage1.w.shape[0])
        include_fft = d > 6
        stabilizer = IntentStabilizer(
            StabilityConfig(
                ewma_alpha=float(args.ewma_alpha),
                p_move_on=float(args.p_move_on),
                p_move_off=float(args.p_move_off),
                move_on_k=int(args.move_on_k),
                move_off_k=int(args.move_off_k),
                p_dir=float(args.p_dir),
                dir_k=int(args.dir_k),
            )
        )
        if model.baseline == "pre_cue":
            fs = float(model.fs_hz)
            base_end = int(round((float(model.cue_start_s) - float(model.guard_s)) * fs))
            base_end = max(base_end, int(round(float(model.window_s) * fs)))
            base_end = min(base_end, int(arr["feature_eeg"].shape[0]))
            if base_end > 0:
                base_feat = eeg_window_features(arr["feature_eeg"][:base_end], fs_hz=fs, include_fft=include_fft)
        print(
            f"[intent_policy] loaded model={args.model} feat_dim={d} include_fft={include_fft} baseline={model.baseline!r}"
        )

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
        infer_ms: list[float] = []

        while True:
            now = time.perf_counter()
            t_sim = (now - start) * float(args.speed)
            if t_sim >= 15.0:
                break

            if str(args.mode) == "oracle":
                if cue_start_s <= t_sim <= cue_end_s and not cue_is_rest(cue):
                    action = move_action
                else:
                    action = Action.STOP
            else:
                assert model is not None
                assert stabilizer is not None
                eeg = arr["feature_eeg"]
                fs = float(model.fs_hz)
                win_n = int(round(float(model.window_s) * fs))
                end = int(round(t_sim * fs))
                start_i = end - win_n
                if start_i < 0 or end <= 0 or end > int(eeg.shape[0]):
                    action = Action.STOP
                else:
                    t_inf0 = time.perf_counter()
                    f = eeg_window_features(eeg[start_i:end], fs_hz=fs, include_fft=include_fft)
                    if base_feat is not None and base_feat.shape == f.shape:
                        f = (f - base_feat).astype(np.float32, copy=False)
                    x = f.reshape(1, -1).astype(np.float32)
                    p_move = float(model.predict_move_proba(x)[0])
                    p_dir = model.predict_direction_proba(x)[0]
                    action = stabilizer.step(p_move=p_move, p_dir=p_dir)
                    infer_ms.append((time.perf_counter() - t_inf0) * 1000.0)

            if action != last_action:
                if str(args.mode) == "model" and model is not None and stabilizer is not None and infer_ms:
                    print(f"[intent_policy] t={t_sim:6.3f}s action={action} infer_ms={infer_ms[-1]:.2f}")
                else:
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
        if infer_ms:
            arr_ms = np.asarray(infer_ms, dtype=np.float32)
            p95 = float(np.percentile(arr_ms, 95))
            print(
                f"[intent_policy] inference_ms mean={float(arr_ms.mean()):.2f} p95={p95:.2f} n={int(arr_ms.shape[0])}"
            )
        ctrl.stop()


if __name__ == "__main__":
    main()
