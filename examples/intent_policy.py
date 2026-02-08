from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

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


class TelemetryOverlay:
    """Minimal OpenCV telemetry window for demo recording.

    Shows: p_move(t), cue window, thresholds, and current discrete action.
    """

    def __init__(self, *, title: str = "ThoughtLink Telemetry", width: int = 720, height: int = 220) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is not available. Install opencv-python or disable --telemetry.")
        self._title = str(title)
        self._w = int(width)
        self._h = int(height)
        self._t: list[float] = []
        self._p: list[float] = []
        self._last_render_ts = 0.0

    def _pt(self, *, t: float, p: float, x0: int, y0: int, w: int, h: int) -> tuple[int, int]:
        t = float(np.clip(t, 0.0, 15.0))
        p = float(np.clip(p, 0.0, 1.0))
        x = int(round(x0 + (t / 15.0) * (w - 1)))
        y = int(round(y0 + (1.0 - p) * (h - 1)))
        return x, y

    def update(
        self,
        *,
        t_sim: float,
        p_move: float,
        action: Action,
        cue_start_s: float,
        cue_end_s: float,
        cfg: StabilityConfig,
        p_move_raw: float | None = None,
        p_dir_top: tuple[str, float] | None = None,
    ) -> None:
        # Collect history for 15s chunk.
        self._t.append(float(t_sim))
        self._p.append(float(p_move))

        # Render every tick. (We keep this cheap; it is fine at 50Hz.)
        bg = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        bg[:] = (18, 18, 18)

        # Plot area.
        pad_l, pad_r, pad_t, pad_b = 60, 10, 10, 40
        x0, y0 = pad_l, pad_t
        pw = self._w - pad_l - pad_r
        ph = self._h - pad_t - pad_b

        # Cue window shading.
        cx0, _ = self._pt(t=float(cue_start_s), p=0.0, x0=x0, y0=y0, w=pw, h=ph)
        cx1, _ = self._pt(t=float(cue_end_s), p=0.0, x0=x0, y0=y0, w=pw, h=ph)
        cx0, cx1 = min(cx0, cx1), max(cx0, cx1)
        if cx1 > cx0:
            bg[y0 : y0 + ph, cx0:cx1] = (28, 24, 50)

        # Axes box.
        cv2.rectangle(bg, (x0, y0), (x0 + pw, y0 + ph), (80, 80, 80), 1)

        # Threshold lines.
        for thr, color, label in [
            (cfg.p_move_on, (60, 140, 255), "on"),
            (cfg.p_move_off, (255, 120, 60), "off"),
        ]:
            _, y = self._pt(t=0.0, p=float(thr), x0=x0, y0=y0, w=pw, h=ph)
            cv2.line(bg, (x0, y), (x0 + pw, y), color, 1)
            cv2.putText(bg, f"{label}={thr:.2f}", (x0 + pw - 120, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # p_move polyline.
        pts: list[tuple[int, int]] = []
        for tt, pp in zip(self._t, self._p, strict=False):
            pts.append(self._pt(t=tt, p=pp, x0=x0, y0=y0, w=pw, h=ph))
        if len(pts) >= 2:
            cv2.polylines(bg, [np.asarray(pts, dtype=np.int32)], False, (80, 220, 120), 2)

        # Current time marker.
        x_now, _ = self._pt(t=float(t_sim), p=0.0, x0=x0, y0=y0, w=pw, h=ph)
        cv2.line(bg, (x_now, y0), (x_now, y0 + ph), (200, 200, 200), 1)

        # Labels.
        cv2.putText(bg, "p_move", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(bg, "0.0", (18, y0 + ph + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(bg, "15s", (x0 + pw - 26, y0 + ph + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(bg, "1.0", (18, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        # Footer text (current values).
        footer_y = self._h - 14
        p_raw_s = "" if p_move_raw is None else f" raw={p_move_raw:.2f}"
        dir_s = ""
        if p_dir_top is not None:
            dir_s = f" dir={p_dir_top[0]}({p_dir_top[1]:.2f})"
        cv2.putText(
            bg,
            f"t={t_sim:5.2f}s  action={action.name:<8}  p_move={p_move:.2f}{p_raw_s}{dir_s}",
            (10, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
        )

        cv2.imshow(self._title, bg)
        cv2.waitKey(1)

    def close(self) -> None:
        if cv2 is None:
            return
        try:
            cv2.destroyWindow(self._title)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="ThoughtLink intent policy (oracle/model) -> bri.Controller (sim).")
    ap.add_argument("--npz", type=str, default=None, help="Path to a dataset .npz chunk (15s).")
    ap.add_argument("--backend", type=str, default="sim", choices=["sim", "robot"], help="bri backend")
    ap.add_argument("--mode", type=str, default="oracle", choices=["oracle", "model"], help="oracle uses labels; model uses a trained decoder")
    ap.add_argument("--model", type=str, default=str(ROOT / "artifacts" / "intent_baseline.npz"), help="Path to IntentModel .npz")
    ap.add_argument("--update-hz", type=float, default=10.0, help="How often to send Action updates")
    ap.add_argument("--hold-s", type=float, default=0.6, help="Controller hold_s (must exceed update interval)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")
    ap.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show an OpenCV window with p_move, action, and cue window (useful for split-screen demo recording).",
    )
    ap.add_argument("--telemetry-w", type=int, default=720, help="Telemetry window width (px)")
    ap.add_argument("--telemetry-h", type=int, default=220, help="Telemetry window height (px)")

    # Stability knobs (defaults match StabilityConfig).
    ap.add_argument("--ewma-alpha", type=float, default=0.2)
    ap.add_argument("--p-move-on", type=float, default=0.6)
    ap.add_argument("--p-move-off", type=float, default=0.4)
    ap.add_argument("--move-on-k", type=int, default=3)
    ap.add_argument("--move-off-k", type=int, default=3)
    ap.add_argument("--p-dir", type=float, default=0.4)
    ap.add_argument("--p-dir-off", type=float, default=0.35)
    ap.add_argument("--dir-k", type=int, default=5)
    ap.add_argument("--dir-off-k", type=int, default=2)
    ap.add_argument("--dir-margin", type=float, default=0.06)
    ap.add_argument(
        "--stop-on-dir-uncertain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, allow direction uncertainty to emit STOP even while in MOVE. If false, hold last direction.",
    )
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

    def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    model: IntentModel | None = None
    stabilizer: IntentStabilizer | None = None
    include_fft = True
    feature_mode = "raw"
    feature_mode_move = "raw"
    feature_mode_dir = "raw"
    base_feat: np.ndarray | None = None
    if str(args.mode) == "model":
        model = IntentModel.load_npz(Path(args.model))
        d = int(model.stage1.w.shape[0])
        include_fft = bool(model.include_fft)
        feature_mode = str(model.feature_mode)
        feature_mode_move = str(model.feature_mode_move or model.feature_mode)
        feature_mode_dir = str(model.feature_mode_dir or model.feature_mode)
        stabilizer = IntentStabilizer(
            StabilityConfig(
                ewma_alpha=float(args.ewma_alpha),
                p_move_on=float(args.p_move_on),
                p_move_off=float(args.p_move_off),
                move_on_k=int(args.move_on_k),
                move_off_k=int(args.move_off_k),
                p_dir=float(args.p_dir),
                p_dir_off=float(args.p_dir_off),
                dir_k=int(args.dir_k),
                dir_off_k=int(args.dir_off_k),
                dir_margin=float(args.dir_margin),
                stop_on_dir_uncertain=bool(args.stop_on_dir_uncertain),
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
            f"[intent_policy] loaded model={args.model} feat_dim={d} include_fft={include_fft} "
            f"baseline={model.baseline!r} feature_mode={feature_mode!r} "
            f"feature_mode_move={feature_mode_move!r} feature_mode_dir={feature_mode_dir!r}"
        )

    overlay: TelemetryOverlay | None = None
    if bool(args.telemetry):
        if cv2 is None:
            print("[intent_policy] WARNING: --telemetry requested but opencv-python is not available; disabling telemetry.")
        else:
            overlay = TelemetryOverlay(width=int(args.telemetry_w), height=int(args.telemetry_h))

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
        p_move_last: float = 0.0
        p_move_last_raw: float | None = None
        p_dir_top: tuple[str, float] | None = None

        # Closed-loop metrics (Requirement_doc.md evaluation criteria).
        t_prev: float | None = None
        action_prev: Action | None = None
        action_switches: int = 0
        dir_switches: int = 0
        stop_toggles: int = 0
        pred_nonstop_move_s: float = 0.0
        pred_nonstop_rest_s: float = 0.0
        onset_t: float | None = None
        release_t: float | None = None
        triggered_in_move: bool = False

        move_start = cue_start_s
        move_end = cue_end_s if not cue_is_rest(cue) else cue_start_s
        move_total_s = max(0.0, move_end - move_start)
        rest_total_s = max(0.0, 15.0 - move_total_s)

        while True:
            now = time.perf_counter()
            t_sim_raw = (now - start) * float(args.speed)
            t_sim = min(15.0, t_sim_raw)

            # Attribute the elapsed interval to the previously-issued action.
            if t_prev is not None and action_prev is not None:
                seg_s = t_sim - t_prev
                if seg_s > 0:
                    move_s = _overlap_s(t_prev, t_sim, move_start, move_end)
                    rest_s = seg_s - move_s
                    if action_prev != Action.STOP:
                        pred_nonstop_move_s += move_s
                        pred_nonstop_rest_s += rest_s

            if t_sim_raw >= 15.0:
                break

            if str(args.mode) == "oracle":
                if cue_start_s <= t_sim <= cue_end_s and not cue_is_rest(cue):
                    action = move_action
                else:
                    action = Action.STOP
                p_move_last = 1.0 if action != Action.STOP else 0.0
                p_move_last_raw = None
                p_dir_top = None
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
                    p_move_last = 0.0
                    p_move_last_raw = None
                    p_dir_top = None
                else:
                    t_inf0 = time.perf_counter()
                    f_raw = eeg_window_features(eeg[start_i:end], fs_hz=fs, include_fft=include_fft)
                    f_delta: np.ndarray | None = None
                    if base_feat is not None and base_feat.shape == f_raw.shape:
                        f_delta = (f_raw - base_feat).astype(np.float32, copy=False)
                    def _select(mode: str) -> np.ndarray:
                        if mode == "raw":
                            return f_raw
                        if mode == "delta":
                            return f_delta if f_delta is not None else f_raw
                        if mode == "raw+delta":
                            if f_delta is None:
                                # Should not happen with baseline='pre_cue', but keep runtime robust.
                                return np.concatenate([f_raw, np.zeros_like(f_raw, dtype=np.float32)], axis=0).astype(
                                    np.float32, copy=False
                                )
                            return np.concatenate([f_raw, f_delta], axis=0).astype(np.float32, copy=False)
                        raise ValueError(f"Unknown feature_mode: {mode!r}")

                    x_move = _select(feature_mode_move).reshape(1, -1).astype(np.float32, copy=False)
                    x_dir = _select(feature_mode_dir).reshape(1, -1).astype(np.float32, copy=False)
                    p_move = float(model.predict_move_proba(x_move)[0])
                    p_dir = model.predict_direction_proba(x_dir)[0]
                    action = stabilizer.step(p_move=p_move, p_dir=p_dir)
                    p_move_last_raw = p_move
                    p_move_last = p_move
                    top_i = int(np.argmax(p_dir))
                    p_dir_top = (["LEFT", "RIGHT", "FORWARD", "BACKWARD"][top_i], float(p_dir[top_i]))
                    infer_ms.append((time.perf_counter() - t_inf0) * 1000.0)

            if action != last_action:
                if last_action is not None:
                    action_switches += 1
                    if (last_action == Action.STOP) != (action == Action.STOP):
                        stop_toggles += 1
                    if last_action != Action.STOP and action != Action.STOP and last_action != action:
                        dir_switches += 1
                if str(args.mode) == "model" and model is not None and stabilizer is not None and infer_ms:
                    print(f"[intent_policy] t={t_sim:6.3f}s action={action} infer_ms={infer_ms[-1]:.2f}")
                else:
                    print(f"[intent_policy] t={t_sim:6.3f}s action={action}")
                last_action = action

            if overlay is not None:
                cfg = stabilizer.cfg if stabilizer is not None else StabilityConfig()
                overlay.update(
                    t_sim=float(t_sim),
                    p_move=float(p_move_last),
                    p_move_raw=p_move_last_raw,
                    p_dir_top=p_dir_top,
                    action=action,
                    cue_start_s=float(cue_start_s),
                    cue_end_s=float(cue_end_s),
                    cfg=cfg,
                )

            if action != Action.STOP and (move_start <= t_sim <= move_end) and not cue_is_rest(cue):
                triggered_in_move = True
                if onset_t is None:
                    onset_t = t_sim
            if triggered_in_move and release_t is None and (t_sim >= move_end) and action == Action.STOP:
                release_t = t_sim

            ctrl.set_action(action)
            t_prev = t_sim
            action_prev = action

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
        if overlay is not None:
            overlay.close()
        if infer_ms:
            arr_ms = np.asarray(infer_ms, dtype=np.float32)
            p95 = float(np.percentile(arr_ms, 95))
            print(
                f"[intent_policy] inference_ms mean={float(arr_ms.mean()):.2f} p95={p95:.2f} n={int(arr_ms.shape[0])}"
            )

        # Report closed-loop metrics (timing/stability/false triggers).
        if move_total_s > 0 and onset_t is not None:
            onset_latency = onset_t - move_start
            onset_s = f"{onset_latency:.3f}s"
        elif move_total_s > 0:
            onset_s = "n/a (no-trigger)"
        else:
            onset_s = "n/a (no-move cue)"

        if move_total_s > 0 and release_t is not None:
            release_latency = release_t - move_end
            release_s = f"{release_latency:.3f}s"
        elif move_total_s > 0:
            release_s = "n/a (no-release)"
        else:
            release_s = "n/a (no-move cue)"

        switches_per_min = float(action_switches) * 60.0 / 15.0
        false_rate = (pred_nonstop_rest_s / rest_total_s) if rest_total_s > 0 else 0.0
        move_cover = (pred_nonstop_move_s / move_total_s) if move_total_s > 0 else 0.0

        print(
            "[intent_policy] metrics "
            f"onset_latency={onset_s} release_latency={release_s} "
            f"switches={action_switches} switches_per_min={switches_per_min:.1f} "
            f"dir_switches={dir_switches} stop_toggles={stop_toggles} "
            f"false_nonstop_rest_s={pred_nonstop_rest_s:.3f} false_rate={false_rate:.3f} "
            f"nonstop_move_s={pred_nonstop_move_s:.3f} move_coverage={move_cover:.3f}"
        )
        ctrl.stop()


if __name__ == "__main__":
    main()
