from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import platform
import sys
import mujoco
import mujoco.viewer
from command_mux import CommandMux
from cv_pose_gestures import PoseGestureDetector
from logger import CsvLogger, LogSample
from sim_controller import ForceConfig, SimController


def _resolve_body_id(mj_model: mujoco.MjModel, body_name: Optional[str]) -> int:
    if body_name:
        return mj_model.body(body_name).id
    for candidate in ("torso_link", "base_link", "pelvis"):
        try:
            return mj_model.body(candidate).id
        except Exception:
            continue
    raise ValueError("No suitable body found. Provide --body name explicitly.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hack Nation MuJoCo + CV teleop")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo scene XML")
    parser.add_argument("--body", default=None, help="Body name to apply force to")
    parser.add_argument("--source", default="both", choices=["keyboard", "cv", "both"])
    parser.add_argument("--log-hz", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--viewer-dt", type=float, default=0.02)
    parser.add_argument("--force", type=float, default=150.0)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--forward", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    parser.add_argument("--left", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--key-timeout", type=float, default=0.35)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--cv-fps", type=float, default=15.0)
    parser.add_argument("--midline", type=float, default=0.5)
    parser.add_argument("--hysteresis-px", type=float, default=20.0)
    parser.add_argument("--stable-ms", type=int, default=150)
    parser.add_argument("--show-cv", action="store_true")
    parser.add_argument("--pose-model", default="models/pose_landmarker_lite.task")
    parser.add_argument("--logs", default="logs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = args.sim_dt

    body_id = _resolve_body_id(mj_model, args.body)
    cmd_mux = CommandMux(key_timeout_s=args.key_timeout)

    force_cfg = ForceConfig(
        magnitude=args.force,
        smoothing=args.smoothing,
        forward=tuple(args.forward),
        left=tuple(args.left),
    )
    sim_ctrl = SimController(body_id=body_id, cfg=force_cfg)

    gesture_detector: Optional[PoseGestureDetector] = None
    if args.source in ("cv", "both"):
        gesture_detector = PoseGestureDetector(
            cam_index=args.cam,
            fps=args.cv_fps,
            midline_ratio=args.midline,
            hysteresis_px=args.hysteresis_px,
            min_stable_ms=args.stable_ms,
            show_window=args.show_cv,
            model_path=args.pose_model,
        )
        gesture_detector.start()

    lock = threading.Lock()

    def key_callback(key: int) -> None:
        glfw = mujoco.glfw.glfw
        mapping = {
            glfw.KEY_W: "forward",
            glfw.KEY_UP: "forward",
            glfw.KEY_A: "left",
            glfw.KEY_LEFT: "left",
            glfw.KEY_D: "right",
            glfw.KEY_RIGHT: "right",
            glfw.KEY_S: "stop",
            glfw.KEY_DOWN: "stop",
            glfw.KEY_SPACE: "stop",
        }
        cmd = mapping.get(key)
        if cmd:
            cmd_mux.update_keyboard(cmd)

    try:
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=key_callback
        )
    except RuntimeError as exc:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "On macOS, MuJoCo viewer must be launched via `mjpython`.\n"
                "Example: mjpython run_sim.py --xml ... --source both --show-cv"
            ) from exc
        raise

    def sim_loop() -> None:
        while viewer.is_running():
            start = time.perf_counter()
            with lock:
                sample = cmd_mux.sample()
                force = sim_ctrl.update(sample.cmd)
                sim_ctrl.apply(mj_data, force)
                mujoco.mj_step(mj_model, mj_data)
            dt = mj_model.opt.timestep - (time.perf_counter() - start)
            if dt > 0:
                time.sleep(dt)

    def viewer_loop() -> None:
        while viewer.is_running():
            with lock:
                viewer.sync()
            time.sleep(args.viewer_dt)

    logger = CsvLogger(log_hz=args.log_hz, out_dir=args.logs)

    def log_sample() -> LogSample:
        state = gesture_detector.latest() if gesture_detector else None
        cmd_sample = cmd_mux.sample()
        return LogSample(
            ts=time.time(),
            cmd=cmd_sample.cmd,
            source=cmd_sample.source,
            left_x=None if not state else state.left.x_px,
            left_y=None if not state else state.left.y_px,
            right_x=None if not state else state.right.x_px,
            right_y=None if not state else state.right.y_px,
            left_vis=None if not state else state.left.visibility,
            right_vis=None if not state else state.right.visibility,
        )

    log_path = logger.start(log_sample)
    print(f"[INFO] Logging to {log_path}")

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    view_thread = threading.Thread(target=viewer_loop, daemon=True)
    sim_thread.start()
    view_thread.start()

    try:
        while viewer.is_running():
            if gesture_detector:
                latest = gesture_detector.latest()
                if latest:
                    cmd_mux.update_cv(latest.cmd, ts=latest.ts)
                if args.show_cv:
                    gesture_detector.render_latest()
            time.sleep(0.01)
    finally:
        logger.stop()
        if gesture_detector:
            gesture_detector.stop()


if __name__ == "__main__":
    main()

