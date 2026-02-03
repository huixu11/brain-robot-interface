from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "unitree_sdk2_python"))

from ..command.cmd_vel import CmdVel, CmdVelSmoother
from ..command.command_mux import CommandMux
from ..command.cv_pose_gestures import PoseGestureDetector
from ..command.csv_logger import CsvLogger, LogSample
from ..backends.unitree_sport_rpc import SportRpcConfig, UnitreeSportRpc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RobotWalk: Unitree Sport RPC")
    parser.add_argument("--interface", default="en0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--cmd-hz", type=float, default=10.0)
    parser.add_argument("--cmd-timeout", type=float, default=0.6)
    parser.add_argument("--source", default="both", choices=["keyboard", "cv", "both"])
    parser.add_argument("--log-hz", type=float, default=30.0)
    parser.add_argument("--max-vx", type=float, default=0.6)
    parser.add_argument("--max-vy", type=float, default=0.4)
    parser.add_argument("--max-yaw", type=float, default=1.5)
    parser.add_argument("--cmd-smoothing", type=float, default=0.2)
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


def _cmd_from_discrete(name: str, max_vx: float, max_vy: float, max_yaw: float) -> tuple[float, float, float]:
    if name == "forward":
        return max_vx, 0.0, 0.0
    if name == "left":
        return 0.0, 0.0, max_yaw
    if name == "right":
        return 0.0, 0.0, -max_yaw
    if name == "yaw_left":
        return 0.0, 0.0, max_yaw
    if name == "yaw_right":
        return 0.0, 0.0, -max_yaw
    return 0.0, 0.0, 0.0


def main() -> None:
    args = _build_arg_parser().parse_args()

    cmd_mux = CommandMux(key_timeout_s=args.key_timeout)
    smoother = CmdVelSmoother(alpha=args.cmd_smoothing)

    gesture_detector = None
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

    robot = UnitreeSportRpc(
        SportRpcConfig(
            domain_id=args.domain_id,
            interface=args.interface,
            cmd_hz=args.cmd_hz,
            cmd_timeout_s=args.cmd_timeout,
        )
    )
    robot.start()

    logger = CsvLogger(log_hz=args.log_hz, out_dir=args.logs)

    def log_sample() -> LogSample:
        state = gesture_detector.latest() if gesture_detector else None
        cmd = cmd_mux.sample()
        return LogSample(
            ts=time.time(),
            vx=cmd.vx,
            vy=cmd.vy,
            yaw_rate=cmd.yaw_rate,
            source=cmd.source,
            left_x=None if not state else state.left.x_px,
            left_y=None if not state else state.left.y_px,
            right_x=None if not state else state.right.x_px,
            right_y=None if not state else state.right.y_px,
            left_vis=None if not state else state.left.visibility,
            right_vis=None if not state else state.right.visibility,
        )

    log_path = logger.start(log_sample)
    print(f"[INFO] Logging to {log_path}")

    try:
        while True:
            if gesture_detector:
                latest = gesture_detector.latest()
                if latest:
                    vx, vy, yaw = _cmd_from_discrete(latest.cmd, args.max_vx, args.max_vy, args.max_yaw)
                    cmd_mux.update_cv(CmdVel(vx, vy, yaw, ts=latest.ts, source="cv"))
                if args.show_cv:
                    gesture_detector.render_latest()

            cmd = smoother.update(cmd_mux.sample())
            robot.set_cmd(cmd)
            robot.step()
            time.sleep(0.01)
    finally:
        logger.stop()
        if gesture_detector:
            gesture_detector.stop()


if __name__ == "__main__":
    main()

