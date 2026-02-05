from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


G1_LOWLEVEL_MOTOR_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_SPORTMODESTATE = "rt/sportmodestate"


@dataclass
class MirrorConfig:
    domain_id: int = 0
    interface: str = "en0"
    robot_type: str = "g1"


class UnitreeLowStateMirror:
    """Subscriber-only mirror: LowState -> MuJoCo qpos/qvel."""

    def __init__(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, cfg: MirrorConfig
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.cfg = cfg

        if cfg.robot_type == "g1":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHg

            self._LowState = LowStateHg
            self._joint_names = G1_LOWLEVEL_MOTOR_JOINT_NAMES
        else:
            raise NotImplementedError(f"Mirror robot_type not implemented: {cfg.robot_type}")

        self._data_lock = threading.Lock()
        self._num_motors = len(self._joint_names)
        self._joint_pos = np.zeros(self._num_motors, dtype=np.float32)
        self._joint_vel = np.zeros(self._num_motors, dtype=np.float32)
        self._imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._base_ang_vel = np.zeros(3, dtype=np.float32)
        self._state_received = False

        self._motor_qposadr: list[int] = []
        self._motor_qveladr: list[int] = []
        for name in self._joint_names:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                self._motor_qposadr.append(-1)
                self._motor_qveladr.append(-1)
            else:
                self._motor_qposadr.append(int(self.mj_model.jnt_qposadr[jid]))
                self._motor_qveladr.append(int(self.mj_model.jnt_dofadr[jid]))

        ChannelFactoryInitialize(cfg.domain_id, cfg.interface)
        self._state_sub = ChannelSubscriber(TOPIC_LOWSTATE, self._LowState)
        self._state_sub.Init(self._lowstate_callback, 10)
        self._sport_sub = ChannelSubscriber(TOPIC_SPORTMODESTATE, SportModeState_)
        self._sport_sub.Init(self._sportstate_callback, 10)

    def _lowstate_callback(self, msg: Any) -> None:
        with self._data_lock:
            for i in range(self._num_motors):
                self._joint_pos[i] = msg.motor_state[i].q
                self._joint_vel[i] = msg.motor_state[i].dq
            self._imu_quat[0] = msg.imu_state.quaternion[0]
            self._imu_quat[1] = msg.imu_state.quaternion[1]
            self._imu_quat[2] = msg.imu_state.quaternion[2]
            self._imu_quat[3] = msg.imu_state.quaternion[3]
            self._base_ang_vel[0] = msg.imu_state.gyroscope[0]
            self._base_ang_vel[1] = msg.imu_state.gyroscope[1]
            self._base_ang_vel[2] = msg.imu_state.gyroscope[2]
            self._state_received = True

    def _sportstate_callback(self, msg: Any) -> None:
        _ = msg

    def apply_to_mujoco(self) -> bool:
        with self._data_lock:
            if not self._state_received:
                return False
            joint_pos = self._joint_pos.copy()
            joint_vel = self._joint_vel.copy()
            imu_quat = self._imu_quat.copy()
            base_ang_vel = self._base_ang_vel.copy()

        self.mj_data.qpos[3:7] = imu_quat
        self.mj_data.qvel[3:6] = base_ang_vel
        for motor_idx in range(self._num_motors):
            qadr = self._motor_qposadr[motor_idx]
            vadr = self._motor_qveladr[motor_idx]
            if qadr >= 0:
                self.mj_data.qpos[qadr] = float(joint_pos[motor_idx])
            if vadr >= 0:
                self.mj_data.qvel[vadr] = float(joint_vel[motor_idx])
        return True


class MirrorBackend:
    def __init__(
        self,
        *,
        bundle_dir: str,
        interface: str = "en0",
        domain_id: int = 0,
        robot_type: str = "g1",
        viewer_dt: float = 0.02,
    ) -> None:
        self._bundle_dir = bundle_dir
        self._cfg = MirrorConfig(domain_id=domain_id, interface=interface, robot_type=robot_type)
        self._viewer_dt = viewer_dt
        self._viewer = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._mirror: UnitreeLowStateMirror | None = None
        self._last_view_sync = 0.0

    def start(self) -> None:
        model_path = self._bundle_dir
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "scene.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        self._model = model
        self._data = data
        self._mirror = UnitreeLowStateMirror(model, data, self._cfg)
        try:
            self._viewer = mujoco.viewer.launch_passive(
                model,
                data,
                show_left_ui=False,
                show_right_ui=False,
            )
        except RuntimeError as exc:
            if platform.system() == "Darwin":
                raise RuntimeError(
                    "On macOS, MuJoCo viewer must be launched via `mjpython`."
                ) from exc
            raise

    def is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def step(self) -> None:
        if not self.is_running():
            return
        assert self._mirror is not None
        assert self._model is not None
        assert self._data is not None
        updated = self._mirror.apply_to_mujoco()
        if updated:
            mujoco.mj_forward(self._model, self._data)
        now = time.time()
        if now - self._last_view_sync >= self._viewer_dt:
            self._viewer.sync()
            self._last_view_sync = now

    def stop(self) -> None:
        if self._viewer is None:
            return
        try:
            if self._viewer.is_running():
                self._viewer.close()
        except Exception:
            pass

