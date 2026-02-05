from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

from ..command.cmd_vel import CmdVel


def _parse_csv_strings(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_csv_floats(s: str) -> np.ndarray:
    vals = [float(p.strip()) for p in s.split(",") if p.strip()]
    return np.array(vals, dtype=np.float32)


class CmdVelController:
    def __init__(self) -> None:
        self._cmd = np.zeros(3, dtype=np.float32)

    def set_cmd(self, cmd: CmdVel) -> None:
        self._cmd[0] = cmd.vx
        self._cmd[1] = cmd.vy
        self._cmd[2] = cmd.yaw_rate

    def get_command(self) -> np.ndarray:
        return self._cmd.copy()


@dataclass
class SimWalkConfig:
    ctrl_dt: float = 0.02
    sim_dt: float = 0.005
    action_scale: float = 0.5
    imu_site_name: str = "imu_in_pelvis"
    gravity_w: tuple[float, float, float] = (0.0, 0.0, -1.0)


class MjlabVelocityOnnxPolicy:
    """MJLab velocity-policy interface for G1 (matches MJLab obs/action contract)."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float,
        input_controller: CmdVelController,
        *,
        imu_site_name: str = "imu_in_pelvis",
        gravity_w: np.ndarray | None = None,
    ) -> None:
        self._policy = ort.InferenceSession(
            policy_path, providers=ort.get_available_providers()
        )
        outputs = [o.name for o in self._policy.get_outputs()]
        if "continuous_actions" in outputs:
            self._output_names = ["continuous_actions"]
        elif "actions" in outputs:
            self._output_names = ["actions"]
        elif outputs:
            self._output_names = [outputs[0]]
        else:
            raise ValueError(f"ONNX policy has no outputs: {policy_path}")

        self._action_scale = float(action_scale)
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._counter = 0
        self._n_substeps = n_substeps
        self._input_controller = input_controller
        self._imu_site_name = imu_site_name
        self._gravity_w = (
            gravity_w if gravity_w is not None else np.array([0.0, 0.0, -1.0])
        )

        meta = self._policy.get_modelmeta().custom_metadata_map
        if (
            "joint_names" not in meta
            or "default_joint_pos" not in meta
            or "action_scale" not in meta
        ):
            raise ValueError(
                "MJLab policy missing metadata keys: joint_names, default_joint_pos, action_scale."
            )
        self._policy_joint_names = _parse_csv_strings(meta["joint_names"])
        self._default_joint_pos_policy = _parse_csv_floats(meta["default_joint_pos"])
        self._action_scale_policy = _parse_csv_floats(meta["action_scale"])

        self._policy_qpos_adr: np.ndarray | None = None
        self._policy_qvel_adr: np.ndarray | None = None
        self._ctrl_policy_idx: np.ndarray | None = None

    def _ensure_mappings(self, model: mujoco.MjModel) -> None:
        if self._policy_qpos_adr is not None:
            return

        names = self._policy_joint_names
        default = self._default_joint_pos_policy
        scale = self._action_scale_policy
        if len(names) != len(default) or len(names) != len(scale):
            raise ValueError(
                f"ONNX metadata mismatch: len(joint_names)={len(names)}, "
                f"len(default_joint_pos)={len(default)}, len(action_scale)={len(scale)}"
            )

        qpos_adr: list[int] = []
        qvel_adr: list[int] = []
        for jname in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Policy joint '{jname}' not found in MuJoCo model.")
            qpos_adr.append(int(model.jnt_qposadr[jid]))
            qvel_adr.append(int(model.jnt_dofadr[jid]))
        self._policy_qpos_adr = np.array(qpos_adr, dtype=np.int32)
        self._policy_qvel_adr = np.array(qvel_adr, dtype=np.int32)

        index_by_name = {n: i for i, n in enumerate(names)}
        ctrl_policy_idx: list[int] = []
        for act_id in range(model.nu):
            jid = int(model.actuator_trnid[act_id, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None or jname not in index_by_name:
                raise ValueError(
                    f"Actuator {act_id} joint '{jname}' not in policy joint_names."
                )
            ctrl_policy_idx.append(index_by_name[jname])
        self._ctrl_policy_idx = np.array(ctrl_policy_idx, dtype=np.int32)

    def _get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        self._ensure_mappings(model)
        linvel = data.sensor("imu_lin_vel").data
        gyro = data.sensor("imu_ang_vel").data
        imu_xmat = data.site_xmat[model.site(self._imu_site_name).id].reshape(3, 3)
        gravity_b = imu_xmat.T @ self._gravity_w

        assert self._policy_qpos_adr is not None
        assert self._policy_qvel_adr is not None
        joint_pos = data.qpos[self._policy_qpos_adr]
        joint_angles = joint_pos - self._default_joint_pos_policy
        joint_velocities = data.qvel[self._policy_qvel_adr]

        obs = np.hstack(
            [
                linvel,
                gyro,
                gravity_b,
                joint_angles,
                joint_velocities,
                self._last_action,
                self._input_controller.get_command(),
            ]
        )
        return obs.astype(np.float32)

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        obs = self._get_obs(model, data)
        onnx_input = {"obs": obs.reshape(1, -1)}
        onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0].astype(
            np.float32
        )
        self._last_action = onnx_pred.copy()

        idx = self._ctrl_policy_idx
        assert idx is not None
        targets = self._default_joint_pos_policy[idx] + (
            onnx_pred[idx] * self._action_scale_policy[idx]
        )
        data.ctrl[:] = targets


def load_sim_model(
    bundle_dir: str, cfg: SimWalkConfig
) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    scene_path = os.path.join(bundle_dir, "scene.xml")
    model_path = scene_path if os.path.exists(scene_path) else os.path.join(bundle_dir, "model.xml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model XML not found: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.sim_dt
    if model.nkey > 0:
        default_qpos = np.array(model.key_qpos[0, 7:], dtype=np.float32)
    else:
        default_qpos = np.array(data.qpos[7:], dtype=np.float32)
    return model, data, default_qpos


def build_sim_policy(
    bundle_dir: str,
    cfg: SimWalkConfig,
    controller: CmdVelController,
    default_qpos: np.ndarray,
) -> MjlabVelocityOnnxPolicy:
    policy_path = os.path.join(bundle_dir, "policy.onnx")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy ONNX not found: {policy_path}")
    n_substeps = round(cfg.ctrl_dt / cfg.sim_dt)
    return MjlabVelocityOnnxPolicy(
        policy_path=policy_path,
        default_angles=default_qpos,
        n_substeps=n_substeps,
        action_scale=cfg.action_scale,
        input_controller=controller,
        imu_site_name=cfg.imu_site_name,
        gravity_w=np.array(cfg.gravity_w, dtype=np.float32),
    )


class SimBackend:
    def __init__(
        self,
        *,
        bundle_dir: str | None = None,
        ctrl_hz: float = 50.0,
        sim_dt: float = 0.005,
        viewer_dt: float = 0.02,
        action_scale: float = 0.5,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._bundle_dir = bundle_dir or str(repo_root / "bundles" / "g1_mjlab")
        self._ctrl_hz = ctrl_hz
        self._sim_dt = sim_dt
        self._viewer_dt = viewer_dt
        self._action_scale = action_scale
        self._viewer = None
        self._last_view_sync = 0.0
        self._cfg: SimWalkConfig | None = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._policy: MjlabVelocityOnnxPolicy | None = None
        self._cmd_controller = CmdVelController()
        self._n_steps = 1

    def start(self) -> None:
        ctrl_dt = 0.02 if self._ctrl_hz <= 0 else 1.0 / self._ctrl_hz
        cfg = SimWalkConfig(ctrl_dt=ctrl_dt, sim_dt=self._sim_dt, action_scale=self._action_scale)
        model, data, default_qpos = load_sim_model(self._bundle_dir, cfg)
        policy = build_sim_policy(self._bundle_dir, cfg, self._cmd_controller, default_qpos)
        self._cfg = cfg
        self._model = model
        self._data = data
        self._policy = policy
        self._n_steps = max(1, round(cfg.ctrl_dt / cfg.sim_dt))
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

    def step(self, cmd: CmdVel) -> None:
        if not self.is_running():
            return
        assert self._model is not None
        assert self._data is not None
        assert self._policy is not None
        self._cmd_controller.set_cmd(cmd)
        for _ in range(self._n_steps):
            self._policy.step(self._model, self._data)
            mujoco.mj_step(self._model, self._data)
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

