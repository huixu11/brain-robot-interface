from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .linear import BinaryLogReg, SoftmaxReg, StandardScaler


DIR_LABELS: list[str] = ["LEFT", "RIGHT", "FORWARD", "BACKWARD"]


@dataclass(frozen=True)
class IntentModel:
    scaler: StandardScaler
    stage1: BinaryLogReg
    stage2: SoftmaxReg
    fs_hz: float
    window_s: float
    hop_s: float
    guard_s: float
    cue_start_s: float
    baseline: str = "none"

    def predict_move_proba(self, x: np.ndarray) -> np.ndarray:
        return self.stage1.predict_proba(self.scaler.transform(x))

    def predict_direction_proba(self, x: np.ndarray) -> np.ndarray:
        return self.stage2.predict_proba(self.scaler.transform(x))

    def predict_pipeline_labels(self, x: np.ndarray, *, p_move: float = 0.5) -> np.ndarray:
        """Return 0..4 labels where 0 is REST/STOP and 1..4 map to DIR_LABELS."""
        xs = self.scaler.transform(x)
        move = self.stage1.predict(xs, threshold=p_move).astype(bool)
        dir_idx = self.stage2.predict(xs)  # 0..3
        out = np.zeros((x.shape[0],), dtype=np.int64)
        out[move] = dir_idx[move] + 1
        return out

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            scaler_mean=self.scaler.mean,
            scaler_std=self.scaler.std,
            stage1_w=self.stage1.w,
            stage1_b=np.array([self.stage1.b], dtype=np.float32),
            stage2_w=self.stage2.w,
            stage2_b=self.stage2.b,
            fs_hz=np.array([self.fs_hz], dtype=np.float32),
            window_s=np.array([self.window_s], dtype=np.float32),
            hop_s=np.array([self.hop_s], dtype=np.float32),
            guard_s=np.array([self.guard_s], dtype=np.float32),
            cue_start_s=np.array([self.cue_start_s], dtype=np.float32),
            baseline=np.array([self.baseline], dtype=object),
            dir_labels=np.array(DIR_LABELS, dtype=object),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "IntentModel":
        arr = np.load(str(path), allow_pickle=True)
        scaler = StandardScaler(mean=arr["scaler_mean"], std=arr["scaler_std"])
        stage1 = BinaryLogReg(w=arr["stage1_w"], b=float(arr["stage1_b"][0]))
        stage2 = SoftmaxReg(w=arr["stage2_w"], b=arr["stage2_b"])
        baseline = "none"
        if "baseline" in getattr(arr, "files", []):
            baseline = str(arr["baseline"][0])
        return cls(
            scaler=scaler,
            stage1=stage1,
            stage2=stage2,
            fs_hz=float(arr["fs_hz"][0]),
            window_s=float(arr["window_s"][0]),
            hop_s=float(arr["hop_s"][0]),
            guard_s=float(arr["guard_s"][0]),
            cue_start_s=float(arr["cue_start_s"][0]),
            baseline=baseline,
        )
