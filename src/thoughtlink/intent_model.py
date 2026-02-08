from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .linear import BinaryLogReg, SoftmaxReg, StandardScaler


DIR_LABELS: list[str] = ["LEFT", "RIGHT", "FORWARD", "BACKWARD"]


@dataclass(frozen=True)
class IntentModel:
    scaler: StandardScaler  # move scaler (and legacy shared scaler)
    stage1: BinaryLogReg
    stage2: SoftmaxReg
    fs_hz: float
    window_s: float
    hop_s: float
    guard_s: float
    cue_start_s: float
    baseline: str = "none"
    include_fft: bool = True
    feature_mode: str = "raw"
    feature_mode_move: str | None = None
    feature_mode_dir: str | None = None
    scaler_dir: StandardScaler | None = None  # direction scaler (optional; falls back to scaler)

    def predict_move_proba(self, x: np.ndarray) -> np.ndarray:
        return self.stage1.predict_proba(self.scaler.transform(x))

    def predict_direction_proba(self, x: np.ndarray) -> np.ndarray:
        sc = self.scaler_dir or self.scaler
        return self.stage2.predict_proba(sc.transform(x))

    def predict_pipeline_labels(self, x: np.ndarray, *, p_move: float = 0.5) -> np.ndarray:
        """Return 0..4 labels where 0 is REST/STOP and 1..4 map to DIR_LABELS."""
        xs = self.scaler.transform(x)
        move = self.stage1.predict(xs, threshold=p_move).astype(bool)
        sc = self.scaler_dir or self.scaler
        xd = sc.transform(x)
        dir_idx = self.stage2.predict(xd)  # 0..3
        out = np.zeros((x.shape[0],), dtype=np.int64)
        out[move] = dir_idx[move] + 1
        return out

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_dir = self.scaler_dir or self.scaler
        feature_mode_move = self.feature_mode_move or self.feature_mode
        feature_mode_dir = self.feature_mode_dir or self.feature_mode
        np.savez_compressed(
            str(path),
            scaler_mean=self.scaler.mean,
            scaler_std=self.scaler.std,
            scaler_dir_mean=scaler_dir.mean,
            scaler_dir_std=scaler_dir.std,
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
            include_fft=np.array([1 if bool(self.include_fft) else 0], dtype=np.int64),
            feature_mode=np.array([self.feature_mode], dtype=object),
            feature_mode_move=np.array([feature_mode_move], dtype=object),
            feature_mode_dir=np.array([feature_mode_dir], dtype=object),
            dir_labels=np.array(DIR_LABELS, dtype=object),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "IntentModel":
        arr = np.load(str(path), allow_pickle=True)
        scaler = StandardScaler(mean=arr["scaler_mean"], std=arr["scaler_std"])
        scaler_dir: StandardScaler | None = None
        if "scaler_dir_mean" in getattr(arr, "files", []) and "scaler_dir_std" in getattr(arr, "files", []):
            scaler_dir = StandardScaler(mean=arr["scaler_dir_mean"], std=arr["scaler_dir_std"])
        stage1 = BinaryLogReg(w=arr["stage1_w"], b=float(arr["stage1_b"][0]))
        stage2 = SoftmaxReg(w=arr["stage2_w"], b=arr["stage2_b"])
        baseline = "none"
        if "baseline" in getattr(arr, "files", []):
            baseline = str(arr["baseline"][0])

        include_fft = stage1.w.shape[0] > 6
        if "include_fft" in getattr(arr, "files", []):
            include_fft = bool(int(arr["include_fft"][0]) != 0)

        feature_mode = "delta" if baseline == "pre_cue" else "raw"
        if "feature_mode" in getattr(arr, "files", []):
            feature_mode = str(arr["feature_mode"][0])
        feature_mode_move: str | None = None
        feature_mode_dir: str | None = None
        if "feature_mode_move" in getattr(arr, "files", []):
            feature_mode_move = str(arr["feature_mode_move"][0])
        if "feature_mode_dir" in getattr(arr, "files", []):
            feature_mode_dir = str(arr["feature_mode_dir"][0])
        return cls(
            scaler=scaler,
            scaler_dir=scaler_dir,
            stage1=stage1,
            stage2=stage2,
            fs_hz=float(arr["fs_hz"][0]),
            window_s=float(arr["window_s"][0]),
            hop_s=float(arr["hop_s"][0]),
            guard_s=float(arr["guard_s"][0]),
            cue_start_s=float(arr["cue_start_s"][0]),
            baseline=baseline,
            include_fft=include_fft,
            feature_mode=feature_mode,
            feature_mode_move=feature_mode_move,
            feature_mode_dir=feature_mode_dir,
        )
