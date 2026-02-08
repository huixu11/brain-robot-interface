from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .labels import CanonicalCue, normalize_cue_label


@dataclass(frozen=True)
class ChunkMeta:
    label_raw: str
    cue: CanonicalCue
    subject_id: str
    session_id: str
    duration_s: float


@dataclass(frozen=True)
class Chunk:
    path: Path
    eeg: np.ndarray  # (n_samples, n_channels)
    meta: ChunkMeta


def default_data_dir(repo_root: Path) -> Path:
    return repo_root / "robot_control_data" / "data"


def iter_npz_files(data_dir: Path) -> Iterator[Path]:
    for p in sorted(data_dir.glob("*.npz")):
        yield p


def load_chunk(path: Path) -> Chunk:
    arr = np.load(str(path), allow_pickle=True)
    eeg = arr["feature_eeg"]
    meta: dict[str, Any] = arr["label"].item()

    label_raw = str(meta.get("label", ""))
    cue = normalize_cue_label(label_raw)
    subject_id = str(meta.get("subject_id", ""))
    session_id = str(meta.get("session_id", ""))
    duration_s = float(meta.get("duration", 0.0))

    return Chunk(
        path=path,
        eeg=eeg,
        meta=ChunkMeta(
            label_raw=label_raw,
            cue=cue,
            subject_id=subject_id,
            session_id=session_id,
            duration_s=duration_s,
        ),
    )


@dataclass(frozen=True)
class WindowConfig:
    fs_hz: float = 500.0
    window_s: float = 0.5
    hop_s: float = 0.1
    cue_start_s: float = 3.0
    guard_s: float = 0.2


@dataclass(frozen=True)
class EegWindow:
    chunk_path: Path
    subject_id: str
    session_id: str
    t0_s: float
    t1_s: float
    x: np.ndarray  # (window_samples, n_channels)
    is_move: bool
    cue: CanonicalCue


def iter_eeg_windows(chunk: Chunk, cfg: WindowConfig) -> Iterator[EegWindow]:
    """Yield sliding windows with move/rest labels derived from cue timing.

    Labels follow the dataset README:
    - rest from t=0..3
    - cue on at t=3 for duration_s
    - rest after cue end

    We exclude transition regions around cue on/off using cfg.guard_s.
    """

    eeg = chunk.eeg
    n_samples, _n_ch = eeg.shape
    total_s = n_samples / cfg.fs_hz

    win_n = int(round(cfg.window_s * cfg.fs_hz))
    hop_n = int(round(cfg.hop_s * cfg.fs_hz))
    if win_n <= 0 or hop_n <= 0:
        raise ValueError("window_s and hop_s must be > 0")
    if n_samples < win_n:
        return

    cue_on = cfg.cue_start_s
    cue_off = min(total_s, cue_on + max(0.0, chunk.meta.duration_s))

    # Transition exclusion windows.
    trans1 = (max(0.0, cue_on - cfg.guard_s), min(total_s, cue_on + cfg.guard_s))
    trans2 = (max(0.0, cue_off - cfg.guard_s), min(total_s, cue_off + cfg.guard_s))

    for start in range(0, n_samples - win_n + 1, hop_n):
        end = start + win_n
        t0 = start / cfg.fs_hz
        t1 = end / cfg.fs_hz

        # Exclude windows that overlap transitions.
        if not (t1 <= trans1[0] or t0 >= trans1[1]):
            continue
        if not (t1 <= trans2[0] or t0 >= trans2[1]):
            continue

        # Define move vs rest windows.
        is_move = (t0 >= cue_on) and (t1 <= cue_off) and (chunk.meta.cue != CanonicalCue.RELAX)

        yield EegWindow(
            chunk_path=chunk.path,
            subject_id=chunk.meta.subject_id,
            session_id=chunk.meta.session_id,
            t0_s=t0,
            t1_s=t1,
            x=eeg[start:end],
            is_move=is_move,
            cue=chunk.meta.cue,
        )


@dataclass(frozen=True)
class SplitConfig:
    train_subjects: int | None = None
    val_subjects: int = 1
    test_subjects: int = 1
    seed: int = 0


@dataclass(frozen=True)
class Split:
    train: list[Chunk]
    val: list[Chunk]
    test: list[Chunk]


def split_by_subject(chunks: list[Chunk], cfg: SplitConfig) -> Split:
    """Split chunks by subject_id to avoid leakage (as recommended by the prompt)."""

    subjects = sorted({c.meta.subject_id for c in chunks})
    rng = np.random.default_rng(cfg.seed)
    perm = list(subjects)
    rng.shuffle(perm)

    n_test = max(0, int(cfg.test_subjects))
    n_val = max(0, int(cfg.val_subjects))
    remaining = len(perm) - (n_test + n_val)
    if remaining < 1:
        raise ValueError("Not enough subjects for requested split sizes")
    n_train = remaining if cfg.train_subjects is None else min(int(cfg.train_subjects), remaining)

    train_sub = set(perm[:n_train])
    val_sub = set(perm[n_train : n_train + n_val])
    test_sub = set(perm[n_train + n_val : n_train + n_val + n_test])

    train = [c for c in chunks if c.meta.subject_id in train_sub]
    val = [c for c in chunks if c.meta.subject_id in val_sub]
    test = [c for c in chunks if c.meta.subject_id in test_sub]

    return Split(train=train, val=val, test=test)


@dataclass(frozen=True)
class SessionSplitConfig:
    """Split chunks by session_id.

    This is useful for realistic per-user calibration: train/val/test all come from the
    same subject(s), but different recording sessions.
    """

    val_sessions: int = 1
    test_sessions: int = 1
    seed: int = 0


def split_by_session(chunks: list[Chunk], cfg: SessionSplitConfig) -> Split:
    """Split chunks by session_id (to avoid leaking within the same recording)."""

    sessions = sorted({c.meta.session_id for c in chunks})
    rng = np.random.default_rng(cfg.seed)
    perm = list(sessions)
    rng.shuffle(perm)

    n = len(perm)
    if n < 2:
        raise ValueError("Not enough sessions to split (need at least 2)")

    n_test = max(0, int(cfg.test_sessions))
    # Ensure at least 1 session remains for training.
    n_test = min(n_test, n - 1)

    n_val = max(0, int(cfg.val_sessions))
    # Ensure at least 1 session remains for training after val+test.
    n_val = min(n_val, n - n_test - 1)

    test_sess = set(perm[:n_test])
    val_sess = set(perm[n_test : n_test + n_val])
    train_sess = set(perm[n_test + n_val :])

    train = [c for c in chunks if c.meta.session_id in train_sess]
    val = [c for c in chunks if c.meta.session_id in val_sess]
    test = [c for c in chunks if c.meta.session_id in test_sess]

    return Split(train=train, val=val, test=test)
