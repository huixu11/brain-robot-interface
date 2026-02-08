from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thoughtlink.data import (
    SessionSplitConfig,
    SplitConfig,
    WindowConfig,
    iter_eeg_windows,
    iter_npz_files,
    load_chunk,
    split_by_session,
    split_by_subject,
)
from thoughtlink.features import eeg_window_features
from thoughtlink.intent_model import IntentModel
from thoughtlink.labels import CanonicalCue
from thoughtlink.linear import fit_scaler, train_binary_logreg, train_softmax_reg
from thoughtlink.metrics import accuracy, confusion_matrix


DIR_MAP: dict[CanonicalCue, int] = {
    CanonicalCue.LEFT_FIST: 0,
    CanonicalCue.RIGHT_FIST: 1,
    CanonicalCue.BOTH_FISTS: 2,  # forward
    CanonicalCue.TONGUE_TAPPING: 3,  # backward proxy
}


@dataclass(frozen=True)
class Dataset:
    x_move: np.ndarray  # (n, d_move)
    x_dir: np.ndarray  # (n, d_dir)
    y_move: np.ndarray  # (n,) 0/1
    y_dir: np.ndarray  # (n,) -1 for rest else 0..3


def build_dataset_shared(
    chunks: list,
    win_cfg: WindowConfig,
    *,
    include_fft: bool,
    baseline: str,
    feature_mode: str,
) -> Dataset:
    # baseline: controls whether we compute per-chunk pre-cue baseline features.
    # feature_mode: controls what we feed into the linear models (raw vs delta vs concatenation).
    feats: list[np.ndarray] = []
    y_move: list[int] = []
    y_dir: list[int] = []

    for chunk in chunks:
        base_feat: np.ndarray | None = None
        if baseline == "pre_cue":
            # Per-chunk baseline correction using the known rest period t=0..cue_start.
            # This is common in BCI pipelines and improves robustness across sessions.
            base_end = int(round((win_cfg.cue_start_s - win_cfg.guard_s) * win_cfg.fs_hz))
            base_end = max(base_end, int(round(win_cfg.window_s * win_cfg.fs_hz)))
            base_end = min(base_end, int(chunk.eeg.shape[0]))
            if base_end > 0:
                base_feat = eeg_window_features(
                    chunk.eeg[:base_end], fs_hz=win_cfg.fs_hz, include_fft=include_fft
                ).astype(np.float32)

        for w in iter_eeg_windows(chunk, win_cfg):
            f_raw = eeg_window_features(w.x, fs_hz=win_cfg.fs_hz, include_fft=include_fft)
            if baseline == "pre_cue" and base_feat is not None and base_feat.shape == f_raw.shape:
                f_delta = (f_raw - base_feat).astype(np.float32, copy=False)
            else:
                f_delta = None

            # Feature modes:
            # - raw: good for move/rest detection (uses absolute energy scale).
            # - delta: good for direction classification (subtract per-chunk pre-cue baseline).
            # - raw+delta: concatenation often helps stage1 while keeping delta signal for stage2.
            if feature_mode == "raw":
                f = f_raw
            elif feature_mode == "delta":
                f = f_delta if f_delta is not None else f_raw
            elif feature_mode == "raw+delta":
                if f_delta is None:
                    raise ValueError("feature_mode='raw+delta' requires baseline='pre_cue'")
                f = np.concatenate([f_raw, f_delta], axis=0).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode!r}")
            feats.append(f)
            y_move.append(1 if w.is_move else 0)
            if w.is_move:
                y_dir.append(DIR_MAP[w.cue])
            else:
                y_dir.append(-1)

    x = np.stack(feats, axis=0).astype(np.float32)
    y_move_arr = np.asarray(y_move, dtype=np.int64)
    y_dir_arr = np.asarray(y_dir, dtype=np.int64)
    return Dataset(x_move=x, x_dir=x, y_move=y_move_arr, y_dir=y_dir_arr)


def build_dataset_dual(chunks: list, win_cfg: WindowConfig, *, include_fft: bool, baseline: str) -> Dataset:
    """Build two aligned feature matrices:

    - x_move: raw window features (good for move/rest).
    - x_dir: baseline-delta features when baseline='pre_cue' else raw (good for direction).
    """

    feats_move: list[np.ndarray] = []
    feats_dir: list[np.ndarray] = []
    y_move: list[int] = []
    y_dir: list[int] = []

    for chunk in chunks:
        base_feat: np.ndarray | None = None
        if baseline == "pre_cue":
            base_end = int(round((win_cfg.cue_start_s - win_cfg.guard_s) * win_cfg.fs_hz))
            base_end = max(base_end, int(round(win_cfg.window_s * win_cfg.fs_hz)))
            base_end = min(base_end, int(chunk.eeg.shape[0]))
            if base_end > 0:
                base_feat = eeg_window_features(
                    chunk.eeg[:base_end], fs_hz=win_cfg.fs_hz, include_fft=include_fft
                ).astype(np.float32)

        for w in iter_eeg_windows(chunk, win_cfg):
            f_raw = eeg_window_features(w.x, fs_hz=win_cfg.fs_hz, include_fft=include_fft)
            if baseline == "pre_cue" and base_feat is not None and base_feat.shape == f_raw.shape:
                f_dir = (f_raw - base_feat).astype(np.float32, copy=False)
            else:
                f_dir = f_raw
            feats_move.append(f_raw)
            feats_dir.append(f_dir)
            y_move.append(1 if w.is_move else 0)
            if w.is_move:
                y_dir.append(DIR_MAP[w.cue])
            else:
                y_dir.append(-1)

    x_move = np.stack(feats_move, axis=0).astype(np.float32)
    x_dir = np.stack(feats_dir, axis=0).astype(np.float32)
    y_move_arr = np.asarray(y_move, dtype=np.int64)
    y_dir_arr = np.asarray(y_dir, dtype=np.int64)
    return Dataset(x_move=x_move, x_dir=x_dir, y_move=y_move_arr, y_dir=y_dir_arr)


def _print_cm(cm: np.ndarray, labels: list[str]) -> None:
    header = " " * 12 + " ".join([f"{l:>10s}" for l in labels])
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>10s}  " + " ".join([f"{int(v):>10d}" for v in row]))


def _split_by_chunk(chunks: list, *, seed: int, val_frac: float = 0.1, test_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(chunks))
    rng.shuffle(idx)
    n = len(chunks)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if n_test + n_val >= n:
        n_test = max(0, min(n_test, n - 1))
        n_val = max(0, min(n_val, n - 1 - n_test))
    test_idx = set(idx[:n_test].tolist())
    val_idx = set(idx[n_test : n_test + n_val].tolist())
    train_idx = set(idx[n_test + n_val :].tolist())
    return type("SplitLike", (), {
        "train": [c for i, c in enumerate(chunks) if i in train_idx],
        "val": [c for i, c in enumerate(chunks) if i in val_idx],
        "test": [c for i, c in enumerate(chunks) if i in test_idx],
    })()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ThoughtLink baseline intent decoder (numpy linear models).")
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "robot_control_data" / "data"))
    ap.add_argument("--max-chunks", type=int, default=0, help="If >0, only load first N chunks (for quick tests).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split",
        type=str,
        default="subject",
        choices=["subject", "session", "chunk"],
        help="How to split train/val/test. Use 'session' for per-user calibration.",
    )
    ap.add_argument(
        "--subject-id",
        action="append",
        default=[],
        help="Filter dataset to subject_id (repeatable, or comma-separated).",
    )
    ap.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Filter dataset to session_id (repeatable, or comma-separated).",
    )

    ap.add_argument("--window-s", type=float, default=0.5)
    ap.add_argument("--hop-s", type=float, default=0.1)
    ap.add_argument("--guard-s", type=float, default=0.2)
    ap.add_argument("--cue-start-s", type=float, default=3.0)
    ap.add_argument("--fs-hz", type=float, default=500.0)
    ap.add_argument("--no-fft", action="store_true", help="Use only time-domain features (faster).")
    ap.add_argument(
        "--feature-layout",
        type=str,
        default="dual",
        choices=["shared", "dual"],
        help="Feature pipeline. 'dual' uses raw features for stage1 and baseline-delta for stage2 (recommended).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="pre_cue",
        choices=["none", "pre_cue"],
        help="Feature baseline correction mode (per chunk). 'pre_cue' subtracts t=0..cue_start features.",
    )
    ap.add_argument(
        "--feature-mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "delta", "raw+delta"],
        help="(shared layout only) What features to feed models. 'auto' uses raw+delta when baseline=pre_cue else raw.",
    )

    ap.add_argument("--val-subjects", type=int, default=1)
    ap.add_argument("--test-subjects", type=int, default=1)
    ap.add_argument("--val-sessions", type=int, default=1)
    ap.add_argument("--test-sessions", type=int, default=1)

    ap.add_argument("--stage1-epochs", type=int, default=50)
    ap.add_argument("--stage1-lr", type=float, default=0.1)
    ap.add_argument("--stage1-l2", type=float, default=1e-3)
    ap.add_argument("--stage1-w-move", type=float, default=1.0, help="Class weight for MOVE (y=1) in stage1.")
    ap.add_argument("--stage1-w-rest", type=float, default=1.0, help="Class weight for REST (y=0) in stage1.")
    ap.add_argument(
        "--stage1-balance",
        action="store_true",
        help="Auto-balance stage1 class weights inversely by class frequency (overrides --stage1-w-*).",
    )

    ap.add_argument("--stage2-epochs", type=int, default=50)
    ap.add_argument("--stage2-lr", type=float, default=0.1)
    ap.add_argument("--stage2-l2", type=float, default=1e-3)

    ap.add_argument("--p-move", type=float, default=0.5, help="Move threshold for reporting pipeline accuracy.")
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "intent_baseline.npz"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = list(iter_npz_files(data_dir))
    if args.max_chunks and args.max_chunks > 0:
        paths = paths[: int(args.max_chunks)]
    if not paths:
        raise SystemExit(f"No .npz files found under: {data_dir}")

    print(f"[train] chunks={len(paths)} data_dir={data_dir}")
    t0 = time.perf_counter()
    chunks = [load_chunk(p) for p in paths]

    def _parse_multi(values: list[str]) -> set[str]:
        out: set[str] = set()
        for v in values:
            for item in str(v).split(","):
                item = item.strip()
                if item:
                    out.add(item)
        return out

    subject_filter = _parse_multi(args.subject_id)
    session_filter = _parse_multi(args.session_id)
    if subject_filter:
        chunks = [c for c in chunks if c.meta.subject_id in subject_filter]
    if session_filter:
        chunks = [c for c in chunks if c.meta.session_id in session_filter]
    if not chunks:
        raise SystemExit("No chunks left after applying --subject-id/--session-id filters.")

    subjects = sorted({c.meta.subject_id for c in chunks})
    sessions = sorted({c.meta.session_id for c in chunks})
    print(f"[train] subjects={len(subjects)} {subjects}")
    print(f"[train] sessions={len(sessions)}")

    split_kind = str(args.split)
    if split_kind == "subject":
        try:
            split = split_by_subject(
                chunks,
                SplitConfig(
                    val_subjects=int(args.val_subjects),
                    test_subjects=int(args.test_subjects),
                    seed=int(args.seed),
                ),
            )
        except ValueError as exc:
            # Useful when running a quick smoke with --max-chunks that only contains 1 subject.
            print(
                f"[train] WARNING: subject split failed ({exc}); falling back to chunk-level split for smoke testing."
            )
            split = _split_by_chunk(chunks, seed=int(args.seed))
    elif split_kind == "session":
        try:
            split = split_by_session(
                chunks,
                SessionSplitConfig(
                    val_sessions=int(args.val_sessions),
                    test_sessions=int(args.test_sessions),
                    seed=int(args.seed),
                ),
            )
        except ValueError as exc:
            print(f"[train] WARNING: session split failed ({exc}); falling back to chunk-level split.")
            split = _split_by_chunk(chunks, seed=int(args.seed))
    else:
        split = _split_by_chunk(chunks, seed=int(args.seed))
    print(
        f"[train] split: train_chunks={len(split.train)} val_chunks={len(split.val)} test_chunks={len(split.test)}"
    )

    win_cfg = WindowConfig(
        fs_hz=float(args.fs_hz),
        window_s=float(args.window_s),
        hop_s=float(args.hop_s),
        guard_s=float(args.guard_s),
        cue_start_s=float(args.cue_start_s),
    )
    include_fft = not bool(args.no_fft)
    feature_layout = str(args.feature_layout)
    baseline = str(args.baseline)
    feature_mode = str(args.feature_mode)
    if feature_layout == "shared":
        if feature_mode == "auto":
            feature_mode = "raw+delta" if baseline == "pre_cue" else "raw"
        if baseline != "pre_cue" and feature_mode == "raw+delta":
            raise SystemExit("feature_mode='raw+delta' requires --baseline pre_cue")
    else:
        # Dual layout does not use --feature-mode; stage1 uses raw, stage2 uses delta if available.
        feature_mode = "n/a"
    print(
        f"[train] window: fs={win_cfg.fs_hz} window_s={win_cfg.window_s} hop_s={win_cfg.hop_s} guard_s={win_cfg.guard_s} "
        f"include_fft={include_fft} layout={feature_layout} baseline={baseline} feature_mode={feature_mode}"
    )

    if feature_layout == "shared":
        ds_train = build_dataset_shared(
            split.train, win_cfg, include_fft=include_fft, baseline=baseline, feature_mode=feature_mode
        )
        ds_val = (
            build_dataset_shared(split.val, win_cfg, include_fft=include_fft, baseline=baseline, feature_mode=feature_mode)
            if split.val
            else None
        )
        ds_test = (
            build_dataset_shared(
                split.test, win_cfg, include_fft=include_fft, baseline=baseline, feature_mode=feature_mode
            )
            if split.test
            else None
        )
        feat_dim_move = int(ds_train.x_move.shape[1])
        feat_dim_dir = feat_dim_move
    else:
        ds_train = build_dataset_dual(split.train, win_cfg, include_fft=include_fft, baseline=baseline)
        ds_val = build_dataset_dual(split.val, win_cfg, include_fft=include_fft, baseline=baseline) if split.val else None
        ds_test = build_dataset_dual(split.test, win_cfg, include_fft=include_fft, baseline=baseline) if split.test else None
        feat_dim_move = int(ds_train.x_move.shape[1])
        feat_dim_dir = int(ds_train.x_dir.shape[1])

    if ds_val is None:
        print("[train] WARNING: empty val split; val metrics will be skipped.")
    if ds_test is None:
        print("[train] WARNING: empty test split; test metrics will be skipped.")
    print(
        f"[train] windows: train={ds_train.x_move.shape[0]} val={(0 if ds_val is None else ds_val.x_move.shape[0])} "
        f"test={(0 if ds_test is None else ds_test.x_move.shape[0])} feat_dim_move={feat_dim_move} feat_dim_dir={feat_dim_dir}"
    )

    scaler_move = fit_scaler(ds_train.x_move)
    xtr_move = scaler_move.transform(ds_train.x_move)
    xva_move = (
        scaler_move.transform(ds_val.x_move) if ds_val is not None else np.empty((0, xtr_move.shape[1]), dtype=np.float32)
    )
    xte_move = (
        scaler_move.transform(ds_test.x_move) if ds_test is not None else np.empty((0, xtr_move.shape[1]), dtype=np.float32)
    )

    mtr = ds_train.y_dir >= 0
    if feature_layout == "dual":
        scaler_dir = fit_scaler(ds_train.x_dir[mtr])
    else:
        scaler_dir = scaler_move
    xtr_dir = scaler_dir.transform(ds_train.x_dir)
    xva_dir = (
        scaler_dir.transform(ds_val.x_dir) if ds_val is not None else np.empty((0, xtr_dir.shape[1]), dtype=np.float32)
    )
    xte_dir = (
        scaler_dir.transform(ds_test.x_dir) if ds_test is not None else np.empty((0, xtr_dir.shape[1]), dtype=np.float32)
    )

    # Stage 1: move/rest
    w_pos = float(args.stage1_w_move)
    w_neg = float(args.stage1_w_rest)
    if bool(args.stage1_balance):
        n_pos = float(ds_train.y_move.sum())
        n_tot = float(ds_train.y_move.shape[0])
        n_neg = max(0.0, n_tot - n_pos)
        # Normalize so average weight is ~1.0 (keeps lr tuning stable).
        w_pos = (n_tot / (2.0 * n_pos)) if n_pos > 0 else 1.0
        w_neg = (n_tot / (2.0 * n_neg)) if n_neg > 0 else 1.0
    stage1 = train_binary_logreg(
        xtr_move,
        ds_train.y_move,
        lr=float(args.stage1_lr),
        epochs=int(args.stage1_epochs),
        l2=float(args.stage1_l2),
        w_pos=w_pos,
        w_neg=w_neg,
        seed=int(args.seed),
    )
    y1_tr = stage1.predict(xtr_move, threshold=float(args.p_move))
    y1_va = stage1.predict(xva_move, threshold=float(args.p_move)) if ds_val is not None else None
    y1_te = stage1.predict(xte_move, threshold=float(args.p_move)) if ds_test is not None else None
    val_acc = None if ds_val is None else accuracy(ds_val.y_move, y1_va)
    test_acc = None if ds_test is None else accuracy(ds_test.y_move, y1_te)
    print(
        f"[stage1] acc train={accuracy(ds_train.y_move, y1_tr):.4f} "
        f"val={'n/a' if val_acc is None else f'{val_acc:.4f}'} "
        f"test={'n/a' if test_acc is None else f'{test_acc:.4f}'}"
    )
    if ds_test is not None and y1_te is not None:
        cm1 = confusion_matrix(ds_test.y_move, y1_te, n_classes=2)
        print("[stage1] confusion_matrix (test)")
        _print_cm(cm1, labels=["REST", "MOVE"])

    # Stage 2: direction on move windows
    mte = np.zeros((0,), dtype=bool) if ds_test is None else (ds_test.y_dir >= 0)
    stage2 = train_softmax_reg(
        xtr_dir[mtr],
        ds_train.y_dir[mtr],
        n_classes=4,
        lr=float(args.stage2_lr),
        epochs=int(args.stage2_epochs),
        l2=float(args.stage2_l2),
        seed=int(args.seed),
    )
    if ds_test is not None and int(mte.sum()) > 0:
        y2_te = stage2.predict(xte_dir[mte])
        print(
            f"[stage2] acc (move windows only) test={accuracy(ds_test.y_dir[mte], y2_te):.4f} n={int(mte.sum())}"
        )
        cm2 = confusion_matrix(ds_test.y_dir[mte], y2_te, n_classes=4)
        print("[stage2] confusion_matrix (test, move windows only)")
        _print_cm(cm2, labels=["LEFT", "RIGHT", "FORWARD", "BACKWARD"])

    # Combined pipeline (0 = STOP/rest, 1..4 = dir)
    def pipeline_pred(x_move: np.ndarray, x_dir: np.ndarray) -> np.ndarray:
        move = stage1.predict(x_move, threshold=float(args.p_move)).astype(bool)
        dir_idx = stage2.predict(x_dir)
        out = np.zeros((x_move.shape[0],), dtype=np.int64)
        out[move] = dir_idx[move] + 1
        return out

    if ds_test is not None:
        ytrue_te = np.where(ds_test.y_dir >= 0, ds_test.y_dir + 1, 0)
        ypred_te = pipeline_pred(xte_move, xte_dir)
        print(f"[pipe] acc test={accuracy(ytrue_te, ypred_te):.4f}")
        cm_pipe = confusion_matrix(ytrue_te, ypred_te, n_classes=5)
        print("[pipe] confusion_matrix (test)")
        _print_cm(cm_pipe, labels=["STOP", "LEFT", "RIGHT", "FORWARD", "BACKWARD"])

    model = IntentModel(
        scaler=scaler_move,
        scaler_dir=None if feature_layout == "shared" else scaler_dir,
        stage1=stage1,
        stage2=stage2,
        fs_hz=win_cfg.fs_hz,
        window_s=win_cfg.window_s,
        hop_s=win_cfg.hop_s,
        guard_s=win_cfg.guard_s,
        cue_start_s=win_cfg.cue_start_s,
        baseline=baseline,
        include_fft=include_fft,
        feature_mode=("delta" if baseline == "pre_cue" else "raw") if feature_layout == "dual" else feature_mode,
        feature_mode_move=("raw" if feature_layout == "dual" else None),
        feature_mode_dir=("delta" if (feature_layout == "dual" and baseline == "pre_cue") else ("raw" if feature_layout == "dual" else None)),
    )
    out = Path(args.out)
    model.save_npz(out)
    print(f"[train] saved model: {out}")

    print(f"[train] done in {(time.perf_counter() - t0):.2f}s")


if __name__ == "__main__":
    main()
