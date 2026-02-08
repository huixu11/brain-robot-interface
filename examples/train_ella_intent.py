from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thoughtlink.data import WindowConfig, iter_eeg_windows, iter_npz_files, load_chunk
from thoughtlink.ella import svd_basis, train_task_binary_via_basis, train_task_softmax_via_basis
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
class FeatureTable:
    x_move: np.ndarray  # (n, d)
    x_dir: np.ndarray  # (n, d)
    y_move: np.ndarray  # (n,) 0/1
    y_dir: np.ndarray  # (n,) -1 for rest else 0..3
    subj: np.ndarray  # (n,) int
    sess: np.ndarray  # (n,) int
    subj_ids: list[str]
    sess_ids: list[str]


def _build_feature_table(
    chunks: list,
    *,
    win_cfg: WindowConfig,
    include_fft: bool,
    baseline: str,
) -> FeatureTable:
    feats_move: list[np.ndarray] = []
    feats_dir: list[np.ndarray] = []
    y_move: list[int] = []
    y_dir: list[int] = []
    subj: list[int] = []
    sess: list[int] = []

    subj_ids = sorted({c.meta.subject_id for c in chunks})
    sess_ids = sorted({c.meta.session_id for c in chunks})
    subj_to_i = {s: i for i, s in enumerate(subj_ids)}
    sess_to_i = {s: i for i, s in enumerate(sess_ids)}

    for chunk in chunks:
        base_feat: np.ndarray | None = None
        if baseline == "pre_cue":
            base_end = int(round((win_cfg.cue_start_s - win_cfg.guard_s) * win_cfg.fs_hz))
            base_end = max(base_end, int(round(win_cfg.window_s * win_cfg.fs_hz)))
            base_end = min(base_end, int(chunk.eeg.shape[0]))
            if base_end > 0:
                base_feat = eeg_window_features(chunk.eeg[:base_end], fs_hz=win_cfg.fs_hz, include_fft=include_fft).astype(
                    np.float32
                )

        for w in iter_eeg_windows(chunk, win_cfg):
            f_raw = eeg_window_features(w.x, fs_hz=win_cfg.fs_hz, include_fft=include_fft)
            if baseline == "pre_cue" and base_feat is not None and base_feat.shape == f_raw.shape:
                f_dir = (f_raw - base_feat).astype(np.float32, copy=False)
            else:
                f_dir = f_raw.astype(np.float32, copy=False)

            feats_move.append(f_raw.astype(np.float32, copy=False))
            feats_dir.append(f_dir)
            y_move.append(1 if w.is_move else 0)
            if w.is_move:
                y_dir.append(DIR_MAP[w.cue])
            else:
                y_dir.append(-1)
            subj.append(subj_to_i[w.subject_id])
            sess.append(sess_to_i[w.session_id])

    x_move = np.stack(feats_move, axis=0).astype(np.float32, copy=False)
    x_dir = np.stack(feats_dir, axis=0).astype(np.float32, copy=False)
    return FeatureTable(
        x_move=x_move,
        x_dir=x_dir,
        y_move=np.asarray(y_move, dtype=np.int64),
        y_dir=np.asarray(y_dir, dtype=np.int64),
        subj=np.asarray(subj, dtype=np.int32),
        sess=np.asarray(sess, dtype=np.int32),
        subj_ids=subj_ids,
        sess_ids=sess_ids,
    )


def _choose_sessions(
    sess_ids: list[str],
    *,
    seed: int,
    eval_sessions: int,
    calib_sessions: int,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(int(seed))
    perm = list(sess_ids)
    rng.shuffle(perm)
    n = len(perm)
    n_eval = max(1, min(int(eval_sessions), n))
    eval_set = set(perm[:n_eval])
    if int(calib_sessions) == 0:
        calib_set = set(perm[n_eval:])
    else:
        n_cal = min(int(calib_sessions), max(0, n - n_eval))
        calib_set = set(perm[n_eval : n_eval + n_cal])
    if not calib_set:
        raise ValueError("Empty calibration session set (adjust --calib-sessions/--eval-sessions).")
    return eval_set, calib_set


def _print_cm(cm: np.ndarray, labels: list[str]) -> None:
    header = " " * 12 + " ".join([f"{l:>10s}" for l in labels])
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>10s}  " + " ".join([f"{int(v):>10d}" for v in row]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an ELLA-style multi-subject intent decoder (numpy).")
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "robot_control_data" / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-subject", type=str, required=True, help="Subject id to adapt/evaluate (task u).")
    ap.add_argument("--eval-sessions", type=int, default=1, help="How many sessions of target to hold out for eval.")
    ap.add_argument(
        "--calib-sessions",
        type=int,
        default=0,
        help="How many sessions of target to use for calibration (0 = all non-eval sessions).",
    )
    ap.add_argument("--basis-k-move", type=int, default=8)
    ap.add_argument("--basis-k-dir", type=int, default=8)
    ap.add_argument(
        "--basis-task",
        type=str,
        default="session",
        choices=["subject", "session"],
        help="What constitutes a task for learning the shared basis. 'session' captures drift better (recommended).",
    )

    ap.add_argument("--window-s", type=float, default=0.5)
    ap.add_argument("--hop-s", type=float, default=0.1)
    ap.add_argument("--guard-s", type=float, default=0.2)
    ap.add_argument("--cue-start-s", type=float, default=3.0)
    ap.add_argument("--fs-hz", type=float, default=500.0)
    ap.add_argument("--no-fft", action="store_true")
    ap.add_argument("--baseline", type=str, default="pre_cue", choices=["none", "pre_cue"])

    # Calibration training hyperparams (latent s).
    ap.add_argument("--stage1-epochs", type=int, default=50)
    ap.add_argument("--stage1-lr", type=float, default=0.1)
    ap.add_argument("--stage1-l2", type=float, default=1e-3)
    ap.add_argument("--stage1-w-move", type=float, default=1.0)
    ap.add_argument("--stage1-w-rest", type=float, default=1.0)

    ap.add_argument("--stage2-epochs", type=int, default=50)
    ap.add_argument("--stage2-lr", type=float, default=0.1)
    ap.add_argument("--stage2-l2", type=float, default=1e-3)

    # Basis subject models training hyperparams (full-space per-subject).
    ap.add_argument("--basis-epochs", type=int, default=30)
    ap.add_argument("--basis-lr", type=float, default=0.1)
    ap.add_argument("--basis-l2", type=float, default=1e-3)

    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = list(iter_npz_files(data_dir))
    if not paths:
        raise SystemExit(f"No .npz files found under: {data_dir}")

    t0 = time.perf_counter()
    chunks = [load_chunk(p) for p in paths]
    subjects = sorted({c.meta.subject_id for c in chunks})
    if str(args.target_subject) not in subjects:
        raise SystemExit(f"target-subject not found: {args.target_subject}. Available: {subjects}")
    print(f"[ella] chunks={len(chunks)} subjects={len(subjects)} target={args.target_subject}")

    # Prepare target vs basis subjects.
    target_subj = str(args.target_subject)
    basis_subjects = [s for s in subjects if s != target_subj]
    if not basis_subjects:
        raise SystemExit("Need at least 2 subjects to learn an ELLA basis.")
    print(f"[ella] basis_subjects={len(basis_subjects)} {basis_subjects}")

    # Build a feature table once for all subjects. We'll fit scalers on basis subjects only.
    win_cfg = WindowConfig(
        fs_hz=float(args.fs_hz),
        window_s=float(args.window_s),
        hop_s=float(args.hop_s),
        guard_s=float(args.guard_s),
        cue_start_s=float(args.cue_start_s),
    )
    include_fft = not bool(args.no_fft)
    baseline = str(args.baseline)

    table = _build_feature_table(chunks, win_cfg=win_cfg, include_fft=include_fft, baseline=baseline)
    subj_id_to_i = {s: i for i, s in enumerate(table.subj_ids)}

    target_i = subj_id_to_i[target_subj]
    basis_mask = table.subj != int(target_i)
    dir_mask_basis = basis_mask & (table.y_dir >= 0)

    scaler_move = fit_scaler(table.x_move[basis_mask])
    scaler_dir = fit_scaler(table.x_dir[dir_mask_basis])
    x_move_sc = scaler_move.transform(table.x_move)
    x_dir_sc = scaler_dir.transform(table.x_dir)

    # Train per-subject models for basis subjects (in the shared scaled space).
    w_move_list: list[np.ndarray] = []
    w_dir_list: list[np.ndarray] = []
    basis_task = str(args.basis_task)
    if basis_task == "subject":
        groups: list[tuple[str, np.ndarray]] = []
        for sid in basis_subjects:
            si = subj_id_to_i[sid]
            groups.append((sid, table.subj == int(si)))
    else:
        # Treat each recording session as a separate task (better captures session drift).
        groups = []
        for s in sorted(set(table.sess[basis_mask].tolist())):
            sess_name = table.sess_ids[int(s)]
            groups.append((sess_name, basis_mask & (table.sess == int(s))))

    print(f"[ella] basis_tasks={len(groups)} basis_task={basis_task!r}")

    for name, m in groups:
        x_m = x_move_sc[m]
        y_m = table.y_move[m]
        stage1 = train_binary_logreg(
            x_m,
            y_m,
            lr=float(args.basis_lr),
            epochs=int(args.basis_epochs),
            l2=float(args.basis_l2),
            w_pos=1.0,
            w_neg=1.0,
            seed=int(args.seed),
        )
        w_move_list.append(stage1.w.astype(np.float32, copy=False))

        md = m & (table.y_dir >= 0)
        x_d = x_dir_sc[md]
        y_d = table.y_dir[md]
        stage2 = train_softmax_reg(
            x_d,
            y_d,
            n_classes=4,
            lr=float(args.basis_lr),
            epochs=int(args.basis_epochs),
            l2=float(args.basis_l2),
            seed=int(args.seed),
        )
        # Use per-class weight vectors as samples for basis learning.
        for c in range(stage2.w.shape[0]):
            w_dir_list.append(stage2.w[c].astype(np.float32, copy=False))

    W_move = np.stack(w_move_list, axis=0)
    W_dir = np.stack(w_dir_list, axis=0)
    basis_move = svd_basis(W_move, k=int(args.basis_k_move), center=False)
    basis_dir = svd_basis(W_dir, k=int(args.basis_k_dir), center=False)
    print(f"[ella] learned bases: move_L={basis_move.L.shape} dir_L={basis_dir.L.shape}")

    # Choose eval vs calib sessions for the target subject.
    target_chunks = [c for c in chunks if c.meta.subject_id == target_subj]
    target_sessions = sorted({c.meta.session_id for c in target_chunks})
    eval_sess, calib_sess = _choose_sessions(
        target_sessions,
        seed=int(args.seed),
        eval_sessions=int(args.eval_sessions),
        calib_sessions=int(args.calib_sessions),
    )
    print(f"[ella] target sessions={len(target_sessions)} eval={sorted(eval_sess)} calib={sorted(calib_sess)}")

    # Build masks for target windows by session.
    eval_sess_i = {i for i, s in enumerate(table.sess_ids) if s in eval_sess}
    calib_sess_i = {i for i, s in enumerate(table.sess_ids) if s in calib_sess}
    target_mask = table.subj == int(target_i)
    calib_mask = target_mask & np.isin(table.sess, np.asarray(sorted(calib_sess_i), dtype=np.int32))
    eval_mask = target_mask & np.isin(table.sess, np.asarray(sorted(eval_sess_i), dtype=np.int32))
    if not np.any(calib_mask):
        raise SystemExit("No calibration windows found after filtering by sessions.")
    if not np.any(eval_mask):
        raise SystemExit("No eval windows found after filtering by sessions.")

    # Calibrate stage1/stage2 in the low-rank latent space.
    stage1_full, s_move = train_task_binary_via_basis(
        x=x_move_sc[calib_mask],
        y=table.y_move[calib_mask],
        basis=basis_move,
        lr=float(args.stage1_lr),
        epochs=int(args.stage1_epochs),
        l2=float(args.stage1_l2),
        w_pos=float(args.stage1_w_move),
        w_neg=float(args.stage1_w_rest),
        seed=int(args.seed),
    )
    md_cal = calib_mask & (table.y_dir >= 0)
    stage2_full, S_dir = train_task_softmax_via_basis(
        x=x_dir_sc[md_cal],
        y=table.y_dir[md_cal],
        n_classes=4,
        basis=basis_dir,
        lr=float(args.stage2_lr),
        epochs=int(args.stage2_epochs),
        l2=float(args.stage2_l2),
        seed=int(args.seed),
    )

    # Quick open-loop eval on target eval windows.
    p_move = stage1_full.predict_proba(x_move_sc[eval_mask])
    yhat_move = (p_move >= 0.5).astype(np.int64)
    y_move = table.y_move[eval_mask]
    acc1 = accuracy(y_move, yhat_move)
    cm1 = confusion_matrix(y_move, yhat_move, n_classes=2)
    print(f"[ella] stage1 acc eval={acc1:.4f} n={int(y_move.shape[0])} basis_k={basis_move.L.shape[1]}")
    print("[ella] stage1 confusion_matrix (eval)")
    _print_cm(cm1, ["REST", "MOVE"])

    md_ev = eval_mask & (table.y_dir >= 0)
    if np.any(md_ev):
        y_dir = table.y_dir[md_ev]
        yhat_dir = stage2_full.predict(x_dir_sc[md_ev])
        acc2 = accuracy(y_dir, yhat_dir)
        cm2 = confusion_matrix(y_dir, yhat_dir, n_classes=4)
        print(f"[ella] stage2 acc eval={acc2:.4f} n={int(y_dir.shape[0])} basis_k={basis_dir.L.shape[1]}")
        print("[ella] stage2 confusion_matrix (eval, move windows only)")
        _print_cm(cm2, ["LEFT", "RIGHT", "FORWARD", "BACKWARD"])
    else:
        print("[ella] WARNING: no move windows in eval split; skipping stage2 eval.")

    # Save an IntentModel that can be used by intent_policy.py / eval_closed_loop.py.
    out = str(args.out).strip()
    if not out:
        out = str(ROOT / "artifacts" / f"intent_ella_{target_subj}.npz")
    out_path = Path(out)
    model = IntentModel(
        scaler=scaler_move,
        scaler_dir=scaler_dir,
        stage1=stage1_full,
        stage2=stage2_full,
        fs_hz=float(win_cfg.fs_hz),
        window_s=float(win_cfg.window_s),
        hop_s=float(win_cfg.hop_s),
        guard_s=float(win_cfg.guard_s),
        cue_start_s=float(win_cfg.cue_start_s),
        baseline=baseline,
        include_fft=include_fft,
        feature_mode="delta" if baseline == "pre_cue" else "raw",
        feature_mode_move="raw",
        feature_mode_dir="delta" if baseline == "pre_cue" else "raw",
    )
    model.save_npz(out_path)
    print(f"[ella] saved model: {out_path}")
    print(f"[ella] done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
