from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bri import Action
from thoughtlink.data import (
    SessionSplitConfig,
    SplitConfig,
    load_chunk,
    iter_npz_files,
    split_by_session,
    split_by_subject,
)
from thoughtlink.features import eeg_window_features
from thoughtlink.intent_model import IntentModel
from thoughtlink.labels import cue_is_rest, cue_to_action, normalize_cue_label
from thoughtlink.stability import IntentStabilizer, StabilityConfig


@dataclass(frozen=True)
class ChunkMetrics:
    path: Path
    subject_id: str
    session_id: str
    cue: str
    duration_s: float
    move_total_s: float
    rest_total_s: float
    pred_nonstop_move_s: float
    pred_nonstop_rest_s: float
    onset_latency_s: float | None
    release_latency_s: float | None
    triggered_in_move: bool
    released_after_move: bool
    action_switches: int
    dir_switches: int
    stop_toggles: int

    @property
    def false_rate(self) -> float:
        if self.rest_total_s <= 0:
            return 0.0
        return float(self.pred_nonstop_rest_s) / float(self.rest_total_s)

    @property
    def move_coverage(self) -> float:
        if self.move_total_s <= 0:
            return 0.0
        return float(self.pred_nonstop_move_s) / float(self.move_total_s)

    @property
    def switches_per_min(self) -> float:
        # Each chunk is 15 seconds.
        return float(self.action_switches) * 60.0 / 15.0


def _parse_multi(values: list[str]) -> set[str]:
    out: set[str] = set()
    for v in values:
        for item in str(v).split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


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
    return type(
        "SplitLike",
        (),
        {
            "train": [c for i, c in enumerate(chunks) if i in train_idx],
            "val": [c for i, c in enumerate(chunks) if i in val_idx],
            "test": [c for i, c in enumerate(chunks) if i in test_idx],
        },
    )()


def eval_chunk_closed_loop(
    *,
    eeg: np.ndarray,
    subject_id: str,
    session_id: str,
    raw_label: str,
    duration_s: float,
    update_hz: float,
    mode: str,
    model: IntentModel | None,
    include_fft: bool,
    feature_mode_move: str,
    feature_mode_dir: str,
    stabilizer_cfg: StabilityConfig,
) -> tuple[ChunkMetrics, np.ndarray]:
    """Evaluate one 15s chunk in a deterministic closed-loop replay.

    Returns (metrics, inference_ms_array).
    """

    cue = normalize_cue_label(raw_label)
    move_action = cue_to_action(cue)

    cue_start_s = float(model.cue_start_s) if model is not None else 3.0
    cue_end_s = min(15.0, cue_start_s + max(0.0, float(duration_s)))

    move_start = cue_start_s
    move_end = cue_end_s if not cue_is_rest(cue) else cue_start_s
    move_total_s = max(0.0, move_end - move_start)
    rest_total_s = max(0.0, 15.0 - move_total_s)

    dt = 0.0 if update_hz <= 0 else 1.0 / float(update_hz)

    base_feat: np.ndarray | None = None
    if mode == "model" and model is not None and model.baseline == "pre_cue":
        fs = float(model.fs_hz)
        base_end = int(round((float(model.cue_start_s) - float(model.guard_s)) * fs))
        base_end = max(base_end, int(round(float(model.window_s) * fs)))
        base_end = min(base_end, int(eeg.shape[0]))
        if base_end > 0:
            base_feat = eeg_window_features(eeg[:base_end], fs_hz=fs, include_fft=include_fft)

    stabilizer = IntentStabilizer(stabilizer_cfg)

    t_prev: float | None = None
    action_prev: Action | None = None
    last_action: Action | None = None

    action_switches = 0
    dir_switches = 0
    stop_toggles = 0
    pred_nonstop_move_s = 0.0
    pred_nonstop_rest_s = 0.0
    onset_t: float | None = None
    release_t: float | None = None
    triggered_in_move = False

    infer_ms: list[float] = []

    # Run deterministic ticks; attribute elapsed time to previously-issued action.
    step = 0
    while True:
        t_sim_raw = step * dt if dt > 0 else float(step)
        t_sim = min(15.0, t_sim_raw)

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

        if mode == "oracle":
            if cue_start_s <= t_sim <= cue_end_s and not cue_is_rest(cue):
                action = move_action
            else:
                action = Action.STOP
        else:
            assert model is not None
            fs = float(model.fs_hz)
            win_n = int(round(float(model.window_s) * fs))
            end = int(round(t_sim * fs))
            start_i = end - win_n
            if start_i < 0 or end <= 0 or end > int(eeg.shape[0]):
                action = Action.STOP
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
                infer_ms.append((time.perf_counter() - t_inf0) * 1000.0)

        if action != last_action:
            if last_action is not None:
                action_switches += 1
                if (last_action == Action.STOP) != (action == Action.STOP):
                    stop_toggles += 1
                if last_action != Action.STOP and action != Action.STOP and last_action != action:
                    dir_switches += 1
            last_action = action

        if action != Action.STOP and (move_start <= t_sim <= move_end) and not cue_is_rest(cue):
            triggered_in_move = True
            if onset_t is None:
                onset_t = t_sim
        if triggered_in_move and release_t is None and (t_sim >= move_end) and action == Action.STOP:
            release_t = t_sim

        t_prev = t_sim
        action_prev = action
        step += 1

    onset_latency_s: float | None = None
    if move_total_s > 0 and onset_t is not None:
        onset_latency_s = onset_t - move_start

    release_latency_s: float | None = None
    if move_total_s > 0 and release_t is not None:
        release_latency_s = release_t - move_end

    metrics = ChunkMetrics(
        path=Path("<in-mem>"),
        subject_id=subject_id,
        session_id=session_id,
        cue=cue.value,
        duration_s=float(duration_s),
        move_total_s=float(move_total_s),
        rest_total_s=float(rest_total_s),
        pred_nonstop_move_s=float(pred_nonstop_move_s),
        pred_nonstop_rest_s=float(pred_nonstop_rest_s),
        onset_latency_s=onset_latency_s,
        release_latency_s=release_latency_s,
        triggered_in_move=bool(triggered_in_move),
        released_after_move=bool(release_t is not None),
        action_switches=int(action_switches),
        dir_switches=int(dir_switches),
        stop_toggles=int(stop_toggles),
    )

    return metrics, np.asarray(infer_ms, dtype=np.float32)


def _summ(x: np.ndarray) -> str:
    if x.size == 0:
        return "n/a"
    p50 = float(np.percentile(x, 50))
    p95 = float(np.percentile(x, 95))
    return f"mean={float(x.mean()):.3f} p50={p50:.3f} p95={p95:.3f} n={int(x.size)}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch closed-loop evaluator (no MuJoCo).")
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "robot_control_data" / "data"))
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--split", type=str, default="subject", choices=["all", "subject", "session", "chunk"])
    ap.add_argument("--subset", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--subject-id", action="append", default=[], help="Filter dataset to subject_id (repeatable, or comma-separated).")
    ap.add_argument("--session-id", action="append", default=[], help="Filter dataset to session_id (repeatable, or comma-separated).")
    ap.add_argument("--val-subjects", type=int, default=1)
    ap.add_argument("--test-subjects", type=int, default=1)
    ap.add_argument("--val-sessions", type=int, default=1)
    ap.add_argument("--test-sessions", type=int, default=1)

    ap.add_argument("--mode", type=str, default="model", choices=["oracle", "model"])
    ap.add_argument("--model", type=str, default=str(ROOT / "artifacts" / "intent_baseline.npz"))
    ap.add_argument("--update-hz", type=float, default=10.0)

    # Stability knobs (same as intent_policy.py).
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

    ap.add_argument("--print-every", type=int, default=0, help="If >0, print per-chunk metrics every N chunks.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = list(iter_npz_files(data_dir))
    if args.max_chunks and args.max_chunks > 0:
        paths = paths[: int(args.max_chunks)]
    if not paths:
        raise SystemExit(f"No .npz files found under: {data_dir}")

    chunks = [load_chunk(p) for p in paths]
    subject_filter = _parse_multi(args.subject_id)
    session_filter = _parse_multi(args.session_id)
    if subject_filter:
        chunks = [c for c in chunks if c.meta.subject_id in subject_filter]
    if session_filter:
        chunks = [c for c in chunks if c.meta.session_id in session_filter]
    if not chunks:
        raise SystemExit("No chunks left after applying --subject-id/--session-id filters.")

    split_kind = str(args.split)
    if split_kind == "all":
        split = type("SplitLike", (), {"train": [], "val": [], "test": chunks})()
        subset = "test"
    elif split_kind == "subject":
        split = split_by_subject(
            chunks,
            SplitConfig(
                val_subjects=int(args.val_subjects),
                test_subjects=int(args.test_subjects),
                seed=int(args.seed),
            ),
        )
        subset = str(args.subset)
    elif split_kind == "session":
        split = split_by_session(
            chunks,
            SessionSplitConfig(
                val_sessions=int(args.val_sessions),
                test_sessions=int(args.test_sessions),
                seed=int(args.seed),
            ),
        )
        subset = str(args.subset)
    else:
        split = _split_by_chunk(chunks, seed=int(args.seed))
        subset = str(args.subset)

    if subset == "train":
        eval_chunks = list(split.train)
    elif subset == "val":
        eval_chunks = list(split.val)
    elif subset == "test":
        eval_chunks = list(split.test)
    else:
        eval_chunks = list(split.train) + list(split.val) + list(split.test)

    if not eval_chunks:
        raise SystemExit(f"Empty eval split: split={split_kind} subset={subset}")

    model: IntentModel | None = None
    include_fft = True
    feature_mode = "raw"
    feature_mode_move = "raw"
    feature_mode_dir = "raw"
    if str(args.mode) == "model":
        model = IntentModel.load_npz(Path(args.model))
        include_fft = bool(model.include_fft)
        feature_mode = str(model.feature_mode)
        feature_mode_move = str(model.feature_mode_move or model.feature_mode)
        feature_mode_dir = str(model.feature_mode_dir or model.feature_mode)

    stabilizer_cfg = StabilityConfig(
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
    )

    print(f"[eval] split={split_kind} subset={subset} chunks={len(eval_chunks)} update_hz={float(args.update_hz):.2f}")
    if subject_filter:
        print(f"[eval] subject_filter={sorted(subject_filter)}")
    if session_filter:
        print(f"[eval] session_filter={sorted(session_filter)}")
    if model is not None:
        print(
            f"[eval] model={args.model} feat_dim={int(model.stage1.w.shape[0])} include_fft={include_fft} "
            f"baseline={model.baseline!r} feature_mode={feature_mode!r} "
            f"feature_mode_move={feature_mode_move!r} feature_mode_dir={feature_mode_dir!r}"
        )
    print(
        "[eval] stabilizer "
        f"ewma_alpha={stabilizer_cfg.ewma_alpha} p_move_on/off={stabilizer_cfg.p_move_on}/{stabilizer_cfg.p_move_off} "
        f"move_on/off_k={stabilizer_cfg.move_on_k}/{stabilizer_cfg.move_off_k} "
        f"p_dir/off={stabilizer_cfg.p_dir}/{stabilizer_cfg.p_dir_off} dir_k/off_k={stabilizer_cfg.dir_k}/{stabilizer_cfg.dir_off_k} "
        f"dir_margin={stabilizer_cfg.dir_margin}"
    )

    # Aggregate, time-weighted denominators.
    sum_rest_total_s = 0.0
    sum_move_total_s = 0.0
    sum_false_rest_s = 0.0
    sum_pred_move_s = 0.0

    # Per-chunk distributions.
    false_rates: list[float] = []
    move_covers: list[float] = []
    switches_per_min: list[float] = []
    onset_lat: list[float] = []
    release_lat: list[float] = []

    n_move_chunks = 0
    n_triggered = 0
    n_released = 0
    n_no_trigger = 0
    n_no_release = 0

    infer_ms_all: list[np.ndarray] = []

    t0 = time.perf_counter()
    for i, c in enumerate(eval_chunks):
        m, inf_ms = eval_chunk_closed_loop(
            eeg=c.eeg,
            subject_id=c.meta.subject_id,
            session_id=c.meta.session_id,
            raw_label=c.meta.label_raw,
            duration_s=c.meta.duration_s,
            update_hz=float(args.update_hz),
            mode=str(args.mode),
            model=model,
            include_fft=include_fft,
            feature_mode_move=feature_mode_move,
            feature_mode_dir=feature_mode_dir,
            stabilizer_cfg=stabilizer_cfg,
        )
        m = ChunkMetrics(
            path=c.path,
            subject_id=m.subject_id,
            session_id=m.session_id,
            cue=m.cue,
            duration_s=m.duration_s,
            move_total_s=m.move_total_s,
            rest_total_s=m.rest_total_s,
            pred_nonstop_move_s=m.pred_nonstop_move_s,
            pred_nonstop_rest_s=m.pred_nonstop_rest_s,
            onset_latency_s=m.onset_latency_s,
            release_latency_s=m.release_latency_s,
            triggered_in_move=m.triggered_in_move,
            released_after_move=m.released_after_move,
            action_switches=m.action_switches,
            dir_switches=m.dir_switches,
            stop_toggles=m.stop_toggles,
        )

        sum_rest_total_s += m.rest_total_s
        sum_move_total_s += m.move_total_s
        sum_false_rest_s += m.pred_nonstop_rest_s
        sum_pred_move_s += m.pred_nonstop_move_s

        false_rates.append(m.false_rate)
        move_covers.append(m.move_coverage)
        switches_per_min.append(m.switches_per_min)

        if m.move_total_s > 0:
            n_move_chunks += 1
            if m.triggered_in_move:
                n_triggered += 1
                if m.onset_latency_s is not None:
                    onset_lat.append(float(m.onset_latency_s))
            else:
                n_no_trigger += 1

            if m.released_after_move:
                n_released += 1
                if m.release_latency_s is not None:
                    release_lat.append(float(m.release_latency_s))
            else:
                n_no_release += 1

        if inf_ms.size:
            infer_ms_all.append(inf_ms)

        pe = int(args.print_every)
        if pe > 0 and ((i + 1) % pe == 0 or (i + 1) == len(eval_chunks)):
            print(
                f"[eval] i={i+1:04d}/{len(eval_chunks)} cue={m.cue} "
                f"false_rate={m.false_rate:.3f} move_coverage={m.move_coverage:.3f} switches_per_min={m.switches_per_min:.1f}"
            )

    elapsed = time.perf_counter() - t0

    false_rate_global = (sum_false_rest_s / sum_rest_total_s) if sum_rest_total_s > 0 else 0.0
    move_cov_global = (sum_pred_move_s / sum_move_total_s) if sum_move_total_s > 0 else 0.0

    print(f"[eval] done in {elapsed:.2f}s")
    print(
        f"[eval] false_rate_global={false_rate_global:.3f} (false_nonstop_rest_s={sum_false_rest_s:.2f} / rest_total_s={sum_rest_total_s:.2f})"
    )
    print(
        f"[eval] move_coverage_global={move_cov_global:.3f} (nonstop_move_s={sum_pred_move_s:.2f} / move_total_s={sum_move_total_s:.2f})"
    )
    print(f"[eval] false_rate_per_chunk: {_summ(np.asarray(false_rates, dtype=np.float32))}")
    if sum_move_total_s > 0:
        print(f"[eval] move_coverage_per_chunk: {_summ(np.asarray(move_covers, dtype=np.float32))}")

    if n_move_chunks:
        trig_rate = float(n_triggered) / float(n_move_chunks)
        rel_rate = float(n_released) / float(n_move_chunks)
        print(
            f"[eval] move_chunks={n_move_chunks} trigger_rate={trig_rate:.3f} release_rate={rel_rate:.3f} "
            f"no_trigger={n_no_trigger} no_release={n_no_release}"
        )
        print(f"[eval] onset_latency_s: {_summ(np.asarray(onset_lat, dtype=np.float32))}")
        print(f"[eval] release_latency_s: {_summ(np.asarray(release_lat, dtype=np.float32))}")

    print(f"[eval] switches_per_min: {_summ(np.asarray(switches_per_min, dtype=np.float32))}")

    if infer_ms_all:
        all_ms = np.concatenate(infer_ms_all, axis=0).astype(np.float32, copy=False)
        p95 = float(np.percentile(all_ms, 95))
        print(f"[eval] inference_ms mean={float(all_ms.mean()):.2f} p95={p95:.2f} n={int(all_ms.size)}")


if __name__ == "__main__":
    main()
