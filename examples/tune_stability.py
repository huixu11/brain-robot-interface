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
    load_chunk,
    iter_npz_files,
    split_by_session,
    split_by_subject,
)
from thoughtlink.features import eeg_window_features
from thoughtlink.intent_model import IntentModel
from thoughtlink.labels import cue_is_rest, normalize_cue_label
from thoughtlink.stability import StabilityConfig


STOP = 0
LEFT = 1
RIGHT = 2
FORWARD = 3
BACKWARD = 4

ACTION_NAMES = {
    STOP: "STOP",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    FORWARD: "FORWARD",
    BACKWARD: "BACKWARD",
}


@dataclass(frozen=True)
class PreChunk:
    path: Path
    subject_id: str
    session_id: str
    cue_value: str
    duration_s: float
    valid: np.ndarray  # (T,) bool
    p_move: np.ndarray  # (T,) float32
    p_dir: np.ndarray  # (T, 4) float32


@dataclass(frozen=True)
class Summary:
    n_chunks: int
    n_move_chunks: int
    false_rate_global: float
    move_coverage_global: float
    trigger_rate: float
    release_rate: float
    onset_latency_mean: float | None
    onset_latency_p95: float | None
    release_latency_mean: float | None
    switches_per_min_mean: float


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


class FastStabilizer:
    """A fast, numpy-free replay of IntentStabilizer for parameter sweeps.

    This mirrors src/thoughtlink/stability.py semantics, but avoids per-tick numpy ops.
    """

    def __init__(self, cfg: StabilityConfig) -> None:
        self.cfg = cfg

        self.p_move_ema: float | None = None
        self.p_dir_ema: list[float] | None = None  # len 4

        self.in_move: bool = False
        self.on_count: int = 0
        self.off_count: int = 0

        self.dir_candidate: int | None = None
        self.dir_count: int = 0
        self.dir_current: int | None = None
        self.dir_off_count: int = 0

    @staticmethod
    def _idx_to_action(idx: int) -> int:
        if idx == 0:
            return LEFT
        if idx == 1:
            return RIGHT
        if idx == 2:
            return FORWARD
        if idx == 3:
            return BACKWARD
        raise ValueError(f"Bad direction idx: {idx}")

    @staticmethod
    def _best_and_margin(p: list[float]) -> tuple[int, float, float]:
        # Returns (best_idx, best_conf, margin=best-second_best).
        best_idx = 0
        best = p[0]
        second = -1.0
        for i in range(1, 4):
            v = p[i]
            if v > best:
                second = best
                best = v
                best_idx = i
            elif v > second:
                second = v
        margin = best - second if second >= 0.0 else best
        return best_idx, best, margin

    def step(self, *, p_move: float, p_dir: np.ndarray) -> int:
        # EWMA smoothing.
        a = float(self.cfg.ewma_alpha)
        if self.p_move_ema is None:
            self.p_move_ema = float(p_move)
        else:
            self.p_move_ema = a * float(p_move) + (1.0 - a) * float(self.p_move_ema)

        if self.p_dir_ema is None:
            self.p_dir_ema = [float(p_dir[0]), float(p_dir[1]), float(p_dir[2]), float(p_dir[3])]
        else:
            self.p_dir_ema[0] = a * float(p_dir[0]) + (1.0 - a) * float(self.p_dir_ema[0])
            self.p_dir_ema[1] = a * float(p_dir[1]) + (1.0 - a) * float(self.p_dir_ema[1])
            self.p_dir_ema[2] = a * float(p_dir[2]) + (1.0 - a) * float(self.p_dir_ema[2])
            self.p_dir_ema[3] = a * float(p_dir[3]) + (1.0 - a) * float(self.p_dir_ema[3])

        assert self.p_dir_ema is not None
        best_idx, best_conf, margin = self._best_and_margin(self.p_dir_ema)

        # Hysteresis + debounce for move/rest state.
        if not self.in_move:
            self.off_count = 0
            if float(self.p_move_ema) >= float(self.cfg.p_move_on):
                self.on_count += 1
            else:
                self.on_count = 0
            if self.on_count >= int(self.cfg.move_on_k):
                self.in_move = True
                self.on_count = 0
                self.dir_current = int(best_idx)
        else:
            self.on_count = 0
            if float(self.p_move_ema) <= float(self.cfg.p_move_off):
                self.off_count += 1
            else:
                self.off_count = 0
            if self.off_count >= int(self.cfg.move_off_k):
                self.in_move = False
                self.off_count = 0
                self.dir_candidate = None
                self.dir_count = 0
                self.dir_current = None
                self.dir_off_count = 0

        if not self.in_move:
            return STOP

        # Direction collapse handling (optional STOP inside MOVE).
        if best_conf < float(self.cfg.p_dir_off) or margin < float(self.cfg.dir_margin):
            self.dir_off_count += 1
        else:
            self.dir_off_count = 0
        if self.dir_off_count >= int(self.cfg.dir_off_k):
            if bool(self.cfg.stop_on_dir_uncertain):
                self.dir_current = None
                self.dir_candidate = None
                self.dir_count = 0
                return STOP
            self.dir_off_count = int(self.cfg.dir_off_k)

        # Direction debounce.
        if best_conf >= float(self.cfg.p_dir) and margin >= float(self.cfg.dir_margin):
            if self.dir_candidate == int(best_idx):
                self.dir_count += 1
            else:
                self.dir_candidate = int(best_idx)
                self.dir_count = 1
            if self.dir_count >= int(self.cfg.dir_k):
                self.dir_current = int(best_idx)
        else:
            self.dir_candidate = None
            self.dir_count = 0

        if self.dir_current is None:
            if not bool(self.cfg.stop_on_dir_uncertain):
                self.dir_current = int(best_idx)
            else:
                return STOP
        return self._idx_to_action(int(self.dir_current))


def _select_features(*, f_raw: np.ndarray, f_delta: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "raw":
        return f_raw
    if mode == "delta":
        return f_delta if f_delta is not None else f_raw
    if mode == "raw+delta":
        if f_delta is None:
            # Keep runtime robust even if baseline isn't available (should not happen in our recommended configs).
            return np.concatenate([f_raw, np.zeros_like(f_raw, dtype=np.float32)], axis=0).astype(np.float32, copy=False)
        return np.concatenate([f_raw, f_delta], axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown feature_mode: {mode!r}")


def precompute_probs(
    chunks: list,
    *,
    model: IntentModel,
    update_hz: float,
) -> tuple[list[PreChunk], np.ndarray]:
    fs = float(model.fs_hz)
    dt = 0.0 if update_hz <= 0 else 1.0 / float(update_hz)
    if dt <= 0:
        raise ValueError("--update-hz must be > 0 for tuning")

    # Tick times mirror eval_closed_loop.py: t = step*dt for step where t < 15.0.
    times = np.arange(0.0, 15.0, dt, dtype=np.float32)
    n_steps = int(times.shape[0])
    win_n = int(round(float(model.window_s) * fs))

    include_fft = bool(model.include_fft)
    feature_mode_move = str(model.feature_mode_move or model.feature_mode)
    feature_mode_dir = str(model.feature_mode_dir or model.feature_mode)

    out: list[PreChunk] = []
    for c in chunks:
        eeg = c.eeg
        eeg_len = int(eeg.shape[0])

        base_feat: np.ndarray | None = None
        if model.baseline == "pre_cue":
            base_end = int(round((float(model.cue_start_s) - float(model.guard_s)) * fs))
            base_end = max(base_end, win_n)
            base_end = min(base_end, eeg_len)
            if base_end > 0:
                base_feat = eeg_window_features(eeg[:base_end], fs_hz=fs, include_fft=include_fft)

        valid = np.zeros((n_steps,), dtype=np.bool_)
        p_move = np.zeros((n_steps,), dtype=np.float32)
        p_dir = np.full((n_steps, 4), 0.25, dtype=np.float32)

        for i in range(n_steps):
            t = float(times[i])
            end = int(round(t * fs))
            start = end - win_n
            if start < 0 or end <= 0 or end > eeg_len:
                continue
            valid[i] = True
            f_raw = eeg_window_features(eeg[start:end], fs_hz=fs, include_fft=include_fft)
            f_delta: np.ndarray | None = None
            if base_feat is not None and base_feat.shape == f_raw.shape:
                f_delta = (f_raw - base_feat).astype(np.float32, copy=False)
            x_move = _select_features(f_raw=f_raw, f_delta=f_delta, mode=feature_mode_move).reshape(1, -1).astype(
                np.float32, copy=False
            )
            x_dir = _select_features(f_raw=f_raw, f_delta=f_delta, mode=feature_mode_dir).reshape(1, -1).astype(
                np.float32, copy=False
            )
            p_move[i] = float(model.predict_move_proba(x_move)[0])
            p_dir[i] = model.predict_direction_proba(x_dir)[0].astype(np.float32, copy=False)

        meta = c.meta
        cue = normalize_cue_label(meta.label_raw)
        out.append(
            PreChunk(
                path=c.path,
                subject_id=meta.subject_id,
                session_id=meta.session_id,
                cue_value=cue.value,
                duration_s=float(meta.duration_s),
                valid=valid,
                p_move=p_move,
                p_dir=p_dir,
            )
        )
    return out, times


def eval_pre_chunks(
    pre: list[PreChunk],
    *,
    times: np.ndarray,
    cue_start_s: float,
    cfg: StabilityConfig,
) -> Summary:
    dt = float(times[1] - times[0]) if times.shape[0] >= 2 else 0.0

    sum_rest_total_s = 0.0
    sum_move_total_s = 0.0
    sum_false_rest_s = 0.0
    sum_pred_move_s = 0.0

    n_move_chunks = 0
    n_triggered = 0
    n_released = 0

    onset_lat: list[float] = []
    release_lat: list[float] = []
    switches_pm: list[float] = []

    for c in pre:
        cue = normalize_cue_label(c.cue_value)
        cue_end_s = min(15.0, cue_start_s + max(0.0, float(c.duration_s)))

        move_start = float(cue_start_s)
        move_end = float(cue_end_s) if not cue_is_rest(cue) else float(cue_start_s)
        move_total_s = max(0.0, move_end - move_start)
        rest_total_s = max(0.0, 15.0 - move_total_s)

        sum_move_total_s += move_total_s
        sum_rest_total_s += rest_total_s

        stab = FastStabilizer(cfg)

        t_prev: float | None = None
        action_prev: int | None = None
        last_action: int | None = None

        action_switches = 0
        pred_nonstop_move_s = 0.0
        pred_nonstop_rest_s = 0.0

        onset_t: float | None = None
        release_t: float | None = None
        triggered_in_move = False

        # Iterate steps, plus a final step at t=15 to attribute the last segment.
        n_steps = int(times.shape[0])
        for step in range(n_steps + 1):
            t_raw = 15.0 if step == n_steps else float(times[step])
            t_sim = min(15.0, t_raw)

            if t_prev is not None and action_prev is not None:
                seg_s = t_sim - t_prev
                if seg_s > 0:
                    move_s = _overlap_s(t_prev, t_sim, move_start, move_end)
                    rest_s = seg_s - move_s
                    if action_prev != STOP:
                        pred_nonstop_move_s += move_s
                        pred_nonstop_rest_s += rest_s

            if step == n_steps:
                break

            if not bool(c.valid[step]):
                action = STOP
            else:
                action = stab.step(p_move=float(c.p_move[step]), p_dir=c.p_dir[step])

            if action != last_action:
                if last_action is not None:
                    action_switches += 1
                last_action = action

            if action != STOP and (move_start <= t_sim <= move_end) and not cue_is_rest(cue):
                triggered_in_move = True
                if onset_t is None:
                    onset_t = t_sim
            if triggered_in_move and release_t is None and (t_sim >= move_end) and action == STOP:
                release_t = t_sim

            t_prev = t_sim
            action_prev = action

        sum_false_rest_s += pred_nonstop_rest_s
        sum_pred_move_s += pred_nonstop_move_s

        switches_pm.append(float(action_switches) * 60.0 / 15.0)

        if move_total_s > 0:
            n_move_chunks += 1
            if onset_t is not None:
                n_triggered += 1
                onset_lat.append(float(onset_t - move_start))
            if release_t is not None:
                n_released += 1
                release_lat.append(float(release_t - move_end))

    false_rate_global = (sum_false_rest_s / sum_rest_total_s) if sum_rest_total_s > 0 else 0.0
    move_cov_global = (sum_pred_move_s / sum_move_total_s) if sum_move_total_s > 0 else 0.0
    trigger_rate = (float(n_triggered) / float(n_move_chunks)) if n_move_chunks else 0.0
    release_rate = (float(n_released) / float(n_move_chunks)) if n_move_chunks else 0.0

    onset_mean = float(np.mean(onset_lat)) if onset_lat else None
    onset_p95 = float(np.percentile(np.asarray(onset_lat, dtype=np.float32), 95)) if onset_lat else None
    release_mean = float(np.mean(release_lat)) if release_lat else None
    switches_mean = float(np.mean(switches_pm)) if switches_pm else 0.0

    return Summary(
        n_chunks=int(len(pre)),
        n_move_chunks=int(n_move_chunks),
        false_rate_global=float(false_rate_global),
        move_coverage_global=float(move_cov_global),
        trigger_rate=float(trigger_rate),
        release_rate=float(release_rate),
        onset_latency_mean=onset_mean,
        onset_latency_p95=onset_p95,
        release_latency_mean=release_mean,
        switches_per_min_mean=switches_mean,
    )


def _fmt_optional(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _print_summary(tag: str, s: Summary) -> None:
    print(
        f"[{tag}] false_rate_global={s.false_rate_global:.3f} move_coverage_global={s.move_coverage_global:.3f} "
        f"trigger_rate={s.trigger_rate:.3f} release_rate={s.release_rate:.3f} "
        f"onset_mean={_fmt_optional(s.onset_latency_mean)} onset_p95={_fmt_optional(s.onset_latency_p95)} "
        f"switches_per_min_mean={s.switches_per_min_mean:.1f}"
    )


def _sample_cfg(rng: np.random.Generator, *, allow_hold_dir: bool) -> StabilityConfig:
    ewma_alpha = float(rng.choice([0.1, 0.15, 0.2, 0.25, 0.3]))
    p_move_on = float(rng.choice([0.55, 0.6, 0.62, 0.65, 0.68, 0.7, 0.72, 0.75]))
    p_move_off = float(rng.choice([0.3, 0.35, 0.4, 0.45, 0.5, 0.55]))
    if p_move_off >= p_move_on:
        p_move_off = max(0.0, p_move_on - 0.15)
    move_on_k = int(rng.choice([2, 3, 5, 7]))
    move_off_k = int(rng.choice([2, 3, 5, 7]))

    # Direction knobs: keep a reasonable, safety-oriented envelope.
    p_dir = float(rng.choice([0.25, 0.3, 0.35, 0.4, 0.5]))
    p_dir_off = float(rng.choice([0.2, 0.25, 0.3, 0.35, 0.4]))
    if p_dir_off > p_dir:
        p_dir_off = max(0.0, p_dir - 0.05)
    dir_k = int(rng.choice([3, 5]))
    dir_off_k = int(rng.choice([2, 5, 10]))
    dir_margin = float(rng.choice([0.03, 0.06, 0.1]))
    stop_on_dir_uncertain = True if not allow_hold_dir else bool(rng.choice([True, False]))

    return StabilityConfig(
        ewma_alpha=ewma_alpha,
        p_move_on=p_move_on,
        p_move_off=p_move_off,
        move_on_k=move_on_k,
        move_off_k=move_off_k,
        p_dir=p_dir,
        p_dir_off=p_dir_off,
        dir_k=dir_k,
        dir_off_k=dir_off_k,
        dir_margin=dir_margin,
        stop_on_dir_uncertain=stop_on_dir_uncertain,
    )


def _seed_candidates() -> list[StabilityConfig]:
    # Include a few deterministic starting points so the tuner never returns a degenerate
    # "all STOP" solution just due to unlucky random sampling.
    return [
        StabilityConfig(),  # current defaults used by eval_closed_loop.py / intent_policy.py
        StabilityConfig(
            ewma_alpha=0.3,
            p_move_on=0.55,
            p_move_off=0.45,
            move_on_k=2,
            move_off_k=3,
            p_dir=0.35,
            p_dir_off=0.25,
            dir_k=3,
            dir_off_k=2,
            dir_margin=0.03,
            stop_on_dir_uncertain=True,
        ),
        StabilityConfig(
            ewma_alpha=0.15,
            p_move_on=0.68,
            p_move_off=0.50,
            move_on_k=5,
            move_off_k=3,
            p_dir=0.40,
            p_dir_off=0.35,
            dir_k=5,
            dir_off_k=2,
            dir_margin=0.06,
            stop_on_dir_uncertain=True,
        ),
    ]


def _cfg_to_args(cfg: StabilityConfig) -> str:
    # Emit args compatible with eval_closed_loop.py / intent_policy.py.
    args = [
        f"--ewma-alpha {cfg.ewma_alpha}",
        f"--p-move-on {cfg.p_move_on}",
        f"--p-move-off {cfg.p_move_off}",
        f"--move-on-k {cfg.move_on_k}",
        f"--move-off-k {cfg.move_off_k}",
        f"--p-dir {cfg.p_dir}",
        f"--p-dir-off {cfg.p_dir_off}",
        f"--dir-k {cfg.dir_k}",
        f"--dir-off-k {cfg.dir_off_k}",
        f"--dir-margin {cfg.dir_margin}",
        ("--stop-on-dir-uncertain" if cfg.stop_on_dir_uncertain else "--no-stop-on-dir-uncertain"),
    ]
    return " ".join(args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-tune StabilityConfig on a split using batch closed-loop replay.")
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "robot_control_data" / "data"))
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--split", type=str, default="session", choices=["subject", "session", "chunk", "all"])
    ap.add_argument("--subject-id", action="append", default=[], help="Filter to subject_id (repeatable, or comma-separated).")
    ap.add_argument("--session-id", action="append", default=[], help="Filter to session_id (repeatable, or comma-separated).")
    ap.add_argument("--val-subjects", type=int, default=1)
    ap.add_argument("--test-subjects", type=int, default=1)
    ap.add_argument("--val-sessions", type=int, default=1)
    ap.add_argument("--test-sessions", type=int, default=1)

    ap.add_argument("--model", type=str, required=True, help="Path to IntentModel .npz")
    ap.add_argument("--update-hz", type=float, default=50.0)
    ap.add_argument("--target-false-rate", type=float, default=0.05)
    ap.add_argument("--min-trigger-rate", type=float, default=0.3)
    ap.add_argument("--min-move-coverage", type=float, default=0.05)
    ap.add_argument(
        "--allow-hold-dir",
        action="store_true",
        help="Allow stop_on_dir_uncertain=False candidates (can improve coverage but often increases false triggers).",
    )
    ap.add_argument("--max-evals", type=int, default=200, help="Number of random candidates to try.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--objective", type=str, default="val_only", choices=["val_only", "robust"], help="robust scores using max(false_rate) over val+test.")
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
    elif split_kind == "subject":
        split = split_by_subject(
            chunks,
            SplitConfig(val_subjects=int(args.val_subjects), test_subjects=int(args.test_subjects), seed=int(args.seed)),
        )
    elif split_kind == "session":
        split = split_by_session(
            chunks,
            SessionSplitConfig(val_sessions=int(args.val_sessions), test_sessions=int(args.test_sessions), seed=int(args.seed)),
        )
    else:
        # Chunk-level split: tune on "val" and report "test".
        rng = np.random.default_rng(int(args.seed))
        idx = np.arange(len(chunks))
        rng.shuffle(idx)
        n = len(chunks)
        n_test = max(1, int(round(n * 0.1)))
        n_val = max(1, int(round(n * 0.1)))
        test_idx = set(idx[:n_test].tolist())
        val_idx = set(idx[n_test : n_test + n_val].tolist())
        split = type(
            "SplitLike",
            (),
            {
                "train": [c for i, c in enumerate(chunks) if i not in test_idx and i not in val_idx],
                "val": [c for i, c in enumerate(chunks) if i in val_idx],
                "test": [c for i, c in enumerate(chunks) if i in test_idx],
            },
        )()

    tune_chunks = list(split.val) if split.val else list(split.train)
    report_chunks = list(split.test) if split.test else []
    if not tune_chunks:
        raise SystemExit("Empty tuning split (val/train).")

    model = IntentModel.load_npz(Path(args.model))
    cue_start_s = float(model.cue_start_s)

    print(f"[tune] split={split_kind} tune_chunks={len(tune_chunks)} report_chunks={len(report_chunks)} update_hz={float(args.update_hz):.2f}")
    print(f"[tune] model={args.model} baseline={model.baseline!r} include_fft={bool(model.include_fft)} feature_mode_move={model.feature_mode_move!r} feature_mode_dir={model.feature_mode_dir!r}")
    print(f"[tune] objective={args.objective} target_false_rate={float(args.target_false_rate):.3f}")

    t0 = time.perf_counter()
    pre_tune, times = precompute_probs(tune_chunks, model=model, update_hz=float(args.update_hz))
    pre_report: list[PreChunk] | None = None
    if report_chunks:
        pre_report, _ = precompute_probs(report_chunks, model=model, update_hz=float(args.update_hz))
    print(f"[tune] precompute done in {time.perf_counter() - t0:.2f}s (ticks={int(times.shape[0])})")

    rng = np.random.default_rng(int(args.seed))
    best_cfg: StabilityConfig | None = None
    best_score: tuple | None = None
    top: list[tuple[tuple, StabilityConfig, Summary, Summary | None]] = []

    def _score(val_s: Summary, test_s: Summary | None) -> tuple:
        # Lexicographic: primarily satisfy false-trigger target, then maximize coverage,
        # while discouraging degenerate "never trigger" configs via trigger_penalty.
        target = float(args.target_false_rate)
        if str(args.objective) == "robust" and test_s is not None:
            false = max(val_s.false_rate_global, test_s.false_rate_global)
            cov = min(val_s.move_coverage_global, test_s.move_coverage_global)
            switches = max(val_s.switches_per_min_mean, test_s.switches_per_min_mean)
            onset = max((val_s.onset_latency_p95 or 99.0), (test_s.onset_latency_p95 or 99.0))
        else:
            false = val_s.false_rate_global
            cov = val_s.move_coverage_global
            switches = val_s.switches_per_min_mean
            onset = val_s.onset_latency_p95 or 99.0

        false_penalty = max(0.0, float(false) - target)
        trig_penalty = max(0.0, float(args.min_trigger_rate) - float(val_s.trigger_rate))
        cov_penalty = max(0.0, float(args.min_move_coverage) - float(val_s.move_coverage_global))

        # Smaller is better; invert coverage so higher coverage sorts earlier.
        return (false_penalty, trig_penalty, cov_penalty, -float(cov), float(switches), float(onset))

    objective = str(args.objective)

    candidates: list[StabilityConfig] = []
    candidates.extend(_seed_candidates())
    for _ in range(int(args.max_evals)):
        candidates.append(_sample_cfg(rng, allow_hold_dir=bool(args.allow_hold_dir)))

    for i, cfg in enumerate(candidates):
        val_s = eval_pre_chunks(pre_tune, times=times, cue_start_s=cue_start_s, cfg=cfg)
        test_s: Summary | None = None
        if pre_report is not None and objective == "robust":
            test_s = eval_pre_chunks(pre_report, times=times, cue_start_s=cue_start_s, cfg=cfg)
        sc = _score(val_s, test_s)

        if best_score is None or sc < best_score:
            best_score = sc
            best_cfg = cfg

        top.append((sc, cfg, val_s, test_s))

    # Sort and print top-k.
    top.sort(key=lambda x: x[0])
    k = max(1, int(args.top_k))
    print(f"[tune] tried {len(top)} candidates. Top {min(k, len(top))}:")
    for rank, (sc, cfg, val_s, test_s) in enumerate(top[:k], start=1):
        print(f"[tune] #{rank:02d} cfg: {_cfg_to_args(cfg)}")
        _print_summary("val", val_s)
        if test_s is not None:
            _print_summary("test", test_s)

    assert best_cfg is not None
    print("[tune] best_cfg (copy/paste into eval_closed_loop.py / intent_policy.py):")
    print(_cfg_to_args(best_cfg))

    if pre_report is not None and objective != "robust":
        best_test = eval_pre_chunks(pre_report, times=times, cue_start_s=cue_start_s, cfg=best_cfg)
        _print_summary("test", best_test)

    # Print reproducible eval commands.
    base_parts: list[str] = [f"python examples\\eval_closed_loop.py --split {split_kind}"]
    for sid in sorted(subject_filter):
        base_parts.append(f"--subject-id {sid}")
    for sid in sorted(session_filter):
        base_parts.append(f"--session-id {sid}")
    base_cmd = " ".join(base_parts).strip()
    if split_kind == "subject":
        base_cmd += f" --val-subjects {int(args.val_subjects)} --test-subjects {int(args.test_subjects)}"
    if split_kind == "session":
        base_cmd += f" --val-sessions {int(args.val_sessions)} --test-sessions {int(args.test_sessions)}"
    base_cmd += f" --mode model --model {args.model} --update-hz {float(args.update_hz)}"

    print("[tune] reproduce on val:")
    print(f"{base_cmd} --subset val {_cfg_to_args(best_cfg)}")
    if report_chunks:
        print("[tune] reproduce on test:")
        print(f"{base_cmd} --subset test {_cfg_to_args(best_cfg)}")


if __name__ == "__main__":
    main()
