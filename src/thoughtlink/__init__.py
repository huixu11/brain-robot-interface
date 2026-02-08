"""ThoughtLink intent decoding layer (brain-signal -> high-level Action)."""

from .data import (
    Chunk,
    ChunkMeta,
    EegWindow,
    SessionSplitConfig,
    Split,
    SplitConfig,
    WindowConfig,
    iter_eeg_windows,
    load_chunk,
    split_by_session,
    split_by_subject,
)
from .labels import CanonicalCue, cue_to_action, normalize_cue_label
from .features import eeg_window_features
from .linear import (
    BinaryLogReg,
    SoftmaxReg,
    StandardScaler,
    fit_scaler,
    train_binary_logreg,
    train_softmax_reg,
)
from .metrics import accuracy, confusion_matrix
from .intent_model import DIR_LABELS, IntentModel
from .stability import IntentStabilizer, StabilityConfig

__all__ = [
    "CanonicalCue",
    "cue_to_action",
    "normalize_cue_label",
    "Chunk",
    "ChunkMeta",
    "EegWindow",
    "SessionSplitConfig",
    "Split",
    "SplitConfig",
    "WindowConfig",
    "iter_eeg_windows",
    "load_chunk",
    "split_by_session",
    "split_by_subject",
    "eeg_window_features",
    "BinaryLogReg",
    "SoftmaxReg",
    "StandardScaler",
    "fit_scaler",
    "train_binary_logreg",
    "train_softmax_reg",
    "accuracy",
    "confusion_matrix",
    "DIR_LABELS",
    "IntentModel",
    "StabilityConfig",
    "IntentStabilizer",
]
