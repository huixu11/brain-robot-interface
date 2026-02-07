"""ThoughtLink intent decoding layer (brain-signal -> high-level Action)."""

from .data import (
    Chunk,
    ChunkMeta,
    EegWindow,
    Split,
    WindowConfig,
    iter_eeg_windows,
    load_chunk,
    split_by_subject,
)
from .labels import CanonicalCue, cue_to_action, normalize_cue_label
from .features import eeg_window_features

__all__ = [
    "CanonicalCue",
    "cue_to_action",
    "normalize_cue_label",
    "Chunk",
    "ChunkMeta",
    "EegWindow",
    "Split",
    "WindowConfig",
    "iter_eeg_windows",
    "load_chunk",
    "split_by_subject",
    "eeg_window_features",
]
