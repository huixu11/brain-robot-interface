"""ThoughtLink intent decoding layer (brain-signal -> high-level Action)."""

from .labels import CanonicalCue, cue_to_action, normalize_cue_label

__all__ = ["CanonicalCue", "cue_to_action", "normalize_cue_label"]

