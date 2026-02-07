from __future__ import annotations

from enum import Enum

from bri import Action


class CanonicalCue(str, Enum):
    LEFT_FIST = "LEFT_FIST"
    RIGHT_FIST = "RIGHT_FIST"
    BOTH_FISTS = "BOTH_FISTS"
    TONGUE_TAPPING = "TONGUE_TAPPING"
    RELAX = "RELAX"


def normalize_cue_label(raw: str) -> CanonicalCue:
    """Normalize dataset label strings to a canonical cue enum.

    The dataset README includes known typos (e.g. 'Left First', 'Both Firsts').
    This function is intentionally tolerant to those variants.
    """

    s = str(raw).strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())

    if not s:
        raise ValueError("Empty label string")

    if "tongue" in s:
        return CanonicalCue.TONGUE_TAPPING
    if "relax" in s:
        return CanonicalCue.RELAX
    if "both" in s and ("fist" in s or "first" in s):
        return CanonicalCue.BOTH_FISTS
    if "left" in s and ("fist" in s or "first" in s):
        return CanonicalCue.LEFT_FIST
    if "right" in s and ("fist" in s or "first" in s):
        return CanonicalCue.RIGHT_FIST

    raise ValueError(f"Unknown label string: {raw!r}")


def cue_to_action(cue: CanonicalCue) -> Action:
    """Map a canonical cue to the repo's Action space.

    Requirement_doc.md's example direction includes backward; the dataset does not have
    an explicit backward cue, so Tongue Tapping is used as a backward proxy.
    """

    if cue == CanonicalCue.LEFT_FIST:
        return Action.LEFT
    if cue == CanonicalCue.RIGHT_FIST:
        return Action.RIGHT
    if cue == CanonicalCue.BOTH_FISTS:
        return Action.FORWARD
    if cue == CanonicalCue.TONGUE_TAPPING:
        return Action.BACKWARD
    if cue == CanonicalCue.RELAX:
        return Action.STOP
    raise ValueError(f"Unhandled cue: {cue}")


def cue_is_rest(cue: CanonicalCue) -> bool:
    return cue == CanonicalCue.RELAX

