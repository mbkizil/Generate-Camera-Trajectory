"""Save/load a trajectory *recipe* -- the list of segment types, their
parameter values, and box motions that produce a `Trajectory` via
`camtraj.sequence.sequence` -- as opposed to `Trajectory` itself, which is
the baked numeric result (positions/rotations per frame).

A recipe is what you'd want to persist across sessions, share with a
teammate, or (eventually) use as the seed structure for randomized batch
generation -- none of which make sense on baked numeric output.
"""

from __future__ import annotations

from typing import Any

from .box_motion import BoxMotion
from .segments import SEGMENT_TYPES, SegmentBase, from_param_dict, to_param_dict

RECIPE_VERSION = 1

_SEGMENT_TYPE_NAMES = {cls: name for name, cls in SEGMENT_TYPES.items()}


def recipe_to_dict(segments: list[SegmentBase], box_motions: list[BoxMotion]) -> dict[str, Any]:
    """The full save-able recipe: every segment's type + params, and every
    segment's box motion, in the same order `sequence()` expects them."""
    if len(segments) != len(box_motions):
        raise ValueError(f"segments ({len(segments)}) and box_motions ({len(box_motions)}) must be the same length")
    return {
        "camtraj_recipe_version": RECIPE_VERSION,
        "segments": [{"type": _SEGMENT_TYPE_NAMES[type(seg)], "params": to_param_dict(seg)} for seg in segments],
        "box_motions": [to_param_dict(motion) for motion in box_motions],
    }


def recipe_from_dict(data: dict[str, Any]) -> tuple[list[SegmentBase], list[BoxMotion]]:
    """Inverse of `recipe_to_dict`."""
    segments = []
    for item in data["segments"]:
        segment_type = item["type"]
        if segment_type not in SEGMENT_TYPES:
            raise ValueError(f"Unknown segment type {segment_type!r} in recipe (known: {list(SEGMENT_TYPES)})")
        segments.append(from_param_dict(SEGMENT_TYPES[segment_type], item["params"]))
    box_motions = [from_param_dict(BoxMotion, item) for item in data["box_motions"]]
    return segments, box_motions
