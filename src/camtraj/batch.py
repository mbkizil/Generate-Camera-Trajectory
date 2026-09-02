"""Randomized batch generation: sample many concrete trajectories from
*ranges* around a fixed skeleton (segment types + count, in order), rather
than the single pinned values `camtraj.recipe` deals with.

A "ranges" dict has the same shape as `to_param_dict()`'s output, except
every numeric `param()` leaf is a `(lo, hi)` pair instead of a single value,
and every `enum_param()` leaf is a list of allowed `.value` strings instead
of one. `default_ranges()` builds the "pinned" ranges dict for an existing
instance (every leaf collapsed to a single point/choice) -- the natural
starting point before a caller widens whichever fields should vary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .box_motion import BoxMotion
from .segments import SEGMENT_TYPES, SegmentBase, get_enum_specs, get_group_specs, get_param_specs, to_param_dict
from .sequence import sequence
from .trajectory import Trajectory

BATCH_RECIPE_VERSION = 1

_SEGMENT_TYPE_NAMES = {cls: name for name, cls in SEGMENT_TYPES.items()}


def default_ranges(instance) -> dict[str, Any]:
    """The "pinned" ranges dict for `instance` (a segment, `BoxMotion`, or a
    nested `group()` instance): every numeric param collapses to `(v, v)`,
    every enum param to `[v.value]` -- sampling from this dict always
    reproduces `instance` exactly, until a caller widens a range or adds an
    enum choice."""
    ranges: dict[str, Any] = {}
    for spec in get_param_specs(instance):
        v = getattr(instance, spec.name)
        ranges[spec.name] = (v, v)
    for spec in get_enum_specs(instance):
        v = getattr(instance, spec.name)
        ranges[spec.name] = [v.value if hasattr(v, "value") else v]
    for spec in get_group_specs(instance):
        ranges[spec.name] = default_ranges(getattr(instance, spec.name))
    return ranges


def sample_from_ranges(cls: type, ranges: dict[str, Any], rng: np.random.Generator):
    """Draw one concrete `cls` instance: uniform within `(lo, hi)` for each
    numeric param, `rng.choice` among the allowed values for each enum
    param, recursing into any nested `group()` field."""
    enum_classes = {spec.name: spec.enum_cls for spec in get_enum_specs(cls)}
    group_classes = {spec.name: spec.group_cls for spec in get_group_specs(cls)}
    kwargs: dict[str, Any] = {}
    for name, value in ranges.items():
        if name in group_classes:
            kwargs[name] = sample_from_ranges(group_classes[name], value, rng)
        elif name in enum_classes:
            kwargs[name] = enum_classes[name](rng.choice(list(value)))
        else:
            lo, hi = value
            # `frames` (the only param whose value must be a whole number --
            # np.linspace's `num` argument) is declared with int literals in
            # every segment file, so `lo`/`hi` are already plain ints here;
            # every other param uses float literals. Branching on that
            # preserves int-ness generically, with no per-field special case.
            if isinstance(lo, int) and isinstance(hi, int):
                kwargs[name] = int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))
            else:
                kwargs[name] = float(lo) if hi <= lo else float(rng.uniform(lo, hi))
    return cls(**kwargs)


def sample_recipe(
    segment_types: list[type[SegmentBase]],
    segment_ranges: list[dict[str, Any]],
    box_motion_ranges: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[list[SegmentBase], list[BoxMotion]]:
    """Draw one full set of concrete segments + box motions. The skeleton
    (segment types, and how many of them) is fixed; only field values vary."""
    segments = [sample_from_ranges(cls, ranges, rng) for cls, ranges in zip(segment_types, segment_ranges)]
    box_motions = [sample_from_ranges(BoxMotion, ranges, rng) for ranges in box_motion_ranges]
    return segments, box_motions


def generate_batch(
    segment_types: list[type[SegmentBase]],
    segment_ranges: list[dict[str, Any]],
    box_motion_ranges: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
    start_position: np.ndarray,
    start_rotation,
    box_start_position: np.ndarray,
) -> list[tuple[dict[str, Any], Trajectory]]:
    """Sample `n` independent draws (deterministic given `seed`), returning
    `(sampled_params, trajectory)` pairs -- `sampled_params` is exactly what
    a manifest entry needs to record what that draw actually chose."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        segments, box_motions = sample_recipe(segment_types, segment_ranges, box_motion_ranges, rng)
        trajectory = sequence(
            segments,
            box_motions,
            start_position=start_position,
            start_rotation=start_rotation,
            box_start_position=box_start_position,
        )
        params = {
            "segments": [{"type": _SEGMENT_TYPE_NAMES[type(seg)], "params": to_param_dict(seg)} for seg in segments],
            "box_motions": [to_param_dict(bm) for bm in box_motions],
        }
        results.append((params, trajectory))
    return results


def batch_ranges_to_dict(
    segment_types: list[type[SegmentBase]],
    segment_ranges: list[dict[str, Any]],
    box_motion_ranges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save-able form of a randomization setup: the skeleton's segment types
    (in order) plus every field's ranges, for both segments and box motions."""
    return {
        "camtraj_batch_recipe_version": BATCH_RECIPE_VERSION,
        "segments": [
            {"type": _SEGMENT_TYPE_NAMES[cls], "ranges": ranges} for cls, ranges in zip(segment_types, segment_ranges)
        ],
        "box_motions": list(box_motion_ranges),
    }


def batch_ranges_from_dict(data: dict[str, Any]) -> tuple[list[type[SegmentBase]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Inverse of `batch_ranges_to_dict`."""
    segment_types: list[type[SegmentBase]] = []
    segment_ranges: list[dict[str, Any]] = []
    for item in data["segments"]:
        segment_type = item["type"]
        if segment_type not in SEGMENT_TYPES:
            raise ValueError(f"Unknown segment type {segment_type!r} in batch recipe (known: {list(SEGMENT_TYPES)})")
        segment_types.append(SEGMENT_TYPES[segment_type])
        segment_ranges.append(item["ranges"])
    box_motion_ranges = list(data["box_motions"])
    return segment_types, segment_ranges, box_motion_ranges
