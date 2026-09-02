"""Randomized batch generation with the axes flipped from `camtraj.batch`:
there, segment/box-motion *values* vary (within ranges) around a fixed
sequence structure. Here, values are fixed -- each candidate is a verbatim,
fully-pinned `(segment, box_motion)` pair, a `Block` -- and what varies is
the *structure*: how many segments a generated trajectory chains, and which
blocks get picked, in what order.

Chaining arbitrary blocks in arbitrary order works "for free" because of
`SegmentBase`'s own continuity contract (`sequence()` always re-derives each
segment's motion relative to wherever the previous one actually ended) --
there's no extra bookkeeping needed here to keep a random reshuffling
geometrically continuous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .box_motion import BoxMotion
from .segments import SEGMENT_TYPES, SegmentBase, to_param_dict
from .sequence import sequence
from .trajectory import Trajectory

_SEGMENT_TYPE_NAMES = {cls: name for name, cls in SEGMENT_TYPES.items()}


@dataclass(frozen=True)
class Block:
    """One candidate `(segment, box_motion)` pair the structure sampler can
    place at any position in a generated sequence -- fully pinned, exactly
    as it appeared wherever it was loaded from (typically one segment of a
    `camtraj.recipe`)."""

    label: str
    segment: SegmentBase
    box_motion: BoxMotion


def sample_structure(
    pool: list[Block],
    length_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[list[SegmentBase], list[BoxMotion], list[str]]:
    """Draw one random sequence: a length uniform in `length_range`
    (inclusive), each position an independent draw *with replacement* from
    `pool` (so the same block can appear more than once, or not at all).
    Returns `(segments, box_motions, labels)` -- `labels` names which pool
    block produced each position, for manifest/preview transparency."""
    if not pool:
        raise ValueError("Pool is empty -- load at least one recipe first.")
    lo, hi = length_range
    n = int(rng.integers(lo, hi + 1))
    chosen = [pool[i] for i in rng.integers(0, len(pool), size=n)]
    return [b.segment for b in chosen], [b.box_motion for b in chosen], [b.label for b in chosen]


def generate_structure_batch(
    pool: list[Block],
    length_range: tuple[int, int],
    *,
    n: int,
    seed: int,
    start_position: np.ndarray,
    start_rotation,
    box_start_position: np.ndarray,
) -> list[tuple[dict[str, Any], Trajectory]]:
    """Sample `n` independent draws (deterministic given `seed`), returning
    `(sampled_params, trajectory)` pairs -- `sampled_params` records exactly
    which blocks (by label) and resulting segment/box-motion values each
    draw chained together, for a manifest entry."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        segments, box_motions, labels = sample_structure(pool, length_range, rng)
        trajectory = sequence(
            segments,
            box_motions,
            start_position=start_position,
            start_rotation=start_rotation,
            box_start_position=box_start_position,
        )
        params = {
            "block_labels": labels,
            "segments": [{"type": _SEGMENT_TYPE_NAMES[type(s)], "params": to_param_dict(s)} for s in segments],
            "box_motions": [to_param_dict(bm) for bm in box_motions],
        }
        results.append((params, trajectory))
    return results
