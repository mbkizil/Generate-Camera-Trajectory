"""Chain segments into one continuous Trajectory."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .segments.base import SegmentBase
from .trajectory import Trajectory


def sequence(
    segments: list[SegmentBase],
    *,
    start_position: np.ndarray | None = None,
    start_rotation: Rotation | None = None,
) -> Trajectory:
    """Build each segment in turn, continuing from the previous segment's end
    pose, and concatenate into one Trajectory with continuous timestamps.

    Each segment's frame count is its own parameter (`seg.frames`), so segment
    length is freely adjustable without affecting any other segment.
    """
    if not segments:
        raise ValueError("sequence() requires at least one segment")

    position = np.zeros(3) if start_position is None else np.asarray(start_position, dtype=np.float64)
    rotation = Rotation.identity() if start_rotation is None else start_rotation

    times_chunks: list[np.ndarray] = []
    position_chunks: list[np.ndarray] = []
    rotation_chunks: list[Rotation] = []
    segment_metadata: list[dict] = []
    t_offset = 0.0

    for i, seg in enumerate(segments):
        sub = seg.build(position, rotation)

        t = sub.times + t_offset
        pos, rot = sub.positions, sub.rotations
        if i > 0:
            # sub's frame 0 duplicates the previous segment's last frame (by
            # SegmentBase.build's contract); drop it so timestamps stay strictly
            # increasing and each pose appears exactly once.
            t, pos, rot = t[1:], pos[1:], rot[1:]

        times_chunks.append(t)
        position_chunks.append(pos)
        rotation_chunks.append(rot)
        segment_metadata.append(sub.metadata)

        t_offset = float(t[-1]) if len(t) else t_offset
        position, rotation = sub.positions[-1], sub.rotations[-1]

    return Trajectory(
        times=np.concatenate(times_chunks),
        positions=np.concatenate(position_chunks, axis=0),
        rotations=Rotation.concatenate(rotation_chunks),
        metadata={"segments": segment_metadata},
    )
