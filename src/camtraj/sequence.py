"""Chain segments into one continuous Trajectory."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .box_motion import BoxMotion
from .segments.base import SegmentBase
from .trajectory import Trajectory


def sequence(
    segments: list[SegmentBase],
    box_motions: list[BoxMotion | None] | None = None,
    *,
    start_position: np.ndarray | None = None,
    start_rotation: Rotation | None = None,
    box_start_position: np.ndarray | None = None,
) -> Trajectory:
    """Build each segment in turn, continuing from the previous segment's end
    pose, and concatenate into one Trajectory with continuous timestamps.

    Each segment's frame count is its own parameter (`seg.frames`), so segment
    length is freely adjustable without affecting any other segment.

    `box_motions`, if given, must have one entry per segment (`None` = the box
    stays put during that segment). The box's resulting position at each frame
    is threaded into every segment's `build()` as `target_positions` -- used by
    target-relative primitives like orbit, ignored by others -- and the full
    per-frame box path is attached to the result as `Trajectory.box_positions`.
    """
    if not segments:
        raise ValueError("sequence() requires at least one segment")
    if box_motions is not None and len(box_motions) != len(segments):
        raise ValueError("box_motions must have the same length as segments")
    box_motions = box_motions if box_motions is not None else [None] * len(segments)

    position = np.zeros(3) if start_position is None else np.asarray(start_position, dtype=np.float64)
    rotation = Rotation.identity() if start_rotation is None else start_rotation
    box_position = np.zeros(3) if box_start_position is None else np.asarray(box_start_position, dtype=np.float64)

    times_chunks: list[np.ndarray] = []
    position_chunks: list[np.ndarray] = []
    rotation_chunks: list[Rotation] = []
    box_chunks: list[np.ndarray] = []
    segment_metadata: list[dict] = []
    t_offset = 0.0

    for i, (seg, box_motion) in enumerate(zip(segments, box_motions)):
        n_frames = seg.frames
        if box_motion is not None:
            target_positions = box_motion.build(box_position, n_frames)
        else:
            target_positions = np.tile(box_position, (n_frames, 1))

        sub = seg.build(position, rotation, target_positions)

        t = sub.times + t_offset
        pos, rot, box_pos = sub.positions, sub.rotations, target_positions
        if i > 0:
            # sub's frame 0 duplicates the previous segment's last frame (by
            # SegmentBase.build's contract); drop it so timestamps stay strictly
            # increasing and each pose appears exactly once.
            t, pos, rot, box_pos = t[1:], pos[1:], rot[1:], box_pos[1:]

        times_chunks.append(t)
        position_chunks.append(pos)
        rotation_chunks.append(rot)
        box_chunks.append(box_pos)
        segment_metadata.append(sub.metadata)

        t_offset = float(t[-1]) if len(t) else t_offset
        position, rotation = sub.positions[-1], sub.rotations[-1]
        box_position = target_positions[-1]

    return Trajectory(
        times=np.concatenate(times_chunks),
        positions=np.concatenate(position_chunks, axis=0),
        rotations=Rotation.concatenate(rotation_chunks),
        box_positions=np.concatenate(box_chunks, axis=0),
        metadata={"segments": segment_metadata},
    )
