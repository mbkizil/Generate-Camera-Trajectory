"""The canonical in-memory trajectory representation.

One data structure, one convention, used everywhere in this codebase. Position
and rotation are never passed around as raw, positionally-ordered floats (that's
how the reference LAMP codebase ended up with a quaternion order that silently
differed between two "interchangeable" scripts) — rotation is always a
`scipy.spatial.transform.Rotation` object, and callers only ever call
`.as_quat()` / `.as_matrix()` / `.as_euler()` explicitly, at the point they
actually need a raw array, with the destination's convention spelled out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Trajectory:
    """An ordered sequence of camera-to-world poses over time.

    Convention (fixed, canonical — see `camtraj.conventions.OPENGL`):
    right-handed, +Y up, camera looks down local -Z, camera-to-world.
    Use `camtraj.conventions.convert_pose` (typically via `camtraj.export`) to
    get poses in another convention — never re-derive axis flips ad hoc.

    Attributes:
        times: (N,) float64 seconds, non-decreasing, starts at 0 by convention.
        positions: (N, 3) float64 world-space camera centers.
        rotations: batched `Rotation` of length N, camera-to-world orientation.
        fps: informational nominal sample rate, if the trajectory was built at one.
        metadata: free-form provenance (which segment(s)/recipe/seed produced this).
    """

    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        n = len(self.times)
        if self.positions.shape != (n, 3):
            raise ValueError(f"positions must have shape ({n}, 3), got {self.positions.shape}")
        if _rotation_len(self.rotations) != n:
            raise ValueError(f"rotations must have length {n}, got {_rotation_len(self.rotations)}")
        if n > 1 and np.any(np.diff(self.times) < 0):
            raise ValueError("times must be non-decreasing")

    def __len__(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self) else 0.0

    def as_matrices(self) -> np.ndarray:
        """(N, 4, 4) camera-to-world matrices, canonical convention."""
        mats = np.tile(np.eye(4), (len(self), 1, 1))
        mats[:, :3, :3] = self.rotations.as_matrix()
        mats[:, :3, 3] = self.positions
        return mats

    def pose_at(self, t) -> tuple[np.ndarray, Rotation]:
        """Interpolate (position, rotation) at time(s) `t` (scalar or array), clamped to range."""
        t = np.clip(np.asarray(t, dtype=np.float64), self.times[0], self.times[-1])
        if len(self) == 1:
            single = np.ndim(t) == 0
            n = 1 if single else len(t)
            pos = np.repeat(self.positions, n, axis=0)
            rot = Rotation.concatenate([self.rotations] * n) if n > 1 else self.rotations
            return (pos[0], rot) if single else (pos, rot)
        pos = np.stack([np.interp(t, self.times, self.positions[:, i]) for i in range(3)], axis=-1)
        rot = Slerp(self.times, self.rotations)(t)
        return pos, rot

    def resample(self, *, times=None, n_frames: int | None = None, fps: float | None = None) -> "Trajectory":
        """Resample to new timestamps, decoupling motion *shape* from *sample rate*."""
        if times is None:
            if n_frames is not None:
                times = np.linspace(self.times[0], self.times[-1], n_frames)
            elif fps is not None:
                n = max(2, int(round(self.duration * fps)) + 1)
                times = np.linspace(self.times[0], self.times[-1], n)
            else:
                raise ValueError("resample() requires one of times, n_frames, fps")
        times = np.asarray(times, dtype=np.float64)
        positions, rotations = self.pose_at(times)
        return Trajectory(
            times=times,
            positions=positions,
            rotations=rotations,
            fps=fps if fps is not None else self.fps,
            metadata=dict(self.metadata),
        )


def _rotation_len(rotation: Rotation) -> int:
    try:
        return len(rotation)
    except TypeError:
        return 1
