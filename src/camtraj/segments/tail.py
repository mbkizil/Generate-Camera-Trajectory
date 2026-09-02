"""The `tail_track` primitive: chase-cam that follows the box.

The original DSL's tail_track had a per-frame IIR "damping" filter driven by a
3-tier categorical follow_style (hard/soft/lazy). This version keeps the same
filter (it's a reasonable, simple model of "catching up") but exposes it as
one continuous `damping` slider, with the old tiers kept as marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from ..look_at import look_at_rotation
from ..trajectory import Trajectory
from .base import DUTCH_MARKS, SegmentBase, enum_param, param, to_param_dict

_DAMPING_MARKS = {1.0: "hard", 0.5: "soft", 0.15: "lazy"}
_AMP_MARKS = {0.0: "locked", 0.5: "0.5", 0.8: "0.8", 1.0: "normal", 1.2: "1.2", 1.5: "1.5"}


class LookMode(str, Enum):
    LOOK_AT_BOX = "look_at_box"
    KEEP_HEADING = "keep_heading"


@dataclass
class TailSegment(SegmentBase):
    """Follows the box, maintaining whatever relative offset the camera
    already had at the start of the segment (like a rigid tow-rod) -- frame 0
    always matches `start_position` exactly, same as every segment, so the
    offset is "whatever it currently is," not a separate free parameter.

    `damping` smooths how tightly the camera tracks the box's motion each
    frame (1.0 = rigid/instant, lower = catches up gradually -- generalizing
    the original DSL's hard/soft/lazy `follow_style`). `look_mode` toggles
    between looking at the box (default) and just facing the direction the
    box is heading (the original DSL's `dont_look`).

    `amp_x/y/z` scale how much of the box's own displacement (since this
    segment started) the camera actually follows, per axis: 1.0 = normal
    1:1 follow, 0.0 = locked (ignore the box's motion on that axis entirely),
    >1.0 = exaggerated. This subsumes the original DSL's separate
    `follow_axis` mask -- locking an axis is just `amp=0` on it, and arbitrary
    per-axis scaling is strictly more expressive than a x/y/z/full choice.
    """

    frames: int = param(label="Frames", min=21, max=300, default=61, step=4, unit="frames")
    damping: float = param(
        label="Damping (1=rigid, lower=laggier)", min=0.05, max=1.0, default=1.0, marks=_DAMPING_MARKS
    )
    amp_x: float = param(label="Amplitude X (0=locked)", min=0.0, max=1.5, default=1.0, marks=_AMP_MARKS)
    amp_y: float = param(label="Amplitude Y (0=locked)", min=0.0, max=1.5, default=1.0, marks=_AMP_MARKS)
    amp_z: float = param(label="Amplitude Z (0=locked)", min=0.0, max=1.5, default=1.0, marks=_AMP_MARKS)
    look_mode: LookMode = enum_param(label="Look", enum_cls=LookMode, default=LookMode.LOOK_AT_BOX)
    dutch_deg: float = param(label="Dutch / roll", min=-45.0, max=45.0, default=0.0, marks=DUTCH_MARKS, unit="deg")

    def build(self, start_position: np.ndarray, start_rotation: Rotation, target_positions: np.ndarray) -> Trajectory:
        n_frames = self.frames

        box_displacement = target_positions - target_positions[0]
        amp = np.array([self.amp_x, self.amp_y, self.amp_z])
        desired_positions = start_position[None, :] + amp[None, :] * box_displacement

        positions = np.empty((n_frames, 3))
        positions[0] = start_position
        for i in range(1, n_frames):
            positions[i] = positions[i - 1] + self.damping * (desired_positions[i] - positions[i - 1])

        if LookMode(self.look_mode) is LookMode.LOOK_AT_BOX:
            look_rotations = Rotation.concatenate(
                [look_at_rotation(p, t) for p, t in zip(positions, target_positions)]
            )
        else:
            heading = target_positions[-1] - target_positions[0]
            heading_norm = np.linalg.norm(heading)
            if heading_norm < 1e-9:
                look_rotations = Rotation.concatenate([start_rotation] * n_frames)
            else:
                aim_points = positions + (heading / heading_norm)[None, :]
                look_rotations = Rotation.concatenate(
                    [look_at_rotation(p, a) for p, a in zip(positions, aim_points)]
                )

        dutch = Rotation.from_euler("z", self.dutch_deg, degrees=True)
        rotations = look_rotations * dutch

        return Trajectory(
            times=np.arange(n_frames, dtype=np.float64),
            positions=positions,
            rotations=rotations,
            metadata={"segment_type": "tail", "params": to_param_dict(self)},
        )
