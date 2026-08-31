"""The `free_form` primitive: raw camera-local translate + rotate.

Needs no target/look-at -- pure dead-reckoning in the camera's own local axes.
This generalizes the original DSL's 7-level categorical translate per axis
(far_left/left/near_left/no/near_right/right/far_right, etc.) into a continuous
[-1, 1] slider, while keeping the old tiers as `marks` for reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..easing import Easing, apply_easing
from ..trajectory import Trajectory
from .base import SegmentBase, enum_param, param

_LATERAL_MARKS = {-1.0: "far_left", -2 / 3: "left", -1 / 3: "near_left", 0.0: "no",
                   1 / 3: "near_right", 2 / 3: "right", 1.0: "far_right"}
_VERTICAL_MARKS = {-1.0: "far_down", -2 / 3: "down", -1 / 3: "near_down", 0.0: "no",
                    1 / 3: "near_up", 2 / 3: "up", 1.0: "far_up"}
_DEPTH_MARKS = {-1.0: "far_in", -2 / 3: "in", -1 / 3: "near_in", 0.0: "no",
                1 / 3: "near_out", 2 / 3: "out", 1.0: "far_out"}


@dataclass
class FreeFormSegment(SegmentBase):
    """Translate along the camera's local right/up/forward axes and rotate by a
    relative yaw/pitch/roll, over `frames` frames.

    `lateral`/`vertical`/`depth` are continuous in [-1, 1] (scaled by
    `distance_scale`); `marks` on each preserve the original DSL's named tiers
    as reference points without limiting the value to only those tiers.
    Sign convention: lateral +1 = right, vertical +1 = up, depth +1 = away from
    the view direction ("out"), depth -1 = along the view direction ("in") --
    each maps directly onto the local +X / +Y / +Z axis respectively (forward
    is local -Z, so "out" being +Z needs no extra sign flip).
    """

    frames: int = param(label="Frames", min=21, max=300, default=61, step=4, unit="frames")

    lateral: float = param(label="Lateral (- left / + right)", min=-1.0, max=1.0, default=0.0, marks=_LATERAL_MARKS)
    vertical: float = param(label="Vertical (- down / + up)", min=-1.0, max=1.0, default=0.0, marks=_VERTICAL_MARKS)
    depth: float = param(label="Depth (- in / + out)", min=-1.0, max=1.0, default=0.0, marks=_DEPTH_MARKS)
    distance_scale: float = param(label="Distance scale", min=0.0, max=20.0, default=6.0, unit="units")

    yaw_deg: float = param(label="Yaw", min=-180.0, max=180.0, default=0.0, step=1.0, unit="deg")
    pitch_deg: float = param(label="Pitch", min=-180.0, max=180.0, default=0.0, step=1.0, unit="deg")
    roll_deg: float = param(label="Roll / dutch", min=-180.0, max=180.0, default=0.0, step=1.0, unit="deg")

    easing: Easing = enum_param(label="Easing", enum_cls=Easing, default=Easing.LINEAR)
    easing_strength: float = param(
        label="Easing strength", min=0.0, max=1.0, default=0.5,
        marks={0.0: "off", 0.5: "default", 1.0: "full"},
    )

    def build(self, start_position: np.ndarray, start_rotation: Rotation, target_positions: np.ndarray) -> Trajectory:
        del target_positions  # free_form needs no target; kept only for interface uniformity
        n_frames = self.frames
        s = np.linspace(0.0, 1.0, n_frames)  # linear normalized progress -> linear frame indices
        s_eased = apply_easing(s, self.easing, self.easing_strength)  # eased motion progress

        # Orientation: compose a progressively-larger *local* (body-frame) delta
        # rotation onto the start orientation. `start_rotation * delta` applies
        # `delta` in the camera's own axes, then re-expresses it in world space --
        # the standard way to add a relative turn on top of an existing pose.
        euler = np.stack([s_eased * self.yaw_deg, s_eased * self.pitch_deg, s_eased * self.roll_deg], axis=-1)
        rotations = start_rotation * Rotation.from_euler("YXZ", euler, degrees=True)

        # Position: integrate the local translation target through the *rotating*
        # frame (like a fly-cam), not a straight line fixed to the start
        # orientation -- so translating and turning in the same segment behaves
        # as expected (e.g. flying forward while panning curves the path).
        local_target = self.distance_scale * np.array([self.lateral, self.vertical, self.depth])
        local_deltas = np.diff(s_eased, prepend=0.0)[:, None] * local_target[None, :]
        rotation_per_step = Rotation.concatenate([start_rotation, rotations[:-1]])
        world_deltas = rotation_per_step.apply(local_deltas)
        positions = start_position + np.cumsum(world_deltas, axis=0)

        return Trajectory(
            times=s * (n_frames - 1),  # frame-index units; the app layer maps this to real seconds
            positions=positions,
            rotations=rotations,
            metadata={"segment_type": "free_form", "params": _params_dict(self)},
        )


def _params_dict(segment: FreeFormSegment) -> dict:
    d = asdict(segment)
    d["easing"] = Easing(segment.easing).value
    return d
