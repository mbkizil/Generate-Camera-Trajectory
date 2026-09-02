"""The `orbit_track` primitive: revolve the camera around a target point.

The original DSL implemented orbiting around x/y/z with three separate,
hand-derived trig formulas (one per axis, "phase-shifted" by inspection). This
version uses one generic rotate-around-an-arbitrary-axis operation (via
`scipy`'s axis-angle `Rotation.from_rotvec`), parametrized by which axis to
use -- adding a fourth (or a genuinely arbitrary) axis later needs no new math.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from ..easing import Easing, apply_easing
from ..look_at import look_at_rotation
from ..trajectory import Trajectory
from .base import DUTCH_MARKS, SegmentBase, enum_param, param, to_param_dict

_DEGREES_MARKS = {30.0: "30", 45.0: "45", 60.0: "60", 90.0: "90", 180.0: "180", 270.0: "270", 360.0: "360"}
_SPIRAL_MARKS = {-1.0: "in", -0.5: "in_0.5", -0.1: "in_0.1", 0.0: "no", 0.1: "out_0.1", 0.5: "out_0.5", 1.0: "out"}


class OrbitAxis(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"


_AXIS_VECTORS = {
    OrbitAxis.X: np.array([1.0, 0.0, 0.0]),
    OrbitAxis.Y: np.array([0.0, 1.0, 0.0]),
    OrbitAxis.Z: np.array([0.0, 0.0, 1.0]),
}


class OrbitDirection(str, Enum):
    CW = "cw"
    CCW = "ccw"


@dataclass
class OrbitSegment(SegmentBase):
    """Revolve around the scene's anchor object ("the box") by `degrees`, on
    the plane perpendicular to `axis`.

    There's no fixed target point: the center is always wherever the box
    currently is at each frame (see `target_positions` in `build()`), even if
    it's moving during this same segment -- the orbit rides along with it.

    Radius is *not* a free parameter: frame 0 always matches the incoming
    `start_position` exactly (like every segment), so the orbit radius is
    whatever the current distance to the box already is -- chain a dolly (or
    just set the camera's starting position) to control it. `spiral` moves the
    radius smoothly during the orbit itself (negative = inward, positive =
    outward), generalizing the original DSL's `spiral_in/out_0.1..0.5` tiers.

    Orientation is a look-at toward the box every frame (plus a constant
    `dutch_deg` roll) -- target-relative primitives define orientation from
    geometry, not by continuing the incoming `start_rotation`.
    """

    frames: int = param(label="Frames", min=21, max=300, default=61, step=4, unit="frames")

    axis: OrbitAxis = enum_param(label="Orbit axis", enum_cls=OrbitAxis, default=OrbitAxis.Y)
    direction: OrbitDirection = enum_param(label="Direction", enum_cls=OrbitDirection, default=OrbitDirection.CW)
    degrees: float = param(label="Orbit degrees", min=1.0, max=720.0, default=90.0, marks=_DEGREES_MARKS, unit="deg")
    spiral: float = param(label="Spiral (- in / + out)", min=-1.0, max=1.0, default=0.0, marks=_SPIRAL_MARKS)
    dutch_deg: float = param(label="Dutch / roll", min=-45.0, max=45.0, default=0.0, marks=DUTCH_MARKS, unit="deg")

    easing: Easing = enum_param(label="Easing", enum_cls=Easing, default=Easing.LINEAR)
    easing_strength: float = param(
        label="Easing strength", min=0.0, max=1.0, default=0.5,
        marks={0.0: "off", 0.5: "default", 1.0: "full"},
    )

    def build(self, start_position: np.ndarray, start_rotation: Rotation, target_positions: np.ndarray) -> Trajectory:
        n_frames = self.frames
        s = np.linspace(0.0, 1.0, n_frames)
        s_eased = apply_easing(s, self.easing, self.easing_strength)

        radius_vector_0 = start_position - target_positions[0]
        axis_vector = _AXIS_VECTORS[OrbitAxis(self.axis)]
        sign = -1.0 if OrbitDirection(self.direction) is OrbitDirection.CW else 1.0

        theta = np.deg2rad(sign * self.degrees) * s_eased  # (n_frames,)
        delta_rotations = Rotation.from_rotvec(axis_vector[None, :] * theta[:, None])
        radius_vectors = delta_rotations.apply(radius_vector_0)  # rotate the fixed start vector, per frame

        radius_scale = 1.0 + self.spiral * s_eased
        positions = target_positions + radius_vectors * radius_scale[:, None]

        look_rotations = Rotation.concatenate(
            [look_at_rotation(p, t) for p, t in zip(positions, target_positions)]
        )
        dutch = Rotation.from_euler("z", self.dutch_deg, degrees=True)
        rotations = look_rotations * dutch

        return Trajectory(
            times=s * (n_frames - 1),
            positions=positions,
            rotations=rotations,
            metadata={"segment_type": "orbit", "params": to_param_dict(self)},
        )
