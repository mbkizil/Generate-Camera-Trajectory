"""The `rotation_track` primitive: mostly-stationary camera that keeps
re-aiming at the box, optionally translating (truck/pedestal/dolly) and/or
pushing in/out along the current sightline while it does.

The original DSL supported two sequential world-space moves in two halves of
the shot; this version uses one continuous, optional world-space move for
simplicity (chain two rotation_track segments back to back for the
two-halves behavior).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from ..easing import Easing, apply_easing
from ..look_at import look_at_rotation
from ..trajectory import Trajectory
from .base import DUTCH_MARKS, SegmentBase, enum_param, optional_group, param

_PUSH_MARKS = {-0.5: "out_0.5", -0.3: "out_0.3", -0.1: "out_0.1", 0.0: "no", 0.1: "in_0.1", 0.3: "in_0.3", 0.5: "in_0.5"}
_WORLD_UP = np.array([0.0, 1.0, 0.0])


class RotAxis(str, Enum):
    PAN = "pan"  # yaw only -- camera stays level, only turns side to side
    TILT = "tilt"  # pitch only -- camera keeps its initial heading, only tilts up/down
    FULL = "full"  # normal look-at: both


@dataclass
class WorldMove:
    """A simple world-space translation over a segment -- truck/pedestal/dolly."""

    move_x: float = param(label="World move X (truck)", min=-10.0, max=10.0, default=0.0, unit="units")
    move_y: float = param(label="World move Y (pedestal)", min=-10.0, max=10.0, default=0.0, unit="units")
    move_z: float = param(label="World move Z (dolly)", min=-10.0, max=10.0, default=0.0, unit="units")


@dataclass
class RotationTrackSegment(SegmentBase):
    """Continuously re-aims at the box, optionally translating (a `WorldMove`,
    off by default -- toggle it on to truck/pedestal/dolly while re-aiming)
    and/or pushing in/out along the current sightline.

    `rot_axis` restricts which rotation axes are used to track the box:
    `pan` locks the camera level and only yaws; `tilt` freezes the camera's
    initial heading and only pitches; `full` (default) does both, like orbit.

    `push` moves the camera toward/away from the box along the current
    sightline, as a fraction of the current distance each frame (so it scales
    naturally rather than risking overshoot at varying distances) --
    generalizes the original DSL's `push_in/out_0.1..0.5`.
    """

    frames: int = param(label="Frames", min=21, max=300, default=61, step=4, unit="frames")

    world_move: WorldMove | None = optional_group(label="World movement", group_cls=WorldMove)
    push: float = param(label="Push (- out / + in)", min=-0.5, max=0.5, default=0.0, marks=_PUSH_MARKS)
    rot_axis: RotAxis = enum_param(label="Rotation axis", enum_cls=RotAxis, default=RotAxis.FULL)
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

        move = self.world_move or WorldMove(move_x=0.0, move_y=0.0, move_z=0.0)
        world_delta = np.array([move.move_x, move.move_y, move.move_z])
        base_positions = start_position[None, :] + s_eased[:, None] * world_delta[None, :]

        to_target = target_positions - base_positions
        distances = np.linalg.norm(to_target, axis=1, keepdims=True)
        directions = np.divide(to_target, distances, out=np.zeros_like(to_target), where=distances > 1e-9)
        push_amount = self.push * s_eased[:, None] * distances
        positions = base_positions + directions * push_amount

        forwards = _aim_forward_vectors(positions, target_positions, RotAxis(self.rot_axis))
        look_rotations = Rotation.concatenate(
            [look_at_rotation(p, p + f) for p, f in zip(positions, forwards)]
        )
        dutch = Rotation.from_euler("z", self.dutch_deg, degrees=True)
        rotations = look_rotations * dutch

        return Trajectory(
            times=s * (n_frames - 1),
            positions=positions,
            rotations=rotations,
            metadata={"segment_type": "rotation_track", "params": _params_dict(self)},
        )


def _aim_forward_vectors(positions: np.ndarray, target_positions: np.ndarray, rot_axis: RotAxis) -> np.ndarray:
    """Per-frame unit "look" direction, respecting `rot_axis`. Feeding
    `eye + forward` back into `look_at_rotation` as a synthetic target reuses
    its well-tested math instead of re-deriving a rotation-from-forward here."""
    to_target = target_positions - positions
    norms = np.linalg.norm(to_target, axis=1, keepdims=True)
    forward = np.divide(to_target, norms, out=np.zeros_like(to_target), where=norms > 1e-9)

    if rot_axis is RotAxis.FULL:
        return forward

    if rot_axis is RotAxis.PAN:
        flat = forward - (forward @ _WORLD_UP)[:, None] * _WORLD_UP[None, :]
        flat_norms = np.linalg.norm(flat, axis=1, keepdims=True)
        fallback = np.tile(np.array([1.0, 0.0, 0.0]), (len(flat), 1))
        return np.divide(flat, flat_norms, out=fallback, where=flat_norms > 1e-9)

    # TILT: freeze the initial (level) heading, only pitch tracks the target
    initial_flat = forward[0] - np.dot(forward[0], _WORLD_UP) * _WORLD_UP
    initial_flat_norm = np.linalg.norm(initial_flat)
    heading = initial_flat / initial_flat_norm if initial_flat_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    horizontal_extent = to_target @ heading
    vertical_extent = to_target @ _WORLD_UP
    aimed = horizontal_extent[:, None] * heading[None, :] + vertical_extent[:, None] * _WORLD_UP[None, :]
    aimed_norms = np.linalg.norm(aimed, axis=1, keepdims=True)
    fallback = np.tile(heading, (len(aimed), 1))
    return np.divide(aimed, aimed_norms, out=fallback, where=aimed_norms > 1e-9)


def _params_dict(segment: RotationTrackSegment) -> dict:
    d = asdict(segment)
    d["easing"] = Easing(segment.easing).value
    d["rot_axis"] = RotAxis(segment.rot_axis).value
    return d
