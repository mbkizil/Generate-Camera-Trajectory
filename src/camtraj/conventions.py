"""
Camera pose conventions, described generically instead of hand-tuned per pair.

A camera pose is always, internally, a camera-to-world rigid transform: a
world-space position (the camera center) plus a rotation that maps a vector
expressed in the camera's *local* axes to world space. What differs between
"OpenGL", "OpenCV", "COLMAP", etc. is only:

  1. Which local axis is "right" / "up" / "forward" (the camera's local basis).
  2. Whether poses are stored camera-to-world (c2w) or world-to-camera (w2c).

Everything else (matrix vs. quaternion, quaternion component order) is a pure
serialization detail handled at the export boundary via
`scipy.spatial.transform.Rotation`, never as raw, positionally-ordered floats.

This module models exactly those two axes of variation and derives conversions
between any two `Convention`s from that description, rather than writing a
separate hand-tuned sign-flip function per pair (which is how the reference
LAMP codebase ended up with four different, mutually-inconsistent axis hacks).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

PoseDirection = Literal["camera_to_world", "world_to_camera"]


class Axis(Enum):
    """A signed, axis-aligned unit direction in a local 3D basis."""

    PX = (1.0, 0.0, 0.0)
    NX = (-1.0, 0.0, 0.0)
    PY = (0.0, 1.0, 0.0)
    NY = (0.0, -1.0, 0.0)
    PZ = (0.0, 0.0, 1.0)
    NZ = (0.0, 0.0, -1.0)

    @property
    def vector(self) -> np.ndarray:
        return np.array(self.value, dtype=np.float64)


@dataclass(frozen=True)
class Convention:
    """Describes a camera's local-axis layout and pose storage direction.

    `right`/`up`/`forward` say which local axis a physical direction maps to.
    E.g. OpenGL/NeRF: right=+X, up=+Y, forward=-Z (camera looks down local -Z).
    `description` is a short, user-facing explanation (axes + c2w/w2c + who
    uses it) -- shown in the app when a user picks this convention to export.
    """

    name: str
    right: Axis
    up: Axis
    forward: Axis
    pose_direction: PoseDirection = "camera_to_world"
    description: str = ""

    def __post_init__(self) -> None:
        used = {tuple(np.abs(v.vector)) for v in (self.right, self.up, self.forward)}
        if len(used) != 3:
            raise ValueError(
                f"Convention {self.name!r}: right/up/forward must be three "
                f"distinct axes, got right={self.right}, up={self.up}, forward={self.forward}"
            )
        # For a physically valid right-handed camera rig, "forward" (the
        # viewing direction) is always the *opposite* of right x up (e.g.
        # OpenGL: right=+X, up=+Y, right x up=+Z, forward=-Z). This also
        # pins every registered convention to the same basis_matrix()
        # determinant sign, which `convert_pose` silently assumes -- getting
        # this wrong wouldn't raise, it'd quietly hand back a reflected
        # (mirror-image) rotation instead of a real one (scipy's
        # `Rotation.from_matrix` orthogonalizes via SVD rather than
        # rejecting a non-rotation matrix).
        cross = np.cross(self.right.vector, self.up.vector)
        if not np.allclose(self.forward.vector, -cross):
            raise ValueError(
                f"Convention {self.name!r}: not a valid right-handed camera basis -- "
                f"forward must equal -(right x up), got right={self.right}, up={self.up}, "
                f"forward={self.forward} (right x up={tuple(cross)})"
            )

    def basis_matrix(self) -> np.ndarray:
        """3x3 matrix whose columns are (right, up, forward) in this convention's own local xyz."""
        return np.stack([self.right.vector, self.up.vector, self.forward.vector], axis=1)


# --- Known conventions -------------------------------------------------------

OPENGL = Convention(
    "opengl",
    right=Axis.PX,
    up=Axis.PY,
    forward=Axis.NZ,
    pose_direction="camera_to_world",
    description=(
        "Right-handed, +Y up, camera-to-world. Camera looks down local -Z. "
        "Used by OpenGL, NeRF/instant-ngp, and most 3D-generation research code "
        "(also camtraj's own internal convention)."
    ),
)
"""Right-handed, +Y up, camera looks down local -Z. Used by OpenGL, NeRF, Blender's
camera object, and most 3D-generation research code. This is camtraj's canonical
internal convention — see `camtraj.trajectory.Trajectory`."""

BLENDER = Convention(
    "blender",
    right=Axis.PX,
    up=Axis.PY,
    forward=Axis.NZ,
    pose_direction="camera_to_world",
    description=(
        "Identical axes and direction to OpenGL (+Y up, c2w, camera looks down "
        "local -Z) -- listed separately since Blender users typically look for "
        "it by name, not by axis convention. Matches a Blender camera object's "
        "own world matrix directly."
    ),
)
"""Same math as OPENGL -- a discoverable alias for users exporting to Blender."""

OPENCV = Convention(
    "opencv",
    right=Axis.PX,
    up=Axis.NY,
    forward=Axis.PZ,
    pose_direction="camera_to_world",
    description=(
        "Right-handed, +Y down, camera-to-world. Camera looks down local +Z "
        "(the pinhole-camera convention used by OpenCV's projection math). "
        "Matches e.g. nerfstudio's 'opencv' camera_model."
    ),
)
"""Right-handed, +Y down, camera looks down local +Z (the pinhole-camera convention
used by OpenCV's projection math). Stored camera-to-world here; some pipelines
(e.g. nerfstudio's "opencv" camera_model) store exactly this."""

COLMAP = Convention(
    "colmap",
    right=Axis.PX,
    up=Axis.NY,
    forward=Axis.PZ,
    pose_direction="world_to_camera",
    description=(
        "Same local axes as OpenCV, but stored world-to-camera: X_cam = R @ "
        "X_world + t. Matches COLMAP's images.txt extrinsics "
        "(QW,QX,QY,QZ,TX,TY,TZ)."
    ),
)
"""Same local axes as OPENCV, but COLMAP natively stores world-to-camera extrinsics
(images.txt QW,QX,QY,QZ,TX,TY,TZ is a world-to-camera rotation+translation)."""

KNOWN_CONVENTIONS: dict[str, Convention] = {c.name: c for c in (OPENGL, BLENDER, OPENCV, COLMAP)}


def axis_change_matrix(from_convention: Convention, to_convention: Convention) -> np.ndarray:
    """3x3 matrix M such that `v_to_local = M @ v_from_local` for the same physical vector."""
    return to_convention.basis_matrix() @ from_convention.basis_matrix().T


def invert_pose(position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, Rotation]:
    """Flip a pose between camera-to-world and world-to-camera. Self-inverse:
    invert_pose(*invert_pose(p, r)) == (p, r)."""
    new_rotation = rotation.inv()
    new_position = -new_rotation.apply(position)
    return new_position, new_rotation


def convert_pose(
    position: np.ndarray,
    rotation: Rotation,
    from_convention: Convention,
    to_convention: Convention,
) -> tuple[np.ndarray, Rotation]:
    """Convert a pose (or a batch of poses, via a batched `Rotation`) between conventions.

    Handles axis relabeling and c2w/w2c direction as two independent steps, always
    pivoting through camera-to-world internally. `position`/`rotation` may be a
    single pose or a batch (scipy `Rotation` supports both transparently).
    """
    pos, rot = np.asarray(position, dtype=np.float64), rotation
    if from_convention.pose_direction != "camera_to_world":
        pos, rot = invert_pose(pos, rot)

    m = axis_change_matrix(from_convention, to_convention)
    rot = Rotation.from_matrix(rot.as_matrix() @ m)
    # `pos` is the camera center in world space; relabeling local axes doesn't move it.

    if to_convention.pose_direction != "camera_to_world":
        pos, rot = invert_pose(pos, rot)
    return pos, rot
