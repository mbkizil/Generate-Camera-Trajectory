"""Shared viser scene-building helpers, reused by every app in apps/.

viser's camera frustums are explicitly documented to follow OpenCV's
[+Z forward, +X right, +Y down] convention with scalar-first (w,x,y,z)
quaternions -- neither of which matches camtraj's canonical OpenGL/c2w
Trajectory or scipy's scalar-last (x,y,z,w) quaternions. We convert explicitly,
once, here (via `camtraj.conventions`) rather than hand-deriving a sign flip
the way the reference LAMP codebase did in several inconsistent places.
"""

from __future__ import annotations

import numpy as np
import camtraj_viser as viser
from scipy.spatial.transform import Rotation

from camtraj.conventions import OPENGL, OPENCV, convert_pose
from camtraj.trajectory import Trajectory

DEFAULT_FOV = float(np.deg2rad(60.0))
DEFAULT_ASPECT = 1.5


def to_frame_wxyz(rotation: Rotation) -> np.ndarray:
    """scipy is scalar-last (x,y,z,w); viser's generic add_frame is scalar-first."""
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


def to_frustum_pose(position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, np.ndarray]:
    """Convert a canonical (OpenGL c2w) pose to what add_camera_frustum expects."""
    cv_position, cv_rotation = convert_pose(position, rotation, OPENGL, OPENCV)
    return cv_position, to_frame_wxyz(cv_rotation)


CUBE_SIZE = 1.0
CUBE_CENTER = np.array([0.0, CUBE_SIZE / 2.0, 0.0])  # sits on the ground (y=0)
DEFAULT_CAMERA_START = np.array([0.0, 1.0, 5.0])  # 1m above ground, 5m from the cube


def add_ground(server: viser.ViserServer, size: float = 20.0) -> None:
    """A solid-looking ground plane (not just floating grid lines), so
    trajectories read against a real sense of scale."""
    server.scene.add_grid(
        "/ground",
        width=size,
        height=size,
        plane="xz",
        cell_size=1.0,
        section_size=5.0,
        cell_color=(180, 180, 175),
        section_color=(130, 130, 125),
        plane_color=(235, 233, 225),
        plane_opacity=0.5,
        shadow_opacity=0.35,
    )


def add_reference_box(server: viser.ViserServer):
    """The scene's anchor object ("the box") -- may move over time, hence a
    handle is returned so its position can be updated during scrub/playback."""
    return server.scene.add_box(
        "/reference_cube",
        color=(210, 140, 60),
        dimensions=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        position=CUBE_CENTER,
    )


def update_reference_box(handle, position: np.ndarray) -> None:
    handle.position = position


def add_path(server: viser.ViserServer, name: str, trajectory: Trajectory, color=(80, 170, 255)):
    return server.scene.add_spline_catmull_rom(
        name,
        points=trajectory.positions,
        color=color,
        thickness=0.015,
        segments=max(len(trajectory) * 2, 32),
    )


def _gradient_colors(n: int, start=(70, 130, 250), end=(250, 90, 60)) -> np.ndarray:
    """Linear RGB interpolation from `start` to `end` across n steps (0-255 ints)."""
    if n <= 1:
        return np.array([start], dtype=int)
    t = np.linspace(0.0, 1.0, n)[:, None]
    colors = np.array(start, dtype=np.float64) * (1 - t) + np.array(end, dtype=np.float64) * t
    return colors.astype(int)


def add_keyframe_frustums(
    server: viser.ViserServer,
    name_prefix: str,
    trajectory: Trajectory,
    stride: int | None = None,
    scale: float = 0.15,
    start_color=(70, 130, 250),
    end_color=(250, 90, 60),
) -> list:
    """Wireframe frustums at a sample of keyframes, colored start->end so the
    direction of travel is visible at a glance instead of a flat gray."""
    n = len(trajectory)
    stride = stride or max(1, n // 12)
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)  # always include the true end frame in the gradient
    colors = _gradient_colors(len(indices), start_color, end_color)
    handles = []
    for idx, color in zip(indices, colors):
        position, wxyz = to_frustum_pose(trajectory.positions[idx], trajectory.rotations[idx])
        handles.append(
            server.scene.add_camera_frustum(
                f"{name_prefix}/kf_{idx:04d}",
                fov=DEFAULT_FOV,
                aspect=DEFAULT_ASPECT,
                scale=scale,
                color=tuple(int(c) for c in color),
                wxyz=wxyz,
                position=position,
            )
        )
    return handles


def add_current_camera(
    server: viser.ViserServer, name: str, trajectory: Trajectory, t: float = 0.0, scale: float = 0.18, color=(230, 60, 50)
):
    """A highlighted frustum for the scrubber position -- shown while paused."""
    position, rotation = trajectory.pose_at(t)
    cv_position, wxyz = to_frustum_pose(position, rotation)
    return server.scene.add_camera_frustum(
        name, fov=DEFAULT_FOV, aspect=DEFAULT_ASPECT, scale=scale, color=color, wxyz=wxyz, position=cv_position, variant="filled"
    )


def update_current_camera(handle, trajectory: Trajectory, t: float) -> None:
    position, rotation = trajectory.pose_at(t)
    cv_position, wxyz = to_frustum_pose(position, rotation)
    handle.position = cv_position
    handle.wxyz = wxyz


def clear_handles(handles: list) -> None:
    for h in handles:
        h.remove()
    handles.clear()
