"""Shared matplotlib plotting helpers for the scripts/sanity_check_*.py scripts.

Fast, no-server way to visually catch a convention/geometry bug before
building on top of it, or before opening the interactive viser app.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe: this box has no display, we only save PNGs

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _screen(points: np.ndarray) -> np.ndarray:
    """Reorder (x, y, z) -> (x, z, y) so world "up" (Y) renders vertically on
    screen -- matplotlib's 3D axes always draw their *third* argument vertical,
    regardless of which world axis it semantically represents."""
    points = np.asarray(points, dtype=np.float64)
    return points[..., [0, 2, 1]]


def camera_frustum_lines(position, rotation, size=0.3, aspect=1.3):
    """Small wireframe camera frustum: apex at `position`, facing local -Z,
    with a short "roof" mark so roll/dutch angle is visible too."""
    hw, hh, d = size * aspect, size, size * 1.6
    local_corners = np.array([[-hw, -hh, -d], [hw, -hh, -d], [hw, hh, -d], [-hw, hh, -d]])
    world_corners = position + rotation.apply(local_corners)
    lines = [[position, c] for c in world_corners]
    lines += [[world_corners[i], world_corners[(i + 1) % 4]] for i in range(4)]
    roof_tip = position + rotation.apply(np.array([0.0, hh * 1.5, -d]))
    lines += [[world_corners[2], roof_tip], [world_corners[3], roof_tip]]
    return lines


def plot_trajectory(
    traj, title: str, out_path: str, frustum_stride: int | None = None, target=None, box_path=None
) -> None:
    """`target`: a single static point (drawn as a star). `box_path`: an
    (N, 3) array for a *moving* box -- drawn as its own colored line with
    start/end markers, for tail_track/rotation_track-style scenarios."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    p = traj.positions
    segs = _screen(np.stack([p[:-1], p[1:]], axis=1))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(traj) - 1, 1)))
    ax.add_collection(Line3DCollection(segs, colors=colors, linewidths=2))

    if frustum_stride is None:
        frustum_stride = max(1, len(traj) // 12)
    frustum_lines = []
    for i in range(0, len(traj), frustum_stride):
        frustum_lines.extend(camera_frustum_lines(p[i], traj.rotations[i]))
    ax.add_collection(Line3DCollection(_screen(np.array(frustum_lines)), colors="black", linewidths=0.8, alpha=0.7))

    ax.scatter(*_screen(p[0]), color="green", s=60, label="start", depthshade=False)
    ax.scatter(*_screen(p[-1]), color="red", s=60, label="end", depthshade=False)
    if target is not None:
        ax.scatter(*_screen(np.asarray(target, dtype=np.float64)), color="orange", marker="*", s=200, label="target", depthshade=False)
    if box_path is not None:
        box_path = np.asarray(box_path, dtype=np.float64)
        box_segs = _screen(np.stack([box_path[:-1], box_path[1:]], axis=1))
        ax.add_collection(Line3DCollection(box_segs, colors=[(0.9, 0.55, 0.1, 0.9)] * len(box_segs), linewidths=4))
        ax.scatter(*_screen(box_path[0]), color="orange", marker="s", s=80, label="box start", depthshade=False)
        ax.scatter(*_screen(box_path[-1]), color="darkorange", marker="s", s=80, label="box end", depthshade=False)

    origin = np.zeros(3)
    for axis, color in zip(np.eye(3), ["r", "g", "b"]):
        ax.plot(*zip(_screen(origin), _screen(origin + axis * 0.5)), color=color, linewidth=1.5)

    ax.set_xlabel("X (right)")
    ax.set_ylabel("Z (forward is -Z)")
    ax.set_zlabel("Y (up)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.view_init(elev=22, azim=-55)
    extra = [arr for arr in (target, box_path) if arr is not None]
    _set_equal_aspect(ax, _screen(np.vstack([p, *[np.atleast_2d(e) for e in extra]])))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}  ({len(traj)} frames)")


def _set_equal_aspect(ax, points: np.ndarray) -> None:
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
