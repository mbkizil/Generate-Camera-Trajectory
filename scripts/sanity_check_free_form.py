"""Phase-1 sanity check: build a free_form trajectory and plot it with
matplotlib, before any web UI exists. This is the fast, no-server way to
visually catch a convention/geometry bug before building on top of it.

Run:
    micromamba run -n camera python scripts/sanity_check_free_form.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe: this box has no display, we only save PNGs

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from camtraj import Easing, sequence
from camtraj.segments import FreeFormSegment


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


def plot_trajectory(traj, title: str, out_path: str, frustum_stride: int | None = None) -> None:
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

    origin = np.zeros(3)
    for axis, color in zip(np.eye(3), ["r", "g", "b"]):
        ax.plot(*zip(_screen(origin), _screen(origin + axis * 0.5)), color=color, linewidth=1.5)

    ax.set_xlabel("X (right)")
    ax.set_ylabel("Z (forward is -Z)")
    ax.set_zlabel("Y (up)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.view_init(elev=22, azim=-55)
    _set_equal_aspect(ax, _screen(p))
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


def main() -> None:
    segments = [
        FreeFormSegment(frames=61, depth=-1.0, distance_scale=3.0, easing=Easing.EASE_IN_OUT, easing_strength=1.0),
        FreeFormSegment(frames=76, lateral=0.8, yaw_deg=70.0, distance_scale=3.0, easing=Easing.EASE_IN_OUT, easing_strength=1.0),
        FreeFormSegment(frames=45, vertical=0.6, pitch_deg=-15.0, distance_scale=2.0, easing=Easing.EASE_OUT, easing_strength=1.0),
    ]
    traj = sequence(segments)
    plot_trajectory(traj, "free_form: dolly-in -> curve+pan -> rise+tilt-down", "outputs/sanity_free_form.png")

    pan_only = sequence([FreeFormSegment(frames=61, yaw_deg=90.0, easing=Easing.EASE_IN_OUT, easing_strength=1.0)])
    plot_trajectory(pan_only, "free_form: pure 90-deg pan in place (no translation)", "outputs/sanity_pan_only.png")


if __name__ == "__main__":
    main()
