"""Sanity check for the tail_track primitive: plot it with matplotlib before
relying on the interactive viser app.

Run:
    micromamba run -n camera python scripts/sanity_check_tail.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from camtraj.segments import LookMode, TailSegment

from _plot_utils import plot_trajectory


def main() -> None:
    frames = 91
    s = np.linspace(0.0, 1.0, frames)
    # box drives in a wide curve (not just a straight line), so the camera's
    # relative offset visibly has to sweep along with it
    target = np.stack([8.0 * s, np.zeros(frames), 4.0 * np.sin(s * np.pi)], axis=-1)
    start_position = target[0] + np.array([0.0, 1.0, 4.0])  # start behind + above the box

    rigid = TailSegment(frames=frames, damping=1.0).build(start_position, Rotation.identity(), target)
    plot_trajectory(rigid, "tail_track: rigid follow (damping=1.0) of a curving box", "outputs/sanity_tail_rigid.png", box_path=target)

    laggy = TailSegment(frames=frames, damping=0.15).build(start_position, Rotation.identity(), target)
    plot_trajectory(laggy, "tail_track: laggy follow (damping=0.15) of the same box", "outputs/sanity_tail_laggy.png", box_path=target)


if __name__ == "__main__":
    main()
