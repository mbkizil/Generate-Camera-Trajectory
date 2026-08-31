"""Sanity check for the rotation_track primitive: plot it with matplotlib
before relying on the interactive viser app.

Run:
    micromamba run -n camera python scripts/sanity_check_rotation_track.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from camtraj.segments import RotationTrackSegment, WorldMove

from _plot_utils import plot_trajectory


def main() -> None:
    target = np.array([0.0, 0.5, 0.0])
    start_position = np.array([5.0, 1.0, 5.0])

    truck_and_push = RotationTrackSegment(
        frames=91, world_move=WorldMove(move_x=-6.0), push=0.3, easing_strength=1.0
    )
    target_positions = np.tile(target, (91, 1))
    traj = truck_and_push.build(start_position, Rotation.identity(), target_positions)
    plot_trajectory(traj, "rotation_track: truck sideways + push in (0.3) while re-aiming", "outputs/sanity_rotation_track.png", target=target)


if __name__ == "__main__":
    main()
