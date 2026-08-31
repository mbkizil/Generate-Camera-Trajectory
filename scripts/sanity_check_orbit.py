"""Sanity check for the orbit primitive: plot it with matplotlib before
relying on the interactive viser app.

Run:
    micromamba run -n camera python scripts/sanity_check_orbit.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from camtraj import Easing, sequence
from camtraj.segments import OrbitAxis, OrbitDirection, OrbitSegment

from _plot_utils import plot_trajectory


def main() -> None:
    target = np.array([0.0, 0.5, 0.0])
    start_position = np.array([0.0, 1.0, 5.0])

    full_orbit = sequence(
        [OrbitSegment(frames=121, degrees=360.0, direction=OrbitDirection.CW, axis=OrbitAxis.Y, easing=Easing.LINEAR)],
        start_position=start_position,
        start_rotation=Rotation.identity(),
        box_start_position=target,
    )
    plot_trajectory(full_orbit, "orbit: 360deg cw around Y, constant radius", "outputs/sanity_orbit_full.png", target=target)

    spiral_in = sequence(
        [OrbitSegment(frames=91, degrees=180.0, direction=OrbitDirection.CCW, spiral=-0.6,
                      easing=Easing.EASE_IN_OUT, easing_strength=1.0)],
        start_position=start_position,
        start_rotation=Rotation.identity(),
        box_start_position=target,
    )
    plot_trajectory(spiral_in, "orbit: 180deg ccw with spiral-in (0.6)", "outputs/sanity_orbit_spiral.png", target=target)


if __name__ == "__main__":
    main()
