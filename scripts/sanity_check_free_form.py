"""Phase-1 sanity check: build a free_form trajectory and plot it with
matplotlib, before any web UI exists.

Run:
    micromamba run -n camera python scripts/sanity_check_free_form.py
"""

from __future__ import annotations

from camtraj import Easing, sequence
from camtraj.segments import FreeFormSegment

from _plot_utils import plot_trajectory


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
