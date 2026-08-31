"""Optional, translation-only motion for the scene's anchor object ("the box").

Every segment can independently say whether the box is moving during that
span of frames -- no rotation, just a straight-line world-space displacement,
mirroring how the original DSL's object trajectories only ever carried
position (rotation fields were parsed but unused for objects).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .easing import Easing, apply_easing
from .segments.base import enum_param, param


@dataclass
class BoxMotion:
    """Moves the box by a fixed world-space delta over one segment's frames,
    starting exactly at whatever position the box was already at."""

    delta_x: float = param(label="Box move X", min=-20.0, max=20.0, default=0.0, unit="units")
    delta_y: float = param(label="Box move Y", min=-20.0, max=20.0, default=0.0, unit="units")
    delta_z: float = param(label="Box move Z", min=-20.0, max=20.0, default=0.0, unit="units")

    easing: Easing = enum_param(label="Easing", enum_cls=Easing, default=Easing.LINEAR)
    easing_strength: float = param(
        label="Easing strength", min=0.0, max=1.0, default=0.5,
        marks={0.0: "off", 0.5: "default", 1.0: "full"},
    )

    def build(self, start_position: np.ndarray, n_frames: int) -> np.ndarray:
        """Return (n_frames, 3) box positions, starting exactly at `start_position`."""
        s = np.linspace(0.0, 1.0, n_frames)
        s_eased = apply_easing(s, self.easing, self.easing_strength)
        delta = np.array([self.delta_x, self.delta_y, self.delta_z])
        return start_position + s_eased[:, None] * delta[None, :]
