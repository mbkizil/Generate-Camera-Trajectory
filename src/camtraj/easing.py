"""Acceleration curves for remapping linear time progress to motion progress.

Frame *timestamps* are always linear (uniform dt) — easing only changes how much
of the motion has happened by a given time, i.e. it reshapes `s = t / duration`
into `s_eased`, which is what segments actually interpolate position/rotation by.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Easing(str, Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EASE_OUT_IN = "ease_out_in"


def _ease_in(t: np.ndarray) -> np.ndarray:
    return t**2


def _ease_out(t: np.ndarray) -> np.ndarray:
    return t * (2.0 - t)


def _full_curve(t: np.ndarray, style: Easing) -> np.ndarray:
    if style is Easing.LINEAR:
        return t
    if style is Easing.EASE_IN:
        return _ease_in(t)
    if style is Easing.EASE_OUT:
        return _ease_out(t)
    if style is Easing.EASE_IN_OUT:
        return np.where(t < 0.5, 0.5 * _ease_in(2.0 * t), 0.5 + 0.5 * _ease_out(2.0 * t - 1.0))
    if style is Easing.EASE_OUT_IN:
        return np.where(t < 0.5, 0.5 * _ease_out(2.0 * t), 0.5 + 0.5 * _ease_in(2.0 * t - 1.0))
    raise ValueError(f"Unknown easing style: {style!r}")


def apply_easing(t, style: Easing | str = Easing.LINEAR, strength: float = 1.0) -> np.ndarray:
    """Map normalized progress `t` in [0, 1] through the named curve, elementwise.

    `ease_in_out` and `ease_out_in` are built by composing `ease_in`/`ease_out`
    on each half, so continuity and endpoint behavior (0 -> 0, 1 -> 1) follow
    automatically instead of needing separately hand-tuned piecewise formulas.

    `strength` in [0, 1] blends between pure linear (0, curve fully disabled)
    and the full named curve (1). E.g. `strength=0.5` is halfway between a
    straight line and the full ease-in-out shape -- lets a segment use "a bit"
    of easing instead of only all-or-nothing.
    """
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    style = Easing(style)
    strength = float(np.clip(strength, 0.0, 1.0))
    if style is Easing.LINEAR or strength == 0.0:
        return t
    curved = _full_curve(t, style)
    return (1.0 - strength) * t + strength * curved
