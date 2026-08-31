"""Look-at orientation solver, shared by target-relative primitives (orbit,
tail, rotation_track) and by anything that just needs an initial "point the
camera at X" pose (e.g. an app's default starting view).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def look_at_rotation(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray = (0.0, 1.0, 0.0)) -> Rotation:
    """Rotation for a camera at `eye` looking at `target`, canonical convention
    (local -Z forward, +Y up, +X right). Falls back to a secondary up-hint if
    the eye->target direction is parallel to `up_hint`; falls back to identity
    if `eye == target`.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up_hint = np.asarray(up_hint, dtype=np.float64)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        return Rotation.identity()
    forward = forward / forward_norm

    if np.linalg.norm(np.cross(forward, up_hint)) < 1e-9:
        up_hint = np.array([0.0, 0.0, 1.0]) if abs(forward[1]) > 0.99 else np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    return Rotation.from_matrix(np.stack([right, true_up, -forward], axis=1))
