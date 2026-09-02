"""Build a static, standing actor mesh from a SOMA body
(https://github.com/NVlabs/SOMA-X) for Auteur's framing demo.

Uses only SOMA-X's default "mhr" identity model -- this project deliberately
never installs or enables any of SOMA-X's other, non-default identity-model
backends.

Requires the `auteur` extra (heavy: torch + NVIDIA Warp):
    pip install -e ".[auteur]"

This module is intentionally *not* imported by `camtraj/__init__.py` --
plain `import camtraj` must keep working without this extra installed.
`camtraj.auteur`'s framing math has no dependency on this module at all;
`ActorState` is just (position, yaw, height), however you obtain it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soma
import torch

# The MHR rig's own rest pose is a T-pose (arms out horizontal); these two
# joints (indices into the 77-entry `poses` array, i.e.
# `SOMALayer._public_joint_names[1:]`) are the upper-arm joints, rotated
# down here to get a standing A-pose instead. Re-derived and verified
# numerically against SOMA's own joint indexing -- not carried over from any
# other body model's pose-array layout.
_LEFT_ARM_POSE_INDEX = 12  # "LeftArm"
_RIGHT_ARM_POSE_INDEX = 40  # "RightArm"
_ARM_DROP_RADIANS = 0.8

_layer_cache: dict[tuple[str, bool], object] = {}


def _get_layer(device: str, low_lod: bool):
    key = (device, low_lod)
    if key not in _layer_cache:
        _layer_cache[key] = soma.SOMALayer(device=device, identity_model_type="mhr", low_lod=low_lod)
    return _layer_cache[key]


@dataclass(frozen=True)
class ActorMesh:
    """A static SOMA body's mesh, in camtraj's canonical convention
    (right-handed, +Y up): feet at local y=0, facing local +Z when unrotated
    (yaw=0 in `camtraj.auteur.ActorState` matches this orientation exactly).
    """

    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64
    height: float  # meters, standing height


def build_a_pose_actor(identity_seed: int | None = None, device: str = "cpu", low_lod: bool = True) -> ActorMesh:
    """A standing A-pose SOMA actor. `identity_seed=None` (the default) gives
    SOMA-X's own neutral/default body shape; an int gives a random (but
    reproducible) body shape instead -- experimental, since this module
    doesn't know what identity-coefficient range/distribution is
    "plausible" beyond SOMA-X's own default of all-zeros.

    `low_lod=True` (the default) asks SOMA-X for its own lower-resolution
    mesh (~4.5k vertices instead of ~18k) -- plenty for a real-time
    interactive viewer, and meaningfully lighter to upload/render than the
    full-resolution mesh, which matters once this is played back live.
    """
    layer = _get_layer(device, low_lod=low_lod)
    num_identity = layer.identity_model.num_identity_coeffs
    num_scale = layer.identity_model.num_scale_params

    if identity_seed is None:
        identity = torch.zeros(1, num_identity, device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(identity_seed)
        identity = torch.randn(1, num_identity, device=device, generator=generator) * 0.5
    scale = torch.zeros(1, num_scale, device=device)

    poses = torch.zeros(1, 77, 3, device=device)
    poses[0, _LEFT_ARM_POSE_INDEX, 2] = -_ARM_DROP_RADIANS
    poses[0, _RIGHT_ARM_POSE_INDEX, 2] = _ARM_DROP_RADIANS

    with torch.no_grad():
        output = layer(poses, identity, scale_params=scale)

    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float64)
    faces = layer.faces.detach().cpu().numpy().astype(np.int64)

    height = float(vertices[:, 1].max() - vertices[:, 1].min())
    vertices[:, 1] -= vertices[:, 1].min()  # feet at local y=0

    return ActorMesh(vertices=vertices, faces=faces, height=height)
