import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from camtraj.conventions import (
    BLENDER,
    COLMAP,
    KNOWN_CONVENTIONS,
    OPENCV,
    OPENGL,
    Axis,
    Convention,
    axis_change_matrix,
    convert_pose,
    invert_pose,
)


def test_left_handed_camera_basis_is_rejected():
    with pytest.raises(ValueError, match="right-handed"):
        Convention("bogus", right=Axis.PX, up=Axis.PY, forward=Axis.PZ)  # forward should be -Z, not +Z


def test_every_known_convention_has_the_same_basis_determinant_sign():
    # convert_pose silently assumes this (see Convention.__post_init__) --
    # a mismatched sign would make Rotation.from_matrix silently return a
    # reflected (mirror-image) rotation instead of raising.
    dets = {round(float(np.linalg.det(c.basis_matrix()))) for c in KNOWN_CONVENTIONS.values()}
    assert dets == {-1}


def test_blender_is_identical_to_opengl():
    rng = np.random.default_rng(2)
    pos = rng.uniform(-5, 5, size=3)
    rot = Rotation.from_quat(rng.normal(size=4))
    opengl_pos, opengl_rot = convert_pose(pos, rot, OPENGL, OPENGL)
    blender_pos, blender_rot = convert_pose(pos, rot, OPENGL, BLENDER)
    np.testing.assert_allclose(blender_pos, opengl_pos)
    np.testing.assert_allclose(blender_rot.as_matrix(), opengl_rot.as_matrix(), atol=1e-12)


def test_opengl_to_opencv_is_the_well_known_yz_flip():
    m = axis_change_matrix(OPENGL, OPENCV)
    np.testing.assert_allclose(m, np.diag([1.0, -1.0, -1.0]), atol=1e-12)


def test_convert_pose_identity_when_same_convention():
    pos = np.array([1.0, 2.0, 3.0])
    rot = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    new_pos, new_rot = convert_pose(pos, rot, OPENGL, OPENGL)
    np.testing.assert_allclose(new_pos, pos)
    np.testing.assert_allclose(new_rot.as_matrix(), rot.as_matrix(), atol=1e-12)


def test_convert_pose_roundtrip_recovers_original():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos = rng.uniform(-5, 5, size=3)
        rot = Rotation.from_quat(rng.normal(size=4))
        for a, b in [(OPENGL, OPENCV), (OPENGL, COLMAP), (OPENCV, COLMAP)]:
            mid_pos, mid_rot = convert_pose(pos, rot, a, b)
            back_pos, back_rot = convert_pose(mid_pos, mid_rot, b, a)
            np.testing.assert_allclose(back_pos, pos, atol=1e-9)
            # rotations may differ by a global sign (quaternion double cover)
            assert np.allclose(back_rot.as_matrix(), rot.as_matrix(), atol=1e-9)


def test_convert_pose_translation_unaffected_by_axis_relabeling_when_both_c2w():
    pos = np.array([1.0, 2.0, 3.0])
    rot = Rotation.identity()
    new_pos, _ = convert_pose(pos, rot, OPENGL, OPENCV)
    np.testing.assert_allclose(new_pos, pos)


def test_colmap_flips_to_world_to_camera():
    # A camera sitting at world (0,0,5) looking at the origin down local -Z
    # (OpenGL convention, identity rotation with camera placed on +Z looking
    # back at origin means forward (-Z) points toward the origin already).
    pos = np.array([0.0, 0.0, 5.0])
    rot = Rotation.identity()
    colmap_pos, colmap_rot = convert_pose(pos, rot, OPENGL, COLMAP)
    # world_to_camera translation should reproject world origin near +Z in camera space
    p_cam = colmap_rot.apply(np.array([0.0, 0.0, 0.0])) + colmap_pos
    np.testing.assert_allclose(p_cam, [0.0, 0.0, 5.0], atol=1e-9)


def test_invert_pose_is_involution():
    rng = np.random.default_rng(1)
    pos = rng.uniform(-3, 3, size=3)
    rot = Rotation.from_quat(rng.normal(size=4))
    p1, r1 = invert_pose(pos, rot)
    p2, r2 = invert_pose(p1, r1)
    np.testing.assert_allclose(p2, pos, atol=1e-9)
    np.testing.assert_allclose(r2.as_matrix(), rot.as_matrix(), atol=1e-9)
