import numpy as np

from camtraj.look_at import look_at_rotation


def test_forward_points_from_eye_to_target():
    cases = [
        (np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([5.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([3.0, 4.0, -2.0]), np.array([-1.0, 0.5, 2.0])),
    ]
    for eye, target in cases:
        rot = look_at_rotation(eye, target)
        forward_world = rot.apply([0.0, 0.0, -1.0])
        expected = (target - eye) / np.linalg.norm(target - eye)
        np.testing.assert_allclose(forward_world, expected, atol=1e-9)


def test_identity_matrix_case_matches_canonical_convention():
    rot = look_at_rotation(eye=np.array([0.0, 0.0, 5.0]), target=np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(rot.as_matrix(), np.eye(3), atol=1e-9)


def test_degenerate_same_point_returns_identity():
    rot = look_at_rotation(eye=np.array([1.0, 2.0, 3.0]), target=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(rot.as_matrix(), np.eye(3), atol=1e-12)


def test_forward_parallel_to_up_hint_does_not_crash():
    rot = look_at_rotation(eye=np.array([0.0, 0.0, 0.0]), target=np.array([0.0, 5.0, 0.0]))
    forward_world = rot.apply([0.0, 0.0, -1.0])
    np.testing.assert_allclose(forward_world, [0.0, 1.0, 0.0], atol=1e-9)
