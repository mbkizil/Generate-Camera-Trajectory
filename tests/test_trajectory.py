import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camtraj import Trajectory


def _line_trajectory(n=5):
    times = np.linspace(0.0, 2.0, n)
    positions = np.stack([times, np.zeros(n), np.zeros(n)], axis=-1)  # moves along +X
    rotations = Rotation.identity() if n == 1 else Rotation.concatenate([Rotation.identity()] * n)
    return Trajectory(times=times, positions=positions, rotations=rotations)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Trajectory(times=np.array([0.0, 1.0]), positions=np.zeros((3, 3)), rotations=Rotation.identity())


def test_rejects_non_monotonic_times():
    with pytest.raises(ValueError):
        Trajectory(
            times=np.array([0.0, 1.0, 0.5]),
            positions=np.zeros((3, 3)),
            rotations=Rotation.concatenate([Rotation.identity()] * 3),
        )


def test_pose_at_interpolates_linearly_and_clamps():
    traj = _line_trajectory()
    pos, _ = traj.pose_at(1.0)
    np.testing.assert_allclose(pos, [1.0, 0.0, 0.0], atol=1e-9)
    pos_clamped, _ = traj.pose_at(999.0)
    np.testing.assert_allclose(pos_clamped, [2.0, 0.0, 0.0], atol=1e-9)


def test_resample_changes_frame_count_not_shape():
    traj = _line_trajectory(n=5)
    resampled = traj.resample(n_frames=101)
    assert len(resampled) == 101
    np.testing.assert_allclose(resampled.positions[0], traj.positions[0], atol=1e-9)
    np.testing.assert_allclose(resampled.positions[-1], traj.positions[-1], atol=1e-9)
    # midpoint in time should still land on the same straight line
    mid = resampled.positions[len(resampled) // 2]
    assert abs(mid[1]) < 1e-9 and abs(mid[2]) < 1e-9


def test_as_matrices_shape_and_identity_block():
    traj = _line_trajectory(n=3)
    mats = traj.as_matrices()
    assert mats.shape == (3, 4, 4)
    np.testing.assert_allclose(mats[:, 3, :], [[0, 0, 0, 1]] * 3, atol=1e-12)
    np.testing.assert_allclose(mats[0, :3, :3], np.eye(3), atol=1e-12)
