import numpy as np
from scipy.spatial.transform import Rotation

from camtraj.segments import RotAxis, RotationTrackSegment, WorldMove


def _build(frames=40, start=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), **kwargs):
    seg = RotationTrackSegment(frames=frames, **kwargs)
    target_positions = np.tile(np.asarray(target, dtype=np.float64), (frames, 1))
    return seg.build(np.array(start), Rotation.identity(), target_positions)


def test_frame_zero_matches_start_position():
    traj = _build(start=(3.0, 1.0, -2.0), world_move=WorldMove(move_x=5.0), push=0.3)
    np.testing.assert_allclose(traj.positions[0], [3.0, 1.0, -2.0], atol=1e-9)


def test_world_move_zero_by_default_camera_stays_put_with_no_push():
    traj = _build(push=0.0)
    assert traj.metadata["params"]["world_move"] == {"move_x": 0.0, "move_y": 0.0, "move_z": 0.0}
    np.testing.assert_allclose(traj.positions - traj.positions[0], 0.0, atol=1e-9)


def test_always_looks_at_a_stationary_target_in_full_mode():
    target = np.array([1.0, 0.5, -1.0])
    traj = _build(target=target, world_move=WorldMove(move_x=2.0, move_y=-1.0))
    for pos, rot in zip(traj.positions, traj.rotations):
        forward = rot.apply([0.0, 0.0, -1.0])
        expected = (target - pos) / np.linalg.norm(target - pos)
        np.testing.assert_allclose(forward, expected, atol=1e-6)


def test_world_move_traces_a_straight_line_with_no_push():
    traj = _build(world_move=WorldMove(move_x=6.0, move_y=3.0, move_z=-2.0), push=0.0)
    start = traj.positions[0]
    end = traj.positions[-1]
    np.testing.assert_allclose(end - start, [6.0, 3.0, -2.0], atol=1e-6)
    # every frame should lie exactly on the straight line from start to end
    for pos in traj.positions:
        segment_vec = end - start
        proj = np.dot(pos - start, segment_vec) / np.dot(segment_vec, segment_vec)
        closest_point_on_line = start + proj * segment_vec
        np.testing.assert_allclose(pos, closest_point_on_line, atol=1e-6)


def test_push_in_reduces_final_distance_by_the_expected_fraction():
    traj = _build(target=(0.0, 0.0, 0.0), push=0.3)
    start_distance = np.linalg.norm(traj.positions[0])
    end_distance = np.linalg.norm(traj.positions[-1])
    np.testing.assert_allclose(end_distance, start_distance * 0.7, atol=1e-6)


def test_push_out_increases_final_distance_by_the_expected_fraction():
    traj = _build(target=(0.0, 0.0, 0.0), push=-0.3)
    start_distance = np.linalg.norm(traj.positions[0])
    end_distance = np.linalg.norm(traj.positions[-1])
    np.testing.assert_allclose(end_distance, start_distance * 1.3, atol=1e-6)


def test_pan_mode_keeps_camera_level():
    # target well above the camera -- a full look-at would tilt up noticeably
    traj = _build(start=(0.0, 0.0, 5.0), target=(0.0, 4.0, 0.0), rot_axis=RotAxis.PAN)
    for rot in traj.rotations:
        forward = rot.apply([0.0, 0.0, -1.0])
        assert abs(forward[1]) < 1e-9, "pan mode must never tilt (zero vertical component)"


def test_tilt_mode_keeps_camera_heading_fixed():
    # target moves side to side -- a full look-at would yaw to follow it, tilt-only must not
    frames = 21
    target_positions = np.stack(
        [np.sin(np.linspace(0, np.pi, frames)) * 3.0, np.zeros(frames), np.full(frames, -5.0)], axis=-1
    )
    seg = RotationTrackSegment(frames=frames, rot_axis=RotAxis.TILT)
    traj = seg.build(np.array([0.0, 0.0, 0.0]), Rotation.identity(), target_positions)
    headings = traj.rotations.apply([0.0, 0.0, -1.0])
    headings_flat = headings.copy()
    headings_flat[:, 1] = 0.0
    headings_flat /= np.linalg.norm(headings_flat, axis=1, keepdims=True)
    np.testing.assert_allclose(headings_flat - headings_flat[0], 0.0, atol=1e-6)


def test_full_mode_is_the_default_and_differs_from_pan():
    target = np.array([0.0, 4.0, -5.0])
    full = _build(start=(0.0, 0.0, 0.0), target=target, rot_axis=RotAxis.FULL)
    pan = _build(start=(0.0, 0.0, 0.0), target=target, rot_axis=RotAxis.PAN)
    assert not np.allclose(full.rotations[0].as_matrix(), pan.rotations[0].as_matrix())
