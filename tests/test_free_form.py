import numpy as np
from scipy.spatial.transform import Rotation

from camtraj import sequence
from camtraj.segments import FreeFormSegment


def _build(frames=30, **kwargs):
    seg = FreeFormSegment(frames=frames, **kwargs)
    return seg.build(np.zeros(3), Rotation.identity())


def test_zero_motion_segment_stays_put():
    traj = _build()
    np.testing.assert_allclose(traj.positions, 0.0, atol=1e-12)
    for r in traj.rotations:
        np.testing.assert_allclose(r.as_matrix(), np.eye(3), atol=1e-12)


def test_frame_zero_always_matches_start_pose_regardless_of_params():
    start_pos = np.array([3.0, -1.0, 2.0])
    start_rot = Rotation.from_euler("xyz", [12, -34, 56], degrees=True)
    seg = FreeFormSegment(frames=20, lateral=0.7, vertical=-0.3, depth=0.9, yaw_deg=40, pitch_deg=-20, roll_deg=15)
    traj = seg.build(start_pos, start_rot)
    np.testing.assert_allclose(traj.positions[0], start_pos, atol=1e-9)
    np.testing.assert_allclose(traj.rotations[0].as_matrix(), start_rot.as_matrix(), atol=1e-9)
    np.testing.assert_allclose(traj.times[0], 0.0)
    np.testing.assert_allclose(traj.times[-1], 19.0)  # frame-index units: frames - 1


def test_pure_lateral_moves_along_world_x_from_identity_start():
    traj = _build(lateral=1.0, distance_scale=5.0)
    np.testing.assert_allclose(traj.positions[-1], [5.0, 0.0, 0.0], atol=1e-9)


def test_pure_vertical_moves_along_world_y_from_identity_start():
    traj = _build(vertical=1.0, distance_scale=4.0)
    np.testing.assert_allclose(traj.positions[-1], [0.0, 4.0, 0.0], atol=1e-9)


def test_depth_sign_convention_out_is_positive_world_z_in_is_negative():
    traj_out = _build(depth=1.0, distance_scale=3.0)
    np.testing.assert_allclose(traj_out.positions[-1], [0.0, 0.0, 3.0], atol=1e-9)
    traj_in = _build(depth=-1.0, distance_scale=3.0)
    np.testing.assert_allclose(traj_in.positions[-1], [0.0, 0.0, -3.0], atol=1e-9)


def test_yaw_90_turns_forward_from_minus_z_to_minus_x():
    """Locks down the exact yaw sign convention with a concrete example, rather
    than relying on prose -- this is the executable source of truth."""
    traj = _build(yaw_deg=90.0)
    forward = traj.rotations[-1].apply([0.0, 0.0, -1.0])
    np.testing.assert_allclose(forward, [-1.0, 0.0, 0.0], atol=1e-9)


def test_translation_direction_rotates_with_the_camera_mid_segment():
    """Translating and yawing in the same segment should curve the path (the
    local translation target is integrated through the rotating frame), not
    move in a straight line fixed to the start orientation."""
    traj = _build(frames=50, lateral=1.0, yaw_deg=90.0, distance_scale=1.0)
    # x should increase at first (facing original orientation) then, once
    # yawed ~90 degrees, further motion is along a different world axis --
    # so the path should NOT be collinear start->mid->end.
    p0, p_mid, p_end = traj.positions[0], traj.positions[len(traj) // 2], traj.positions[-1]
    v1 = p_mid - p0
    v2 = p_end - p_mid
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    assert cos_angle < 0.999, "expected a curved path when translating while yawing"


def test_sequence_chains_segments_with_continuous_pose_and_time():
    seg1 = FreeFormSegment(frames=21, lateral=1.0, distance_scale=2.0)
    seg2 = FreeFormSegment(frames=21, vertical=1.0, distance_scale=1.0)
    traj = sequence([seg1, seg2])

    # no duplicate timestamp at the seam
    assert np.all(np.diff(traj.times) > 0)
    np.testing.assert_allclose(traj.times[-1], 40.0, atol=1e-9)

    # seg1 alone should land at (2, 0, 0); seg2 continues from there straight up
    np.testing.assert_allclose(traj.positions[-1], [2.0, 1.0, 0.0], atol=1e-6)


def test_frames_field_controls_length():
    assert len(_build(frames=21)) == 21
    assert len(_build(frames=101)) == 101
