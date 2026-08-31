import numpy as np
from scipy.spatial.transform import Rotation

from camtraj.segments import LookMode, TailSegment


def _moving_target(frames, start=(0.0, 0.0, 0.0), total_displacement=(10.0, 0.0, 0.0)):
    """A box moving in a straight line from `start` by `total_displacement`."""
    start = np.asarray(start, dtype=np.float64)
    total_displacement = np.asarray(total_displacement, dtype=np.float64)
    s = np.linspace(0.0, 1.0, frames)
    return start[None, :] + s[:, None] * total_displacement[None, :]


def test_frame_zero_matches_start_position():
    frames = 30
    target = _moving_target(frames)
    seg = TailSegment(frames=frames, damping=0.4)
    traj = seg.build(np.array([5.0, 1.0, 2.0]), Rotation.identity(), target)
    np.testing.assert_allclose(traj.positions[0], [5.0, 1.0, 2.0], atol=1e-9)


def test_rigid_follow_maintains_exact_offset():
    frames = 40
    target = _moving_target(frames, total_displacement=(2.0, 0.5, -1.0))
    start = np.array([0.0, 0.0, 0.0]) + target[0] + np.array([1.0, 2.0, 3.0])
    seg = TailSegment(frames=frames, damping=1.0)
    traj = seg.build(start, Rotation.identity(), target)
    offsets = traj.positions - target
    np.testing.assert_allclose(offsets - offsets[0], 0.0, atol=1e-9)


def test_damped_follow_lags_behind_rigid_target():
    frames = 60
    target = _moving_target(frames, total_displacement=(3.0, 0.0, 0.0))
    start = target[0] + np.array([0.0, 0.0, 5.0])
    rigid = TailSegment(frames=frames, damping=1.0).build(start, Rotation.identity(), target)
    damped = TailSegment(frames=frames, damping=0.2).build(start, Rotation.identity(), target)
    # rigid always matches the offset exactly; a damped follower measurably lags
    assert not np.allclose(damped.positions[-1], rigid.positions[-1], atol=1e-3)


def test_looks_at_target_by_default():
    frames = 25
    target = _moving_target(frames, total_displacement=(0.5, 1.5, -0.5))
    seg = TailSegment(frames=frames, damping=0.6)
    traj = seg.build(target[0] + np.array([2.0, 0.0, 2.0]), Rotation.identity(), target)
    for pos, rot, t in zip(traj.positions, traj.rotations, target):
        forward = rot.apply([0.0, 0.0, -1.0])
        expected = (t - pos) / np.linalg.norm(t - pos)
        np.testing.assert_allclose(forward, expected, atol=1e-6)


def test_keep_heading_faces_the_boxs_direction_of_travel_not_the_box_itself():
    frames = 20
    target = _moving_target(frames, total_displacement=(1.0, 0.0, 0.0))  # box drives straight along +X
    seg = TailSegment(frames=frames, damping=1.0, look_mode=LookMode.KEEP_HEADING)
    traj = seg.build(target[0] + np.array([0.0, 1.0, 3.0]), Rotation.identity(), target)
    # camera rigidly follows a box moving in a straight line -> constant heading
    # -> orientation should not change frame to frame in keep_heading mode.
    for rot in traj.rotations:
        np.testing.assert_allclose(rot.as_matrix(), traj.rotations[0].as_matrix(), atol=1e-9)
    forward = traj.rotations[0].apply([0.0, 0.0, -1.0])
    np.testing.assert_allclose(forward, [1.0, 0.0, 0.0], atol=1e-9)


def test_locking_an_axis_ignores_the_boxs_motion_on_it():
    frames = 30
    target = _moving_target(frames, total_displacement=(10.0, 5.0, 0.0))
    start = target[0] + np.array([0.0, 0.0, 3.0])
    traj = TailSegment(frames=frames, damping=1.0, amp_x=1.0, amp_y=0.0).build(start, Rotation.identity(), target)
    # Y is locked (amp_y=0): the camera's height should never change even
    # though the box climbs by 5 units; X should still track normally.
    np.testing.assert_allclose(traj.positions[:, 1], start[1], atol=1e-9)
    np.testing.assert_allclose(traj.positions[-1, 0] - traj.positions[0, 0], 10.0, atol=1e-6)


def test_amplifying_an_axis_exaggerates_the_boxs_motion():
    frames = 20
    target = _moving_target(frames, total_displacement=(10.0, 0.0, 0.0))
    start = target[0] + np.array([0.0, 0.0, 3.0])
    traj = TailSegment(frames=frames, damping=1.0, amp_x=2.0).build(start, Rotation.identity(), target)
    np.testing.assert_allclose(traj.positions[-1, 0] - traj.positions[0, 0], 20.0, atol=1e-6)
