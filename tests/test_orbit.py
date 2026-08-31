import numpy as np
from scipy.spatial.transform import Rotation

from camtraj.segments import OrbitAxis, OrbitDirection, OrbitSegment


def _build(frames=40, start_position=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), start_rotation=None, **kwargs):
    seg = OrbitSegment(frames=frames, **kwargs)
    target_positions = np.tile(np.asarray(target, dtype=np.float64), (frames, 1))
    return seg.build(np.array(start_position), start_rotation or Rotation.identity(), target_positions)


def test_frame_zero_always_matches_start_position():
    start = np.array([2.0, 1.0, -3.0])
    traj = _build(start_position=start, target=(1.0, 0.5, 0.0), degrees=200.0, spiral=0.7)
    np.testing.assert_allclose(traj.positions[0], start, atol=1e-9)


def test_constant_radius_when_no_spiral():
    traj = _build(degrees=360.0, spiral=0.0, target=(0.0, 0.0, 0.0))
    radii = np.linalg.norm(traj.positions, axis=1)  # target is the origin here
    np.testing.assert_allclose(radii, radii[0], atol=1e-9)


def test_always_looking_at_target():
    target = np.array([1.0, 0.5, -2.0])
    traj = _build(degrees=270.0, target=target)
    for pos, rot in zip(traj.positions, traj.rotations):
        forward = rot.apply([0.0, 0.0, -1.0])
        expected = (target - pos) / np.linalg.norm(target - pos)
        np.testing.assert_allclose(forward, expected, atol=1e-6)


def test_cw_90_from_plus_z_moves_toward_minus_x():
    """Locks down the cw/ccw sign convention with a concrete example (orbiting
    around the default Y axis, the most common "horizontal circle" case)."""
    traj = _build(start_position=(0.0, 0.0, 5.0), degrees=90.0, direction=OrbitDirection.CW, axis=OrbitAxis.Y)
    np.testing.assert_allclose(traj.positions[-1], [-5.0, 0.0, 0.0], atol=1e-6)


def test_ccw_is_the_opposite_of_cw():
    cw = _build(degrees=45.0, direction=OrbitDirection.CW)
    ccw = _build(degrees=45.0, direction=OrbitDirection.CCW)
    assert not np.allclose(cw.positions[-1], ccw.positions[-1])
    # mirror images across the plane containing the start point and the axis
    np.testing.assert_allclose(cw.positions[-1][0], -ccw.positions[-1][0], atol=1e-6)


def test_spiral_in_shrinks_radius_by_the_expected_fraction():
    traj = _build(degrees=0.0, spiral=-0.4, target=(0.0, 0.0, 0.0))
    start_radius = np.linalg.norm(traj.positions[0])
    end_radius = np.linalg.norm(traj.positions[-1])
    np.testing.assert_allclose(end_radius, start_radius * 0.6, atol=1e-6)


def test_spiral_out_grows_radius_by_the_expected_fraction():
    traj = _build(degrees=0.0, spiral=0.5, target=(0.0, 0.0, 0.0))
    start_radius = np.linalg.norm(traj.positions[0])
    end_radius = np.linalg.norm(traj.positions[-1])
    np.testing.assert_allclose(end_radius, start_radius * 1.5, atol=1e-6)


def test_orbit_axis_choice_changes_the_plane_of_motion():
    # Orbiting a start point that's off the Y axis, around Y, should keep Y constant.
    traj_y = _build(start_position=(3.0, 2.0, 0.0), axis=OrbitAxis.Y, degrees=180.0)
    np.testing.assert_allclose(traj_y.positions[:, 1], 2.0, atol=1e-9)

    # Orbiting the same start point around Z should keep Z constant instead.
    traj_z = _build(start_position=(3.0, 2.0, 0.0), axis=OrbitAxis.Z, degrees=180.0)
    np.testing.assert_allclose(traj_z.positions[:, 2], 0.0, atol=1e-9)
    assert not np.allclose(traj_z.positions[:, 1], 2.0)


def test_orbit_follows_a_moving_target():
    """The center should track a moving box, not a fixed point in space."""
    frames = 61
    target_positions = np.stack(
        [np.linspace(0.0, 10.0, frames), np.zeros(frames), np.zeros(frames)], axis=-1
    )  # box drives along +X while we orbit it
    seg = OrbitSegment(frames=frames, degrees=0.0, spiral=0.0)  # no rotation/spiral: isolate the "ride along" behavior
    traj = seg.build(np.array([0.0, 0.0, 5.0]), Rotation.identity(), target_positions)

    # radius relative to the *moving* target should stay constant...
    radii = np.linalg.norm(traj.positions - target_positions, axis=1)
    np.testing.assert_allclose(radii, radii[0], atol=1e-6)
    # ...even though the absolute position clearly moved along with the box.
    np.testing.assert_allclose(traj.positions[-1, 0], 10.0, atol=1e-6)
