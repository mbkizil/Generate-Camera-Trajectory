import numpy as np
import pytest

from camtraj.auteur import (
    ActorMotion,
    ActorState,
    AuteurKeyframe,
    FramingState,
    apply_camera_jitter,
    build_auteur_trajectory,
    decode_framing,
    freeze_camera_after,
)
from camtraj.auteur import DEFAULT_MIN_DISTANCE_FACTOR
from camtraj.easing import Easing


def _actor(position=(0.0, 0.9, 0.0), yaw=0.0, height=1.8) -> ActorState:
    return ActorState(position=np.array(position), yaw=yaw, height=height)


def _static_motion(yaw=0.0, height=1.8) -> ActorMotion:
    """A motion that never leaves the origin -- the moving-actor equivalent
    of a fixed `ActorState` at the origin, for tests that don't care about
    actor motion specifically."""
    return ActorMotion(end_position=np.zeros(3), end_yaw=yaw, height=height)


def test_rejects_invalid_shot_scale():
    actor = _actor()
    with pytest.raises(ValueError, match="shot_scale"):
        decode_framing(FramingState(shot_scale=0.0), actor)
    with pytest.raises(ValueError, match="shot_scale"):
        decode_framing(FramingState(shot_scale=-1.0), actor)
    with pytest.raises(ValueError, match="shot_scale"):
        decode_framing(FramingState(shot_scale=25.0), actor)


def test_shot_scale_above_one_is_a_tighter_close_up():
    # shot_scale > 1 means the frame shows *less* than the actor's full
    # height (e.g. a face-only close-up) -- deliberately allowed, unlike the
    # Auteur DSL's own (0, 1] writeup; see decode_framing's docstring.
    actor = _actor()
    close_up, _ = decode_framing(FramingState(shot_scale=8.0), actor)
    full_body, _ = decode_framing(FramingState(shot_scale=1.0), actor)
    close_dist = np.linalg.norm(close_up[[0, 2]] - actor.position[[0, 2]])
    full_dist = np.linalg.norm(full_body[[0, 2]] - actor.position[[0, 2]])
    assert close_dist < full_dist


def test_min_distance_floor_prevents_the_camera_reaching_the_actor():
    # An extreme shot_scale at a wide fov would otherwise put the camera at
    # a tiny fraction of a meter from the actor's own position -- easily
    # inside their body mesh. min_distance (default a fraction of actor
    # height) must always be respected, even if that means the achieved
    # framing is looser than shot_scale nominally asks for.
    actor = _actor(position=(0.0, 0.0, 0.0))
    pos, _ = decode_framing(FramingState(shot_scale=20.0), actor, fov=np.deg2rad(60.0))
    dist = np.linalg.norm(pos[[0, 2]] - actor.position[[0, 2]])
    expected_min = DEFAULT_MIN_DISTANCE_FACTOR * actor.height
    assert dist >= expected_min - 1e-9


def test_min_distance_can_be_overridden():
    actor = _actor(position=(0.0, 0.0, 0.0))
    pos, _ = decode_framing(FramingState(shot_scale=20.0), actor, fov=np.deg2rad(60.0), min_distance=0.05)
    dist = np.linalg.norm(pos[[0, 2]] - actor.position[[0, 2]])
    assert dist < DEFAULT_MIN_DISTANCE_FACTOR * actor.height


def test_framing_offset_still_consistent_when_min_distance_clamps():
    # Regression: frame_width (used for framing_offset's world-space shift)
    # must be recomputed from the *actual* clamped distance, not the
    # nominal pre-clamp frame_height -- otherwise the lateral shift would be
    # too small for the frame that's actually visible once clamped.
    actor = _actor(position=(0.0, 0.0, 0.0))
    pos_a, rot_a = decode_framing(FramingState(shot_scale=20.0, framing_offset=0.0), actor)
    pos_b, rot_b = decode_framing(FramingState(shot_scale=20.0, framing_offset=0.3), actor)
    np.testing.assert_allclose(rot_a.as_matrix(), rot_b.as_matrix(), atol=1e-9)
    shift = np.linalg.norm(pos_b - pos_a)
    # With the bug (nominal frame_height), the shift would be tiny (< 1cm);
    # with the fix, it should be a meaningful fraction of a meter.
    assert shift > 0.05


def test_larger_shot_scale_moves_camera_closer():
    actor = _actor()
    _, _ = decode_framing(FramingState(shot_scale=0.2), actor)
    wide_pos, _ = decode_framing(FramingState(shot_scale=0.2), actor)
    close_pos, _ = decode_framing(FramingState(shot_scale=0.9), actor)
    wide_dist = np.linalg.norm(wide_pos[[0, 2]] - actor.position[[0, 2]])
    close_dist = np.linalg.norm(close_pos[[0, 2]] - actor.position[[0, 2]])
    assert close_dist < wide_dist


def test_orientation_orbits_at_constant_ground_distance_and_camera_level():
    actor = _actor()
    distances = []
    for deg in [0.0, 45.0, 90.0, 180.0, 270.0]:
        pos, _ = decode_framing(FramingState(orientation=np.deg2rad(deg)), actor)
        assert pos[1] == pytest.approx(1.6)  # camera_level default, unaffected by orbiting
        distances.append(np.linalg.norm(pos[[0, 2]] - actor.position[[0, 2]]))
    np.testing.assert_allclose(distances, distances[0], atol=1e-9)


def test_camera_level_is_absolute_not_relative_to_actor_height():
    low_actor = _actor(position=(0.0, 0.0, 0.0))
    high_actor = _actor(position=(0.0, 5.0, 0.0))
    pos_low, _ = decode_framing(FramingState(camera_level=1.6), low_actor)
    pos_high, _ = decode_framing(FramingState(camera_level=1.6), high_actor)
    assert pos_low[1] == pytest.approx(1.6)
    assert pos_high[1] == pytest.approx(1.6)


def test_framing_offset_translates_without_changing_rotation():
    actor = _actor()
    pos_a, rot_a = decode_framing(FramingState(framing_offset=0.0), actor)
    pos_b, rot_b = decode_framing(FramingState(framing_offset=0.2), actor)
    np.testing.assert_allclose(rot_a.as_matrix(), rot_b.as_matrix(), atol=1e-9)
    assert np.linalg.norm(pos_b - pos_a) > 1e-6


def test_positive_framing_offset_puts_the_actor_on_the_right_of_frame():
    # A positive offset is labeled "right" (see apps/auteur_designer.py's
    # _FO_MARKS, straight from the DSL's own vocabulary table) -- so the
    # actor must land in the camera's local +X (right) half of the image,
    # not the left. This is exactly the class of sign bug the project's
    # testing philosophy calls for a locked-down check on.
    actor = _actor(position=(0.0, 0.9, 0.0))
    pos, rot = decode_framing(FramingState(orientation=0.0, framing_offset=0.2), actor)
    local_to_actor = rot.inv().apply(actor.position - pos)
    assert local_to_actor[0] > 0.0


def test_front_orientation_looks_back_toward_the_actor():
    # Front (orientation=0) places the camera along the actor's own forward
    # axis (+Z when yaw=0), looking back toward them.
    actor = _actor(position=(0.0, 0.0, 0.0), yaw=0.0, height=1.8)
    pos, rot = decode_framing(FramingState(orientation=0.0, lookat_level=0.9), actor)
    assert pos[2] > 0.0  # camera sits in front of the actor, on +Z
    forward = rot.apply([0.0, 0.0, -1.0])
    to_actor = np.array([0.0, 0.9, 0.0]) - pos
    to_actor /= np.linalg.norm(to_actor)
    np.testing.assert_allclose(forward, to_actor, atol=1e-6)


def test_locked_down_numeric_example():
    actor = ActorState(position=np.array([0.0, 0.0, 0.0]), yaw=0.0, height=2.0)
    state = FramingState(camera_level=1.0, lookat_level=1.0, shot_scale=0.5, orientation=0.0)
    pos, rot = decode_framing(state, actor, fov=np.deg2rad(60.0), aspect=1.5)
    # frame_height = 2.0 / 0.5 = 4.0; distance = 2.0 / tan(30deg) = 2.0 / 0.57735 = 3.4641
    expected_distance = 2.0 / np.tan(np.deg2rad(30.0))
    np.testing.assert_allclose(pos, [0.0, 1.0, expected_distance], atol=1e-6)
    # camera and look-at are both at height 1.0 -- level shot, zero pitch/roll
    np.testing.assert_allclose(rot.as_euler("yxz", degrees=True)[1:], [0.0, 0.0], atol=1e-6)


def test_build_auteur_trajectory_requires_at_least_two_keyframes():
    with pytest.raises(ValueError, match="at least 2"):
        build_auteur_trajectory([AuteurKeyframe(FramingState())], _static_motion())


def test_build_auteur_trajectory_frame_zero_matches_first_keyframe():
    motion = _static_motion()
    actor = motion.state_at(0.0)
    keyframes = [
        AuteurKeyframe(FramingState(shot_scale=0.3)),
        AuteurKeyframe(FramingState(shot_scale=0.7), frames=20),
    ]
    trajectory = build_auteur_trajectory(keyframes, motion)
    expected_pos, expected_rot = decode_framing(keyframes[0].state, actor)
    np.testing.assert_allclose(trajectory.positions[0], expected_pos, atol=1e-9)
    np.testing.assert_allclose(trajectory.rotations[0].as_matrix(), expected_rot.as_matrix(), atol=1e-9)
    # last frame matches the second (final) keyframe exactly
    expected_end_pos, _ = decode_framing(keyframes[1].state, actor)
    np.testing.assert_allclose(trajectory.positions[-1], expected_end_pos, atol=1e-9)


def test_build_auteur_trajectory_total_frame_count():
    keyframes = [
        AuteurKeyframe(FramingState()),
        AuteurKeyframe(FramingState(shot_scale=0.7), frames=15),
        AuteurKeyframe(FramingState(shot_scale=0.3), frames=25),
    ]
    trajectory = build_auteur_trajectory(keyframes, _static_motion())
    assert len(trajectory) == 1 + (15 - 1) + (25 - 1)


def test_orientation_interpolation_takes_the_short_way_around():
    keyframes = [
        AuteurKeyframe(FramingState(orientation=np.deg2rad(350.0))),
        AuteurKeyframe(FramingState(orientation=np.deg2rad(10.0)), frames=37, easing=Easing.LINEAR),
    ]
    trajectory = build_auteur_trajectory(keyframes, _static_motion())
    # sweeping the short way (350 -> 360/0 -> 10, total 20deg) means every
    # intermediate frame's ground-distance-preserving azimuth stays close to
    # the endpoints -- check no frame's position is near the *opposite* side
    # of the orbit (which the long way around, 340deg of sweep, would visit).
    positions = trajectory.positions
    xz = positions[:, [0, 2]]  # actor stays at the origin the whole time
    angles = np.degrees(np.arctan2(xz[:, 0], xz[:, 1])) % 360.0
    # the short path never travels through the far side (~170-190 deg away
    # from the endpoints' ~0/350/10 neighborhood)
    assert np.all((angles < 60.0) | (angles > 300.0))


def test_actor_motion_slides_linearly_from_the_origin():
    motion = ActorMotion(end_position=np.array([4.0, 0.0, 2.0]), end_yaw=0.0, height=1.8)
    start = motion.state_at(0.0)
    mid = motion.state_at(0.5)
    end = motion.state_at(1.0)
    np.testing.assert_allclose(start.position, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mid.position, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(end.position, [4.0, 0.0, 2.0])


def test_actor_motion_turns_linearly_from_yaw_zero():
    motion = ActorMotion(end_position=np.zeros(3), end_yaw=np.pi / 2, height=1.8)
    assert motion.state_at(0.0).yaw == pytest.approx(0.0)
    assert motion.state_at(0.5).yaw == pytest.approx(np.pi / 4)
    assert motion.state_at(1.0).yaw == pytest.approx(np.pi / 2)


def test_build_auteur_trajectory_moves_the_actor_and_records_box_positions():
    motion = ActorMotion(end_position=np.array([10.0, 0.0, 0.0]), end_yaw=0.0, height=1.8)
    keyframes = [AuteurKeyframe(FramingState()), AuteurKeyframe(FramingState(), frames=11)]
    trajectory = build_auteur_trajectory(keyframes, motion)
    assert trajectory.box_positions is not None
    np.testing.assert_allclose(trajectory.box_positions[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(trajectory.box_positions[-1], [10.0, 0.0, 0.0], atol=1e-9)
    # the camera keeps the same framing relative to the actor at every frame,
    # so it must have moved by the same displacement as the actor
    camera_displacement = trajectory.positions[-1] - trajectory.positions[0]
    np.testing.assert_allclose(camera_displacement, [10.0, 0.0, 0.0], atol=1e-9)


def test_freeze_camera_after_locks_position_and_rotation_but_not_the_actor():
    motion = ActorMotion(end_position=np.array([10.0, 0.0, 0.0]), end_yaw=0.0, height=1.8)
    keyframes = [AuteurKeyframe(FramingState(orientation=0.0)), AuteurKeyframe(FramingState(orientation=np.pi / 2), frames=21)]
    trajectory = build_auteur_trajectory(keyframes, motion)
    n = len(trajectory)
    freeze_frame = n // 2

    frozen = freeze_camera_after(trajectory, freeze_frame)

    # camera pose is identical to the pre-freeze pose at freeze_frame, for
    # every later frame -- it doesn't keep tracking the actor
    assert (frozen.positions[freeze_frame:] == frozen.positions[freeze_frame]).all()
    tail_len = n - freeze_frame
    np.testing.assert_allclose(
        frozen.rotations[freeze_frame:].as_quat(),
        np.tile(frozen.rotations[freeze_frame].as_quat(), (tail_len, 1)),
    )
    # unaffected before the freeze point
    np.testing.assert_allclose(frozen.positions[:freeze_frame], trajectory.positions[:freeze_frame])
    # the actor itself keeps moving regardless
    assert not np.allclose(frozen.box_positions[freeze_frame], frozen.box_positions[-1])
    np.testing.assert_allclose(frozen.box_positions, trajectory.box_positions)


def test_freeze_camera_after_rejects_out_of_range_frame():
    trajectory = build_auteur_trajectory(
        [AuteurKeyframe(FramingState()), AuteurKeyframe(FramingState(), frames=11)], _static_motion()
    )
    with pytest.raises(ValueError, match="frame"):
        freeze_camera_after(trajectory, -1)
    with pytest.raises(ValueError, match="frame"):
        freeze_camera_after(trajectory, len(trajectory))


def test_shot_scale_sweep_has_roughly_constant_dolly_speed():
    # Regression: a plain linear ramp in shot_scale drives camera distance
    # through a sharp hyperbola (fast at the wide end, crawling at the tight
    # end) since distance ~ 1/shot_scale -- interpolating the *reciprocal*
    # of shot_scale instead keeps distance changing at a constant rate.
    # A narrow fov (matching the app's own AUTEUR_FOV) and a shot_scale range
    # that stays clear of decode_framing's min_distance floor, so the
    # measured distances reflect the interpolation alone.
    motion = _static_motion()
    keyframes = [
        AuteurKeyframe(FramingState(shot_scale=0.4, orientation=0.0)),
        AuteurKeyframe(FramingState(shot_scale=3.0, orientation=0.0), frames=61, easing=Easing.LINEAR),
    ]
    trajectory = build_auteur_trajectory(keyframes, motion, fov=np.deg2rad(28.0))
    distances = np.linalg.norm(trajectory.positions[:, [0, 2]], axis=1)
    steps = np.diff(distances)
    assert steps.std() / np.abs(steps.mean()) < 0.01


def _still_trajectory(n=40):
    motion = _static_motion()
    keyframes = [AuteurKeyframe(FramingState()), AuteurKeyframe(FramingState(), frames=n)]
    return build_auteur_trajectory(keyframes, motion)


def test_jitter_rejects_out_of_range_strength():
    trajectory = _still_trajectory()
    with pytest.raises(ValueError, match="strength"):
        apply_camera_jitter(trajectory, -0.1)
    with pytest.raises(ValueError, match="strength"):
        apply_camera_jitter(trajectory, 1.1)


def test_jitter_zero_strength_leaves_trajectory_unchanged():
    trajectory = _still_trajectory()
    jittered = apply_camera_jitter(trajectory, 0.0)
    assert jittered is trajectory


def test_jitter_leaves_box_positions_untouched():
    trajectory = _still_trajectory()
    jittered = apply_camera_jitter(trajectory, 0.7)
    np.testing.assert_array_equal(jittered.box_positions, trajectory.box_positions)


def test_jitter_perturbs_position_and_scales_with_strength():
    trajectory = _still_trajectory()
    low = apply_camera_jitter(trajectory, 0.25)
    high = apply_camera_jitter(trajectory, 1.0)

    low_delta = np.abs(low.positions - trajectory.positions).max()
    high_delta = np.abs(high.positions - trajectory.positions).max()

    assert low_delta > 0.0
    assert high_delta > low_delta


def test_jitter_is_deterministic():
    trajectory = _still_trajectory()
    a = apply_camera_jitter(trajectory, 0.6)
    b = apply_camera_jitter(trajectory, 0.6)
    np.testing.assert_allclose(a.rotations.as_quat(), b.rotations.as_quat())
    np.testing.assert_allclose(a.positions, b.positions)


def test_jitter_perturbs_rotation_and_scales_with_strength():
    trajectory = _still_trajectory()
    base_angles = trajectory.rotations.as_euler("xyz")

    low = apply_camera_jitter(trajectory, 0.25)
    high = apply_camera_jitter(trajectory, 1.0)

    low_delta = np.abs(low.rotations.as_euler("xyz") - base_angles).max()
    high_delta = np.abs(high.rotations.as_euler("xyz") - base_angles).max()

    assert low_delta > 0.0
    assert high_delta > low_delta
