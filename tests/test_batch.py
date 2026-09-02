import json

import numpy as np
import pytest

from camtraj import BoxMotion, look_at_rotation
from camtraj.batch import (
    batch_ranges_from_dict,
    batch_ranges_to_dict,
    default_ranges,
    generate_batch,
    sample_from_ranges,
    sample_recipe,
)
from camtraj.segments import (
    FreeFormSegment,
    OrbitAxis,
    OrbitDirection,
    OrbitSegment,
    RotAxis,
    RotationTrackSegment,
    WorldMove,
)


def test_default_ranges_are_pinned_points():
    segment = OrbitSegment(frames=81, axis=OrbitAxis.Y, direction=OrbitDirection.CW, degrees=45.0)
    ranges = default_ranges(segment)
    assert ranges["frames"] == (81, 81)
    assert ranges["degrees"] == (45.0, 45.0)
    assert ranges["axis"] == ["y"]
    assert ranges["direction"] == ["cw"]


def test_sampling_a_pinned_range_always_reproduces_the_original():
    segment = RotationTrackSegment(frames=41, world_move=WorldMove(move_x=3.0), rot_axis=RotAxis.PAN)
    ranges = default_ranges(segment)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert sample_from_ranges(RotationTrackSegment, ranges, rng) == segment


def test_sampling_stays_within_the_given_range():
    ranges = default_ranges(OrbitSegment())
    ranges["degrees"] = (10.0, 20.0)
    ranges["direction"] = ["cw", "ccw"]
    rng = np.random.default_rng(1)
    seen_directions = set()
    for _ in range(200):
        sample = sample_from_ranges(OrbitSegment, ranges, rng)
        assert 10.0 <= sample.degrees <= 20.0
        seen_directions.add(sample.direction)
    assert seen_directions == {OrbitDirection.CW, OrbitDirection.CCW}


def test_sample_recipe_respects_nested_group_ranges():
    ranges = default_ranges(RotationTrackSegment())
    ranges["world_move"]["move_x"] = (-5.0, 5.0)
    rng = np.random.default_rng(2)
    values = [sample_from_ranges(RotationTrackSegment, ranges, rng).world_move.move_x for _ in range(50)]
    assert min(values) >= -5.0 and max(values) <= 5.0
    assert len(set(values)) > 1  # actually varies, not silently pinned


def test_generate_batch_is_deterministic_given_a_seed():
    segment_types = [FreeFormSegment]
    segment_ranges = [default_ranges(FreeFormSegment())]
    segment_ranges[0]["lateral"] = (-1.0, 1.0)
    box_motion_ranges = [default_ranges(BoxMotion())]

    kwargs = dict(
        segment_types=segment_types,
        segment_ranges=segment_ranges,
        box_motion_ranges=box_motion_ranges,
        n=5,
        seed=42,
        start_position=np.array([0.0, 1.0, 5.0]),
        start_rotation=look_at_rotation(np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 0.0])),
        box_start_position=np.array([0.0, 0.5, 0.0]),
    )
    results_a = generate_batch(**kwargs)
    results_b = generate_batch(**kwargs)
    assert len(results_a) == 5
    for (params_a, traj_a), (params_b, traj_b) in zip(results_a, results_b):
        assert params_a == params_b
        np.testing.assert_allclose(traj_a.positions, traj_b.positions)


def test_batch_ranges_round_trip_through_json():
    segment_types = [OrbitSegment, RotationTrackSegment]
    segment_ranges = [default_ranges(OrbitSegment()), default_ranges(RotationTrackSegment())]
    segment_ranges[0]["degrees"] = (0.0, 90.0)
    box_motion_ranges = [default_ranges(BoxMotion()), default_ranges(BoxMotion())]

    data = batch_ranges_to_dict(segment_types, segment_ranges, box_motion_ranges)
    reloaded = json.loads(json.dumps(data))
    rebuilt_types, rebuilt_segment_ranges, rebuilt_box_ranges = batch_ranges_from_dict(reloaded)
    expected_box_ranges = json.loads(json.dumps(box_motion_ranges))  # normalize tuples -> lists, same as a real save/load

    assert rebuilt_types == segment_types
    assert rebuilt_segment_ranges[0]["degrees"] == [0.0, 90.0]
    assert rebuilt_box_ranges == expected_box_ranges


def test_batch_ranges_from_dict_rejects_unknown_segment_type():
    data = {"camtraj_batch_recipe_version": 1, "segments": [{"type": "nope", "ranges": {}}], "box_motions": []}
    with pytest.raises(ValueError, match="Unknown segment type"):
        batch_ranges_from_dict(data)
