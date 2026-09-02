import json

import pytest

from camtraj import BoxMotion, recipe_from_dict, recipe_to_dict
from camtraj.easing import Easing
from camtraj.segments import OrbitAxis, OrbitDirection, OrbitSegment, RotAxis, RotationTrackSegment, WorldMove
from camtraj.segments.base import from_param_dict, to_param_dict


def test_to_param_dict_round_trips_through_json():
    segment = RotationTrackSegment(
        frames=41,
        world_move=WorldMove(move_x=1.5, move_y=-2.0, move_z=0.0),
        rot_axis=RotAxis.PAN,
        easing=Easing.EASE_IN_OUT,
    )
    d = to_param_dict(segment)
    # Nothing but JSON-safe primitives should be left -- no Enum members, no
    # nested dataclass instances (this is what makes it save-able at all).
    reloaded = json.loads(json.dumps(d))
    assert reloaded["rot_axis"] == "pan"
    assert reloaded["easing"] == "ease_in_out"
    assert reloaded["world_move"] == {"move_x": 1.5, "move_y": -2.0, "move_z": 0.0}


def test_from_param_dict_is_the_inverse_of_to_param_dict():
    segment = OrbitSegment(frames=81, axis=OrbitAxis.Y, direction=OrbitDirection.CCW, degrees=45.0)
    rebuilt = from_param_dict(OrbitSegment, to_param_dict(segment))
    assert rebuilt == segment


def test_recipe_round_trip_preserves_segments_and_box_motions():
    segments = [
        RotationTrackSegment(frames=30, world_move=WorldMove(move_x=3.0), rot_axis=RotAxis.TILT),
        OrbitSegment(frames=61, axis=OrbitAxis.Y, direction=OrbitDirection.CW),
    ]
    box_motions = [BoxMotion(delta_x=2.0), BoxMotion(easing=Easing.EASE_OUT)]

    data = recipe_to_dict(segments, box_motions)
    json.dumps(data)  # must be plain-JSON-safe end to end, not just per-field
    rebuilt_segments, rebuilt_box_motions = recipe_from_dict(json.loads(json.dumps(data)))

    assert rebuilt_segments == segments
    assert rebuilt_box_motions == box_motions


def test_recipe_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        recipe_to_dict([OrbitSegment()], [])


def test_recipe_from_dict_rejects_unknown_segment_type():
    data = {"camtraj_recipe_version": 1, "segments": [{"type": "not_a_real_type", "params": {}}], "box_motions": [{}]}
    with pytest.raises(ValueError, match="Unknown segment type"):
        recipe_from_dict(data)
