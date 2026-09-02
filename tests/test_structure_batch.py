import numpy as np
import pytest

from camtraj import BoxMotion, look_at_rotation
from camtraj.segments import FreeFormSegment, OrbitAxis, OrbitDirection, OrbitSegment, RotationTrackSegment, RotAxis
from camtraj.structure_batch import Block, generate_structure_batch, sample_structure


def _pool() -> list[Block]:
    return [
        Block("free_form_default", FreeFormSegment(), BoxMotion()),
        Block("orbit_90_cw", OrbitSegment(degrees=90.0, direction=OrbitDirection.CW, axis=OrbitAxis.Y), BoxMotion()),
        Block("rotation_track_pan", RotationTrackSegment(rot_axis=RotAxis.PAN), BoxMotion(delta_x=1.0)),
    ]


def test_sample_structure_rejects_empty_pool():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Pool is empty"):
        sample_structure([], (1, 3), rng)


def test_sample_structure_length_stays_within_range():
    pool = _pool()
    rng = np.random.default_rng(1)
    for _ in range(100):
        segments, box_motions, labels = sample_structure(pool, (2, 4), rng)
        assert 2 <= len(segments) <= 4
        assert len(segments) == len(box_motions) == len(labels)


def test_sample_structure_only_uses_pool_values_verbatim():
    pool = _pool()
    rng = np.random.default_rng(2)
    pool_by_label = {b.label: b for b in pool}
    for _ in range(50):
        segments, box_motions, labels = sample_structure(pool, (1, 5), rng)
        for segment, box_motion, label in zip(segments, box_motions, labels):
            assert segment == pool_by_label[label].segment
            assert box_motion == pool_by_label[label].box_motion


def test_sample_structure_fixed_length_when_range_collapsed():
    pool = _pool()
    rng = np.random.default_rng(3)
    for _ in range(20):
        segments, _, _ = sample_structure(pool, (3, 3), rng)
        assert len(segments) == 3


def test_generate_structure_batch_is_deterministic_given_a_seed():
    kwargs = dict(
        pool=_pool(),
        length_range=(1, 3),
        n=6,
        seed=7,
        start_position=np.array([0.0, 1.0, 5.0]),
        start_rotation=look_at_rotation(np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 0.0])),
        box_start_position=np.array([0.0, 0.5, 0.0]),
    )
    results_a = generate_structure_batch(**kwargs)
    results_b = generate_structure_batch(**kwargs)
    assert len(results_a) == 6
    for (params_a, traj_a), (params_b, traj_b) in zip(results_a, results_b):
        assert params_a == params_b
        np.testing.assert_allclose(traj_a.positions, traj_b.positions)


def test_generate_structure_batch_varies_which_blocks_are_used():
    results = generate_structure_batch(
        pool=_pool(),
        length_range=(1, 1),
        n=50,
        seed=11,
        start_position=np.array([0.0, 1.0, 5.0]),
        start_rotation=look_at_rotation(np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 0.0])),
        box_start_position=np.array([0.0, 0.5, 0.0]),
    )
    seen_labels = {params["block_labels"][0] for params, _ in results}
    assert len(seen_labels) > 1  # with 50 draws over 3 blocks, should see more than one


def test_chained_blocks_stay_continuous_across_a_random_reshuffle():
    pool = _pool()
    rng = np.random.default_rng(4)
    from camtraj.sequence import sequence

    for _ in range(20):
        segments, box_motions, _ = sample_structure(pool, (2, 4), rng)
        trajectory = sequence(
            segments,
            box_motions,
            start_position=np.array([0.0, 1.0, 5.0]),
            start_rotation=look_at_rotation(np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 0.0])),
            box_start_position=np.array([0.0, 0.5, 0.0]),
        )
        assert np.all(np.isfinite(trajectory.positions))
        # no discontinuous jump between consecutive frames anywhere in the chain
        jumps = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1)
        assert np.max(jumps) < 5.0
