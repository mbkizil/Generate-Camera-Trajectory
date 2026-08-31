import numpy as np

from camtraj import BoxMotion
from camtraj.easing import Easing


def test_zero_delta_stays_put():
    positions = BoxMotion().build(np.array([1.0, 2.0, 3.0]), n_frames=10)
    np.testing.assert_allclose(positions, np.tile([1.0, 2.0, 3.0], (10, 1)), atol=1e-12)


def test_frame_zero_matches_start_position():
    motion = BoxMotion(delta_x=5.0, delta_y=-2.0, delta_z=1.0)
    positions = motion.build(np.array([0.0, 0.0, 0.0]), n_frames=15)
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(positions[-1], [5.0, -2.0, 1.0], atol=1e-9)


def test_linear_easing_is_uniform_progress():
    motion = BoxMotion(delta_x=10.0, easing=Easing.LINEAR)
    positions = motion.build(np.zeros(3), n_frames=11)
    np.testing.assert_allclose(positions[:, 0], np.linspace(0.0, 10.0, 11), atol=1e-9)
