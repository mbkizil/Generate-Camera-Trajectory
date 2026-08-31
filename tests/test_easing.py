import numpy as np

from camtraj.easing import Easing, apply_easing


def test_strength_zero_is_always_linear_regardless_of_style():
    t = np.linspace(0.0, 1.0, 11)
    for style in Easing:
        np.testing.assert_allclose(apply_easing(t, style, strength=0.0), t, atol=1e-12)


def test_strength_one_matches_full_curve():
    # ease_in(0.5) = 0.25 by construction (t**2)
    np.testing.assert_allclose(apply_easing(0.5, Easing.EASE_IN, strength=1.0), 0.25, atol=1e-12)


def test_strength_half_is_the_midpoint_blend():
    linear = 0.5
    full = 0.25  # ease_in(0.5)
    expected = 0.5 * linear + 0.5 * full
    np.testing.assert_allclose(apply_easing(0.5, Easing.EASE_IN, strength=0.5), expected, atol=1e-12)


def test_endpoints_always_zero_and_one():
    for style in Easing:
        for strength in (0.0, 0.3, 0.5, 1.0):
            result = apply_easing(np.array([0.0, 1.0]), style, strength)
            np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)


def test_default_linear_style_ignores_strength():
    t = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(apply_easing(t, Easing.LINEAR, strength=1.0), t, atol=1e-12)
