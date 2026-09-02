"""Auteur: actor-relative camera *framing*, not world-space motion.

Every primitive in `camtraj.segments` describes camera motion directly in
world space (translate here, orbit around there). Auteur instead describes
how a camera should *frame* a standing actor via six continuous,
cinematographically-meaningful axes -- camera height, look-at height, shot
scale, lateral framing offset, orientation angle, and dutch tilt -- and
derives the resulting 6-DoF camera pose from that (never the reverse).
Keyframes are placed directly in this framing space `K`, one exact set of
coordinates per keyframe, and interpolated axis-by-axis (never in raw SE(3)
position+rotation space): a ramp in shot scale reads as a roughly
constant-speed dolly/zoom (see `_interpolate_shot_scale` -- shot_scale
itself is interpolated reciprocally, since it's inversely related to
distance), and a linear ramp in orientation reads as a uniform orbit, in a
way that interpolating world-space camera poses directly would not
reproduce.

This module has no dependency on any particular body model -- `ActorState`
is just (position, yaw, height), and `ActorMotion` just a start-at-origin
linear slide (and start-at-yaw-0 linear turn) between two such states.
`camtraj.soma_actor` supplies one way to get an actor's height (and a mesh
to render) from a SOMA body; nothing here requires it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .easing import Easing, apply_easing
from .look_at import look_at_rotation
from .trajectory import Trajectory

DEFAULT_FOV = float(np.deg2rad(60.0))
DEFAULT_ASPECT = 1.5


@dataclass(frozen=True)
class ActorState:
    """A standing actor: pelvis position, facing yaw, and standing height.

    `yaw` is defined relative to the actor's own rest orientation -- yaw=0
    means "facing world +Z, unrotated," matching a SOMA body's own A-pose
    rest orientation (see `camtraj.soma_actor`). Only `position`'s X/Z
    (ground-plane) components affect framing; `height` is measured on the
    actor's own rest pose and does not change with `position`.
    """

    position: np.ndarray
    yaw: float = 0.0
    height: float = 1.7

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64))
        if self.position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {self.position.shape}")


@dataclass(frozen=True)
class ActorMotion:
    """The actor's motion for a whole Auteur trajectory: a direct,
    constant-velocity slide from the world origin to `end_position`, facing
    world +Z (yaw 0) at the start and turning at a constant rate to face
    `end_yaw` by the end -- no easing, no curve, "nothing fancy" for either.
    Zero `end_position`/`end_yaw` (the defaults) mean the actor doesn't move
    or turn, the same convention `BoxMotion` uses for camtraj's world-space
    primitives. `height` is fixed for the whole trajectory.
    """

    end_position: np.ndarray = None  # type: ignore[assignment]
    end_yaw: float = 0.0
    height: float = 1.7

    def __post_init__(self) -> None:
        end_position = np.zeros(3) if self.end_position is None else self.end_position
        object.__setattr__(self, "end_position", np.asarray(end_position, dtype=np.float64))
        if self.end_position.shape != (3,):
            raise ValueError(f"end_position must have shape (3,), got {self.end_position.shape}")

    def state_at(self, t: float) -> ActorState:
        """The actor's state a fraction `t` (in [0, 1]) of the way through
        the trajectory -- position slides linearly from the origin to
        `end_position`, yaw sweeps linearly from 0 to `end_yaw`; height
        stays fixed."""
        return ActorState(position=t * self.end_position, yaw=t * self.end_yaw, height=self.height)


@dataclass(frozen=True)
class FramingState:
    """One point in Auteur's 6D actor-relative framing space `K`.

    All heights are absolute meters above the ground plane (not relative to
    the actor's pelvis). `shot_scale` is the fraction of frame height the
    actor's body occupies -- smaller is a wider shot. `orientation` and
    `dutch` are radians.
    """

    camera_level: float = 1.6
    lookat_level: float = 0.9
    shot_scale: float = 2.0  # ~"MS" (waist-up) -- see apps/auteur_designer.py's _SS_MARKS
    framing_offset: float = 0.0
    orientation: float = 0.0
    dutch: float = 0.0


DEFAULT_MIN_DISTANCE_FACTOR = 0.4
"""Default `min_distance` in `decode_framing`, as a multiple of actor
height -- a generous clearance past a typical adult's front-to-back body
depth (roughly 0.2-0.25x height), so a tight shot_scale can't put the
camera inside the actor's own mesh."""


def decode_framing(
    state: FramingState,
    actor: ActorState,
    fov: float = DEFAULT_FOV,
    aspect: float = DEFAULT_ASPECT,
    min_distance: float | None = None,
) -> tuple[np.ndarray, Rotation]:
    """Auteur's geometric decoder `Phi`: framing state + actor state + fixed
    intrinsics -> a (position, rotation) camera pose, in camtraj's canonical
    convention (right-handed, +Y up, camera-to-world, looks down local -Z).

    `shot_scale` fixes the actor-to-camera distance via a pinhole relation
    (bigger actor-in-frame => closer camera); `framing_offset` then slides
    the camera sideways *without* re-aiming, so the actor lands off-center
    in the final image (lead-room / rule-of-thirds), exactly like sliding a
    real camera on a slider while it stays locked onto its subject.

    `shot_scale` > 1 is intentional and necessary, despite the Auteur DSL's
    own writeup capping it at 1: a true close-up shows *less* than the
    actor's full height (e.g. just the face), which literally means the
    actor's full height is bigger than the frame -- shot_scale (frame
    fraction the actor's full height occupies) exceeding 1 is exactly how
    that's expressed here, not a bug. See `apps/auteur_designer.py`'s
    `_SS_MARKS` for calibrated values (ECU~8, MS~2, FS~0.65, EWS~0.25) and
    its log-scaled slider (shot_scale is a multiplicative "how much closer"
    quantity, so equal ratios -- not equal differences -- should feel like
    equal steps).

    A large `shot_scale` at a wide `fov` implies a very short distance --
    real cinematography avoids this by switching to a longer (narrower-fov)
    lens for tight shots rather than walking the camera into the subject.
    This decoder doesn't vary `fov` per shot, so instead `min_distance`
    (default `DEFAULT_MIN_DISTANCE_FACTOR * actor.height`) puts a floor
    under the computed distance: past that point, a shot_scale that would
    require getting even closer instead shows a bit more of the actor than
    literally requested, rather than clipping the camera through their mesh.
    """
    if not 0.0 < state.shot_scale <= 20.0:
        raise ValueError(f"shot_scale must be in (0, 20], got {state.shot_scale}")
    if min_distance is None:
        min_distance = DEFAULT_MIN_DISTANCE_FACTOR * actor.height

    frame_height = actor.height / state.shot_scale
    distance = max(min_distance, (frame_height / 2.0) / np.tan(fov / 2.0))
    # Recompute the *actual* frame extent at the distance we're really
    # using -- if the min_distance floor kicked in, that's larger than
    # frame_height above, and framing_offset's lateral shift (in world
    # units) needs to match what's actually visible, not the nominal ask.
    actual_frame_height = 2.0 * distance * np.tan(fov / 2.0)
    frame_width = aspect * actual_frame_height

    theta = state.orientation + actor.yaw
    offset_direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    base_position = actor.position + offset_direction * distance
    base_position[1] = state.camera_level

    look_at_target = np.array([actor.position[0], state.lookat_level, actor.position[2]])
    rotation = look_at_rotation(base_position, look_at_target)

    # Slide along the camera's own (pre-roll) right axis -- ground-plane-
    # perpendicular to the view direction regardless of pitch, since pitch
    # rotates about that same right axis and so never changes it. Sliding
    # the *camera* toward its own right moves the (still-centered-on-target)
    # look direction the other way, so the actor ends up appearing on the
    # LEFT of frame -- the opposite of what a positive ("right") offset
    # should do. Sliding the camera left instead puts the actor on the right.
    right = rotation.apply([1.0, 0.0, 0.0])
    position = base_position - right * (state.framing_offset * frame_width)

    dutch = Rotation.from_euler("z", state.dutch)
    rotation = rotation * dutch
    return position, rotation


@dataclass(frozen=True)
class AuteurKeyframe:
    """One keyframe in an Auteur sequence. `frames`/`easing`/`easing_strength`
    describe the transition *into* this keyframe from the previous one, and
    are ignored on the first keyframe (which is just the starting state)."""

    state: FramingState
    frames: int = 30
    easing: Easing = Easing.LINEAR
    easing_strength: float = 0.5


def _interpolate_angle(a: float, b: float, t: np.ndarray) -> np.ndarray:
    """Shortest-path interpolation between two angles (radians) -- avoids
    the wraparound bug of naive linear interpolation (e.g. 350deg -> 10deg
    should sweep 20deg forward, not the long way through 180deg)."""
    delta = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + t * delta


def _interpolate_shot_scale(a: float, b: float, t: np.ndarray) -> np.ndarray:
    """Interpolate shot_scale so the camera-to-actor *distance* it implies
    moves at a roughly constant rate, rather than shot_scale itself.
    `decode_framing` makes distance inversely proportional to shot_scale, so
    a plain linear ramp in shot_scale drives distance through a sharp
    hyperbolic curve -- fast at the wide end, crawling to a stop at the
    tight end, not the "constant dolly speed" a linear sweep should feel
    like. Interpolating the *reciprocal* linearly instead keeps distance
    itself changing at a roughly even rate across the sweep."""
    inv_a, inv_b = 1.0 / a, 1.0 / b
    return 1.0 / (inv_a + t * (inv_b - inv_a))


def _interpolate_state(a: FramingState, b: FramingState, t: np.ndarray) -> list[FramingState]:
    camera_level = a.camera_level + t * (b.camera_level - a.camera_level)
    lookat_level = a.lookat_level + t * (b.lookat_level - a.lookat_level)
    shot_scale = _interpolate_shot_scale(a.shot_scale, b.shot_scale, t)
    framing_offset = a.framing_offset + t * (b.framing_offset - a.framing_offset)
    orientation = _interpolate_angle(a.orientation, b.orientation, t)
    dutch = _interpolate_angle(a.dutch, b.dutch, t)
    return [
        FramingState(
            camera_level=camera_level[i],
            lookat_level=lookat_level[i],
            shot_scale=shot_scale[i],
            framing_offset=framing_offset[i],
            orientation=orientation[i],
            dutch=dutch[i],
        )
        for i in range(len(t))
    ]


def build_auteur_trajectory(
    keyframes: list[AuteurKeyframe],
    actor_motion: ActorMotion,
    fov: float = DEFAULT_FOV,
    aspect: float = DEFAULT_ASPECT,
) -> Trajectory:
    """Interpolate a sequence of Auteur keyframes (in framing space `K`) into
    a dense camera `Trajectory`, decoding each interpolated state -- against
    the actor's own position *at that frame* -- with `decode_framing`.
    Requires at least 2 keyframes (a single fixed framing is two identical
    keyframes; see the "static camera" pattern in `apps/auteur_designer.py`).

    The actor's per-frame ground position is also returned as
    `Trajectory.box_positions` (the same field `camtraj.sequence` uses for
    the world-space "box" -- the actor plays the same "scene anchor" role
    here), so `Trajectory.box_position_at(t)` works for the actor too.
    """
    if len(keyframes) < 2:
        raise ValueError(f"build_auteur_trajectory needs at least 2 keyframes, got {len(keyframes)}")

    all_states: list[FramingState] = [keyframes[0].state]
    for prev, kf in zip(keyframes[:-1], keyframes[1:]):
        if kf.frames < 2:
            raise ValueError(f"each keyframe transition needs >= 2 frames, got {kf.frames}")
        s = np.linspace(0.0, 1.0, kf.frames)[1:]  # skip 0 -- already have the previous endpoint
        s_eased = apply_easing(s, kf.easing, kf.easing_strength)
        all_states.extend(_interpolate_state(prev.state, kf.state, s_eased))

    n = len(all_states)
    positions = np.empty((n, 3))
    box_positions = np.empty((n, 3))
    rotations = []
    for i, framing_state in enumerate(all_states):
        t_fraction = i / (n - 1) if n > 1 else 0.0
        actor_state = actor_motion.state_at(t_fraction)
        position, rotation = decode_framing(framing_state, actor_state, fov=fov, aspect=aspect)
        positions[i] = position
        box_positions[i] = actor_state.position
        rotations.append(rotation)

    return Trajectory(
        times=np.arange(n, dtype=np.float64),
        positions=positions,
        rotations=Rotation.concatenate(rotations),
        box_positions=box_positions,
        metadata={"segment_type": "auteur", "actor_height": actor_motion.height},
    )


def freeze_camera_after(trajectory: Trajectory, frame: int) -> Trajectory:
    """Post-process a built `Trajectory` so the camera holds its exact world
    pose from `frame` onward -- e.g. for a trajectory whose last keyframe
    ends before the last frame, where the "static hold" should mean an
    actually still camera, not one that keeps re-deriving its position every
    frame to maintain the same relative framing on a still-moving actor.
    `box_positions` (the actor's own motion) is left untouched -- only the
    camera locks, the actor keeps moving as normal.
    """
    if not 0 <= frame < len(trajectory):
        raise ValueError(f"frame must be in [0, {len(trajectory)}), got {frame}")
    positions = trajectory.positions.copy()
    positions[frame:] = positions[frame]
    frozen_rotation = trajectory.rotations[frame]
    rotations = Rotation.concatenate(
        [trajectory.rotations[:frame]] + [frozen_rotation] * (len(trajectory) - frame)
    )
    return dataclasses.replace(trajectory, positions=positions, rotations=rotations)


DEFAULT_JITTER_MAX_DEGREES = 1.5
"""Peak pitch/yaw wobble amplitude (roll is scaled down further, see
`apply_camera_jitter`) at `strength == 1.0` -- a visible but not disorienting
handheld shake. (Halved from an earlier 3.0: that value's `strength == 0.5`
read as the right amount of shake, so it became the new `strength == 1.0`.)"""

DEFAULT_JITTER_MAX_TRANSLATION = 0.03
"""Peak lateral/vertical positional wobble, in meters, at `strength == 1.0`
(forward/back and vertical components are scaled down further, see
`apply_camera_jitter`) -- a few centimeters, like a hand's natural sway
rather than a tripod's perfect stillness."""


def _wobble(i: np.ndarray, phase: float, strength: float) -> np.ndarray:
    """A blend of a few fixed, mutually-incommensurate sine frequencies at a
    given phase offset -- reused for every jittered axis (rotation and
    translation alike) so each one wanders independently without ever
    reading as one obvious metronome tick, while staying fully deterministic
    given the frame index `i`.

    The faster two components fade in *faster than linearly* with
    `strength`, so a low strength is dominated by the slow sway alone (a
    smooth, natural drift) and only picks up the busier high-frequency
    jiggle -- which is what makes small movements read as a "vibrating"
    jitter rather than a hand's sway -- as strength climbs toward the
    "handshake" end.
    """
    slow = np.sin(2 * np.pi * i / 29.0 + phase * 1.3)
    medium = np.sin(2 * np.pi * i / 13.0 + phase)
    fast = np.sin(2 * np.pi * i / 6.0 + phase * 2.1)
    return 0.55 * slow + 0.30 * (strength**1.5) * medium + 0.15 * (strength**3) * fast


def apply_camera_jitter(
    trajectory: Trajectory,
    strength: float,
    max_degrees: float = DEFAULT_JITTER_MAX_DEGREES,
    max_translation: float = DEFAULT_JITTER_MAX_TRANSLATION,
) -> Trajectory:
    """Post-process a built `Trajectory` with a small, deterministic
    handheld-camera wobble: a sum of a few fixed, incommensurate sine waves
    driving both rotation (pitch/yaw/roll) and position (right/up/forward),
    applied in the *camera's own* local frame -- so it reads as the camera
    nodding and drifting slightly in the hand, not the world moving around
    it. Rotation and translation use independent phases, so they don't
    visibly lock-step into a single obvious oscillation.

    Indexed by frame number `i`, not by time -- matching every other
    periodic/rate-based quantity in this codebase, frames are the canonical
    unit, so a 300-frame trajectory shakes at the same per-frame rate as a
    30-frame one rather than stretching to fit.

    `strength` in [0, 1] scales overall amplitude linearly, and additionally
    reshapes *frequency content* (see `_wobble`): 0 leaves the trajectory
    bit-for-bit unchanged (returns the same object), 0.5 is a gentle, mostly
    single-frequency sway, and 1.0 is the peak, busiest ("handshake") wobble.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if strength == 0.0:
        return trajectory

    n = len(trajectory)
    i = np.arange(n, dtype=np.float64)

    angle_amplitude = np.deg2rad(max_degrees * strength)
    pitch = angle_amplitude * _wobble(i, 0.0, strength)
    yaw = angle_amplitude * _wobble(i, 1.3, strength)
    roll = angle_amplitude * 0.6 * _wobble(i, 2.6, strength)
    jitter_rotation = Rotation.from_euler("xyz", np.stack([pitch, yaw, roll], axis=1))
    rotations = trajectory.rotations * jitter_rotation  # local wobble: base pose first, then jitter about its own axes

    translation_amplitude = max_translation * strength
    right_shift = translation_amplitude * _wobble(i, 4.1, strength)
    up_shift = translation_amplitude * 0.7 * _wobble(i, 5.7, strength)
    forward_shift = translation_amplitude * 0.4 * _wobble(i, 6.9, strength)
    right = rotations.apply([1.0, 0.0, 0.0])
    up = rotations.apply([0.0, 1.0, 0.0])
    forward = rotations.apply([0.0, 0.0, -1.0])  # camera looks down its local -Z
    offset = right_shift[:, None] * right + up_shift[:, None] * up + forward_shift[:, None] * forward
    positions = trajectory.positions + offset

    return dataclasses.replace(trajectory, positions=positions, rotations=rotations)
