"""Auteur framing designer -- a first test demo, separate from the
world-space-motion apps (trajectory_designer / batch_generator /
structure_batch_generator).

Instead of moving a camera through world space around a box, Auteur frames a
standing actor via six actor-relative axes -- camera level, look-at level,
shot scale, lateral framing offset, orientation angle, dutch tilt -- at a
sequence of keyframes, and interpolates *those six numbers* (not raw
position/rotation) to produce the camera path. The actor always starts at
the world origin and, optionally, slides in a straight line to wherever you
place it -- "nothing fancy," no easing -- while the camera's framing keeps
recalculating relative to wherever the actor currently is. The actor is a
static-shape SOMA body (https://github.com/NVlabs/SOMA-X, "mhr" identity
model only) in A-pose.

Requires the `auteur` extra (heavy: torch + NVIDIA Warp):
    pip install -e ".[auteur]"

Run:
    micromamba run -n camera python -m apps.auteur_designer
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import os
import tempfile
import threading
import time
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import camtraj_viser as viser
from scipy.spatial.transform import Rotation

from camtraj.auteur import (
    ActorMotion,
    AuteurKeyframe,
    DEFAULT_ASPECT,
    FramingState,
    apply_camera_jitter,
    build_auteur_trajectory,
    freeze_camera_after,
)
from camtraj.conventions import KNOWN_CONVENTIONS, OPENGL, convert_pose
from camtraj.easing import Easing
from camtraj.soma_actor import build_a_pose_actor

from ._shared import viz

PLAYBACK_FPS = 30.0
MIN_KEYFRAMES = 1  # K1 alone is a valid trajectory -- static for its whole length
MAX_KEYFRAMES = 8
TOTAL_FRAMES_MIN = 21
TOTAL_FRAMES_MAX = 300
POV_SIZE = (256, 384)  # (height, width) -- both divisible by 16 (h264 macroblock size), matches DEFAULT_ASPECT = 1.5
POV_MAX_FRAMES = 90  # subsample long trajectories so a render pass stays bounded
POV_PLAYBACK_FPS = 12.0  # minimum playback rate for the rendered POV sequence
DEFAULT_TOTAL_FRAMES = 121

# A single fixed 60deg FOV (the shared camtraj.auteur.DEFAULT_FOV, tuned for
# the box-and-frustum apps) puts tight shot_scales absurdly close -- an ECU
# would sit ~0.19m from the actor's centerline, well inside their own body
# mesh. Real cinematography solves exactly this by swapping to a longer,
# narrower-fov lens for close-ups rather than moving the camera closer; this
# app doesn't vary FOV per shot (yet), so a narrower *fixed* FOV moves the
# whole range back to a comfortable distance instead. decode_framing's own
# min_distance floor (see camtraj.auteur) is the hard backstop for whatever
# this doesn't fully solve at the tightest end of the shot_scale slider.
AUTEUR_FOV = float(np.deg2rad(28.0))

# (min, max, marks) for each of Auteur's six framing axes -- domains and
# reference marks from its own DSL vocabulary (radians for angles, meters
# for heights, unitless fractions otherwise).
#
# Shot scale's marks are recalibrated against real framing intent, not the
# Auteur DSL writeup's own numbers (which were both backwards -- ECU at the
# *smallest* number, EWS at the *largest*, the opposite of "ECU closest" --
# and, even corrected for direction, far too loose: its (0, 1] domain caps
# out at "whole body exactly fills frame," which can't represent a real
# close-up at all). shot_scale = actor_height / frame_height, so a true
# close-up -- showing *less* than the full body -- needs shot_scale > 1;
# see decode_framing's docstring. Anchored on ECU and MS, then spaced at a
# roughly constant *ratio* (not difference) between neighbors, since
# shot_scale is a multiplicative "how much closer" quantity -- a linear
# slider bunches every wide-shot mark together near 0 while leaving a huge
# gap between the two closest ones. "Full shot" is deliberately pushed out
# to where "extreme wide" used to sit (whole body only 3/5 of frame height),
# with EWS further out still, and the slider itself is log-scaled (see
# _ss_to_slider/_slider_from_ss) so equal drag distance means equal zoom
# ratio everywhere along it, not just near the marks:
#   ECU  ~ only the face fills the frame       (face ~= height / 8)
#   MS   ~ waist-up                            (waist ~= height / 2)
#   FS   ~ whole body is 3/5 of frame height   (shot_scale == 0.65)
#   EWS  ~ whole body is 1/4 of frame height   (shot_scale == 0.25)
_CL_MARKS = {0.1: "ground", 0.6: "low", 1.6: "eye", 2.5: "high", 4.0: "overhead"}
_SS_MARKS = {8.0: "ECU", 5.0: "CU", 3.2: "MCU", 2.0: "MS", 1.2: "MLS", 0.65: "FS", 0.4: "LS", 0.25: "EWS"}
_SS_MARKS_BY_NAME = {label: value for value, label in _SS_MARKS.items()}
_SS_MIN, _SS_MAX = 0.15, 10.0


def _ss_to_slider(shot_scale: float) -> float:
    """shot_scale -> log10-space slider position (see _SS_MARKS' comment)."""
    return float(np.log10(shot_scale))


def _slider_to_ss(slider_value: float) -> float:
    return float(10.0**slider_value)
_FO_MARKS = {-0.35: "far_left", -0.17: "left", 0.0: "center", 0.17: "right", 0.35: "far_right"}
_OA_MARKS = {
    0.0: "front", 45.0: "front_r", 90.0: "side_r", 135.0: "back_r",
    180.0: "back", -135.0: "back_l", -90.0: "side_l", -45.0: "front_l",
}
_DA_MARKS = {-25.0: "strong_l", -10.0: "slight_l", 0.0: "none", 10.0: "slight_r", 25.0: "strong_r"}


def _frame_text(t: float, duration: float) -> str:
    return f"**Frame {t:0.1f}** / {duration:0.0f}"


def _default_keyframe(shot_scale: float, orientation: float, actor_height: float) -> FramingState:
    # A simple default sweep so a fresh session already shows something
    # meaningful, both keyframes aimed at eye level.
    return FramingState(
        camera_level=1.6, lookat_level=0.95 * actor_height, shot_scale=shot_scale, orientation=orientation
    )


AUTEUR_RECIPE_VERSION = 1


def _auteur_recipe_to_dict(state: dict) -> dict:
    """The editable Auteur *design* (actor motion + keyframes + transition
    settings) as a JSON-able dict -- as opposed to Export, which bakes the
    *current* trajectory into numeric arrays. Angles are stored in degrees
    (matching every angle slider in this app) so a saved recipe reads
    naturally if hand-edited."""
    return {
        "auteur_recipe_version": AUTEUR_RECIPE_VERSION,
        "actor_end_position": list(state["actor_end_position"]),
        "actor_yaw_deg": state["actor_yaw_deg"],
        "total_frames": state["total_frames"],
        "easing": state["easing"].value,
        "easing_strength": state["easing_strength"],
        "jitter": state["jitter"],
        "keyframes": [
            {
                "frame": kf["frame"],
                "camera_level": kf["state"].camera_level,
                "lookat_level": kf["state"].lookat_level,
                "shot_scale": kf["state"].shot_scale,
                "framing_offset": kf["state"].framing_offset,
                "orientation_deg": np.rad2deg(kf["state"].orientation),
                "dutch_deg": np.rad2deg(kf["state"].dutch),
            }
            for kf in state["keyframes"]
        ],
    }


def _auteur_recipe_from_dict(data: dict) -> dict:
    """Inverse of `_auteur_recipe_to_dict` -- returns a dict of the fields in
    `state` that a recipe covers, ready to assign in on the caller's own
    `state` (transient fields like `selected`/`t`/`playing` aren't part of a
    recipe and are left for the caller to reset)."""
    keyframes = data["keyframes"]
    if not keyframes:
        raise ValueError("Recipe has no keyframes.")
    return {
        "actor_end_position": np.array(data["actor_end_position"], dtype=np.float64),
        "actor_yaw_deg": data["actor_yaw_deg"],
        "total_frames": data["total_frames"],
        "easing": Easing(data["easing"]),
        "easing_strength": data["easing_strength"],
        "jitter": data["jitter"],
        "keyframes": [
            {
                "frame": kf["frame"],
                "state": FramingState(
                    camera_level=kf["camera_level"],
                    lookat_level=kf["lookat_level"],
                    shot_scale=kf["shot_scale"],
                    framing_offset=kf["framing_offset"],
                    orientation=np.deg2rad(kf["orientation_deg"]),
                    dutch=np.deg2rad(kf["dutch_deg"]),
                ),
            }
            for kf in keyframes
        ],
    }


def main(port: int = 8083, share: bool = True) -> None:
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(brand_color=(70, 150, 220))
    viz.add_ground(server)

    server.scene.set_up_direction("+y")
    server.initial_camera.position = (3.0, 2.2, 6.0)
    server.initial_camera.look_at = (0.0, 0.9, 0.0)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    print("Loading SOMA actor (first run downloads model weights, can take a minute)...")
    actor_mesh = build_a_pose_actor(device="cpu")
    print(f"Actor loaded -- height {actor_mesh.height:.2f} m.")

    # Look-at level marks, same idea as camera level's but expressed as
    # fractions of *this* actor's own height (anatomically-ish: feet/knees/
    # waist/chest/eyes), since "where on the actor am I aiming" only makes
    # sense relative to them -- unlike camera level, which is a world-space
    # height independent of anybody's size.
    _LOOKAT_MARKS = {
        0.0: "feet",
        0.25 * actor_mesh.height: "knees",
        0.50 * actor_mesh.height: "waist",
        0.75 * actor_mesh.height: "chest",
        0.95 * actor_mesh.height: "eyes",
    }

    state = {
        "actor_end_position": np.array([0.0, 0.0, 0.0]),  # actor always *starts* at the origin
        "actor_yaw_deg": 0.0,
        "total_frames": DEFAULT_TOTAL_FRAMES,
        # Kept sorted by "frame" at all times -- the timeline widget's own
        # neighbor-clamped dragging (mirrored server-side in the action
        # handler below) guarantees this invariant never breaks. K1 (index 0)
        # is always frame 0 and is never moved/removed.
        "keyframes": [
            # K1: long shot from behind; K2: close-up from front-right (45deg).
            {"frame": 0, "state": _default_keyframe(_SS_MARKS_BY_NAME["LS"], np.pi, actor_mesh.height)},
            {
                "frame": DEFAULT_TOTAL_FRAMES - 1,
                "state": _default_keyframe(_SS_MARKS_BY_NAME["CU"], np.pi / 4, actor_mesh.height),
            },
        ],
        "selected": 0,
        "easing": Easing.LINEAR,
        "easing_strength": 0.5,
        "jitter": 0.0,
        "t": 0.0,
        "playing": False,
    }

    actor_handle = server.scene.add_mesh_simple(
        "/actor", actor_mesh.vertices, actor_mesh.faces, color=(210, 170, 140),
        position=(0.0, 0.0, 0.0), wxyz=viz.to_frame_wxyz(Rotation.identity()),
    )

    def current_actor_motion() -> ActorMotion:
        return ActorMotion(
            end_position=state["actor_end_position"],
            end_yaw=np.deg2rad(state["actor_yaw_deg"]),
            height=actor_mesh.height,
        )

    def build_trajectory():
        kfs = state["keyframes"]
        auteur_keyframes = [AuteurKeyframe(kfs[0]["state"])]
        for prev, cur in zip(kfs[:-1], kfs[1:]):
            span = cur["frame"] - prev["frame"] + 1
            auteur_keyframes.append(
                AuteurKeyframe(cur["state"], frames=span, easing=state["easing"], easing_strength=state["easing_strength"])
            )
        # If the last keyframe isn't at the final frame, hold for the
        # remainder -- but frozen at the camera's exact world pose from the
        # last keyframe, not re-derived every frame from the actor's
        # continuing position. Otherwise a still-moving actor would keep
        # dragging a "static" camera along to maintain the same relative
        # framing, which isn't static at all. The actor itself (box_positions
        # / its rendered mesh) keeps moving as normal; only the camera locks.
        last = kfs[-1]
        tail = (state["total_frames"] - 1) - last["frame"]
        if tail > 0:
            auteur_keyframes.append(AuteurKeyframe(last["state"], frames=tail + 1))
        traj = build_auteur_trajectory(auteur_keyframes, current_actor_motion(), fov=AUTEUR_FOV, aspect=DEFAULT_ASPECT)
        if tail > 0:
            traj = freeze_camera_after(traj, last["frame"])
        return apply_camera_jitter(traj, state["jitter"])

    trajectory = build_trajectory()
    path_handle = viz.add_path(server, "/path", trajectory)
    frustum_handles = viz.add_keyframe_frustums(server, "/keyframes", trajectory, stride=max(1, len(trajectory) // 8))
    current_cam = viz.add_current_camera(server, "/current_camera", trajectory, t=0.0)

    def update_actor_pose(t: float) -> None:
        actor_handle.position = trajectory.box_position_at(t)
        t_fraction = (t / trajectory.duration) if trajectory.duration > 0 else 0.0
        yaw = current_actor_motion().state_at(t_fraction).yaw
        actor_handle.wxyz = viz.to_frame_wxyz(Rotation.from_euler("y", yaw))

    def rebuild_and_redraw() -> None:
        nonlocal trajectory, path_handle
        trajectory = build_trajectory()
        path_handle.remove()
        path_handle = viz.add_path(server, "/path", trajectory)
        viz.clear_handles(frustum_handles)
        frustum_handles.extend(
            viz.add_keyframe_frustums(server, "/keyframes", trajectory, stride=max(1, len(trajectory) // 8))
        )
        for h in frustum_handles:
            h.visible = not state["playing"]
        state["t"] = min(state["t"], trajectory.duration)
        scrubber.max = trajectory.duration
        scrubber.value = state["t"]
        viz.update_current_camera(current_cam, trajectory, state["t"])
        update_actor_pose(state["t"])
        frame_readout.content = _frame_text(state["t"], trajectory.duration)

    # --- Guide: a closeable quick-reference note ------------------------------

    GUIDE_MARKDOWN = """**Quick guide**

- Click the timeline bar to add a keyframe; drag a pin to move it.
- Click a pin to select it, then edit its framing below.
- **Delete** / **Backspace** removes the selected keyframe (K1 can't be removed).
- **Space** plays/pauses; playback loops automatically.
- Shot scale is log-scaled: bigger = camera closer (ECU), smaller = farther (EWS).

Press **i** to reopen this note."""

    with server.gui.add_folder("Guide", order=-1000) as guide_folder:
        server.gui.add_markdown(GUIDE_MARKDOWN)
        guide_close_button = server.gui.add_button("Close", icon=viser.Icon.X)

        @guide_close_button.on_click
        def _(_) -> None:
            guide_folder.visible = False

    guide_toggle_command = server.gui.add_command("Toggle guide note", hotkey="i", icon=viser.Icon.INFO_CIRCLE)

    @guide_toggle_command.on_trigger
    def _(_) -> None:
        guide_folder.visible = not guide_folder.visible

    # --- Actor -----------------------------------------------------------
    # Always starts at the world origin facing +Z (yaw 0); "end position"/
    # "end yaw" are the last point/facing it reaches, via a direct linear
    # slide/turn -- no easing, no curve.

    with server.gui.add_folder("Actor"):
        end_x_slider = server.gui.add_slider("End position X", min=-15.0, max=15.0, step=0.1, initial_value=0.0)
        end_z_slider = server.gui.add_slider("End position Z", min=-15.0, max=15.0, step=0.1, initial_value=0.0)
        yaw_slider = server.gui.add_slider("End yaw", min=-180.0, max=180.0, step=1.0, initial_value=0.0, hint="degrees -- the actor turns to face this by the end")

        @end_x_slider.on_update
        def _(_) -> None:
            state["actor_end_position"][0] = end_x_slider.value
            rebuild_and_redraw()

        @end_z_slider.on_update
        def _(_) -> None:
            state["actor_end_position"][2] = end_z_slider.value
            rebuild_and_redraw()

        @yaw_slider.on_update
        def _(_) -> None:
            state["actor_yaw_deg"] = yaw_slider.value
            rebuild_and_redraw()

    # --- Camera keyframe timeline ---------------------------------------
    # A standalone overlay (like the other apps' segment timeline) rather
    # than sidebar widgets: a fixed-width bar spanning the whole trajectory,
    # with draggable pins at each keyframe's exact frame. K1 is pinned to
    # frame 0 and can't move or be removed; every other pin is constrained
    # between its neighbors (no crossing, no overlap). If the last keyframe
    # isn't at the final frame, the remainder holds static (see
    # build_trajectory above) -- so there's no separate "static" toggle
    # needed: put K2 right after K1 for an (almost) fully static shot, or
    # leave a gap after the last keyframe for a "settle and hold" ending.

    def sync_keyframe_timeline() -> None:
        labels_and_frames = [(f"K{i + 1}", kf["frame"]) for i, kf in enumerate(state["keyframes"])]
        server.gui.set_keyframe_timeline(
            labels_and_frames, state["selected"], state["total_frames"], TOTAL_FRAMES_MIN, TOTAL_FRAMES_MAX
        )

    def _remove_keyframe(index: int) -> None:
        """Shared by the timeline's per-pin "x" and the Delete/Backspace
        hotkey (see below) -- same guards either way: K1 (index 0) can never
        be removed. A lone K1 is itself a perfectly valid trajectory (static
        for the whole length -- see build_trajectory's tail logic), so
        that's the only real floor; MIN_KEYFRAMES stays as an explicit,
        defensive second guard."""
        kfs = state["keyframes"]
        if index == 0 or not (0 <= index < len(kfs)) or len(kfs) <= MIN_KEYFRAMES:
            return
        kfs.pop(index)
        state["selected"] = min(state["selected"], len(kfs) - 1)
        sync_keyframe_timeline()
        rebuild_keyframe_sliders()
        rebuild_and_redraw()

    def _on_keyframe_timeline_action(action, index: int, frame: int) -> None:
        kfs = state["keyframes"]

        if action == "select":
            if 0 <= index < len(kfs) and index != state["selected"]:
                state["selected"] = index
                sync_keyframe_timeline()
                rebuild_keyframe_sliders()
            return

        if action == "add":
            if len(kfs) >= MAX_KEYFRAMES:
                return
            new_frame = max(1, min(state["total_frames"] - 1, frame))
            insert_at = sum(1 for kf in kfs if kf["frame"] < new_frame)
            if insert_at > 0 and kfs[insert_at - 1]["frame"] == new_frame:
                new_frame = min(state["total_frames"] - 1, new_frame + 1)
            if insert_at < len(kfs) and kfs[insert_at]["frame"] == new_frame:
                new_frame = max(1, new_frame - 1)
            base_state = kfs[max(0, insert_at - 1)]["state"]
            kfs.insert(insert_at, {"frame": new_frame, "state": base_state})
            state["selected"] = insert_at
            sync_keyframe_timeline()
            rebuild_keyframe_sliders()
            rebuild_and_redraw()
            return

        if action == "remove":
            _remove_keyframe(index)
            return

        if action == "move":
            if index == 0 or not (0 <= index < len(kfs)):
                return
            lo = kfs[index - 1]["frame"] + 1
            hi = (state["total_frames"] - 1) if index == len(kfs) - 1 else kfs[index + 1]["frame"] - 1
            kfs[index]["frame"] = max(lo, min(hi, frame))
            sync_keyframe_timeline()
            rebuild_and_redraw()
            return

        if action == "set_total_frames":
            new_total = max(TOTAL_FRAMES_MIN, min(TOTAL_FRAMES_MAX, frame))
            new_total = max(new_total, kfs[-1]["frame"] + 1)  # never truncate past the last keyframe
            state["total_frames"] = new_total
            sync_keyframe_timeline()
            rebuild_and_redraw()
            return

    server.gui.on_keyframe_timeline_action(_on_keyframe_timeline_action)

    # Delete/Backspace: remove the *selected* keyframe -- handy since the
    # timeline's embedded "x" is small and only ever targets whichever pin
    # it's clicked on, not necessarily the selected one.
    delete_command = server.gui.add_command("Delete selected keyframe", hotkey="delete", icon=viser.Icon.TRASH)
    backspace_command = server.gui.add_command("Delete selected keyframe (backspace)", hotkey="backspace")

    @delete_command.on_trigger
    def _(_) -> None:
        _remove_keyframe(state["selected"])

    @backspace_command.on_trigger
    def _(_) -> None:
        _remove_keyframe(state["selected"])
    sync_keyframe_timeline()

    # --- Selected keyframe's framing sliders ------------------------------

    keyframe_folder = server.gui.add_folder("Keyframe")
    keyframe_handles: list = []

    def rebuild_keyframe_sliders() -> None:
        for h in keyframe_handles:
            h.remove()
        keyframe_handles.clear()

        index = state["selected"]
        kf = state["keyframes"][index]
        fstate = kf["state"]

        def on_change(name: str, value: float) -> None:
            state["keyframes"][index]["state"] = dataclasses.replace(state["keyframes"][index]["state"], **{name: value})
            rebuild_and_redraw()

        with keyframe_folder:
            heading = server.gui.add_markdown(f"**K{index + 1}** -- frame {kf['frame']}")
            keyframe_handles.append(heading)

            cl = server.gui.add_slider("Camera level", min=0.1, max=4.0, step=0.05, initial_value=fstate.camera_level,
                                        marks=tuple(sorted(_CL_MARKS.items())), hint="meters above ground")
            ll = server.gui.add_slider("Look-at level", min=0.0, max=2.2, step=0.05, initial_value=fstate.lookat_level,
                                        marks=tuple(sorted(_LOOKAT_MARKS.items())), hint="meters above ground, aimed height on the actor")
            ss = server.gui.add_slider(
                "Shot scale", min=_ss_to_slider(_SS_MIN), max=_ss_to_slider(_SS_MAX), step=0.01,
                initial_value=_ss_to_slider(fstate.shot_scale),
                marks=tuple(sorted((_ss_to_slider(v), label) for v, label in _SS_MARKS.items())),
                hint="bigger = camera closer (ECU); smaller = farther (EWS) -- log scale",
            )
            fo = server.gui.add_slider("Framing offset", min=-0.5, max=0.5, step=0.01, initial_value=fstate.framing_offset,
                                        marks=tuple(sorted(_FO_MARKS.items())), hint="lateral lead-room")
            oa = server.gui.add_slider("Orientation", min=-180.0, max=180.0, step=1.0, initial_value=np.rad2deg(fstate.orientation),
                                        marks=tuple(sorted(_OA_MARKS.items())), hint="degrees -- camera azimuth around the actor")
            da = server.gui.add_slider("Dutch angle", min=-45.0, max=45.0, step=1.0, initial_value=np.rad2deg(fstate.dutch),
                                        marks=tuple(sorted(_DA_MARKS.items())), hint="degrees")

            cl.on_update(lambda _: on_change("camera_level", cl.value))
            ll.on_update(lambda _: on_change("lookat_level", ll.value))
            ss.on_update(lambda _: on_change("shot_scale", _slider_to_ss(ss.value)))
            fo.on_update(lambda _: on_change("framing_offset", fo.value))
            oa.on_update(lambda _: on_change("orientation", np.deg2rad(oa.value)))
            da.on_update(lambda _: on_change("dutch", np.deg2rad(da.value)))
            keyframe_handles.extend([cl, ll, ss, fo, oa, da])

    rebuild_keyframe_sliders()

    with server.gui.add_folder("Transition", expand_by_default=False):
        easing_dropdown = server.gui.add_dropdown("Easing", options=[e.value for e in Easing], initial_value=state["easing"].value)
        easing_strength_slider = server.gui.add_slider(
            "Easing strength", min=0.0, max=1.0, step=0.05, initial_value=state["easing_strength"],
            marks=((0.0, "off"), (0.5, "default"), (1.0, "full")),
        )
        jitter_slider = server.gui.add_slider(
            "Jitter", min=0.0, max=1.0, step=0.05, initial_value=state["jitter"],
            marks=((0.0, "none"), (0.5, "low"), (1.0, "high")),
            hint="handheld-camera wobble, applied on top of the whole trajectory",
        )

        @easing_dropdown.on_update
        def _(_) -> None:
            state["easing"] = Easing(easing_dropdown.value)
            rebuild_and_redraw()

        @easing_strength_slider.on_update
        def _(_) -> None:
            state["easing_strength"] = easing_strength_slider.value
            rebuild_and_redraw()

        @jitter_slider.on_update
        def _(_) -> None:
            state["jitter"] = jitter_slider.value
            rebuild_and_redraw()

    # --- Playback ------------------------------------------------------------

    with server.gui.add_folder("Playback"):
        scrubber = server.gui.add_slider("Frame", min=0.0, max=trajectory.duration, step=0.1, initial_value=0.0)
        frame_readout = server.gui.add_markdown(_frame_text(0.0, trajectory.duration))
        play_pause_button = server.gui.add_button(
            "Play", icon=viser.Icon.PLAYER_PLAY, hint="Space bar also toggles play/pause"
        )

        @scrubber.on_update
        def _(_) -> None:
            state["t"] = scrubber.value
            viz.update_current_camera(current_cam, trajectory, state["t"])
            update_actor_pose(state["t"])
            frame_readout.content = _frame_text(state["t"], trajectory.duration)

        def set_playing(value: bool) -> None:
            state["playing"] = value
            play_pause_button.icon = viser.Icon.PLAYER_PAUSE if value else viser.Icon.PLAYER_PLAY
            for h in frustum_handles:
                h.visible = not value

        @play_pause_button.on_click
        def _(_) -> None:
            set_playing(not state["playing"])

    space_command = server.gui.add_command(
        "Play / pause trajectory", hotkey="space", icon=viser.Icon.PLAYER_PLAY
    )

    @space_command.on_trigger
    def _(_) -> None:
        set_playing(not state["playing"])

    # --- Camera POV: a separate, on-demand player -----------------------------
    # Same decoupled-from-the-scrubber pattern as the other apps: only an
    # explicit click renders a fresh batch of frames. Unlike the other apps'
    # flipbook-in-a-loop player, the rendered frames are encoded to an actual
    # .mp4 (via imageio + ffmpeg) and embedded as a native <video> element --
    # real scrub/pause/replay controls, courtesy of the browser itself,
    # instead of a custom sleep-loop with no way to stop or rewind.
    # Renders the low-lod actor mesh (see camtraj.soma_actor) -- rendering
    # full-resolution here would be considerably heavier per frame.

    pov_busy = [False]

    def _run_pov_job(target) -> None:
        if pov_busy[0]:
            return

        def _guarded() -> None:
            pov_busy[0] = True
            try:
                target()
            finally:
                pov_busy[0] = False

        threading.Thread(target=_guarded, daemon=True).start()

    def _encode_pov_video(frames: list[np.ndarray], fps: float) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        try:
            with imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(frame)
            with open(path, "rb") as f:
                return f.read()
        finally:
            os.unlink(path)

    def render_and_play_pov() -> None:
        clients = server.get_clients()
        if not clients:
            pov_status.content = "No connected browser to render from."
            return
        client = next(iter(clients.values()))

        n = max(2, min(len(trajectory), POV_MAX_FRAMES))
        query_times = np.linspace(trajectory.times[0], trajectory.times[-1], n)

        was_playing = state["playing"]
        set_playing(False)  # avoid the playback loop fighting over current_cam mid-capture
        path_handle.visible = False
        for h in frustum_handles:
            h.visible = False
        current_cam.visible = False

        frames: list[np.ndarray] = []
        try:
            for i, t in enumerate(query_times):
                pov_status.content = f"Rendering frame {i + 1} / {n}..."
                position, rotation = trajectory.pose_at(t)
                cv_position, wxyz = viz.to_frustum_pose(position, rotation)
                update_actor_pose(t)
                try:
                    frames.append(client.get_render(*POV_SIZE, wxyz=wxyz, position=cv_position, fov=AUTEUR_FOV, timeout=2.0))
                except TimeoutError:
                    break
        finally:
            path_handle.visible = True
            for h in frustum_handles:
                h.visible = not was_playing
            current_cam.visible = True
            update_actor_pose(state["t"])
            if was_playing:
                set_playing(True)

        if not frames:
            pov_status.content = "Render failed -- no frames came back."
            return

        pov_status.content = "Encoding video..."
        fps = max(POV_PLAYBACK_FPS, len(frames) / max(trajectory.duration / PLAYBACK_FPS, 1e-6))
        try:
            video_bytes = _encode_pov_video(frames, fps)
        except Exception as e:
            pov_status.content = f"Video encoding failed: {e}"
            return
        b64 = base64.b64encode(video_bytes).decode("ascii")
        pov_video_html.content = (
            f'<video controls autoplay muted loop style="width: 100%; border-radius: 0.3em;" '
            f'src="data:video/mp4;base64,{b64}"></video>'
        )
        pov_status.content = f"Done -- {len(frames)} frames encoded."

    with server.gui.add_folder("Camera POV", expand_by_default=False):
        pov_video_html = server.gui.add_html("")
        pov_status = server.gui.add_markdown("Not rendered yet.")
        pov_play_button = server.gui.add_button("Render & Play POV", icon=viser.Icon.PLAYER_PLAY)

        @pov_play_button.on_click
        def _(_) -> None:
            _run_pov_job(render_and_play_pov)

    # --- Recipe: save/load the keyframe design itself -------------------------
    # As opposed to Export below, which bakes the *current* trajectory into
    # numeric arrays, a recipe is the editable design (actor motion +
    # keyframes + transition settings) -- what you'd persist across
    # sessions or share with a teammate.

    with server.gui.add_folder("Recipe"):
        save_recipe_button = server.gui.add_button("Save recipe (.json)", icon=viser.Icon.DEVICE_FLOPPY)
        load_recipe_button = server.gui.add_upload_button(
            "Load recipe (.json)", icon=viser.Icon.UPLOAD, mime_type=".json,application/json"
        )
        recipe_status = server.gui.add_markdown("")

        @save_recipe_button.on_click
        def _(event: viser.GuiEvent) -> None:
            data = _auteur_recipe_to_dict(state)
            content = json.dumps(data, indent=2).encode("utf-8")
            target = event.client if event.client is not None else server
            target.send_file_download("camtraj_auteur_recipe.json", content)

        @load_recipe_button.on_upload
        def _(_) -> None:
            try:
                fields = _auteur_recipe_from_dict(json.loads(load_recipe_button.value.content.decode("utf-8")))
            except Exception as e:
                recipe_status.content = f"**Load failed:** {e}"
                return
            state.update(fields)
            state["selected"] = 0
            recipe_status.content = f"Loaded {len(fields['keyframes'])} keyframe(s)."
            end_x_slider.value = state["actor_end_position"][0]
            end_z_slider.value = state["actor_end_position"][2]
            yaw_slider.value = state["actor_yaw_deg"]
            easing_dropdown.value = state["easing"].value
            easing_strength_slider.value = state["easing_strength"]
            jitter_slider.value = state["jitter"]
            sync_keyframe_timeline()
            rebuild_keyframe_sliders()
            rebuild_and_redraw()

    # --- Export: camera + actor, as two separate .npz files ------------------
    # The camera trajectory converts (via camtraj.conventions.convert_pose)
    # from the canonical OpenGL/c2w convention into whichever is picked, same
    # as the other apps. Actor positions are always plain world-space
    # coordinates (not a camera pose), so they're exported as-is regardless
    # of the selected camera convention.

    with server.gui.add_folder("Export"):
        convention_dropdown = server.gui.add_dropdown(
            "Convention", options=list(KNOWN_CONVENTIONS.keys()), initial_value=OPENGL.name
        )
        convention_info = server.gui.add_markdown(KNOWN_CONVENTIONS[OPENGL.name].description)

        @convention_dropdown.on_update
        def _(_) -> None:
            convention_info.content = KNOWN_CONVENTIONS[convention_dropdown.value].description

        export_button = server.gui.add_button("Download trajectory (.npz)", icon=viser.Icon.DOWNLOAD)

        @export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            target_convention = KNOWN_CONVENTIONS[convention_dropdown.value]
            cam_positions, cam_rotations = convert_pose(
                trajectory.positions, trajectory.rotations, OPENGL, target_convention
            )

            camera_buf = io.BytesIO()
            np.savez(
                camera_buf,
                times=trajectory.times,
                positions=cam_positions,
                quaternions_xyzw=cam_rotations.as_quat(),
                convention=target_convention.name,
            )
            actor_buf = io.BytesIO()
            np.savez(actor_buf, times=trajectory.times, positions=trajectory.box_positions)

            target = event.client if event.client is not None else server
            target.send_file_download(f"camtraj_auteur_camera_{target_convention.name}.npz", camera_buf.getvalue())
            target.send_file_download("camtraj_auteur_actor.npz", actor_buf.getvalue())

    def playback_loop() -> None:
        last = time.time()
        while True:
            now = time.time()
            dt = now - last
            last = now
            if state["playing"]:
                new_t = state["t"] + dt * PLAYBACK_FPS
                if new_t >= trajectory.duration:
                    new_t = 0.0  # loop back to the start rather than stopping
                state["t"] = new_t
                scrubber.value = new_t
                viz.update_current_camera(current_cam, trajectory, new_t)
                update_actor_pose(new_t)
                frame_readout.content = _frame_text(new_t, trajectory.duration)
            time.sleep(1.0 / PLAYBACK_FPS)

    threading.Thread(target=playback_loop, daemon=True).start()

    print(f"\ncamtraj auteur designer running -- local: http://localhost:{server.get_port()}")
    if share:
        url = server.request_share_url()
        if url:
            print(f"camtraj auteur designer running -- public share URL: {url}\n")
        else:
            print("Could not obtain a public share URL (relay unreachable?); use the local URL above.\n")
    server.sleep_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()
    main(port=args.port, share=args.share)
