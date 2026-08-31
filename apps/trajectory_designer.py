"""Interactive multi-segment free_form trajectory designer.

Run:
    micromamba run -n camera python -m apps.trajectory_designer
A public share URL (via viser's built-in relay) is requested by default and
printed alongside the local one -- pass --no-share to skip it and use
localhost / manual port-forwarding instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import threading
import time
from typing import Optional

import numpy as np
import viser

from camtraj import Easing, look_at_rotation, sequence
from camtraj.segments import FreeFormSegment
from camtraj.trajectory import Trajectory

from ._shared import viz
from ._shared.param_forms import add_param_sliders

MAX_SEGMENTS = 10
PLAYBACK_FPS = 30.0  # nominal real-time rate mapping frame-units to wall-clock seconds
POV_MIN_INTERVAL = 0.2  # seconds; also throttles how often the main view's own
# frustums/path have to blink out to keep them out of the POV capture
POV_SIZE = (240, 360)  # (height, width), matches viz.DEFAULT_ASPECT = 1.5

_START_POSITION = viz.DEFAULT_CAMERA_START
# Face the cube's heading (forward/backward, left/right) but stay level -- zero
# pitch -- rather than tilting to center the cube exactly, by aiming at a point
# at the *camera's own height* instead of the cube's.
_LEVEL_TARGET = np.array([viz.CUBE_CENTER[0], _START_POSITION[1], viz.CUBE_CENTER[2]])
_START_ROTATION = look_at_rotation(_START_POSITION, _LEVEL_TARGET)


def _frame_text(t: float, duration: float) -> str:
    return f"**Frame {t:0.1f}** / {duration:0.0f}"


def main(port: int = 8080, share: bool = True) -> None:
    server = viser.ViserServer(port=port)
    viz.add_ground(server)

    # Lock the viewer's orbit controls to a Y-up turntable (drag = yaw/pitch
    # only, never roll), and start the view behind the camera+cube looking in.
    server.scene.set_up_direction("+y")
    server.initial_camera.position = (3.0, 2.5, 10.0)
    server.initial_camera.look_at = (0.0, 0.75, 1.5)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    state = {
        "segments": [FreeFormSegment()],
        "selected": 0,
        "t": 0.0,
        "playing": False,
    }

    def build_full_trajectory() -> Trajectory:
        return sequence(state["segments"], start_position=_START_POSITION, start_rotation=_START_ROTATION)

    state["trajectory"] = build_full_trajectory()

    path_handle = viz.add_path(server, "/path", state["trajectory"])
    frustum_handles = viz.add_keyframe_frustums(server, "/keyframes", state["trajectory"])
    current_cam = viz.add_current_camera(server, "/current_camera", state["trajectory"], t=0.0)

    scrubber: Optional[viser.GuiSliderHandle] = None
    frame_readout: Optional[viser.GuiMarkdownHandle] = None
    play_pause_button: Optional[viser.GuiButtonHandle] = None
    segment_selector: Optional[viser.GuiDropdownHandle] = None
    pov_checkbox: Optional[viser.GuiCheckboxHandle] = None
    pov_image: Optional[viser.GuiImageHandle] = None
    pov_last_render = [0.0]

    def update_pov(force: bool = False) -> None:
        if pov_checkbox is None or not pov_checkbox.value:
            return
        now = time.time()
        if not force and (now - pov_last_render[0]) < POV_MIN_INTERVAL:
            return
        clients = server.get_clients()
        if not clients:
            return
        client = next(iter(clients.values()))
        position, rotation = state["trajectory"].pose_at(state["t"])
        cv_position, wxyz = viz.to_frustum_pose(position, rotation)

        # Hide our own visualization-only annotations first -- a real camera
        # wouldn't see its own path curve or the other frustums drawn for our
        # benefit, only actual scene content (ground + cube).
        path_handle.visible = False
        for h in frustum_handles:
            h.visible = False
        current_cam.visible = False
        try:
            image = client.get_render(
                *POV_SIZE, wxyz=wxyz, position=cv_position, fov=viz.DEFAULT_FOV, timeout=1.0
            )
            pov_image.image = image
            pov_last_render[0] = now
        except TimeoutError:
            pass
        finally:
            path_handle.visible = True
            for h in frustum_handles:
                h.visible = not state["playing"]
            current_cam.visible = True

    def set_playing(value: bool) -> None:
        state["playing"] = value
        play_pause_button.icon = viser.Icon.PLAYER_PAUSE if value else viser.Icon.PLAYER_PLAY
        for h in frustum_handles:  # decluttering: hide the *other* constant keyframe frustums while playing
            h.visible = not value
        # current_cam (the one moving frustum) stays visible either way

    def rebuild_and_redraw() -> None:
        nonlocal path_handle
        trajectory = build_full_trajectory()
        state["trajectory"] = trajectory

        path_handle.remove()
        path_handle = viz.add_path(server, "/path", trajectory)
        viz.clear_handles(frustum_handles)
        frustum_handles.extend(viz.add_keyframe_frustums(server, "/keyframes", trajectory))
        for h in frustum_handles:
            h.visible = not state["playing"]

        state["t"] = min(state["t"], trajectory.duration)
        scrubber.max = trajectory.duration
        scrubber.value = state["t"]
        viz.update_current_camera(current_cam, trajectory, state["t"])
        frame_readout.content = _frame_text(state["t"], trajectory.duration)
        update_pov(force=True)

    # --- Sequence management (up to MAX_SEGMENTS segments) -----------------

    with server.gui.add_folder("Sequence"):
        segment_selector = server.gui.add_dropdown("Editing segment", options=["1"], initial_value="1")
        add_segment_button = server.gui.add_button("Add segment", icon=viser.Icon.PLUS)
        remove_segment_button = server.gui.add_button("Remove segment", icon=viser.Icon.TRASH)

    segment_params_folder = server.gui.add_folder("Segment parameters")
    segment_param_handles: list = []

    def rebuild_segment_param_ui() -> None:
        for h in segment_param_handles:
            h.remove()
        segment_param_handles.clear()

        selected_segment = state["segments"][state["selected"]]

        with segment_params_folder:

            def on_param_change(name: str, value) -> None:
                idx = state["selected"]
                state["segments"][idx] = dataclasses.replace(state["segments"][idx], **{name: value})
                rebuild_and_redraw()

            sliders = add_param_sliders(server, selected_segment, on_param_change)
            segment_param_handles.extend(sliders.values())

            easing_dropdown = server.gui.add_dropdown(
                "Easing", options=[e.value for e in Easing], initial_value=Easing(selected_segment.easing).value
            )
            segment_param_handles.append(easing_dropdown)

            @easing_dropdown.on_update
            def _(_) -> None:
                idx = state["selected"]
                state["segments"][idx] = dataclasses.replace(state["segments"][idx], easing=Easing(easing_dropdown.value))
                rebuild_and_redraw()

    def select_segment(idx: int) -> None:
        state["selected"] = idx
        segment_selector.value = str(idx + 1)
        rebuild_segment_param_ui()

    def refresh_segment_selector_options() -> None:
        segment_selector.options = [str(i + 1) for i in range(len(state["segments"]))]

    @segment_selector.on_update
    def _(_) -> None:
        idx = int(segment_selector.value) - 1
        if idx != state["selected"]:
            select_segment(idx)

    @add_segment_button.on_click
    def _(_) -> None:
        if len(state["segments"]) >= MAX_SEGMENTS:
            return
        state["segments"].append(FreeFormSegment())
        refresh_segment_selector_options()
        select_segment(len(state["segments"]) - 1)
        rebuild_and_redraw()

    @remove_segment_button.on_click
    def _(_) -> None:
        if len(state["segments"]) <= 1:
            return
        idx = state["selected"]
        state["segments"].pop(idx)
        refresh_segment_selector_options()
        select_segment(min(idx, len(state["segments"]) - 1))
        rebuild_and_redraw()

    rebuild_segment_param_ui()  # populate for the initial single segment

    # --- Playback ------------------------------------------------------------

    with server.gui.add_folder("Playback"):
        scrubber = server.gui.add_slider("Frame", min=0.0, max=state["trajectory"].duration, step=0.1, initial_value=0.0)

        @scrubber.on_update
        def _(_) -> None:
            state["t"] = scrubber.value
            viz.update_current_camera(current_cam, state["trajectory"], state["t"])
            frame_readout.content = _frame_text(state["t"], state["trajectory"].duration)
            update_pov()

        frame_readout = server.gui.add_markdown(_frame_text(0.0, state["trajectory"].duration))
        play_pause_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        reset_button = server.gui.add_button("Reset", icon=viser.Icon.PLAYER_SKIP_BACK)
        speed_dropdown = server.gui.add_dropdown("Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"], initial_value="1x")
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

        @play_pause_button.on_click
        def _(_) -> None:
            set_playing(not state["playing"])

        @reset_button.on_click
        def _(_) -> None:
            set_playing(False)
            state["t"] = 0.0
            scrubber.value = 0.0
            viz.update_current_camera(current_cam, state["trajectory"], 0.0)
            frame_readout.content = _frame_text(0.0, state["trajectory"].duration)
            update_pov(force=True)

    with server.gui.add_folder("Camera POV"):
        pov_checkbox = server.gui.add_checkbox("Show POV", initial_value=False)
        pov_image = server.gui.add_image(np.full((*POV_SIZE, 3), 200, dtype=np.uint8), label="Camera view")

        @pov_checkbox.on_update
        def _(_) -> None:
            if pov_checkbox.value:
                update_pov(force=True)

    set_playing(False)  # establish consistent initial visibility

    def playback_loop() -> None:
        last = time.time()
        while True:
            now = time.time()
            dt = now - last
            last = now
            if state["playing"]:
                duration = max(state["trajectory"].duration, 1e-6)
                speed = float(speed_dropdown.value.rstrip("x"))
                new_t = state["t"] + dt * speed * PLAYBACK_FPS
                if new_t >= duration:
                    new_t = new_t % duration if loop_checkbox.value else duration
                    if not loop_checkbox.value:
                        set_playing(False)
                state["t"] = new_t
                scrubber.value = new_t
                viz.update_current_camera(current_cam, state["trajectory"], new_t)
                frame_readout.content = _frame_text(new_t, duration)
                update_pov()
            time.sleep(1.0 / PLAYBACK_FPS)

    threading.Thread(target=playback_loop, daemon=True).start()

    print(f"\ncamtraj trajectory designer running -- local: http://localhost:{port}")
    if share:
        url = server.request_share_url()
        if url:
            print(f"camtraj trajectory designer running -- public share URL: {url}\n")
        else:
            print("Could not obtain a public share URL (relay unreachable?); use the local URL above.\n")
    server.sleep_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()
    main(port=args.port, share=args.share)
