"""Interactive multi-segment trajectory designer (free_form, orbit, ...).

Run:
    micromamba run -n camera python -m apps.trajectory_designer
A public share URL (via viser's built-in relay) is requested by default and
printed alongside the local one -- pass --no-share to skip it and use
localhost / manual port-forwarding instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import threading
import time
from typing import Optional

import numpy as np
import camtraj_viser as viser

from camtraj import BoxMotion, look_at_rotation, recipe_from_dict, recipe_to_dict, sequence
from camtraj.conventions import KNOWN_CONVENTIONS, OPENGL, convert_pose
from camtraj.segments import SEGMENT_TYPES, FreeFormSegment
from camtraj.trajectory import Trajectory

from ._shared import viz
from ._shared.example_recipe import EXAMPLE_RECIPE_PATH
from ._shared.param_forms import add_enum_dropdowns, add_groups, add_param_sliders

_SEGMENT_TYPE_NAMES = {cls: name for name, cls in SEGMENT_TYPES.items()}

MAX_SEGMENTS = 10
PLAYBACK_FPS = 30.0  # nominal real-time rate mapping frame-units to wall-clock seconds

POV_SIZE = (240, 360)  # (height, width), matches viz.DEFAULT_ASPECT = 1.5
POV_MAX_FRAMES = 90  # subsample long trajectories so a render pass stays bounded
POV_PLAYBACK_FPS = 12.0  # minimum playback rate for the rendered POV sequence

_START_POSITION = viz.DEFAULT_CAMERA_START
# Face the box's heading (forward/backward, left/right) but stay level -- zero
# pitch -- rather than tilting to center it exactly, by aiming at a point at
# the *camera's own height* instead of the box's.
_LEVEL_TARGET = np.array([viz.CUBE_CENTER[0], _START_POSITION[1], viz.CUBE_CENTER[2]])
_START_ROTATION = look_at_rotation(_START_POSITION, _LEVEL_TARGET)


def _frame_text(t: float, duration: float) -> str:
    return f"**Frame {t:0.1f}** / {duration:0.0f}"


def main(port: int = 8080, share: bool = True) -> None:
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(brand_color=(70, 150, 220))  # sky blue accent -- the main panel's own color,
    # independent of the segment timeline's orange (which sets its own explicit colors, unrelated to this)
    viz.add_ground(server)
    box_handle = viz.add_reference_box(server)

    # Lock the viewer's orbit controls to a Y-up turntable (drag = yaw/pitch
    # only, never roll), and start the view behind the camera+box looking in.
    server.scene.set_up_direction("+y")
    server.initial_camera.position = (3.0, 2.5, 10.0)
    server.initial_camera.look_at = (0.0, 0.75, 1.5)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    state = {
        "segments": [FreeFormSegment()],
        "box_motions": [BoxMotion()],  # always present, zero-motion by default -- no separate on/off toggle
        "selected": 0,
        "t": 0.0,
        "playing": False,
    }

    def build_full_trajectory() -> Trajectory:
        return sequence(
            state["segments"],
            state["box_motions"],
            start_position=_START_POSITION,
            start_rotation=_START_ROTATION,
            box_start_position=viz.CUBE_CENTER,
        )

    state["trajectory"] = build_full_trajectory()

    path_handle = viz.add_path(server, "/path", state["trajectory"])
    frustum_handles = viz.add_keyframe_frustums(server, "/keyframes", state["trajectory"])
    current_cam = viz.add_current_camera(server, "/current_camera", state["trajectory"], t=0.0)
    viz.update_reference_box(box_handle, state["trajectory"].box_position_at(0.0))

    scrubber: Optional[viser.GuiSliderHandle] = None
    frame_readout: Optional[viser.GuiMarkdownHandle] = None
    play_pause_button: Optional[viser.GuiButtonHandle] = None
    pov_image: Optional[viser.GuiImageHandle] = None
    pov_status: Optional[viser.GuiMarkdownHandle] = None
    pov_busy = [False]

    def update_camera_and_box(t: float) -> None:
        viz.update_current_camera(current_cam, state["trajectory"], t)
        viz.update_reference_box(box_handle, state["trajectory"].box_position_at(t))

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
        update_camera_and_box(state["t"])
        frame_readout.content = _frame_text(state["t"], trajectory.duration)

    # --- Sequence timeline: a standalone overlay, not a GUI panel -----------
    #
    # This app vendors a patched build of viser (src/camtraj_viser) that adds
    # a dedicated `set_segment_timeline`/`on_segment_timeline_action` API
    # (see camtraj_viser/client/src/SegmentTimeline.tsx): a small, chromeless,
    # always-centered row of colored boxes with an embedded "x" per removable
    # segment, rendered outside the normal panel/dock system entirely (which
    # always draws a header/background around its content, and floats at a
    # fixed pixel position rather than staying centered). Segment 1 is
    # permanently anchored -- never removable -- which also means the
    # sequence can never be emptied out from under it.

    _SEGMENT_COLOR = (190, 110, 40)  # slightly dark orange, deliberately distinct
    # from the main panel's own sky-blue brand_color -- the two accents are
    # unrelated (this one is sent explicitly per segment, not derived from theme)

    def _segment_extent(i: int) -> float:
        """0-1 normalized position of this segment's frame count between the
        `frames` slider's actual min/max (21/300, shared by every primitive),
        sqrt-scaled so short segments aren't visually crushed to nothing.
        Drives both the timeline box's width and label text (see
        SegmentTimeline.tsx) -- both grow together as this approaches 1."""
        frames = state["segments"][i].frames
        return min((max(frames - 21, 0) / 279.0) ** 0.5, 1.0)

    def select_segment(idx: int) -> None:
        state["selected"] = idx
        rebuild_timeline()
        rebuild_segment_param_ui()

    def rebuild_timeline() -> None:
        """Full rebuild: segment count, selection, and/or frame counts may have changed."""
        n = len(state["segments"])
        segments = [
            (f"seg {i + 1}", f"segment {i + 1}", _SEGMENT_COLOR, _segment_extent(i), i != 0) for i in range(n)
        ]
        server.gui.set_segment_timeline(segments, state["selected"])

    def _on_timeline_action(action, index: int) -> None:
        if action == "add":
            if len(state["segments"]) >= MAX_SEGMENTS:
                return
            state["segments"].append(FreeFormSegment())
            state["box_motions"].append(BoxMotion())
            select_segment(len(state["segments"]) - 1)
            rebuild_and_redraw()
        elif action == "remove":
            if index == 0 or index >= len(state["segments"]):
                return
            state["segments"].pop(index)
            state["box_motions"].pop(index)
            select_segment(min(index, len(state["segments"]) - 1))
            rebuild_and_redraw()
        else:  # "select"
            if index != state["selected"]:
                select_segment(index)

    server.gui.on_segment_timeline_action(_on_timeline_action)
    rebuild_timeline()

    # --- Segment: the selected segment's own properties ----------------------

    segment_params_folder = server.gui.add_folder("Segment")
    segment_param_handles: list = []

    def rebuild_segment_param_ui() -> None:
        for h in segment_param_handles:
            h.remove()
        segment_param_handles.clear()

        selected_segment = state["segments"][state["selected"]]

        with segment_params_folder:
            type_dropdown = server.gui.add_dropdown(
                "Primitive type",
                options=list(SEGMENT_TYPES.keys()),
                initial_value=_SEGMENT_TYPE_NAMES[type(selected_segment)],
            )
            segment_param_handles.append(type_dropdown)

            @type_dropdown.on_update
            def _(_) -> None:
                new_cls = SEGMENT_TYPES[type_dropdown.value]
                if new_cls is type(state["segments"][state["selected"]]):
                    return
                state["segments"][state["selected"]] = new_cls()  # fresh defaults; old params rarely transfer meaningfully
                rebuild_segment_param_ui()
                rebuild_and_redraw()

            def on_param_change(name: str, value) -> None:
                idx = state["selected"]
                state["segments"][idx] = dataclasses.replace(state["segments"][idx], **{name: value})
                rebuild_and_redraw()
                if name == "frames":
                    rebuild_timeline()  # relabel this segment's length in the timeline bar

            sliders = add_param_sliders(server, selected_segment, on_param_change)
            segment_param_handles.extend(sliders.values())

            def on_enum_change(name: str, value) -> None:
                idx = state["selected"]
                state["segments"][idx] = dataclasses.replace(state["segments"][idx], **{name: value})
                rebuild_and_redraw()

            dropdowns = add_enum_dropdowns(server, selected_segment, on_enum_change)
            segment_param_handles.extend(dropdowns.values())

            reset_button = server.gui.add_button(
                "Reset to defaults", icon=viser.Icon.RESTORE, hint="Reset this segment's own parameters (not box motion)"
            )
            segment_param_handles.append(reset_button)

            @reset_button.on_click
            def _(_) -> None:
                idx = state["selected"]
                state["segments"][idx] = type(state["segments"][idx])()
                rebuild_segment_param_ui()
                rebuild_and_redraw()
                rebuild_timeline()  # frame count reset to its default too

            # Box motion is a separate, collapsed-by-default sub-section: it's a
            # scene-level concern (does the box move here?), not one of the
            # primitive's own parameters, so it's visually set apart. Every
            # segment always has a BoxMotion (never None) -- a zero-motion one
            # just means "doesn't move," so its sliders are always visible here
            # (no separate on/off checkbox to fight with).
            box_motion_folder = server.gui.add_folder("Box motion", expand_by_default=False)
            segment_param_handles.append(box_motion_folder)

            with box_motion_folder:
                motion = state["box_motions"][state["selected"]]

                def on_motion_change(name: str, value) -> None:
                    idx = state["selected"]
                    state["box_motions"][idx] = dataclasses.replace(state["box_motions"][idx], **{name: value})
                    rebuild_and_redraw()

                motion_sliders = add_param_sliders(server, motion, on_motion_change)
                segment_param_handles.extend(motion_sliders.values())
                motion_dropdowns = add_enum_dropdowns(server, motion, on_motion_change)
                segment_param_handles.extend(motion_dropdowns.values())

                motion_reset_button = server.gui.add_button(
                    "Reset box motion to defaults", icon=viser.Icon.RESTORE
                )
                segment_param_handles.append(motion_reset_button)

                @motion_reset_button.on_click
                def _(_) -> None:
                    idx = state["selected"]
                    state["box_motions"][idx] = BoxMotion()
                    rebuild_segment_param_ui()
                    rebuild_and_redraw()

            # Primitive-specific groups (e.g. rotation_track's world movement)
            # -- always present, no on/off checkbox, own reset button. Same
            # collapsed-by-default treatment as box motion above.
            def get_group_value(name: str):
                return getattr(state["segments"][state["selected"]], name)

            def set_group_value(name: str, value) -> None:
                idx = state["selected"]
                state["segments"][idx] = dataclasses.replace(state["segments"][idx], **{name: value})
                rebuild_and_redraw()

            add_groups(server, selected_segment, segment_param_handles, get_group_value, set_group_value, rebuild_segment_param_ui)

    rebuild_segment_param_ui()  # populate for the initial single segment

    # --- Playback ------------------------------------------------------------

    with server.gui.add_folder("Playback"):
        scrubber = server.gui.add_slider("Frame", min=0.0, max=state["trajectory"].duration, step=0.1, initial_value=0.0)

        @scrubber.on_update
        def _(_) -> None:
            state["t"] = scrubber.value
            update_camera_and_box(state["t"])
            frame_readout.content = _frame_text(state["t"], state["trajectory"].duration)

        frame_readout = server.gui.add_markdown(_frame_text(0.0, state["trajectory"].duration))
        play_pause_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY, hint="Space bar also toggles play/pause")
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
            update_camera_and_box(0.0)
            frame_readout.content = _frame_text(0.0, state["trajectory"].duration)

    space_command = server.gui.add_command(
        "Play / pause trajectory", hotkey="space", icon=viser.Icon.PLAYER_PLAY
    )

    @space_command.on_trigger
    def _(_) -> None:
        set_playing(not state["playing"])

    # --- Camera POV: a separate, on-demand player -----------------------------
    # Deliberately decoupled from the main scrubber/playback: it never
    # re-renders on its own just because a param or the scrubber changed --
    # only an explicit click renders a fresh batch of frames and plays them once.

    pov_state = {"frames": [], "duration": 0.0}

    with server.gui.add_folder("Camera POV", expand_by_default=False):
        pov_image = server.gui.add_image(np.full((*POV_SIZE, 3), 200, dtype=np.uint8), label="Camera view")
        pov_status = server.gui.add_markdown("Not rendered yet.")
        pov_play_button = server.gui.add_button("Render & Play POV", icon=viser.Icon.PLAYER_PLAY)
        pov_play_again_button = server.gui.add_button("Play Again", icon=viser.Icon.REPEAT)

        @pov_play_button.on_click
        def _(_) -> None:
            _run_pov_job(render_and_play_pov)

        @pov_play_again_button.on_click
        def _(_) -> None:
            _run_pov_job(play_pov_frames)

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

    def play_pov_frames() -> None:
        frames = pov_state["frames"]
        if not frames:
            pov_status.content = "Nothing rendered yet -- click Render & Play POV first."
            return
        interval = max((pov_state["duration"] / PLAYBACK_FPS) / len(frames), 1.0 / POV_PLAYBACK_FPS)
        for i, frame in enumerate(frames):
            pov_image.image = frame
            pov_status.content = f"Playing {i + 1} / {len(frames)}"
            time.sleep(interval)
        pov_status.content = f"Done -- played {len(frames)} rendered frames."

    def render_and_play_pov() -> None:
        clients = server.get_clients()
        if not clients:
            pov_status.content = "No connected browser to render from."
            return
        client = next(iter(clients.values()))

        trajectory = state["trajectory"]
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
                viz.update_reference_box(box_handle, trajectory.box_position_at(t))
                try:
                    frames.append(client.get_render(*POV_SIZE, wxyz=wxyz, position=cv_position, fov=viz.DEFAULT_FOV, timeout=2.0))
                except TimeoutError:
                    break
        finally:
            path_handle.visible = True
            for h in frustum_handles:
                h.visible = not was_playing
            current_cam.visible = True
            update_camera_and_box(state["t"])
            if was_playing:
                set_playing(True)

        if not frames:
            pov_status.content = "Render failed -- no frames came back."
            return

        pov_state["frames"] = frames
        pov_state["duration"] = trajectory.duration
        play_pov_frames()

    # --- Recipe: save/load the segment + box-motion definitions themselves ---
    # As opposed to Export below, which bakes the *current* trajectory into
    # numeric arrays, a recipe is the editable design itself (segment types +
    # their param values + box motions) -- what you'd persist across
    # sessions, share with a teammate, or eventually randomize around for
    # batch generation. See `camtraj.recipe`.

    with server.gui.add_folder("Recipe"):
        save_recipe_button = server.gui.add_button("Save recipe (.json)", icon=viser.Icon.DEVICE_FLOPPY)
        example_recipe_button = server.gui.add_button("Load example recipe", icon=viser.Icon.SPARKLES)
        load_recipe_button = server.gui.add_upload_button(
            "Load recipe (.json)", icon=viser.Icon.UPLOAD, mime_type=".json,application/json"
        )
        recipe_status = server.gui.add_markdown("")

        @save_recipe_button.on_click
        def _(event: viser.GuiEvent) -> None:
            data = recipe_to_dict(state["segments"], state["box_motions"])
            content = json.dumps(data, indent=2).encode("utf-8")
            target = event.client if event.client is not None else server
            target.send_file_download("camtraj_recipe.json", content)

        def _apply_recipe_data(data) -> None:
            segments, box_motions = recipe_from_dict(data)
            if not segments:
                raise ValueError("Recipe has no segments.")
            state["segments"] = segments
            state["box_motions"] = box_motions
            state["selected"] = 0
            recipe_status.content = f"Loaded {len(segments)} segment(s)."
            rebuild_timeline()
            rebuild_segment_param_ui()
            rebuild_and_redraw()

        @example_recipe_button.on_click
        def _(_) -> None:
            try:
                _apply_recipe_data(json.loads(EXAMPLE_RECIPE_PATH.read_text()))
            except Exception as e:
                recipe_status.content = f"**Load failed:** {e}"

        @load_recipe_button.on_upload
        def _(_) -> None:
            try:
                _apply_recipe_data(json.loads(load_recipe_button.value.content.decode("utf-8")))
            except Exception as e:
                recipe_status.content = f"**Load failed:** {e}"

    # --- Export: camera + box, as two separate .npz files --------------------
    # The camera trajectory is converted (via `camtraj.conventions.convert_pose`)
    # from the app's canonical OpenGL/c2w convention into whichever convention
    # is picked -- no new sign-flip code, just the same generic conversion used
    # internally for viser's frustums. Box positions are always plain
    # world-space coordinates (not a camera pose), so they're exported as-is
    # regardless of the selected camera convention.

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
            trajectory = state["trajectory"]
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
            box_buf = io.BytesIO()
            np.savez(box_buf, times=trajectory.times, positions=trajectory.box_positions)

            target = event.client if event.client is not None else server
            target.send_file_download(f"camtraj_camera_{target_convention.name}.npz", camera_buf.getvalue())
            target.send_file_download("camtraj_box.npz", box_buf.getvalue())

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
                update_camera_and_box(new_t)
                frame_readout.content = _frame_text(new_t, duration)
            time.sleep(1.0 / PLAYBACK_FPS)

    threading.Thread(target=playback_loop, daemon=True).start()

    print(f"\ncamtraj trajectory designer running -- local: http://localhost:{server.get_port()}")
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
