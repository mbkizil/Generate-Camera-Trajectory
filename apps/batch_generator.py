"""Interactive batch/randomized trajectory generator.

Load a recipe (built in `apps/trajectory_designer.py`) as a fixed skeleton --
segment types and count don't change here. Every numeric slider becomes a
draggable min/max *range*; every dropdown becomes a set of checkboxes (an
"allowed values" multi-select). A range collapsed to a single point (or a
multi-select with exactly one box checked) means "pinned" -- nothing is
randomized until you deliberately widen something. "Preview a random draw"
samples one trajectory and shows it in the 3D view, so you can sanity-check
ranges before committing to a full batch. "Generate & download" samples N
independent draws (deterministic given a seed) and bundles them into one
.zip: a manifest.json recording exactly what each draw sampled, plus a
camera/box .npz pair per trajectory, in your chosen convention.

Run:
    micromamba run -n camera python -m apps.batch_generator
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import zipfile
from typing import Optional

import numpy as np
import camtraj_viser as viser

from camtraj import BoxMotion, default_ranges, generate_batch, look_at_rotation, sample_recipe, sequence
from camtraj.conventions import KNOWN_CONVENTIONS, OPENGL, convert_pose
from camtraj.recipe import recipe_from_dict
from camtraj.segments import SEGMENT_TYPES, FreeFormSegment, to_param_dict

from ._shared import viz
from ._shared.example_recipe import EXAMPLE_RECIPE_PATH
from ._shared.range_forms import add_enum_multiselects, add_range_groups, add_range_sliders

MAX_BATCH_SIZE = 500

_SEGMENT_TYPE_NAMES = {cls: name for name, cls in SEGMENT_TYPES.items()}

_START_POSITION = viz.DEFAULT_CAMERA_START
_LEVEL_TARGET = np.array([viz.CUBE_CENTER[0], _START_POSITION[1], viz.CUBE_CENTER[2]])
_START_ROTATION = look_at_rotation(_START_POSITION, _LEVEL_TARGET)


def main(port: int = 8081, share: bool = True) -> None:
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(brand_color=(70, 150, 220))
    viz.add_ground(server)
    box_handle = viz.add_reference_box(server)

    server.scene.set_up_direction("+y")
    server.initial_camera.position = (3.0, 2.5, 10.0)
    server.initial_camera.look_at = (0.0, 0.75, 1.5)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    state = {
        "segment_types": [FreeFormSegment],
        "skeleton_segments": [FreeFormSegment()],
        "segment_ranges": [default_ranges(FreeFormSegment())],
        "skeleton_box_motions": [BoxMotion()],
        "box_motion_ranges": [default_ranges(BoxMotion())],
        "selected": 0,
    }
    preview_handles: list = []

    server.gui.add_markdown(
        "### How this works\n"
        "1. **Load a recipe** below (built in the single-trajectory app) as your skeleton -- "
        "segment types and count are fixed here, only their parameter values vary.\n"
        "2. Every slider becomes a **range**: drag both handles to the same spot to pin it "
        "exactly, or spread them apart to let it vary per generated trajectory.\n"
        "3. Every dropdown becomes a set of **checkboxes**: check one to pin that choice, "
        "check several to let it vary.\n"
        "4. **Preview a random draw** to sanity-check your ranges before generating a full batch."
    )

    # --- Skeleton --------------------------------------------------------

    def describe_skeleton() -> str:
        names = [_SEGMENT_TYPE_NAMES[cls] for cls in state["segment_types"]]
        return f"**Skeleton:** {len(names)} segment(s) -- " + " -> ".join(names)

    skeleton_summary = server.gui.add_markdown(describe_skeleton())

    with server.gui.add_folder("Skeleton"):
        example_recipe_button = server.gui.add_button("Load example recipe", icon=viser.Icon.SPARKLES)
        load_recipe_button = server.gui.add_upload_button(
            "Load recipe (.json)", icon=viser.Icon.UPLOAD, mime_type=".json,application/json"
        )
        skeleton_status = server.gui.add_markdown("")

        def _apply_recipe_data(data) -> None:
            segments, box_motions = recipe_from_dict(data)
            if not segments:
                raise ValueError("Recipe has no segments.")
            state["segment_types"] = [type(seg) for seg in segments]
            state["skeleton_segments"] = segments
            state["segment_ranges"] = [default_ranges(seg) for seg in segments]
            state["skeleton_box_motions"] = box_motions
            state["box_motion_ranges"] = [default_ranges(bm) for bm in box_motions]
            state["selected"] = 0
            skeleton_status.content = f"Loaded {len(segments)} segment(s)."
            skeleton_summary.content = describe_skeleton()
            rebuild_segment_selector()
            rebuild_range_ui()

        @example_recipe_button.on_click
        def _(_) -> None:
            try:
                _apply_recipe_data(json.loads(EXAMPLE_RECIPE_PATH.read_text()))
            except Exception as e:
                skeleton_status.content = f"**Load failed:** {e}"

        @load_recipe_button.on_upload
        def _(_) -> None:
            try:
                _apply_recipe_data(json.loads(load_recipe_button.value.content.decode("utf-8")))
            except Exception as e:
                skeleton_status.content = f"**Load failed:** {e}"

    # --- Segment selector --------------------------------------------------

    segment_selector_folder = server.gui.add_folder("Select segment")
    segment_dropdown: Optional[viser.GuiDropdownHandle] = None

    def _segment_labels() -> list[str]:
        return [f"{i + 1}: {_SEGMENT_TYPE_NAMES[cls]}" for i, cls in enumerate(state["segment_types"])]

    def rebuild_segment_selector() -> None:
        nonlocal segment_dropdown
        if segment_dropdown is not None:
            segment_dropdown.remove()
        with segment_selector_folder:
            segment_dropdown = server.gui.add_dropdown("Segment", options=_segment_labels())
            segment_dropdown.value = _segment_labels()[state["selected"]]

            @segment_dropdown.on_update
            def _(_) -> None:
                idx = _segment_labels().index(segment_dropdown.value)
                if idx != state["selected"]:
                    state["selected"] = idx
                    rebuild_range_ui()

    rebuild_segment_selector()

    # --- Segment + box-motion ranges ----------------------------------------

    ranges_folder = server.gui.add_folder("Ranges")
    range_handles: list = []

    def rebuild_range_ui() -> None:
        for h in range_handles:
            h.remove()
        range_handles.clear()

        idx = state["selected"]
        cls = state["segment_types"][idx]

        with ranges_folder:
            heading = server.gui.add_markdown(f"**Segment {idx + 1}: {_SEGMENT_TYPE_NAMES[cls]}**")
            range_handles.append(heading)

            def on_seg_change(name: str, value) -> None:
                state["segment_ranges"][idx][name] = value

            sliders = add_range_sliders(server, cls, state["segment_ranges"][idx], on_seg_change)
            range_handles.extend(sliders.values())

            for checkboxes in add_enum_multiselects(server, cls, state["segment_ranges"][idx], on_seg_change).values():
                range_handles.extend(checkboxes)

            def get_seg_current(name: str):
                return state["segment_ranges"][idx][name]

            def set_seg_current(name: str, value) -> None:
                state["segment_ranges"][idx][name] = value

            def get_seg_pinned(name: str):
                return default_ranges(getattr(state["skeleton_segments"][idx], name))

            add_range_groups(
                server, cls, state["segment_ranges"][idx], range_handles,
                get_seg_current, set_seg_current, get_seg_pinned, rebuild_range_ui,
            )

            reset_seg_button = server.gui.add_button("Reset segment ranges to pinned", icon=viser.Icon.RESTORE)
            range_handles.append(reset_seg_button)

            @reset_seg_button.on_click
            def _(_) -> None:
                state["segment_ranges"][idx] = default_ranges(state["skeleton_segments"][idx])
                rebuild_range_ui()

            box_motion_folder = server.gui.add_folder("Box motion ranges", expand_by_default=False)
            range_handles.append(box_motion_folder)

            with box_motion_folder:

                def on_box_change(name: str, value) -> None:
                    state["box_motion_ranges"][idx][name] = value

                box_sliders = add_range_sliders(server, BoxMotion, state["box_motion_ranges"][idx], on_box_change)
                range_handles.extend(box_sliders.values())
                for checkboxes in add_enum_multiselects(server, BoxMotion, state["box_motion_ranges"][idx], on_box_change).values():
                    range_handles.extend(checkboxes)

                reset_box_button = server.gui.add_button("Reset box motion ranges to pinned", icon=viser.Icon.RESTORE)
                range_handles.append(reset_box_button)

                @reset_box_button.on_click
                def _(_) -> None:
                    state["box_motion_ranges"][idx] = default_ranges(state["skeleton_box_motions"][idx])
                    rebuild_range_ui()

    rebuild_range_ui()

    # --- Preview -------------------------------------------------------------

    def do_preview() -> dict:
        rng = np.random.default_rng()
        segments, box_motions = sample_recipe(
            state["segment_types"], state["segment_ranges"], state["box_motion_ranges"], rng
        )
        trajectory = sequence(
            segments, box_motions,
            start_position=_START_POSITION, start_rotation=_START_ROTATION, box_start_position=viz.CUBE_CENTER,
        )
        viz.clear_handles(preview_handles)
        preview_handles.append(viz.add_path(server, "/preview_path", trajectory))
        preview_handles.extend(viz.add_keyframe_frustums(server, "/preview_keyframes", trajectory))
        viz.update_reference_box(box_handle, trajectory.box_position_at(0.0))
        return {
            "segments": [{"type": _SEGMENT_TYPE_NAMES[type(s)], "params": to_param_dict(s)} for s in segments],
            "box_motions": [to_param_dict(bm) for bm in box_motions],
        }

    with server.gui.add_folder("Preview"):
        preview_button = server.gui.add_button("Preview a random draw", icon=viser.Icon.DICE)
        preview_info = server.gui.add_markdown("")

        @preview_button.on_click
        def _(_) -> None:
            sampled = do_preview()
            preview_info.content = "```json\n" + json.dumps(sampled, indent=2) + "\n```"

    do_preview()  # populate the scene immediately instead of starting empty

    # --- Generate batch --------------------------------------------------

    busy = [False]
    with server.gui.add_folder("Generate batch"):
        n_input = server.gui.add_number("Count", 10, min=1, max=MAX_BATCH_SIZE, step=1)
        seed_input = server.gui.add_number("Seed", 0, min=0, step=1)
        convention_dropdown = server.gui.add_dropdown(
            "Convention", options=list(KNOWN_CONVENTIONS.keys()), initial_value=OPENGL.name
        )
        convention_info = server.gui.add_markdown(KNOWN_CONVENTIONS[OPENGL.name].description)

        @convention_dropdown.on_update
        def _(_) -> None:
            convention_info.content = KNOWN_CONVENTIONS[convention_dropdown.value].description

        generate_button = server.gui.add_button("Generate & download batch (.zip)", icon=viser.Icon.DOWNLOAD)
        generate_status = server.gui.add_markdown("")

        @generate_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if busy[0]:
                return
            client = event.client

            def _run() -> None:
                busy[0] = True
                generate_status.content = "Generating..."
                try:
                    n = int(n_input.value)
                    seed = int(seed_input.value)
                    target_convention = KNOWN_CONVENTIONS[convention_dropdown.value]
                    results = generate_batch(
                        state["segment_types"], state["segment_ranges"], state["box_motion_ranges"],
                        n=n, seed=seed,
                        start_position=_START_POSITION, start_rotation=_START_ROTATION,
                        box_start_position=viz.CUBE_CENTER,
                    )
                    manifest = {"seed": seed, "convention": target_convention.name, "count": n, "draws": []}
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for i, (params, trajectory) in enumerate(results):
                            manifest["draws"].append({"index": i, **params})
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
                            zf.writestr(f"traj_{i:04d}_camera.npz", camera_buf.getvalue())
                            box_buf = io.BytesIO()
                            np.savez(box_buf, times=trajectory.times, positions=trajectory.box_positions)
                            zf.writestr(f"traj_{i:04d}_box.npz", box_buf.getvalue())
                        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

                    target = client if client is not None else server
                    target.send_file_download("camtraj_batch.zip", buf.getvalue())
                    generate_status.content = f"Done -- generated {n} trajectories."
                except Exception as e:
                    generate_status.content = f"**Generation failed:** {e}"
                finally:
                    busy[0] = False

            threading.Thread(target=_run, daemon=True).start()

    print(f"\ncamtraj batch generator running -- local: http://localhost:{server.get_port()}")
    if share:
        url = server.request_share_url()
        if url:
            print(f"camtraj batch generator running -- public share URL: {url}\n")
        else:
            print("Could not obtain a public share URL (relay unreachable?); use the local URL above.\n")
    server.sleep_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()
    main(port=args.port, share=args.share)
