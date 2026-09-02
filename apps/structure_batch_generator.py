"""Interactive structure-randomized trajectory generator.

The complement to `apps.batch_generator`: there, segment types/count are
fixed and each field's *value* varies within a range. Here, every value is
fixed -- each candidate segment is used verbatim, exactly as it appeared in
whichever recipe it was loaded from -- and what varies is the *structure*:
how many segments a generated trajectory chains, and which ones, in what
order.

Load one or more recipes (built in `apps/trajectory_designer.py`); every
segment of every loaded recipe becomes one candidate "block" in a shared
pool. Uncheck a block to exclude it without losing it. "Preview a random
draw" samples a random-length sequence of random blocks (with replacement)
and shows it in the 3D view. "Generate & download" samples N independent,
reproducible sequences and bundles them into one .zip: a manifest.json
recording which blocks each draw chained together, plus a camera/box .npz
pair per trajectory, in your chosen convention.

Run:
    micromamba run -n camera python -m apps.structure_batch_generator
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

from camtraj import BoxMotion, generate_structure_batch, look_at_rotation, sample_structure, sequence
from camtraj.conventions import KNOWN_CONVENTIONS, OPENGL, convert_pose
from camtraj.recipe import recipe_from_dict
from camtraj.segments import SEGMENT_TYPES, to_param_dict
from camtraj.structure_batch import Block

from ._shared import viz
from ._shared.example_recipe import EXAMPLE_RECIPE_PATH

MAX_BATCH_SIZE = 500
MAX_SEQUENCE_LENGTH = 20

_SEGMENT_TYPE_NAMES = {cls: name for name, cls in SEGMENT_TYPES.items()}

_START_POSITION = viz.DEFAULT_CAMERA_START
_LEVEL_TARGET = np.array([viz.CUBE_CENTER[0], _START_POSITION[1], viz.CUBE_CENTER[2]])
_START_ROTATION = look_at_rotation(_START_POSITION, _LEVEL_TARGET)


def main(port: int = 8082, share: bool = True) -> None:
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(brand_color=(70, 150, 220))
    viz.add_ground(server)
    box_handle = viz.add_reference_box(server)

    server.scene.set_up_direction("+y")
    server.initial_camera.position = (3.0, 2.5, 10.0)
    server.initial_camera.look_at = (0.0, 0.75, 1.5)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    state = {
        "pool": [],  # list[Block]
        "enabled": [],  # list[bool], parallel to pool
    }
    preview_handles: list = []

    server.gui.add_markdown(
        "### How this works\n"
        "1. **Load one or more recipes** below (built in the single-trajectory "
        "app) -- every segment of every recipe you load becomes one candidate "
        "**block** in a shared pool. Loading another recipe adds to the pool, "
        "it doesn't replace it.\n"
        "2. Uncheck a block to exclude it from sampling without losing it.\n"
        "3. Set a **sequence length range** -- how many blocks a generated "
        "trajectory chains together.\n"
        "4. **Preview a random draw** to see one sampled sequence before "
        "generating a full batch. Each block is used exactly as-is -- only "
        "*which* blocks and *how many* varies, not their own parameters."
    )

    # --- Block pool ----------------------------------------------------

    pool_folder = server.gui.add_folder("Block pool")
    pool_handles: list = []

    def rebuild_pool_ui() -> None:
        for h in pool_handles:
            h.remove()
        pool_handles.clear()
        with pool_folder:
            if not state["pool"]:
                empty_msg = server.gui.add_markdown("_No blocks yet -- load a recipe below._")
                pool_handles.append(empty_msg)
            for i, block in enumerate(state["pool"]):
                checkbox = server.gui.add_checkbox(block.label, initial_value=state["enabled"][i])
                pool_handles.append(checkbox)

                def _make_callback(i=i, checkbox=checkbox):
                    def _(_) -> None:
                        state["enabled"][i] = checkbox.value

                    return _

                checkbox.on_update(_make_callback())

    rebuild_pool_ui()

    with pool_folder:
        example_recipe_button = server.gui.add_button("Load example recipe", icon=viser.Icon.SPARKLES)
        load_recipe_button = server.gui.add_upload_button(
            "Load recipe (.json)", icon=viser.Icon.UPLOAD, mime_type=".json,application/json"
        )
        pool_status = server.gui.add_markdown("")
        clear_pool_button = server.gui.add_button("Clear pool", icon=viser.Icon.TRASH)

        def _add_recipe_to_pool(data, filename: str) -> None:
            segments, box_motions = recipe_from_dict(data)
            if not segments:
                raise ValueError("Recipe has no segments.")
            for i, (seg, motion) in enumerate(zip(segments, box_motions)):
                label = f"{filename}#{i + 1} ({_SEGMENT_TYPE_NAMES[type(seg)]})"
                state["pool"].append(Block(label, seg, motion))
                state["enabled"].append(True)
            pool_status.content = f"Added {len(segments)} block(s) -- pool now has {len(state['pool'])}."
            rebuild_pool_ui()

        @example_recipe_button.on_click
        def _(_) -> None:
            try:
                _add_recipe_to_pool(json.loads(EXAMPLE_RECIPE_PATH.read_text()), EXAMPLE_RECIPE_PATH.name)
            except Exception as e:
                pool_status.content = f"**Load failed:** {e}"

        @load_recipe_button.on_upload
        def _(_) -> None:
            try:
                filename = load_recipe_button.value.name or "recipe"
                _add_recipe_to_pool(json.loads(load_recipe_button.value.content.decode("utf-8")), filename)
            except Exception as e:
                pool_status.content = f"**Load failed:** {e}"

        @clear_pool_button.on_click
        def _(_) -> None:
            state["pool"] = []
            state["enabled"] = []
            pool_status.content = "Pool cleared."
            rebuild_pool_ui()

    # --- Sequence length -------------------------------------------------

    with server.gui.add_folder("Sequence length"):
        length_slider = server.gui.add_multi_slider(
            "Blocks per trajectory", min=1, max=MAX_SEQUENCE_LENGTH, step=1, initial_value=(2, 4)
        )

    def _enabled_pool() -> list[Block]:
        return [b for b, on in zip(state["pool"], state["enabled"]) if on]

    # --- Preview -----------------------------------------------------------

    def do_preview() -> dict:
        pool = _enabled_pool()
        if not pool:
            raise ValueError("No enabled blocks in the pool -- load a recipe and/or check at least one block.")
        rng = np.random.default_rng()
        lo, hi = length_slider.value
        segments, box_motions, labels = sample_structure(pool, (lo, hi), rng)
        trajectory = sequence(
            segments, box_motions,
            start_position=_START_POSITION, start_rotation=_START_ROTATION, box_start_position=viz.CUBE_CENTER,
        )
        viz.clear_handles(preview_handles)
        preview_handles.append(viz.add_path(server, "/preview_path", trajectory))
        preview_handles.extend(viz.add_keyframe_frustums(server, "/preview_keyframes", trajectory))
        viz.update_reference_box(box_handle, trajectory.box_position_at(0.0))
        return {
            "block_labels": labels,
            "segments": [{"type": _SEGMENT_TYPE_NAMES[type(s)], "params": to_param_dict(s)} for s in segments],
        }

    with server.gui.add_folder("Preview"):
        preview_button = server.gui.add_button("Preview a random draw", icon=viser.Icon.DICE)
        preview_info = server.gui.add_markdown("")

        @preview_button.on_click
        def _(_) -> None:
            try:
                sampled = do_preview()
            except Exception as e:
                preview_info.content = f"**Preview failed:** {e}"
                return
            preview_info.content = (
                f"**Blocks used:** {' -> '.join(sampled['block_labels'])}\n\n"
                "```json\n" + json.dumps(sampled["segments"], indent=2) + "\n```"
            )

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
                    pool = _enabled_pool()
                    if not pool:
                        raise ValueError("No enabled blocks in the pool -- load a recipe and/or check at least one block.")
                    n = int(n_input.value)
                    seed = int(seed_input.value)
                    lo, hi = length_slider.value
                    target_convention = KNOWN_CONVENTIONS[convention_dropdown.value]
                    results = generate_structure_batch(
                        pool, (lo, hi),
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
                    target.send_file_download("camtraj_structure_batch.zip", buf.getvalue())
                    generate_status.content = f"Done -- generated {n} trajectories."
                except Exception as e:
                    generate_status.content = f"**Generation failed:** {e}"
                finally:
                    busy[0] = False

            threading.Thread(target=_run, daemon=True).start()

    print(f"\ncamtraj structure batch generator running -- local: http://localhost:{server.get_port()}")
    if share:
        url = server.request_share_url()
        if url:
            print(f"camtraj structure batch generator running -- public share URL: {url}\n")
        else:
            print("Could not obtain a public share URL (relay unreachable?); use the local URL above.\n")
    server.sleep_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args()
    main(port=args.port, share=args.share)
