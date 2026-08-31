# camtraj

Camera-trajectory authoring, randomization, visualization, and export, built
for user-friendliness first. Runs in the micromamba env `camera`.

## Where this came from

`/home/colligo/LAMP` is a "text-to-camera-trajectory via LLM" research repo.
Its DSL (`LAMP/DSL.txt`, `LAMP/src/scripts/generate_camera_trajectory.py`,
`generate_relative_trajectory.py`) is the *logic* camtraj reimplements — same
motion vocabulary (free_form / orbit_track / tail_track / rotation_track,
common modifiers like dutch/ease) — but we explicitly do **not** want the
LLM/text-to-DSL part, and the original code has real, confirmed bugs: two
generators write incompatible field orders for "the same" quaternion+position
format, positions/rotations get quantized to `[0,512]` ints (an artifact of
feeding an LLM discrete tokens, not needed here), and Blender/OpenCV export
scripts each hand-roll a different, sometimes-wrong axis-flip hack. None of
that is ported forward. See conversation history / git log for the full
bug-by-bug analysis that motivated the rewrite; the short version is captured
in the "Design principles" below.

## Design principles (read this before changing core math)

- **One canonical internal convention.** `Trajectory` (in `src/camtraj/trajectory.py`)
  is always right-handed, +Y up, camera-to-world, camera looks down local -Z
  (the "OpenGL/NeRF" convention). Never introduce a second in-memory
  convention or scramble quaternion component order positionally — rotation
  is always a `scipy.spatial.transform.Rotation` object, never raw floats
  passed around in some order.
- **Conventions are generic, not hand-tuned per pair.** `src/camtraj/conventions.py`
  describes a convention as (which local axis is right/up/forward) + (c2w or
  w2c), and derives conversion between *any* two from that description
  (`convert_pose`). Adding PyTorch3D/Blender/ARKit/etc. later is a ~5-line
  `Convention(...)` declaration, not new bespoke math. When viser's camera
  frustums turned out to need OpenCV convention internally, this is the
  function that handled it — no new sign-flip hack was written.
- **Segments are composable, continuity is the contract.** Each primitive is a
  `SegmentBase` subclass (`src/camtraj/segments/`). `build(start_position,
  start_rotation, target_positions) -> Trajectory` must return frame 0 exactly
  matching the incoming pose (so `sequence()` can chain segments of different
  primitive types with no position jump). Target-relative primitives (orbit,
  tail, rotation_track) are the exception for *rotation*: they redefine
  orientation via look-at geometry every frame rather than continuing the
  incoming rotation, and that's intentional (documented on each class).
- **Frame count is a segment parameter, not seconds × fps.** `Trajectory.times`
  are frame-index units, not wall-clock seconds — this matches how these
  trajectories actually get consumed (N frames into a video/3D pipeline). The
  interactive app maps frame-units to real seconds only for its own playback
  pacing (`PLAYBACK_FPS` in `apps/trajectory_designer.py`), never in the core
  library.
- **The "box" is a first-class, always-present scene anchor.** Every segment
  gets a `target_positions` array (the box's position at each of that
  segment's frames) whether it uses it or not (free_form ignores it). Box
  motion (`src/camtraj/box_motion.py`, `BoxMotion`) is translation-only, no
  on/off flag — a zero-delta `BoxMotion` *is* "doesn't move." `sequence()`
  threads box position across segment boundaries the same way it does camera
  pose.
- **Continuous sliders, not categorical tags — but keep the old tiers as
  reference marks.** The original DSL's 7-level categoricals
  (far_left/left/near_left/no/...) become one continuous slider with `marks`
  at those old values (see `param()` in `src/camtraj/segments/base.py`). Never
  reintroduce a hard-coded enum where a continuous range with reference marks
  works.
- **One generic mechanism per UI concept, not per-primitive UI code.** Segment
  dataclass fields opt into GUI generation via `param()` (slider),
  `enum_param()` (dropdown), or `optional_group()` (collapsible nested
  sub-dataclass, e.g. rotation_track's world movement) — see
  `src/camtraj/segments/base.py` and `apps/_shared/param_forms.py`. Adding a
  new segment type or a new field means the app UI updates itself; you should
  not need to touch `apps/trajectory_designer.py` for that. Known limit: the
  app's teardown bookkeeping currently assumes at most one `optional_group()`
  per segment (documented in `add_optional_groups`'s docstring) — revisit that
  if a second one is ever needed.
- **Every geometric primitive gets invariant tests, not just smoke tests.**
  E.g. orbit: constant radius when spiral=0, always-looking-at-target, a
  *locked-down concrete numeric example* for the cw/ccw sign convention (not
  just "it runs"). This caught real bugs before they reached the app. Mirror
  this pattern for any new primitive.

## Repo layout

```
src/camtraj/            core library, no UI deps (numpy + scipy only)
  conventions.py          convention framework + convert_pose
  trajectory.py           Trajectory: the one canonical data structure
  easing.py               easing curves + a 0-1 "strength" blend toward linear
  look_at.py              look_at_rotation (shared by orbit/tail/rotation_track)
  box_motion.py           BoxMotion (translation-only scene-anchor motion)
  sequence.py             chain segments (+ box motions) into one Trajectory
  segments/
    base.py                 SegmentBase ABC + param()/enum_param()/optional_group()
    free_form.py             camera-local translate+rotate, no target needed
    orbit.py                 revolve around the box, generic axis-angle rotation
    tail.py                  chase-cam, damped follow, per-axis amplitude
    rotation_track.py        re-aim at the box while optionally translating/pushing

src/camtraj_viser/      vendored + patched viser (import as camtraj_viser, NOT
                        viser -- see CAMTRAJ_PATCH.md in that dir for exactly
                        what's patched and why, and how to rebuild the client)

apps/
  trajectory_designer.py  the interactive viser app (multi-segment editor,
                          playback, on-demand POV renderer)
  _shared/viz.py           scene-building helpers (ground, box, frustums, path)
  _shared/param_forms.py   generic slider/dropdown/optional-group GUI builders

scripts/
  sanity_check_*.py       matplotlib, no server needed -- run these first when
                          adding/changing a primitive, before touching the app

tests/                  pytest; one file per module, invariant-style (see above)
```

## Environment

- micromamba env `camera` (Python 3.10). `pip install -e ".[dev,viz]"` from
  repo root installs everything, including the vendored viser's own runtime
  deps (they're no longer pulled in transitively from PyPI, since we don't
  depend on real `viser`).
- Node.js is only needed to *rebuild* `src/camtraj_viser`'s frontend after
  patching it further (see that directory's `CAMTRAJ_PATCH.md`). Never needed
  to just run the app — the built client is committed.
- Run the app: `micromamba run -n camera python -m apps.trajectory_designer`.
  Run tests: `micromamba run -n camera python -m pytest tests/ -q`.

## Deliberately deferred / not yet done

- Original DSL's `jitter`, `ver` (vertical-angle bias), and `object` (framing
  offset) common modifiers aren't implemented for any primitive yet.
- Export to other conventions (OpenCV/COLMAP/etc.) is fully supported by
  `conventions.py` but not yet wired into the app as an actual export button.
- General UI/UX polish beyond what's been explicitly requested so far
  (progressive disclosure of "advanced" params, etc.) is an open, ongoing
  thread, not a finished feature.
