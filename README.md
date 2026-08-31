# camtraj

User-friendly camera trajectory authoring, randomization, visualization, and
export. Successor to the trajectory-generation logic buried inside LAMP's DSL
scripts — same motion vocabulary, rebuilt on one clean, documented, tested
convention instead of four inconsistent hand-tuned ones.

## Status

Early. Core math library only so far: `free_form` is the one motion primitive
implemented; no interactive viewer or export formats yet. See the plan for the
full roadmap.

## Install

```bash
micromamba activate camera
pip install -e ".[dev,viz]"
```

## Layout

```
src/camtraj/
  conventions.py   # camera convention framework (right/up/forward + c2w/w2c),
                    # generic conversion instead of hand-tuned per-pair hacks
  trajectory.py     # Trajectory: the one canonical in-memory representation
  easing.py         # acceleration curves (linear/ease_in/out/in_out/out_in)
  segments/
    base.py         # SegmentBase + the generic param()/get_param_specs() UI-metadata mechanism
    free_form.py     # free_form primitive: camera-local translate + rotate
  sequence.py        # chain segments into one continuous Trajectory
scripts/
  sanity_check_free_form.py   # matplotlib 3D plot, no server needed -- run this first
tests/               # geometric invariants (convention round-trips, motion signs, continuity)
```

## Try it

```bash
micromamba run -n camera python -m pytest tests/ -v
micromamba run -n camera python scripts/sanity_check_free_form.py
# then open outputs/sanity_free_form.png and outputs/sanity_pan_only.png
```

## Conventions, in one place

Internally, a `Trajectory` is always camera-to-world, right-handed, +Y up,
camera looks down local -Z (the "OpenGL/NeRF" convention) — see
`camtraj.conventions.OPENGL`. Other conventions (OpenCV, COLMAP, ...) are
described the same way (which local axis is right/up/forward, plus whether
poses are stored camera-to-world or world-to-camera) and converted to/from
generically via `camtraj.conventions.convert_pose` — never with an ad hoc,
per-target sign-flip hack.
