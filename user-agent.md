# Working in this repo (for coding agents)

CamTraj generates and visualizes camera (and object) trajectories: pick a
motion primitive, shape it with continuous sliders, get an exportable 3D
trajectory — one handcrafted shot, or a batch of randomized variations. See
`README.md` for the full pitch and use cases.

## The one rule that matters

**Prefer changing the UI (`apps/`) over changing the math (`src/camtraj/`).**

Almost every user request — a new button, a different layout, another export
option, a tweaked default, a new way to preview something — is a UI/UX
change. The trajectory math underneath (`src/camtraj/segments/`,
`conventions.py`, `sequence.py`, `batch.py`, ...) is small, deliberately
designed, and covered by invariant tests (constant-radius checks, locked-down
sign conventions, continuity guarantees) that took real effort to get right.
Editing it casually risks silently reintroducing the exact class of bugs this
project was built to avoid (axis-flip mistakes, broken pose conventions,
discontinuous chained segments).

If a request truly needs new core behavior (a new motion primitive, a new
export convention, a new randomization mode), that's fine — it's meant to be
extensible — but:
- follow the existing patterns (a new primitive is a `SegmentBase` subclass
  using `param()`/`enum_param()`/`group()` for its fields; a new convention is
  a `Convention(...)` declaration, not new sign-flip math)
- add invariant tests for it, not just a smoke test
- run `python -m pytest tests/ -q` before considering it done

## Where things are

```
src/camtraj/          core math/logic -- numpy + scipy only, no UI. Change
                       carefully; see "the one rule" above.
  segments/              the motion primitives (free_form, orbit, tail_track, rotation_track)
  conventions.py         pose conventions (OpenGL/Blender/OpenCV/COLMAP) + conversion
  sequence.py            chains segments into one continuous Trajectory
  recipe.py              save/load a trajectory design as JSON
  batch.py               randomized-values sampling (ranges around a fixed design)
  structure_batch.py     randomized-structure sampling (random sequences of fixed segments)

src/camtraj_viser/    vendored, patched viser frontend -- the UI framework the
                       apps run on. Only touch this for genuinely new UI
                       primitives the stock framework can't do (see
                       CAMTRAJ_PATCH.md for precedent); prefer plain app code
                       in apps/ otherwise.

apps/                 the three interactive apps -- this is where most
                       requested changes belong
  trajectory_designer.py        handcraft one trajectory
  auteur_designer.py            handcraft Auteur trajectory
  batch_generator.py            randomize parameter values around a fixed design
  structure_batch_generator.py  randomize which/how many segments, values fixed
  _shared/                      reusable UI-building helpers (sliders, scene setup)

scripts/              matplotlib sanity-check plots, no server needed
tests/                pytest, one file per module
outputs/              gitignored scratch space, except the three README images
```

## Practical notes

- Environment: micromamba env `camera`, Python 3.10+. `pip install -e ".[dev,viz]"`. If working with Auteur, then also `pip install -e ".[auteur]"`
- Run an app: `python -m apps.trajectory_designer` (or `batch_generator` /
  `structure_batch_generator` — each binds its own port, can run together).
- Run tests: `python -m pytest tests/ -q`.
- No Node.js needed to run anything — the vendored viser frontend is
  prebuilt and committed. Node is only needed if you patch
  `src/camtraj_viser/client/` itself (rare; see its `CAMTRAJ_PATCH.md`).
