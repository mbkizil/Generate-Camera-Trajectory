# camtraj's fork of viser

This directory is a vendored, renamed copy of [viser](https://github.com/nerfstudio-project/viser)
1.1.0 (Apache-2.0, see `LICENSE_UPSTREAM`), imported as `camtraj_viser` instead
of `viser` so it can't be confused with a real PyPI install of viser.

It's vendored (not a pip dependency, not a git submodule) so that cloning the
`camtraj` repo alone is enough to run the app -- no separate fetch step, and
no Node.js/npm needed at runtime, since the frontend is already built and
committed under `client/build/`.

## What's actually patched

Only `add_button_group` gained two new optional parameters, `colors` and
`sizes`, used by `apps/trajectory_designer.py`'s segment timeline (real
per-segment color instead of viser's default plain outline buttons, and real
proportional width instead of equal-width buttons). Everything else is
unmodified stock viser 1.1.0.

Changed files:
- `_messages.py`: `GuiButtonGroupProps` gained `colors` / `sizes` fields.
- `_gui_api.py`: `add_button_group()` gained `colors` / `sizes` parameters.
- `client/src/WebsocketMessages.ts`: matching fields on `GuiButtonGroupMessage`
  (this file is normally auto-generated from the Python dataclasses by a
  codegen script upstream doesn't ship in the PyPI wheel, so this was edited
  by hand -- keep it in sync manually if `_messages.py` changes again).
- `client/src/components/ButtonGroup.tsx`: renders per-option `color` (via the
  existing `toMantineColor` helper, same as the singular Button component) and
  `flexGrow` from `sizes`; the currently-selected option renders `variant="filled"`,
  others `variant="light"` (both colored), falling back to the original
  `variant="outline"` when no color is given -- so any *other* code calling
  `add_button_group()` without `colors`/`sizes` renders exactly as before.
- Six files had an absolute `import viser` / `from viser import ...` changed
  to `camtraj_viser` (internal self-imports; see `git log` on this directory
  for the exact lines) so the renamed package still imports itself correctly.

## Rebuilding after a further patch

```bash
micromamba run -n camera bash -c "cd src/camtraj_viser/client && npm install && npm run build"
```

`npm run build` runs `tsc` first, so a type error in a `.ts`/`.tsx` change
fails the build loudly instead of shipping silently. The build output
(`client/build/index.html`, a single self-contained file) must be committed --
nothing rebuilds it automatically for end users.

## Pulling in upstream viser updates

There's no submodule/upstream remote wired up for this. To update: download a
newer viser wheel, diff its `viser/` against this directory to see what
changed upstream, re-apply the four patches above on top, and rebuild. Given
the diff is small and localized, this should stay a manageable, occasional
manual step rather than a constant maintenance burden -- but it *is* a real
one, worth remembering next time viser ships something we want.
