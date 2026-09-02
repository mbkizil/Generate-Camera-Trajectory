# camtraj's fork of viser

This directory is a vendored, renamed copy of [viser](https://github.com/nerfstudio-project/viser)
1.1.0 (Apache-2.0, see `LICENSE_UPSTREAM`), imported as `camtraj_viser` instead
of `viser` so it can't be confused with a real PyPI install of viser.

It's vendored (not a pip dependency, not a git submodule) so that cloning the
`camtraj` repo alone is enough to run the app -- no separate fetch step, and
no Node.js/npm needed at runtime, since the frontend is already built and
committed under `client/build/`.

## What's actually patched

1. `add_button_group` gained two optional parameters, `colors` and `sizes`
   (real per-option color instead of viser's default plain outline buttons,
   and real proportional width instead of equal-width buttons). No longer
   used by the app (superseded by patch 2 below for the segment timeline
   specifically), but left in place as a general-purpose addition.
2. A new, standalone `set_segment_timeline` / `on_segment_timeline_action`
   API (`ViserServer.gui...`), used by `apps/trajectory_designer.py`'s
   segment timeline. This is **not** a GUI panel component -- it's a small
   dedicated message type + a React component (`SegmentTimeline.tsx`) mounted
   unconditionally at the App root, next to `CommandPalette`/`Titlebar`,
   deliberately bypassing viser's panel/dock system
   (`dock/FloatingWindowView.tsx`, `dock/TabGroupFrame.tsx`) entirely. That
   system always renders window chrome (header, background, drag/resize
   affordances) with no chromeless escape hatch, and floats at a fixed pixel
   position rather than staying centered -- neither of which the timeline
   wants. The overlay is fixed-positioned (`left: 50%; transform:
   translateX(-50%)`), so it stays centered regardless of segment count, and
   has no drag/resize code at all, so it's inherently not repositionable.
   Each segment box embeds its own "x" (hover-highlighted, hidden for segment
   0, which is a permanent anchor); "+" is a fixed trailing element.
3. A second, similarly standalone `set_keyframe_timeline` /
   `on_keyframe_timeline_action` API + `KeyframeTimeline.tsx` component, used
   by `apps/auteur_designer.py`. Same "bypass the panel/dock system"
   rationale as patch 2, but a different interaction model: a fixed-width
   bar spanning the whole trajectory (not resizable, sequentially-appended
   boxes), with draggable pins at absolute frame positions, click-on-empty-
   bar to add, and an embedded native `<input type="number">` for the total
   frame count. Pin 0 is permanently fixed at frame 0 (like segment 0 in
   patch 2); every other pin is drag-clamped between its immediate neighbors
   server-side (the client does an optimistic local clamp during drag for a
   smooth feel, but the server re-validates and always re-broadcasts the
   authoritative state after every action).

Changed files:
- `_messages.py`: `GuiButtonGroupProps` gained `colors` / `sizes` fields
  (patch 1). New `SegmentTimelineMessage`/`SegmentTimelineActionMessage`
  (patch 2) and `KeyframeTimelineMessage`/`KeyframeTimelineActionMessage`
  (patch 3) message types, all server->client full-replace broadcasts paired
  with a client->server action message.
- `_gui_api.py`: `add_button_group()` gained `colors` / `sizes` parameters
  (patch 1). New `GuiApi.set_segment_timeline()` / `.on_segment_timeline_action()`
  (patch 2) and `.set_keyframe_timeline()` / `.on_keyframe_timeline_action()`
  (patch 3) methods, each with a handler registration for their action message.
- `client/src/WebsocketMessages.ts`: matching fields on `GuiButtonGroupMessage`
  (patch 1); new message interfaces for patches 2 and 3, added to the
  `Message` union. This file is normally auto-generated from the Python
  dataclasses by a codegen script upstream doesn't ship in the PyPI wheel,
  so it's edited by hand -- keep it in sync manually if `_messages.py`
  changes again.
- `client/src/components/ButtonGroup.tsx`: renders per-option `color` (via the
  existing `toMantineColor` helper, same as the singular Button component) and
  `flexGrow` from `sizes`; the currently-selected option renders `variant="filled"`,
  others `variant="light"` (both colored), falling back to the original
  `variant="outline"` when no color is given -- so any *other* code calling
  `add_button_group()` without `colors`/`sizes` renders exactly as before
  (patch 1).
- `client/src/SegmentTimeline.tsx` (patch 2) / `client/src/KeyframeTimeline.tsx`
  (patch 3): the standalone overlay components described above.
- `client/src/ControlPanel/GuiState.ts`: new `segmentTimeline` (patch 2) /
  `keyframeTimeline` (patch 3) state fields + setter actions, mirroring the
  existing `commands` map's wiring but as a single nullable value, not an
  entity-keyed map (there's always exactly one of each timeline).
- `client/src/MessageHandler.tsx`: dispatches `SegmentTimelineMessage` /
  `KeyframeTimelineMessage` to their setters.
- `client/src/App.tsx`: mounts `<SegmentTimeline />` (patch 2) and
  `<KeyframeTimeline />` (patch 3) unconditionally, next to `<CommandPalette />`.
  Only one of the two ever actually renders anything in a given app, since
  each Python app only ever calls one of `set_segment_timeline` /
  `set_keyframe_timeline`, leaving the other's state permanently `null`.
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
