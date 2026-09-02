"""Generic GUI-form builder for the batch generator: the range-editing
counterpart to `param_forms.py`. Every `param()` field becomes a 2-handle
range slider (both handles at the same value = pinned); every
`enum_param()` field becomes one checkbox per allowed member (checking only
one pins it); every `group()` field recurses the same way inside its own
collapsed-by-default folder. A new segment type gets a working range-editing
form for free, exactly like the single-trajectory app's sliders do.
"""

from __future__ import annotations

from typing import Any, Callable

import camtraj_viser as viser

from camtraj.segments.base import get_enum_specs, get_group_specs, get_param_specs


def add_range_sliders(
    server: viser.ViserServer,
    cls: type,
    ranges: dict[str, Any],
    on_change: Callable[[str, tuple[float, float]], None],
) -> dict[str, viser.GuiMultiSliderHandle]:
    """One 2-handle range slider per `param()` field of `cls`, bound to
    `ranges[name]` (a `(lo, hi)` pair). `on_change(name, (lo, hi))` fires
    whenever either handle moves. Returns {field_name: slider_handle}."""
    handles: dict[str, viser.GuiMultiSliderHandle] = {}
    for spec in get_param_specs(cls):
        marks = tuple(sorted(spec.marks.items())) if spec.marks else None
        lo, hi = ranges[spec.name]
        handles[spec.name] = server.gui.add_multi_slider(
            spec.label,
            min=spec.min,
            max=spec.max,
            step=spec.step,
            initial_value=(lo, hi),
            marks=marks,
            hint=spec.name + (f" ({spec.unit})" if spec.unit else "") + " -- drag both handles together to pin",
        )

    def _make_callback(name: str):
        def _callback(_) -> None:
            on_change(name, handles[name].value)

        return _callback

    for name, handle in handles.items():
        handle.on_update(_make_callback(name))

    return handles


def add_enum_multiselects(
    server: viser.ViserServer,
    cls: type,
    ranges: dict[str, Any],
    on_change: Callable[[str, list[str]], None],
) -> dict[str, list[viser.GuiCheckboxHandle]]:
    """One checkbox per member of each `enum_param()` field of `cls`, bound
    to `ranges[name]` (a list of allowed `.value` strings). Checking only
    one box pins that field; checking several lets it vary per draw. At
    least one box must stay checked (an empty allowed-list can't be sampled
    from), enforced by snapping the last unchecked box back on.
    Returns {field_name: [checkbox_handle, ...]}."""
    handles: dict[str, list[viser.GuiCheckboxHandle]] = {}
    for spec in get_enum_specs(cls):
        allowed = set(ranges[spec.name])
        checkboxes: list[viser.GuiCheckboxHandle] = []
        for member in spec.enum_cls:
            checkbox = server.gui.add_checkbox(f"{spec.label}: {member.value}", initial_value=member.value in allowed)
            checkboxes.append(checkbox)
        handles[spec.name] = checkboxes

        def _make_callback(spec=spec, checkboxes=checkboxes):
            def _callback(event) -> None:
                chosen = [m.value for m, cb in zip(spec.enum_cls, checkboxes) if cb.value]
                if not chosen:
                    event.target.value = True  # can't allow zero choices -- revert
                    return
                on_change(spec.name, chosen)

            return _callback

        callback = _make_callback()
        for checkbox in checkboxes:
            checkbox.on_update(callback)

    return handles


def add_range_groups(
    server: viser.ViserServer,
    cls: type,
    ranges: dict[str, Any],
    handles: list,
    get_current: Callable[[str], Any],
    on_replace: Callable[[str, Any], None],
    get_pinned: Callable[[str], Any],
    rebuild: Callable[[], None],
) -> None:
    """For each `group()` field of `cls`: a folder (collapsed by default)
    with that field's own range sliders / multi-selects, plus a "Reset ... to
    pinned" button that collapses every field in the group back to the
    skeleton's original single value (via `get_pinned(name)`, expected to
    return a fresh `default_ranges(...)`-shaped dict). Mirrors `add_groups`
    (the single-trajectory app's pinned-value version) but for ranges. Every
    created handle (the folder itself, its sliders/checkboxes, the reset
    button) is appended to the caller's own `handles` list, the same
    teardown convention every other builder in this module and in
    `param_forms.py` follows -- omitting this would leak a duplicate folder
    on every rebuild.
    """
    for spec in get_group_specs(cls):
        folder = server.gui.add_folder(spec.label, expand_by_default=False)
        handles.append(folder)
        with folder:
            group_ranges = get_current(spec.name)

            def on_field_change(field_name: str, value, spec=spec) -> None:
                current = dict(get_current(spec.name))
                current[field_name] = value
                on_replace(spec.name, current)

            handles.extend(add_range_sliders(server, spec.group_cls, group_ranges, on_field_change).values())
            for checkboxes in add_enum_multiselects(server, spec.group_cls, group_ranges, on_field_change).values():
                handles.extend(checkboxes)

            reset_button = server.gui.add_button(f"Reset {spec.label.lower()} to pinned", icon=viser.Icon.RESTORE)
            handles.append(reset_button)

            @reset_button.on_click
            def _(_, spec=spec) -> None:
                on_replace(spec.name, get_pinned(spec.name))
                rebuild()
