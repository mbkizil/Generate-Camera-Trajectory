"""Generic GUI-form builder: turns any segment's `param()` fields into sliders
via `get_param_specs`. A new segment type gets a working form for free -- no
per-primitive form-building code, and the old DSL's named tiers survive as
slider `marks` instead of hard limits.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import camtraj_viser as viser

from camtraj.segments.base import get_enum_specs, get_group_specs, get_param_specs


def add_param_sliders(
    server: viser.ViserServer,
    segment_obj,
    on_change: Callable[[str, float], None],
) -> dict[str, viser.GuiSliderHandle]:
    """Add one slider per `param()` field of `segment_obj` (a class, for
    defaults, or an instance, to reflect its current values). `on_change(name,
    value)` fires whenever any slider moves. Returns {field_name: slider_handle}."""
    handles: dict[str, viser.GuiSliderHandle] = {}
    for spec in get_param_specs(segment_obj):
        marks = tuple(sorted(spec.marks.items())) if spec.marks else None
        handles[spec.name] = server.gui.add_slider(
            spec.label,
            min=spec.min,
            max=spec.max,
            step=spec.step,
            initial_value=getattr(segment_obj, spec.name),
            marks=marks,
            hint=spec.name + (f" ({spec.unit})" if spec.unit else ""),
        )

    def _make_callback(name: str):
        def _callback(_) -> None:
            on_change(name, handles[name].value)

        return _callback

    for name, handle in handles.items():
        handle.on_update(_make_callback(name))

    return handles


def add_optional_groups(
    server: viser.ViserServer,
    segment_obj,
    handles: list,
    get_current: Callable[[str], Any | None],
    on_replace: Callable[[str, Any | None], None],
) -> list:
    """For each `optional_group()` field of `segment_obj`: appends a checkbox
    to `handles`, and (if enabled) that group's own `param()`/`enum_param()`
    widgets directly under it -- mutating the caller's own `handles` list in
    place (like `rebuild_and_redraw`-driven code elsewhere in the app), so its
    usual `for h in handles: h.remove()` teardown stays correct even across a
    later checkbox toggle. `get_current(name)` fetches the field's live value
    (a group instance or None); `on_replace(name, value)` writes a new one
    back and should trigger whatever re-render the caller needs. Returns the
    list of checkbox handles created (one per group), so a caller that adds
    more dynamic sections *before* this call can bound its own teardown by
    "up to the first of these checkboxes" instead of "to the end of `handles`".

    Call this **last**, after any other dynamic sections in the same handles
    list (e.g. box motion) -- each group's own nested widgets are torn down as
    "everything after this group's checkbox," which is only safe when nothing
    else follows it in `handles`.
    """
    checkboxes: list = []
    for spec in get_group_specs(segment_obj):
        checkbox = server.gui.add_checkbox(spec.label, initial_value=get_current(spec.name) is not None)
        handles.append(checkbox)
        checkboxes.append(checkbox)

        def rebuild_nested(spec=spec, checkbox=checkbox) -> None:
            start = handles.index(checkbox) + 1
            for h in handles[start:]:
                h.remove()
            del handles[start:]
            if not checkbox.value:
                return
            group = get_current(spec.name) or spec.group_cls()

            def on_field_change(field_name: str, value) -> None:
                current = get_current(spec.name) or spec.group_cls()
                on_replace(spec.name, dataclasses.replace(current, **{field_name: value}))

            sliders = add_param_sliders(server, group, on_field_change)
            handles.extend(sliders.values())
            dropdowns = add_enum_dropdowns(server, group, on_field_change)
            handles.extend(dropdowns.values())

        @checkbox.on_update
        def _(_, spec=spec, rebuild_nested=rebuild_nested) -> None:
            on_replace(spec.name, spec.group_cls() if checkbox.value else None)
            rebuild_nested()

        rebuild_nested()

    return checkboxes


def add_enum_dropdowns(
    server: viser.ViserServer,
    segment_obj,
    on_change: Callable[[str, Any], None],
) -> dict[str, viser.GuiDropdownHandle]:
    """Add one dropdown per `enum_param()` field of `segment_obj`. `on_change(name,
    enum_member)` fires whenever any dropdown changes. Returns {field_name: handle}."""
    specs = {spec.name: spec for spec in get_enum_specs(segment_obj)}
    handles: dict[str, viser.GuiDropdownHandle] = {}
    for name, spec in specs.items():
        current = getattr(segment_obj, name)
        current_value = current.value if hasattr(current, "value") else current
        handles[name] = server.gui.add_dropdown(
            spec.label,
            options=[member.value for member in spec.enum_cls],
            initial_value=current_value,
        )

    def _make_callback(name: str):
        enum_cls = specs[name].enum_cls

        def _callback(_) -> None:
            on_change(name, enum_cls(handles[name].value))

        return _callback

    for name, handle in handles.items():
        handle.on_update(_make_callback(name))

    return handles
