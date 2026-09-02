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


def add_groups(
    server: viser.ViserServer,
    segment_obj,
    handles: list,
    get_current: Callable[[str], Any],
    on_replace: Callable[[str, Any], None],
    rebuild: Callable[[], None],
) -> None:
    """For each `group()` field of `segment_obj`: a folder (collapsed by
    default) containing that field's own `param()`/`enum_param()` widgets,
    plus a "Reset ... to defaults" button -- no on/off checkbox, since the
    field is always present (a zero-valued group instance already means "no
    effect," the same convention `BoxMotion` uses). Every created handle is
    appended to the caller's own `handles` list, like the rest of this
    module's builders. `get_current(name)` fetches the field's live value;
    `on_replace(name, value)` writes a new one back. `rebuild()` is called
    after a reset so the caller can redraw its whole form with fresh slider
    values (mirrors how a segment's own "Reset to defaults" button works) --
    unlike the old checkbox-driven version, there's no positional teardown
    assumption here, so multiple groups per segment are fine.
    """
    for spec in get_group_specs(segment_obj):
        folder = server.gui.add_folder(spec.label, expand_by_default=False)
        handles.append(folder)
        with folder:
            group_value = get_current(spec.name)

            def on_field_change(field_name: str, value, spec=spec) -> None:
                current = get_current(spec.name)
                on_replace(spec.name, dataclasses.replace(current, **{field_name: value}))

            sliders = add_param_sliders(server, group_value, on_field_change)
            handles.extend(sliders.values())
            dropdowns = add_enum_dropdowns(server, group_value, on_field_change)
            handles.extend(dropdowns.values())

            reset_button = server.gui.add_button(f"Reset {spec.label.lower()} to defaults", icon=viser.Icon.RESTORE)
            handles.append(reset_button)

            @reset_button.on_click
            def _(_, spec=spec) -> None:
                on_replace(spec.name, spec.group_cls())
                rebuild()


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
