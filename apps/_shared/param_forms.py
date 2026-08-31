"""Generic GUI-form builder: turns any segment's `param()` fields into sliders
via `get_param_specs`. A new segment type gets a working form for free -- no
per-primitive form-building code, and the old DSL's named tiers survive as
slider `marks` instead of hard limits.
"""

from __future__ import annotations

from typing import Callable

import viser

from camtraj.segments.base import get_param_specs


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
