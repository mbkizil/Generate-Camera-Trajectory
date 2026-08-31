"""Shared infrastructure for motion primitives ("segments").

Design principle: a segment is a plain dataclass of continuous parameters. Any
field can carry UI-facing metadata (range, unit, and named "reference marks" —
e.g. the old DSL's `far_left/left/near_left/no/...` tiers become slider tick
marks instead of the only allowed values) via the `param()` field wrapper below.
A single generic `get_param_specs()` walks any segment class's fields to build
its UI form — so a new segment type gets a working GUI for free, with no
per-primitive form-building code and no separate "spec" class to keep in sync.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..trajectory import Trajectory

_META_KEY = "camtraj_param"


@dataclass(frozen=True)
class ParamSpec:
    """UI-facing description of one segment parameter."""

    name: str
    label: str
    min: float
    max: float
    default: float
    step: float
    unit: str = ""
    marks: dict[float, str] | None = None
    """Named reference values worth exposing as slider tick marks, e.g.
    {-1.0: "far_left", -0.667: "left", ..., 1.0: "far_right"} — kept for
    continuity with the original DSL's discrete presets, without limiting the
    parameter to only those values."""


def param(
    *,
    label: str,
    min: float,
    max: float,
    default: float,
    step: float | None = None,
    unit: str = "",
    marks: dict[float, str] | None = None,
) -> dataclasses.Field:
    """Dataclass field wrapper that attaches a `ParamSpec` to a segment parameter."""
    spec = ParamSpec(
        name="",
        label=label,
        min=min,
        max=max,
        default=default,
        step=step if step is not None else _auto_step(min, max),
        unit=unit,
        marks=marks,
    )
    return dataclasses.field(default=default, metadata={_META_KEY: spec})


def _auto_step(lo: float, hi: float) -> float:
    span = abs(hi - lo) or 1.0
    return round(span / 100.0, 6)


def get_param_specs(segment) -> list[ParamSpec]:
    """Introspect a segment instance or class, returning its `param()` fields' specs."""
    specs = []
    for f in dataclasses.fields(segment):
        spec = f.metadata.get(_META_KEY)
        if spec is not None:
            specs.append(dataclasses.replace(spec, name=f.name))
    return specs


class SegmentBase(ABC):
    """Base class for a motion primitive.

    A segment's frame count (`frames`) is one of its own parameters -- length is
    per-segment and freely adjustable, not imposed by whoever calls `build()`.
    A segment is always built relative to the pose it continues from, so
    segments can be freely chained (see `camtraj.sequence.sequence`) regardless
    of how many frames the previous segment used.
    """

    frames: int

    @abstractmethod
    def build(self, start_position: np.ndarray, start_rotation: Rotation) -> Trajectory:
        """Return a `self.frames`-frame Trajectory starting exactly at
        (start_position, start_rotation) and ending at this segment's
        fully-applied motion."""
        raise NotImplementedError
