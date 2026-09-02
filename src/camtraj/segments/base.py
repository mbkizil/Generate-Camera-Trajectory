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
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from ..trajectory import Trajectory

_META_KEY = "camtraj_param"
_ENUM_META_KEY = "camtraj_enum_param"
_GROUP_META_KEY = "camtraj_group"

DUTCH_MARKS = {v: str(int(v)) for v in (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)}
"""Shared reference marks for a dutch/roll angle -- a "common modifier" in the
original DSL, reused by every target-relative primitive (orbit, tail, ...)."""


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


@dataclass(frozen=True)
class EnumParamSpec:
    """UI-facing description of one segment dropdown (categorical) parameter."""

    name: str
    label: str
    enum_cls: type[Enum]
    default: Enum


def enum_param(*, label: str, enum_cls: type[Enum], default: Enum) -> dataclasses.Field:
    """Dataclass field wrapper attaching an `EnumParamSpec` -- the dropdown
    equivalent of `param()`, kept as metadata (not a type-annotation lookup)
    since `from __future__ import annotations` makes annotations plain strings
    at runtime."""
    spec = EnumParamSpec(name="", label=label, enum_cls=enum_cls, default=default)
    return dataclasses.field(default=default, metadata={_ENUM_META_KEY: spec})


def get_enum_specs(segment) -> list[EnumParamSpec]:
    """Introspect a segment instance or class, returning its `enum_param()` fields' specs."""
    specs = []
    for f in dataclasses.fields(segment):
        spec = f.metadata.get(_ENUM_META_KEY)
        if spec is not None:
            specs.append(dataclasses.replace(spec, name=f.name))
    return specs


@dataclass(frozen=True)
class GroupSpec:
    """UI-facing description of one nested sub-group of parameters on a
    segment (e.g. rotation_track's "World movement"), always present -- a
    zero-valued instance of `group_cls` means "this group's effect is
    disabled," the same convention `BoxMotion` uses."""

    name: str
    label: str
    group_cls: type


def group(*, label: str, group_cls: type) -> dataclasses.Field:
    """Dataclass field wrapper for a nested sub-dataclass of its own
    `param()`/`enum_param()` fields -- e.g. rotation_track's world movement.
    Always present (never `None`); the app renders it as its own
    collapsed-by-default folder with a dedicated reset button, no on/off
    checkbox -- a zero-valued `group_cls()` already means "no effect."
    `group_cls` must be a plain dataclass built the same way segments are."""
    spec = GroupSpec(name="", label=label, group_cls=group_cls)
    return dataclasses.field(default_factory=group_cls, metadata={_GROUP_META_KEY: spec})


def get_group_specs(segment) -> list[GroupSpec]:
    """Introspect a segment instance or class, returning its `group()` fields' specs."""
    specs = []
    for f in dataclasses.fields(segment):
        spec = f.metadata.get(_GROUP_META_KEY)
        if spec is not None:
            specs.append(dataclasses.replace(spec, name=f.name))
    return specs


def to_param_dict(instance) -> dict:
    """Recursively convert a `param()`/`enum_param()`/`group()` dataclass (a
    segment, `BoxMotion`, or a nested group like `WorldMove`) into a plain
    JSON-safe dict: enum fields become their `.value`, nested dataclass
    fields (`group()`) recurse the same way. Used both for a `Trajectory`'s
    own `metadata["params"]` provenance and for `camtraj.recipe`'s save/load."""
    result = {}
    for f in dataclasses.fields(instance):
        value = getattr(instance, f.name)
        if isinstance(value, Enum):
            result[f.name] = value.value
        elif dataclasses.is_dataclass(value):
            result[f.name] = to_param_dict(value)
        else:
            result[f.name] = value
    return result


def from_param_dict(cls: type, data: dict):
    """Inverse of `to_param_dict`: reconstruct a `cls` instance from a dict,
    using `cls`'s own `enum_param()`/`group()` field metadata to know which
    values need converting back (a plain dict can't carry that on its own)."""
    enum_classes = {spec.name: spec.enum_cls for spec in get_enum_specs(cls)}
    group_classes = {spec.name: spec.group_cls for spec in get_group_specs(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in enum_classes:
            kwargs[key] = enum_classes[key](value)
        elif key in group_classes:
            kwargs[key] = from_param_dict(group_classes[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


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
    def build(self, start_position: np.ndarray, start_rotation: Rotation, target_positions: np.ndarray) -> Trajectory:
        """Return a `self.frames`-frame Trajectory starting exactly at
        (start_position, start_rotation) and ending at this segment's
        fully-applied motion.

        `target_positions` is a (self.frames, 3) array giving the scene's
        anchor object ("the box")'s position at each frame of this segment --
        required for the interface to stay uniform, even though primitives
        that don't need a target (e.g. free_form) just ignore it."""
        raise NotImplementedError
