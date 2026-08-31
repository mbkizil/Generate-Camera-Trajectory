from .base import (
    EnumParamSpec,
    OptionalGroupSpec,
    ParamSpec,
    SegmentBase,
    enum_param,
    get_enum_specs,
    get_group_specs,
    get_param_specs,
    optional_group,
    param,
)
from .free_form import FreeFormSegment
from .orbit import OrbitAxis, OrbitDirection, OrbitSegment
from .tail import LookMode, TailSegment
from .rotation_track import RotAxis, RotationTrackSegment, WorldMove

SEGMENT_TYPES: dict[str, type[SegmentBase]] = {
    "free_form": FreeFormSegment,
    "orbit": OrbitSegment,
    "tail": TailSegment,
    "rotation_track": RotationTrackSegment,
}

__all__ = [
    "ParamSpec",
    "EnumParamSpec",
    "OptionalGroupSpec",
    "SegmentBase",
    "get_param_specs",
    "get_enum_specs",
    "get_group_specs",
    "param",
    "enum_param",
    "optional_group",
    "FreeFormSegment",
    "OrbitSegment",
    "OrbitAxis",
    "OrbitDirection",
    "TailSegment",
    "LookMode",
    "RotationTrackSegment",
    "RotAxis",
    "WorldMove",
    "SEGMENT_TYPES",
]
