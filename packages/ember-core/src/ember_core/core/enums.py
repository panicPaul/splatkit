"""Closed Ember enum concepts."""

from __future__ import annotations

from enum import StrEnum


class RenderMode(StrEnum):
    """Execution intent for a render request."""

    INFERENCE = "inference"
    TRAINING = "training"
    METRIC_PROBE = "metric_probe"


class ExecutionMode(StrEnum):
    """Runtime implementation mode."""

    PYTHON_REF = "python_ref"
    TORCH = "torch"
    NATIVE_MOJO = "native_mojo"


class BufferLifetime(StrEnum):
    """Lifetime class for planned buffers."""

    INPUT = "input"
    OUTPUT = "output"
    PERSISTENT = "persistent"
    SCRATCH = "scratch"
    ALIAS = "alias"


class AccessMode(StrEnum):
    """How a stage accesses a buffer."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class ParameterScope(StrEnum):
    """Closed parameter target scopes."""

    SCENE = "scene"
    MODULES = "modules"
    PARAMETERS = "parameters"


class BuiltinOptimizerKind(StrEnum):
    """Built-in optimizer kinds."""

    ADAM = "adam"
    SGD = "sgd"


class DeviceKind(StrEnum):
    """Device placement class used by buffer specs."""

    CPU = "cpu"
    CUDA = "cuda"


class DTypeName(StrEnum):
    """Portable dtype names for buffer specs."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT64 = "int64"
    INT32 = "int32"
    UINT16 = "uint16"
    UINT8 = "uint8"
    BOOL = "bool"
