"""Helpers for packing and unpacking FasterGS depth runtime outputs."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ember_native_faster_gs.faster_gs_depth.runtime.types import (
    BlendResult,
    RenderResult,
)
from ember_native_faster_gs.faster_gs.runtime.types import (
    PreprocessResult,
    SortResult,
)

_PREPROCESS_OUTPUT_COUNT = 10
_SORT_OUTPUT_COUNT = 4
_BLEND_OUTPUT_COUNT = 8
_RENDER_OUTPUT_COUNT = 2 + _PREPROCESS_OUTPUT_COUNT + _SORT_OUTPUT_COUNT + (
    _BLEND_OUTPUT_COUNT - 2
)


@dataclass(frozen=True)
class ParsedRenderOutputs:
    """Structured view over flattened depth render outputs."""

    preprocess: PreprocessResult
    sort: SortResult
    blend: BlendResult

    @property
    def image(self) -> Tensor:
        """Convenience access to the rendered RGB image."""
        return self.blend.image

    @property
    def depth(self) -> Tensor:
        """Convenience access to the rendered depth image."""
        return self.blend.depth


def parse_blend_outputs(outputs: tuple[Tensor, ...]) -> BlendResult:
    """Convert raw blend outputs into a structured result."""
    return BlendResult.from_tensors(*outputs)


def pack_render_outputs(
    preprocess_outputs: tuple[Tensor, ...],
    sort_outputs: tuple[Tensor, ...],
    blend_outputs: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Flatten staged outputs for the combined depth render ops."""
    return (
        blend_outputs[0],
        blend_outputs[1],
        *preprocess_outputs,
        *sort_outputs,
        *blend_outputs[2:],
    )


def parse_render_outputs(outputs: tuple[Tensor, ...]) -> ParsedRenderOutputs:
    """Convert raw render outputs into an internal structured staged view."""
    if len(outputs) != _RENDER_OUTPUT_COUNT:
        raise ValueError(
            "Unexpected faster_gs_depth::render output arity: "
            f"expected {_RENDER_OUTPUT_COUNT}, got {len(outputs)}."
        )
    preprocess_start = 2
    preprocess_stop = preprocess_start + _PREPROCESS_OUTPUT_COUNT
    sort_stop = preprocess_stop + _SORT_OUTPUT_COUNT
    preprocess = PreprocessResult.from_tensors(*outputs[preprocess_start:preprocess_stop])
    sort = SortResult.from_tensors(*outputs[preprocess_stop:sort_stop])
    blend = BlendResult.from_tensors(outputs[0], outputs[1], *outputs[sort_stop:])
    return ParsedRenderOutputs(preprocess=preprocess, sort=sort, blend=blend)


def make_render_result(outputs: tuple[Tensor, ...]) -> RenderResult:
    """Build the public render result from raw render op outputs."""
    return RenderResult(image=outputs[0], depth=outputs[1])
