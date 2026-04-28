"""Helpers for packing and unpacking staged FasterGS runtime outputs."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ember_native_faster_gs.faster_gs.runtime.types import (
    BlendResult,
    PreprocessResult,
    RenderResult,
    SortResult,
)

_PREPROCESS_OUTPUT_COUNT = 10
_SORT_OUTPUT_COUNT = 4
_BLEND_OUTPUT_COUNT = 6
_RENDER_OUTPUT_COUNT = (
    1
    + _PREPROCESS_OUTPUT_COUNT
    + _SORT_OUTPUT_COUNT
    + (_BLEND_OUTPUT_COUNT - 1)
)


@dataclass(frozen=True)
class ParsedRenderOutputs:
    """Internal parsed view over the flattened combined render outputs."""

    preprocess: PreprocessResult
    sort: SortResult
    blend: BlendResult

    @property
    def image(self) -> Tensor:
        """Convenience access to the rendered image."""
        return self.blend.image


def parse_preprocess_outputs(outputs: tuple[Tensor, ...]) -> PreprocessResult:
    """Convert raw preprocess op outputs into a structured result."""
    return PreprocessResult.from_tensors(*outputs)


def parse_sort_outputs(outputs: tuple[Tensor, ...]) -> SortResult:
    """Convert raw sort op outputs into a structured result."""
    return SortResult.from_tensors(*outputs)


def parse_blend_outputs(outputs: tuple[Tensor, ...]) -> BlendResult:
    """Convert raw blend op outputs into a structured result."""
    return BlendResult.from_tensors(*outputs)


def pack_render_outputs(
    preprocess_outputs: tuple[Tensor, ...],
    sort_outputs: tuple[Tensor, ...],
    blend_outputs: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Flatten staged outputs for the combined render ops."""
    return (
        blend_outputs[0],
        *preprocess_outputs,
        *sort_outputs,
        *blend_outputs[1:],
    )


def parse_render_outputs(outputs: tuple[Tensor, ...]) -> ParsedRenderOutputs:
    """Convert raw render op outputs into an internal structured staged view."""
    if len(outputs) != _RENDER_OUTPUT_COUNT:
        raise ValueError(
            "Unexpected faster_gs::render output arity: "
            f"expected {_RENDER_OUTPUT_COUNT}, got {len(outputs)}."
        )
    preprocess_start = 1
    preprocess_stop = preprocess_start + _PREPROCESS_OUTPUT_COUNT
    sort_stop = preprocess_stop + _SORT_OUTPUT_COUNT
    image = outputs[0]
    preprocess = parse_preprocess_outputs(outputs[preprocess_start:preprocess_stop])
    sort = parse_sort_outputs(outputs[preprocess_stop:sort_stop])
    blend = BlendResult.from_tensors(image, *outputs[sort_stop:])
    return ParsedRenderOutputs(preprocess=preprocess, sort=sort, blend=blend)


def make_render_result(outputs: tuple[Tensor, ...]) -> RenderResult:
    """Build the public render result from raw render op outputs."""
    return RenderResult(image=outputs[0])
