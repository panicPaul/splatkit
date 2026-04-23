"""Public staged runtime API for the FasterGS Mojo backend."""

from __future__ import annotations

from typing import Any

from splatkit_native_faster_gs.faster_gs.runtime.packing import (
    make_render_result,
    parse_blend_outputs,
    parse_preprocess_outputs,
    parse_sort_outputs,
)
from splatkit_native_faster_gs.faster_gs.runtime.types import (
    BlendResult,
    PreprocessResult,
    RenderResult,
    SortResult,
)
from splatkit_native_faster_gs_mojo.core.runtime.ops import (
    blend_image_only,
    blend_op,
    preprocess_op,
    render_op,
    sort_op,
)
from splatkit_native_faster_gs_mojo.core.runtime.ops._common import (
    requires_grad,
)


_RENDER_ARG_NAMES = (
    "center_positions",
    "log_scales",
    "unnormalized_rotations",
    "opacities",
    "sh_coefficients_0",
    "sh_coefficients_rest",
    "world_2_camera",
    "camera_position",
    "near_plane",
    "far_plane",
    "width",
    "height",
    "focal_x",
    "focal_y",
    "center_x",
    "center_y",
    "bg_color",
    "proper_antialiasing",
    "active_sh_bases",
)


def _bind_render_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if len(args) > len(_RENDER_ARG_NAMES):
        msg = (
            "render() received too many positional arguments for the "
            "FasterGS Mojo runtime."
        )
        raise TypeError(msg)

    bound_inputs = {
        name: value for name, value in zip(_RENDER_ARG_NAMES[: len(args)], args)
    }
    for name in _RENDER_ARG_NAMES[len(args) :]:
        if name not in kwargs:
            msg = f"render() missing required argument: {name!r}"
            raise TypeError(msg)
        bound_inputs[name] = kwargs[name]
    return bound_inputs


def preprocess(*args, **kwargs) -> PreprocessResult:
    """Run the preprocess stage for the FasterGS Mojo family."""
    return parse_preprocess_outputs(preprocess_op(*args, **kwargs))


def sort(*args, **kwargs) -> SortResult:
    """Run the sort stage for the FasterGS Mojo family."""
    return parse_sort_outputs(sort_op(*args, **kwargs))


def blend(*args, **kwargs) -> BlendResult:
    """Run the blend stage for the FasterGS Mojo family."""
    return parse_blend_outputs(blend_op(*args, **kwargs))


def render(*args, **kwargs) -> RenderResult:
    """Run the full render stage for the FasterGS Mojo family."""
    if len(args) == 8 and all(name in kwargs for name in _RENDER_ARG_NAMES[8:]):
        center_positions = args[0]
        log_scales = args[1]
        unnormalized_rotations = args[2]
        opacities = args[3]
        sh_coefficients_0 = args[4]
        sh_coefficients_rest = args[5]
        world_2_camera = args[6]
        camera_position = args[7]
        near_plane = kwargs["near_plane"]
        far_plane = kwargs["far_plane"]
        width = kwargs["width"]
        height = kwargs["height"]
        focal_x = kwargs["focal_x"]
        focal_y = kwargs["focal_y"]
        center_x = kwargs["center_x"]
        center_y = kwargs["center_y"]
        bg_color = kwargs["bg_color"]
        proper_antialiasing = kwargs["proper_antialiasing"]
        active_sh_bases = kwargs["active_sh_bases"]
    else:
        bound_inputs = _bind_render_inputs(args, kwargs)
        center_positions = bound_inputs["center_positions"]
        log_scales = bound_inputs["log_scales"]
        unnormalized_rotations = bound_inputs["unnormalized_rotations"]
        opacities = bound_inputs["opacities"]
        sh_coefficients_0 = bound_inputs["sh_coefficients_0"]
        sh_coefficients_rest = bound_inputs["sh_coefficients_rest"]
        world_2_camera = bound_inputs["world_2_camera"]
        camera_position = bound_inputs["camera_position"]
        near_plane = bound_inputs["near_plane"]
        far_plane = bound_inputs["far_plane"]
        width = bound_inputs["width"]
        height = bound_inputs["height"]
        focal_x = bound_inputs["focal_x"]
        focal_y = bound_inputs["focal_y"]
        center_x = bound_inputs["center_x"]
        center_y = bound_inputs["center_y"]
        bg_color = bound_inputs["bg_color"]
        proper_antialiasing = bound_inputs["proper_antialiasing"]
        active_sh_bases = bound_inputs["active_sh_bases"]
    if not requires_grad(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
    ):
        preprocess_outputs = preprocess_op(
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane=near_plane,
            far_plane=far_plane,
            width=width,
            height=height,
            focal_x=focal_x,
            focal_y=focal_y,
            center_x=center_x,
            center_y=center_y,
            proper_antialiasing=proper_antialiasing,
            active_sh_bases=active_sh_bases,
        )
        sort_outputs = sort_op(
            preprocess_outputs[4],
            preprocess_outputs[5],
            preprocess_outputs[6],
            preprocess_outputs[7],
            preprocess_outputs[0],
            preprocess_outputs[1],
            preprocess_outputs[8],
            preprocess_outputs[9],
            width=width,
            height=height,
        )
        image = blend_image_only(
            sort_outputs[0],
            sort_outputs[1],
            sort_outputs[2],
            sort_outputs[3],
            preprocess_outputs[0],
            preprocess_outputs[1],
            preprocess_outputs[2],
            bg_color,
            width=width,
            height=height,
        )
        return RenderResult(image=image)
    return make_render_result(render_op(*args, **kwargs))


__all__ = [
    "BlendResult",
    "PreprocessResult",
    "RenderResult",
    "SortResult",
    "blend",
    "preprocess",
    "render",
    "sort",
]
