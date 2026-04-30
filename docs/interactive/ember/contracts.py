"""Interactive Ember contracts tutorial."""

# ruff: noqa: ANN001, ANN202

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _():
    import ember_core as sk
    import marimo as mo
    import torch

    mo.md(
        r"""
        # Ember Contracts

        Ember starts with explicit scene, camera, render output, and backend
        contracts. Backends register render functions by name, and notebooks
        request optional outputs through the shared wrapper.
        """
    )
    return mo, sk, torch


@app.cell
def _(sk, torch):
    scene = sk.GaussianScene3D(
        center_position=torch.zeros(4, 3),
        log_scales=torch.zeros(4, 3),
        quaternion_orientation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4,
        ),
        logit_opacity=torch.zeros(4),
        feature=torch.rand(4, 3),
        sh_degree=0,
    )
    camera = sk.CameraState(
        width=torch.tensor([96]),
        height=torch.tensor([64]),
        fov_degrees=torch.tensor([60.0]),
        cam_to_world=torch.eye(4).unsqueeze(0),
    )
    return camera, scene


@app.cell
def _(sk, torch):
    class ToyOptions(sk.RenderOptions):
        pass

    return (ToyOptions,)


@app.cell
def _(ToyOptions, sk, torch):
    def render_toy_backend(
        scene,
        camera,
        *,
        return_alpha=False,
        return_depth=False,
        return_gaussian_impact_score=False,
        return_normals=False,
        return_2d_projections=False,
        return_projective_intersection_transforms=False,
        options=None,
    ):
        del (
            scene,
            return_alpha,
            return_depth,
            return_gaussian_impact_score,
            return_normals,
            return_2d_projections,
            return_projective_intersection_transforms,
            options,
        )
        height = int(camera.height[0])
        width = int(camera.width[0])
        xs = torch.linspace(0.0, 1.0, width)
        ys = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        image = torch.stack((grid_x, grid_y, 1.0 - grid_x), dim=-1)
        return sk.RenderOutput(render=image.unsqueeze(0))

    sk.register_backend(
        name="docs.toy",
        default_options=ToyOptions(),
        accepted_scene_types=(sk.GaussianScene3D,),
    )(render_toy_backend)
    return (render_toy_backend,)


@app.cell
def _(camera, scene, sk):
    output = sk.render(scene, camera, backend="docs.toy")
    return (output,)


@app.cell(hide_code=True)
def _(mo, output, sk):
    mo.md(
        f"""
        ---

        ## Registered Backend

        `docs.toy` rendered a tensor with shape `{tuple(output.render.shape)}`.

        Registered backend names now include:

        ```text
        {", ".join(sorted(sk.BACKEND_REGISTRY))}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## What To Carry Forward

        - Core owns contracts, not heavy backend logic.
        - Backends are activated explicitly with `register_backend(...)`.
        - Optional outputs are declared as capabilities before callers can
          request them.
        - New research code can start with a local backend and be promoted
          later if the boundary proves useful.
        """
    )
    return


if __name__ == "__main__":
    app.run()
