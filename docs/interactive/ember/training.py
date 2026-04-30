"""Interactive Ember training tutorial."""

# ruff: noqa: ANN001, ANN202

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _():
    import ember_core as sk
    import marimo as mo

    mo.md(
        r"""
        # Ember Training

        Training is configured declaratively. User-facing notebooks can expose
        a compact Pydantic model and materialize Ember's lower-level
        `TrainingConfig` only when the run starts.
        """
    )
    return mo, sk


@app.cell
def _(mo):
    backend = mo.ui.dropdown(
        options=["adapter.gsplat", "faster_gs.core", "docs.toy"],
        value="adapter.gsplat",
        label="Render backend",
    )
    steps = mo.ui.slider(start=10, stop=1000, step=10, value=100, label="Steps")
    mo.vstack([backend, steps])
    return backend, steps


@app.cell
def _(sk, backend, steps):
    training_config = sk.TrainingConfig(
        runtime=sk.RuntimeConfig(max_steps=steps.value),
        render=sk.RenderPipelineSpec(backend=backend.value),
        batching=sk.BatchingConfig(batch_size=1),
        loss=sk.LossConfig(
            target=sk.CallableSpec(target="my_project.losses.rgb_l2_loss"),
        ),
        optimization=sk.OptimizationConfig(parameter_groups=()),
    )
    return (training_config,)


@app.cell(hide_code=True)
def _(mo, training_config):
    mo.md(
        f"""
        ---

        ## Materialized Training Config

        ```python
        {training_config.model_dump()}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Real Run Shape

        ```python
        scene_record = sk.load_scene_record(scene_config)
        dataset = sk.prepare_frame_dataset(scene_record, frame_config)
        result = sk.run_training(dataset, training_config)
        checkpoint = sk.load_checkpoint_dir(result.checkpoint_dir)
        ```

        Keep paper-specific ergonomics in the notebook config. Convert to
        `TrainingConfig` at the boundary where reproducible execution starts.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Training Boundaries

        - Runtime config controls seed/device behavior.
        - Render config chooses a registered backend.
        - Loss and hook configs point to importable functions.
        - Optimization config targets scene/model parameters declaratively.
        - Checkpoints export config, metadata, model tensors, and optional
          scene files.
        """
    )
    return


if __name__ == "__main__":
    app.run()
