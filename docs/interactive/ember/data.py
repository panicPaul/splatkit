"""Interactive Ember data tutorial."""

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
        # Ember Data

        Data flows through typed scene-load configs and prepared-frame configs.
        The same config objects work in notebooks, scripts, presets, and
        checkpoint metadata.
        """
    )
    return mo, sk


@app.cell
def _(mo):
    width = mo.ui.slider(
        start=320,
        stop=1920,
        step=80,
        value=960,
        label="Target resize width",
    )
    split = mo.ui.dropdown(
        options=["train", "val", "all"],
        value="train",
        label="Frame split",
    )
    mo.vstack([width, split])
    return split, width


@app.cell
def _(sk, split, width):
    image_prep = sk.ImagePreparationConfig(
        resize_width_target=width.value,
        normalize=True,
    )
    prepared_config = sk.PreparedFrameDatasetConfig(
        image_preparation=image_prep,
        split=sk.SplitConfig(
            target=split.value,
            every_n=None if split.value == "all" else 8,
        ),
    )
    return image_prep, prepared_config


@app.cell(hide_code=True)
def _(mo, prepared_config):
    mo.md(
        f"""
        ---

        ## Prepared Frame Config

        ```python
        {prepared_config.model_dump()}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Loading A Real Scene

        ```python
        scene_record = sk.load_scene_record(
            sk.ColmapSceneConfig(
                path="data/garden",
                source_pipes=(sk.HorizonAlignPipeConfig(),),
            )
        )

        dataset = sk.prepare_frame_dataset(scene_record, prepared_config)
        ```

        `ColmapSceneConfig` is the default entrypoint for COLMAP-style scenes.
        Prepared-frame policy, resizing, splitting, normalization, and
        materialization stay in `PreparedFrameDatasetConfig`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Data Boundaries

        - Scene records describe the loaded scene and source sensors.
        - Prepared-frame datasets describe training/evaluation samples.
        - Source pipes transform records before frame materialization.
        - Image preparation stays explicit so experiments are reproducible.
        """
    )
    return


if __name__ == "__main__":
    app.run()
