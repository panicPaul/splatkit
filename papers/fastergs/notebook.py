"""FasterGS paper notebook with shared declarative training + script mode."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import ember_core as sk
    import marimo as mo

    _NOTEBOOK_DIR = Path(__file__).resolve().parent
    if str(_NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(_NOTEBOOK_DIR))

    from config import (
        FasterGSExperimentConfig,
        build_prepared_frame_dataset_config,
        build_scene_load_config,
        build_training_config,
        load_default_experiment_config,
        load_experiment_script_config,
        register_fastergs_backends,
    )
    from marimo_config_gui import (
        config_error,
        config_form,
        config_json,
        config_value,
        create_config_state,
    )

    (
        experiment_form_state,
        experiment_json_state,
        experiment_bindings,
    ) = create_config_state(
        FasterGSExperimentConfig,
        value=load_default_experiment_config(),
        script_loader=load_experiment_script_config,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # FasterGS

    Paper notebook for **Faster-GS: Analyzing and Improving Gaussian Splatting
    Optimization**.

    This artifact is intentionally dual-use:

    - interactive `marimo` notebook for editing a resolved experiment config
    - direct Python script for server-side runs via `cli` or `json`

    Paper:
    https://arxiv.org/abs/2602.09999
    """)
    return


@app.cell(hide_code=True)
def _():
    config_form(
        experiment_bindings,
        form_gui_state=experiment_form_state,
    )
    return


@app.cell(hide_code=True)
def _():
    config_json(
        experiment_bindings,
        form_gui_state=experiment_form_state,
        json_gui_state=experiment_json_state,
    )
    return


@app.cell(hide_code=True)
def _():
    run_button = mo.ui.button(
        value=0,
        label="Run FasterGS Training",
        on_click=lambda value: (0 if value is None else int(value)) + 1,
    )
    return (run_button,)


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    experiment_config = config_value(
        experiment_bindings,
        form_gui_state=experiment_form_state,
        json_gui_state=experiment_json_state,
    )
    return (experiment_config,)


@app.cell
def _(experiment_config):
    scene_config = (
        None
        if experiment_config is None
        else build_scene_load_config(experiment_config)
    )
    prepared_data_config = (
        None
        if experiment_config is None
        else build_prepared_frame_dataset_config(experiment_config)
    )
    training_config = (
        None
        if experiment_config is None
        else build_training_config(experiment_config)
    )
    return prepared_data_config, scene_config, training_config


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run

    The selected default JSON config is loaded first, then any CLI or JSON
    overrides win field-by-field. In interactive mode you edit the fully
    resolved config directly and launch training with the run button. In script
    mode the notebook runs automatically.
    """)
    return


@app.cell
def _(
    experiment_config,
    is_script_mode,
    prepared_data_config,
    run_button,
    scene_config,
    training_config,
):
    if experiment_config is None:
        available_backend = None
        frame_dataset = None
        registered_backend_modules = ()
        scene_record = None
        should_run = False
        training_result = None
    else:
        should_run = is_script_mode or (run_button.value or 0) > 0
        registered_backend_modules = (
            register_fastergs_backends() if should_run else ()
        )
        available_backend = (
            experiment_config.backend in sk.BACKEND_REGISTRY
            if should_run
            else None
        )
        scene_record = (
            sk.load_scene_record(scene_config)
            if should_run and available_backend
            else None
        )
        frame_dataset = (
            sk.prepare_frame_dataset(scene_record, prepared_data_config)
            if scene_record is not None
            else None
        )
        training_result = (
            sk.run_training(frame_dataset, training_config)
            if frame_dataset is not None
            else None
        )
    return (
        available_backend,
        frame_dataset,
        registered_backend_modules,
        scene_record,
        should_run,
        training_result,
    )


@app.cell(hide_code=True)
def _(run_button):
    run_button
    return


@app.cell(hide_code=True)
def _():
    config_error(
        experiment_bindings,
        form_gui_state=experiment_form_state,
        json_gui_state=experiment_json_state,
    )
    return


@app.cell(hide_code=True)
def _(
    available_backend,
    experiment_config,
    frame_dataset,
    registered_backend_modules,
    scene_record,
    should_run,
    training_result,
):
    if experiment_config is None:
        _run_status = mo.callout(
            "Choose a valid experiment config.",
            kind="warn",
        )
    elif not should_run:
        _run_status = mo.callout(
            "Press 'Run FasterGS Training' to start the experiment.",
            kind="info",
        )
    elif not registered_backend_modules:
        _run_status = mo.callout(
            "No FasterGS backend modules could be imported in this environment.",
            kind="warn",
        )
    elif not available_backend:
        _run_status = mo.callout(
            f"Backend `{experiment_config.backend}` is not registered.",
            kind="warn",
        )
    elif (
        scene_record is None or frame_dataset is None or training_result is None
    ):
        _run_status = mo.callout(
            "Training did not produce a result.",
            kind="warn",
        )
    else:
        _run_status = mo.vstack(
            [
                mo.callout(
                    (
                        f"Completed `{experiment_config.preset}` on "
                        f"`{experiment_config.backend}` with "
                        f"{len(frame_dataset)} training frames."
                    ),
                    kind="success",
                ),
                mo.md(
                    "```json\n"
                    + json.dumps(
                        {
                            "checkpoint_dir": training_result.checkpoint_dir,
                            "steps": len(training_result.history),
                            "last_metrics": training_result.history[-1],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n```"
                ),
            ],
            gap=0.75,
        )
    _run_status
    return


if __name__ == "__main__":
    app.run()
