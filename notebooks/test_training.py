"""Minimal notebook-first training script using splatkit.training."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import splatkit as sk
    import torch
    from marimo_3dv import form_gui
    from pydantic import BaseModel, Field


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Config
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Recipe
    """)
    return


@app.class_definition
class NotebookTrainingConfig(BaseModel):
    """User-editable config for the notebook training runtime demo."""

    checkpoint_path: Path = Field(
        Path("tmp/test_training_runtime.pt"),
        description="Checkpoint path used for save and load actions.",
    )
    run_steps: int = Field(
        25,
        ge=1,
        le=1000,
        description="Number of optimization steps for the full run action.",
    )
    step_chunk: int = Field(
        5,
        ge=1,
        le=100,
        description="Number of optimization steps executed by the step action.",
    )
    batch_value: float = Field(
        2.0,
        gt=0.0,
        description="Single synthetic batch value fed into the minimal recipe.",
    )
    target_value: float = Field(
        1.0,
        description="Target scalar value the linear model tries to match.",
    )
    initial_weight: float = Field(
        0.0,
        description="Initial parameter value for the synthetic scene weight.",
    )
    learning_rate: float = Field(
        1e-1,
        gt=0.0,
        description="Learning rate for the SGD optimizer.",
    )
    weight_decay: float = Field(
        0.0,
        ge=0.0,
        description="Weight decay applied by the SGD optimizer.",
    )
    seed: int = Field(
        0,
        description="Random seed forwarded into the splatkit runtime config.",
    )


@app.cell
def _():
    config_form = form_gui(
        NotebookTrainingConfig,
        value=NotebookTrainingConfig(),
        label="Training Config",
        live_update=False,
    )
    return (config_form,)


@app.cell
def _(config_form):
    config_form
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Status
    """)
    return


@app.class_definition
class LinearScene(torch.nn.Module):
    """Single-parameter scene module used by the minimal training example."""

    def __init__(self, initial_weight: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([initial_weight], dtype=torch.float32)
        )


@app.class_definition
class LinearSceneAdapter:
    """Adapter that serializes and restores the minimal scene state."""

    def state_dict(self, scene: LinearScene) -> Mapping[str, Any]:
        """Return a checkpointable state dictionary for the scene."""
        return scene.state_dict()

    def load_state_dict(
        self,
        scene: LinearScene,
        state_dict: Mapping[str, Any],
    ) -> None:
        """Load a checkpoint state dictionary back into the scene."""
        scene.load_state_dict(dict(state_dict))


@app.class_definition
class LinearRecipe:
    """Minimal training recipe used to exercise the notebook runtime."""

    scene_adapter = LinearSceneAdapter()

    def __init__(self, config: NotebookTrainingConfig) -> None:
        self.config = config

    def create_scene(
        self,
        experiment_config: sk.ExperimentConfig,
    ) -> LinearScene:
        """Create the trainable scene for the runtime."""
        del experiment_config
        return LinearScene(self.config.initial_weight)

    def create_optimizer(
        self,
        scene: LinearScene,
        experiment_config: sk.ExperimentConfig,
    ) -> torch.optim.Optimizer:
        """Build the optimizer for the synthetic scene parameter."""
        return torch.optim.SGD(
            [scene.weight],
            lr=experiment_config.optimizer.lr,
            weight_decay=experiment_config.optimizer.weight_decay,
        )

    def create_batches(
        self,
        experiment_config: sk.ExperimentConfig,
    ) -> list[torch.Tensor]:
        """Create the repeated synthetic training batches."""
        del experiment_config
        return [torch.tensor([self.config.batch_value], dtype=torch.float32)]

    def compute_train_loss(
        self,
        scene: LinearScene,
        batch: torch.Tensor,
        step: int,
        experiment_config: sk.ExperimentConfig,
    ) -> sk.TrainStepResult:
        """Compute the scalar training loss for one synthetic batch."""
        del step, experiment_config
        prediction = scene.weight * batch
        loss = ((prediction - self.config.target_value) ** 2).mean()
        return sk.TrainStepResult(
            loss=loss,
            metrics={
                "weight": float(scene.weight.detach().item()),
                "prediction": float(prediction.detach().item()),
            },
        )

    def evaluate(
        self,
        scene: LinearScene,
        step: int,
        experiment_config: sk.ExperimentConfig,
    ) -> dict[str, float]:
        """Return simple scalar evaluation metrics for the current scene."""
        del step, experiment_config
        prediction = scene.weight.detach().item() * self.config.batch_value
        error = prediction - self.config.target_value
        return {
            "weight": float(scene.weight.detach().item()),
            "prediction": float(prediction),
            "abs_error": float(abs(error)),
        }


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Runtime
    """)
    return


@app.function
def build_experiment_config(
    config: NotebookTrainingConfig,
) -> sk.ExperimentConfig:
    """Build the runtime config from the notebook form values."""
    experiment_config = sk.ExperimentConfig()
    experiment_config.optimizer.lr = config.learning_rate
    experiment_config.optimizer.weight_decay = config.weight_decay
    experiment_config.runtime.seed = config.seed
    return experiment_config


@app.function
def create_runtime(config: NotebookTrainingConfig) -> sk.TrainerRuntime:
    """Create a fresh trainer runtime for the current notebook config."""
    recipe = LinearRecipe(config)
    experiment_config = build_experiment_config(config)
    return sk.TrainerRuntime.create(recipe, experiment_config)


@app.function
def load_runtime_from_checkpoint(
    config: NotebookTrainingConfig,
) -> sk.TrainerRuntime:
    """Load a trainer runtime from the configured checkpoint path."""
    recipe = LinearRecipe(config)
    return sk.TrainerRuntime.from_checkpoint(
        config.checkpoint_path,
        recipe,
    )


@app.cell
def _():
    runtime_state, set_runtime_state = mo.state({})
    return runtime_state, set_runtime_state


@app.cell
def _(config_form, runtime_state):
    config: NotebookTrainingConfig = config_form.value
    runtime = runtime_state().get("runtime")
    return config, runtime


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Controls
    """)
    return


@app.cell
def _():
    create_button = (
        mo.ui.run_button(label="Create Runtime")
        if mo.running_in_notebook()
        else None
    )
    return (create_button,)


@app.cell
def _():
    step_button = (
        mo.ui.run_button(label="Step Chunk")
        if mo.running_in_notebook()
        else None
    )
    return (step_button,)


@app.cell
def _():
    run_button = (
        mo.ui.run_button(label="Run Full") if mo.running_in_notebook() else None
    )
    return (run_button,)


@app.cell
def _():
    eval_button = (
        mo.ui.run_button(label="Evaluate") if mo.running_in_notebook() else None
    )
    return (eval_button,)


@app.cell
def _():
    save_button = (
        mo.ui.run_button(label="Save Checkpoint")
        if mo.running_in_notebook()
        else None
    )
    return (save_button,)


@app.cell
def _():
    load_button = (
        mo.ui.run_button(label="Load Checkpoint")
        if mo.running_in_notebook()
        else None
    )
    return (load_button,)


@app.cell
def _(config: NotebookTrainingConfig, runtime, save_button, set_runtime_state):
    if save_button is not None and save_button.value and runtime is not None:
        runtime.save_checkpoint(config.checkpoint_path)
        set_runtime_state({"runtime": runtime})
    return


@app.cell
def _(config: NotebookTrainingConfig, set_runtime_state):
    if not mo.running_in_notebook():
        script_runtime = create_runtime(config)
        script_runtime.run_steps(config.run_steps)
        set_runtime_state({"runtime": script_runtime})
    return


@app.cell
def _(config: NotebookTrainingConfig, runtime, set_runtime_state, step_button):
    if step_button is not None and step_button.value and runtime is not None:
        runtime.run_steps(config.step_chunk)
        set_runtime_state({"runtime": runtime})
    return


@app.cell
def _(config: NotebookTrainingConfig, run_button, runtime, set_runtime_state):
    if run_button is not None and run_button.value and runtime is not None:
        runtime.run_steps(config.run_steps)
        set_runtime_state({"runtime": runtime})
    return


@app.cell
def _(eval_button, runtime, set_runtime_state):
    if eval_button is not None and eval_button.value and runtime is not None:
        runtime.evaluate()
        set_runtime_state({"runtime": runtime})
    return


@app.cell
def _(config: NotebookTrainingConfig, create_button, set_runtime_state):
    if create_button is not None and create_button.value:
        new_runtime = create_runtime(config)
        set_runtime_state({"runtime": new_runtime})
    return


@app.cell
def _(config: NotebookTrainingConfig, load_button, set_runtime_state):
    if load_button is not None and load_button.value:
        loaded_runtime = load_runtime_from_checkpoint(config)
        set_runtime_state({"runtime": loaded_runtime})
    return


@app.cell
def _(
    create_button,
    eval_button,
    load_button,
    run_button,
    save_button,
    step_button,
):
    controls = (
        mo.md("")
        if not mo.running_in_notebook()
        else mo.hstack(
            [
                create_button,
                step_button,
                run_button,
                eval_button,
                save_button,
                load_button,
            ],
            justify="start",
            gap=0.5,
        )
    )
    controls
    return


@app.cell(hide_code=True)
def _(config: NotebookTrainingConfig, runtime):
    if runtime is None:
        status = mo.callout(
            "Create a runtime to begin stepping the minimal recipe.",
            kind="warn",
        )
    else:
        latest_metrics = runtime.history[-1] if runtime.history else {}
        latest_eval = runtime.eval_history[-1] if runtime.eval_history else {}
        status = mo.vstack(
            [
                mo.callout(
                    (
                        f"Global step: `{runtime.global_step}`  \n"
                        f"Weight: `{float(runtime.scene.weight.detach().item()):.6f}`"
                    ),
                    kind="success",
                ),
                mo.md(f"Latest train metrics: `{latest_metrics}`"),
                mo.md(f"Latest eval metrics: `{latest_eval}`"),
                mo.md(f"Checkpoint path: `{config.checkpoint_path}`"),
            ]
        )
    status
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Test Training

    Minimal notebook/script training example built on top of
    `splatkit.training`. It uses a tiny synthetic recipe so the notebook
    stays useful while the scene-specific training stack is still evolving.
    """)
    return


if __name__ == "__main__":
    app.run()
