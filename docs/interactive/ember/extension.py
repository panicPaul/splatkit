"""Interactive Ember extension tutorial."""

# ruff: noqa: ANN001, ANN202, B018

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # Extending Ember

        New ideas should usually enter Ember additively: a new backend, trait,
        pass, notebook, config, or package. Promote code into `ember-core` only
        after the boundary is broadly reusable.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    extension_kind = mo.ui.dropdown(
        options=[
            "local backend",
            "adapter backend",
            "native backend stage",
            "densification method",
            "paper notebook",
        ],
        value="paper notebook",
        label="Extension kind",
    )
    extension_kind
    return (extension_kind,)


@app.cell(hide_code=True)
def _(extension_kind, mo):
    notes = {
        "paper notebook": (
            "Start notebook-first. Keep paper-faithful code local until the "
            "reusable pieces are obvious."
        ),
        "local backend": (
            "Register a local render function against `ember-core` contracts "
            "before creating a package."
        ),
        "adapter backend": (
            "Wrap external code behind `register_backend(...)` and keep "
            "upstream-specific options inside the adapter package."
        ),
        "native backend stage": (
            "Expose reusable stages such as preprocess, sort, and blend when "
            "they are useful independently."
        ),
        "densification method": (
            "Prefer a typed method/pass that composes with existing training "
            "runtime state."
        ),
    }
    mo.callout(notes[extension_kind.value], kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Backend Skeleton

        ```python
        import ember_core as sk


        class MyOptions(sk.RenderOptions):
            ...


        def render_my_backend(scene, camera, *, options=None, **kwargs):
            ...


        def register() -> None:
            sk.register_backend(
                name="my.backend",
                default_options=MyOptions(),
                accepted_scene_types=(sk.GaussianScene3D,),
                supported_outputs=frozenset({"alpha", "depth"}),
            )(render_my_backend)
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Promotion Checklist

        - Does the idea compose through an existing contract?
        - Is a new backend safer than mutating an existing backend?
        - Can this start as a notebook?
        - Is the reusable boundary a trait, config, pass, or backend stage?
        - Are defaults easy without hiding important choices?
        """
    )
    return


if __name__ == "__main__":
    app.run()
