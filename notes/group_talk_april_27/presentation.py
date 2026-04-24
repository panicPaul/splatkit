import marimo

__generated_with = "0.23.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/presentation.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Group Talk April 27th

    ## Splatkit and 3D elipsoids that are truly 2D
    """)
    return


@app.cell(hide_code=True)
def _(mo, slider):
    mo.md(rf"""
    # Marimo
    ## or what is this weird looking power point


    - bullet point
    - another bullet point
    - graphic
    - we can embed gui elements like this slider: {slider}
    - They update automatically: slider = {slider.value}
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=1, step=0.01)
    return (slider,)


@app.cell
def _(mo, slider):
    left = mo.md(f"""
    # Marimo

    ## or what is this weird looking PowerPoint

    - bullet point
    - another bullet point
    - graphic
    - we can embed GUI elements like this slider: {slider}
    - They update automatically: slider = **{slider.value}**
    """)


    return (left,)


@app.cell
def _(mo, slider):

    right = mo.md(f"""
    - The slider value updates automatically
    {slider.value}
    """)
    return (right,)


@app.cell
def _(left, mo, right):
    mo.hstack(
        [left, right],
        widths=[1, 1],
        align="end",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo, slider):
    mo.md(rf"""
    # Slide 2

    | A | B |
    |---|---|
    | {slider}| 2 |
    """)
    return


if __name__ == "__main__":
    app.run()
