# /// script
# dependencies = [
#     "marimo",
#     "torch @ https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl",
#     "splatkit[cu130] @ file:///home/schlack/Repositories/splatkit/packages/splatkit",
#     "splatkit-native-3dgrt[cu130] @ file:///home/schlack/Repositories/splatkit/packages/splatkit-native-3dgrt",
#     "splatkit-native-faster-gs[cu130] @ file:///home/schlack/Repositories/splatkit/packages/splatkit-native-faster-gs",
#     "splatkit-native-faster-gs-mojo[cu130] @ file:///home/schlack/Repositories/splatkit/packages/splatkit-native-faster-gs-mojo",
#     "splatkit-native-svraster[cu130] @ file:///home/schlack/Repositories/splatkit/packages/splatkit-native-svraster",
# ]
# requires-python = ">=3.14"
#
# [tool.uv]
# prerelease = "allow"
#
# [tool.uv.sources]
# max = { index = "modular-nightly" }
# mojo = { index = "modular-nightly" }
#
# [[tool.uv.index]]
# name = "modular-nightly"
# url = "https://whl.modular.com/nightly/simple/"
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import importlib

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(
        """
        # Local Packaging Sandbox

        This notebook installs `splatkit` native packages from local checkout
        paths via PEP 723 metadata and verifies that the packages import.
        """
    )
    return


@app.cell
def _():
    module_names = [
        "splatkit",
        "splatkit_native_faster_gs",
        "splatkit_native_faster_gs.faster_gs",
        "splatkit_native_faster_gs.faster_gs_depth",
        "splatkit_native_faster_gs.gaussian_pop",
        "splatkit_native_faster_gs_mojo",
        "splatkit_native_faster_gs_mojo.core",
        "splatkit_native_3dgrt",
        "splatkit_native_3dgrt.stoch3dgs",
        "splatkit_native_svraster",
        "splatkit_native_svraster.core",
    ]
    return (module_names,)


@app.cell
def _(module_names):
    imported_modules = {
        module_name: importlib.import_module(module_name)
        for module_name in module_names
    }
    return (imported_modules,)


@app.cell
def _(imported_modules):
    versions = {
        module_name: getattr(module, "__version__", None)
        for module_name, module in imported_modules.items()
    }
    versions
    return


if __name__ == "__main__":
    app.run()
