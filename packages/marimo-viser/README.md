# marimo-viser

Viser-backed viewer helpers for marimo notebooks that expose Ember
`CameraState` objects to render functions.

The first viewer runtime follows the `nerfview` model: Viser owns browser
navigation and the Python side renders background images from each client
camera. Notebook authors should keep experiment controls in marimo, while the
Viser panel is reserved for built-in viewer controls and fallback interaction.
