# models/

Put model files here. Every `.onnx` / `.pt` file in this folder appears in the app's model list.

Model files are **not stored in git** (see `.gitignore`). Get them from the release zip
(`validation_tool_<version>.zip`) or the team's model share:

| File | What it is |
|---|---|
| `open_adas_v13s_single.onnx` | Open ADAS V13-S single model (detector + sign/light classifier + plate stage), with its description embedded |

To add a new model, package it with `tools/package_model.py` (see "Packaging a new model" in the main README).
