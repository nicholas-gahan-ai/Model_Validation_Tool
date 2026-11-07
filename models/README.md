# models/

Put packaged model files (`.onnx`, or Ultralytics `.pt`) here. Every model file in this folder appears in the app's
model list; a model anywhere else can be opened with **Open…**.

Model files are **not stored in git** (see `.gitignore`); they are distributed in the release zip.

To add a new model, package it with `tools/package_model.py` (see "Packaging a new model" in the main README).
