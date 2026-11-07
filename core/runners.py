"""Pick the runner for a model file."""
from __future__ import annotations

from .model_file import ModelFile, open_model


def make_runner(model: ModelFile | str, device: str = "auto"):
    if not isinstance(model, ModelFile):
        model = open_model(model)
    if model.kind == "ultralytics":
        from .runner_ultralytics import UltralyticsRunner  # optional dependency, imported lazily
        return UltralyticsRunner(model, device=device)
    from .runner_onnx import OnnxRunner
    return OnnxRunner(model, device=device)
