"""Open one model file: ONNX (description embedded in metadata, or a sidecar YAML) or Ultralytics .pt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .description import METADATA_KEY, DescriptionError, ModelDescription, check_against_outputs, parse

MODEL_SUFFIXES = (".onnx", ".pt")


class NeedsDescription(Exception):
    """ONNX without embedded description and without a sidecar: the UI asks for one once."""

    def __init__(self, path: Path, inputs: dict, outputs: dict):
        super().__init__(f"{path.name} has no embedded model description")
        self.path, self.inputs, self.outputs = path, inputs, outputs


@dataclass
class ModelFile:
    path: Path
    kind: str  # onnx | ultralytics
    description: ModelDescription | None
    sha256: str
    description_source: str = "embedded"  # embedded | sidecar | model (ultralytics)
    io: dict = field(default_factory=dict)  # {"inputs": {name: shape}, "outputs": {name: shape}}

    @property
    def display_name(self) -> str:
        d = self.description
        return f"{d.name} ({d.version})" if d and d.version else (d.name if d else self.path.stem)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".description.yaml") if path.suffix != ".onnx" else \
        path.with_suffix(".description.yaml")


def onnx_io(path: Path) -> tuple[dict, dict, dict]:
    """(metadata, inputs, outputs) without loading weights into memory twice."""
    import onnx
    m = onnx.load(str(path), load_external_data=False)

    def shp(v):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or None) for d in v.type.tensor_type.shape.dim]
    inits = {t.name for t in m.graph.initializer}
    return ({p.key: p.value for p in m.metadata_props},
            {i.name: shp(i) for i in m.graph.input if i.name not in inits},
            {o.name: shp(o) for o in m.graph.output})


def open_model(path: str | Path, description_override: dict | None = None) -> ModelFile:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    if path.suffix.lower() == ".pt":
        return ModelFile(path=path, kind="ultralytics", description=None, sha256=sha256_file(path),
                         description_source="model")
    if path.suffix.lower() != ".onnx":
        raise DescriptionError(f"unsupported model type '{path.suffix}' (use .onnx or .pt)")
    meta, inputs, outputs = onnx_io(path)
    source = "embedded"
    if description_override is not None:
        d, source = description_override, "sidecar"
    elif METADATA_KEY in meta:
        try:
            d = json.loads(meta[METADATA_KEY])
        except json.JSONDecodeError as e:
            raise DescriptionError(f"embedded description is not valid JSON: {e}") from e
    elif sidecar_path(path).is_file():
        try:
            d = yaml.safe_load(sidecar_path(path).read_text())
        except yaml.YAMLError as e:
            raise DescriptionError(f"{sidecar_path(path).name}: {e}") from e
        source = "sidecar"
    else:
        raise NeedsDescription(path, inputs, outputs)
    desc = parse(d)
    if desc.input.name not in inputs:
        raise DescriptionError(f"input '{desc.input.name}' not in the model (has: {', '.join(inputs)})")
    errs = check_against_outputs(desc, outputs)
    if errs:
        raise DescriptionError("description does not match the model:\n  " + "\n  ".join(errs))
    return ModelFile(path=path, kind="onnx", description=desc, sha256=sha256_file(path), description_source=source,
                     io={"inputs": inputs, "outputs": outputs})


def save_sidecar(path: Path, d: dict) -> Path:
    """Validate and save a description beside a metadata-less ONNX (the only sidecar case)."""
    open_model(path, description_override=d)  # validates against the graph
    sp = sidecar_path(Path(path))
    sp.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    return sp


def list_models(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in MODEL_SUFFIXES)


def pick_variant(paths: list[Path], device: str) -> list[Path]:
    """When fp16 and fp32 twins of the same name+version exist, keep fp32 for CPU and fp16 otherwise."""
    groups: dict[tuple, list[tuple[str, Path]]] = {}
    other = []
    for p in paths:
        try:
            meta, _, _ = onnx_io(p) if p.suffix == ".onnx" else ({}, {}, {})
            d = json.loads(meta[METADATA_KEY])
            groups.setdefault((d.get("name"), d.get("version")), []).append((str(d.get("precision", "")), p))
        except Exception:
            other.append(p)
    out = []
    want = "fp32" if device == "cpu" else "fp16"
    for items in groups.values():
        match = [p for prec, p in items if prec == want]
        out.append(match[0] if match else items[0][1])
    return sorted(out + other)


def description_template(path: Path, inputs: dict, outputs: dict) -> str:
    """A starting YAML for a metadata-less ONNX, guessed from its inputs and outputs (the user edits it)."""
    name, shape = next(iter(inputs.items()))
    dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    nchw = len(dims) == 4 and dims[1] in (1, 3)
    h, w = (dims[2], dims[3]) if nchw else ((dims[1], dims[2]) if len(dims) == 4 else (None, None))
    lines = ["schema: gahan-model-description/1", f"name: {path.stem}", "version: ''", "input:",
             f"  name: {name}", f"  batch: {dims[0] or 1}", f"  layout: {'NCHW' if nchw else 'NHWC'}",
             f"  width: {w or 640}", f"  height: {h or 640}", "  resize: letterbox   # letterbox | stretch",
             "  pad_value: 114", "  color: RGB", "  scale: 0.00392156862745098", "  dtype: float32"]
    det = [n for n, s in outputs.items() if s and s[-1] == 6]
    det.sort(key=lambda n: (0 if ("raw" in n or "det" in n) else 1, n))
    if det:
        lines += ["detection:", f"  output: {det[0]}", "  format: xyxy_conf_cls", "  coords: input_px",
                  "  classes: [class_0, class_1]   # EDIT: class names in id order", "  conf_threshold: 0.25"]
    masks = [(n, s) for n, s in outputs.items() if len(s) >= 3 and s[-2:] == [h, w] and h]
    if masks:
        lines.append("masks:")
        for n, s in masks:
            c = s[-3] if len(s) >= 3 and isinstance(s[-3], int) else 1
            if c == 1:
                lines.append(f"  - {{name: {n.replace('_logits', '')}, output: {n}, type: binary, activation: sigmoid, threshold: 0.5}}")
            else:
                cls = ", ".join(f"c{i}" for i in range(c))
                lines.append(f"  - {{name: {n.replace('_logits', '')}, output: {n}, type: classes, "
                             f"class_names: [{cls}], ignore: [c0]}}   # EDIT class names")
    lines.append("# outputs in the graph (name: shape):")
    lines += [f"#   {n}: {s}" for n, s in outputs.items()]
    lines.append("# attribute heads, extra box sources and tags: see README 'Adding a model'")
    return "\n".join(lines) + "\n"
