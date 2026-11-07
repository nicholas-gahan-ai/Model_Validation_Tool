"""Package an exported ONNX + a description YAML into ONE model file (description embedded in metadata_props).

  python tools/package_model.py --onnx exported.onnx --description desc.yaml --out models/<name>.onnx

The graph and weights are not changed; only the metadata entry `gahan_model_description` is added/replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.description import METADATA_KEY, DescriptionError, check_against_outputs, parse  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_output_shapes(model: onnx.ModelProto) -> dict[str, list]:
    out = {}
    for o in model.graph.output:
        out[o.name] = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
                       for d in o.type.tensor_type.shape.dim]
    return out


def package(onnx_path: Path, desc_path: Path, out_path: Path) -> dict:
    d = yaml.safe_load(desc_path.read_text())
    desc = parse(d)  # raises DescriptionError
    model = onnx.load(str(onnx_path))
    inputs = {i.name for i in model.graph.input} - {t.name for t in model.graph.initializer}
    if desc.input.name not in inputs:
        raise DescriptionError(f"input '{desc.input.name}' not in model inputs {sorted(inputs)}")
    errs = check_against_outputs(desc, graph_output_shapes(model))
    if errs:
        raise DescriptionError("description does not match the graph:\n  " + "\n  ".join(errs))
    existing = {p.key: p.value for p in model.metadata_props}
    if METADATA_KEY in existing and json.loads(existing[METADATA_KEY]).get("sha256_graph"):
        # re-packaging an already packaged file: keep the original graph hash
        d["sha256_graph"] = json.loads(existing[METADATA_KEY])["sha256_graph"]
    else:
        d["sha256_graph"] = sha256_file(onnx_path)
    keep = [p for p in model.metadata_props if p.key != METADATA_KEY]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    model.metadata_props.add(key=METADATA_KEY, value=json.dumps(d, ensure_ascii=False, default=str))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--description", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        d = package(a.onnx, a.description, a.out)
    except DescriptionError as e:
        sys.exit(f"ERROR: {e}")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)  name={d['name']} version={d.get('version')} "
          f"sha256_graph={d['sha256_graph'][:16]}…")


if __name__ == "__main__":
    main()
