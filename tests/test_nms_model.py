import json

import numpy as np
import onnx

from core.description import METADATA_KEY
from core.model_file import open_model
from core.nms import nms


def naive_nms(b, s, t):
    order = sorted(range(len(s)), key=lambda i: -s[i]); keep = []
    def iou(a, c):
        w = max(0, min(a[2], c[2]) - max(a[0], c[0])); h = max(0, min(a[3], c[3]) - max(a[1], c[1]))
        i = w * h; return i / ((a[2]-a[0])*(a[3]-a[1]) + (c[2]-c[0])*(c[3]-c[1]) - i)
    for i in order:
        if all(iou(b[i], b[k]) <= t for k in keep):
            keep.append(i)
    return keep


def test_nms_matches_naive():
    rng = np.random.default_rng(0)
    for _ in range(50):
        xy = rng.uniform(0, 100, (40, 2)); wh = rng.uniform(5, 40, (40, 2))
        b = np.concatenate([xy, xy + wh], 1).astype(np.float32); s = rng.uniform(0, 1, 40).astype(np.float32)
        assert nms(b, s, 0.5).tolist() == naive_nms(b, s, 0.5)


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).tolist() == []


def test_packaged_model_roundtrip(model_path):
    m = open_model(model_path)
    assert m.kind == "onnx" and m.description_source == "embedded"
    meta = {p.key: p.value for p in onnx.load(str(model_path), load_external_data=False).metadata_props}
    d = json.loads(meta[METADATA_KEY])
    assert len(d["sha256_graph"]) == 64 and d["name"] == m.description.name
