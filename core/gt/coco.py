"""COCO JSON: images[file_name], annotations[bbox xywh, category_id, segmentation, attributes?], categories."""
from __future__ import annotations

import json
from pathlib import Path

from . import GtBox, GtError, GtFrame, GtSet


def read_coco(path: Path) -> GtSet:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GtError(f"{Path(path).name}: JSON error at line {e.lineno}, column {e.colno}: {e.msg}") from e
    try:
        cats = {c["id"]: c["name"] for c in d["categories"]}
        frames: dict[str, GtFrame] = {}
        by_id = {}
        for im in d["images"]:
            f = GtFrame(im["file_name"], im.get("width"), im.get("height"))
            frames[f.key] = f
            by_id[im["id"]] = f
        for a in d["annotations"]:
            f = by_id.get(a["image_id"])
            if f is None:
                continue
            label = cats.get(a["category_id"], str(a["category_id"]))
            if a.get("iscrowd"):
                continue  # crowd regions are not objects to match
            # boxes and polygons are both kept; the label mapping decides which one is used
            # (a model class -> the box, a mask -> the polygon)
            if a.get("bbox") is not None:
                x, y, w, h = a["bbox"]
                f.boxes.append(GtBox((float(x), float(y), float(x + w), float(y + h)), label,
                                     dict(a.get("attributes") or {})))
            seg = a.get("segmentation")
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        f.polygons.append((label, list(zip(poly[0::2], poly[1::2]))))
    except (KeyError, TypeError, ValueError) as e:
        raise GtError(f"{Path(path).name}: not a valid COCO file ({type(e).__name__}: {e})") from e
    return GtSet(frames, "coco", path)
