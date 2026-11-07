"""Test helper (GT reader check): Open ADAS per-image XML -> COCO JSON, YOLO txt (+ classes.txt), one CVAT XML 1.1.

  python tools/convert_gt.py --xml-root <annotations_xml> --out <dir>

Labels and attributes are copied as they are (raw), so every reader sees the same GT. Keys are the
<scene>/<file>.jpg paths relative to --xml-root.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.gt.cvat_xml import read_openadas_tree  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    gt = read_openadas_tree(a.xml_root)
    frames = sorted(gt.frames.values(), key=lambda f: f.key)
    labels = sorted({b.label for f in frames for b in f.boxes})
    a.out.mkdir(parents=True, exist_ok=True)

    # COCO (boxes + raw attributes)
    cats = [{"id": i, "name": n} for i, n in enumerate(labels)]
    cid = {n: i for i, n in enumerate(labels)}
    images, anns = [], []
    for i, f in enumerate(frames):
        images.append({"id": i, "file_name": f.key, "width": f.width, "height": f.height})
        for b in f.boxes:
            x1, y1, x2, y2 = b.xyxy
            anns.append({"id": len(anns), "image_id": i, "category_id": cid[b.label], "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "attributes": b.attributes})
    (a.out / "coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))

    # YOLO (boxes only; normalised, 10 decimals so the boxes round-trip exactly enough)
    yd = a.out / "yolo"
    yd.mkdir(exist_ok=True)
    (yd / "classes.txt").write_text("\n".join(labels) + "\n")
    for f in frames:
        p = yd / Path(f.key).with_suffix(".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for b in f.boxes:
            x1, y1, x2, y2 = b.xyxy
            lines.append(f"{cid[b.label]} {(x1 + x2) / 2 / f.width:.10f} {(y1 + y2) / 2 / f.height:.10f} "
                         f"{(x2 - x1) / f.width:.10f} {(y2 - y1) / f.height:.10f}")
        p.write_text("\n".join(lines) + ("\n" if lines else ""))

    # one CVAT XML 1.1 (CVAT for images), with polylines / polygons and attributes
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    for i, f in enumerate(frames):
        im = ET.SubElement(root, "image", id=str(i), name=f.key, width=str(f.width), height=str(f.height))
        for b in f.boxes:
            el = ET.SubElement(im, "box", label=b.label, occluded="0", xtl=repr(b.xyxy[0]), ytl=repr(b.xyxy[1]),
                               xbr=repr(b.xyxy[2]), ybr=repr(b.xyxy[3]))
            for k, v in b.attributes.items():
                ET.SubElement(el, "attribute", name=k).text = v
        for lab, pts in f.polylines:
            ET.SubElement(im, "polyline", label=lab, points=";".join(f"{x!r},{y!r}" for x, y in pts))
        for lab, pts in f.polygons:
            ET.SubElement(im, "polygon", label=lab, points=";".join(f"{x!r},{y!r}" for x, y in pts))
    ET.ElementTree(root).write(a.out / "cvat_for_images.xml", encoding="utf-8", xml_declaration=True)
    print(f"{len(frames)} frames, {len(anns)} boxes, labels {labels}")
    print("wrote", a.out / "coco.json", yd, a.out / "cvat_for_images.xml")


if __name__ == "__main__":
    main()
