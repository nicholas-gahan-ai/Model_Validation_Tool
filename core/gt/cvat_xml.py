"""CVAT XML 1.1 ("CVAT for images" and "CVAT for video") and the Open ADAS original layout
(the same XML, one file per image, in a folder tree)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from . import GtBox, GtError, GtFrame, GtSet


def parse_points(s: str) -> list[tuple[float, float]]:
    """'x,y;x,y' -> [(x, y), ...] (scripts/convert_open_adas_to_canonical.py parse_points)."""
    pts = []
    for p in (s or "").strip().split(";"):
        if not p.strip():
            continue
        x, y = p.split(",")
        pts.append((float(x), float(y)))
    return pts


def parse_attributes(el) -> dict:
    return {a.get("name"): (a.text or "").strip() for a in el.findall("attribute")}


def _parse(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as e:
        line, col = getattr(e, "position", (0, 0))
        raise GtError(f"{path.name}: XML error at line {line}, column {col}: {e.msg if hasattr(e, 'msg') else e}") from e
    except OSError as e:
        raise GtError(f"{path}: {e}") from e


def _shapes_into(frame: GtFrame, el, path: Path):
    for b in el.findall("box"):
        try:
            xyxy = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
        except (TypeError, ValueError) as e:
            raise GtError(f"{path.name}: <box> in image '{frame.key}' has bad coordinates") from e
        frame.boxes.append(GtBox(xyxy, b.get("label", ""), parse_attributes(b)))
    for p in el.findall("polyline"):
        frame.polylines.append((p.get("label", ""), parse_points(p.get("points", ""))))
    for p in el.findall("polygon"):
        frame.polygons.append((p.get("label", ""), parse_points(p.get("points", ""))))


def _image_frame(img, path: Path) -> GtFrame:
    name = img.get("name")
    if not name:
        raise GtError(f"{path.name}: <image> without a name")
    f = GtFrame(name, int(float(img.get("width", 0))) or None, int(float(img.get("height", 0))) or None)
    _shapes_into(f, img, path)
    return f


def read_cvat_xml(path: Path) -> GtSet:
    root = _parse(path)
    frames: dict[str, GtFrame] = {}
    for img in root.iter("image"):
        f = _image_frame(img, path)
        frames[f.key] = f
    by_index: dict[int, GtFrame] = {}
    tracks = root.findall("track")
    if tracks:  # CVAT for video
        size = root.find("meta/task/original_size") or root.find("meta/job/original_size")
        w = int(size.findtext("width", "0")) if size is not None else None
        h = int(size.findtext("height", "0")) if size is not None else None

        def fr(i):
            if i not in by_index:
                by_index[i] = GtFrame(f"frame_{i:06d}", w, h)
            return by_index[i]
        for t in tracks:
            label = t.get("label", "")
            for b in t.findall("box"):
                if b.get("outside") == "1":
                    continue
                i = int(b.get("frame"))
                fr(i).boxes.append(GtBox(tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")), label,
                                         parse_attributes(b)))
            for kind, tgt in (("polyline", "polylines"), ("polygon", "polygons")):
                for p in t.findall(kind):
                    if p.get("outside") == "1":
                        continue
                    getattr(fr(int(p.get("frame"))), tgt).append((label, parse_points(p.get("points", ""))))
    if not frames and not by_index:
        raise GtError(f"{path.name}: no <image> or <track> elements (is this CVAT XML 1.1?)")
    return GtSet(frames, "cvat", path, by_index)


def read_openadas_tree(folder: Path) -> GtSet:
    files = sorted(Path(folder).rglob("*.xml"))
    if not files:
        raise GtError(f"{folder}: no .xml files")
    frames: dict[str, GtFrame] = {}
    for p in files:
        root = _parse(p)
        imgs = list(root.iter("image"))
        if not imgs:
            raise GtError(f"{p.name}: no <image> element")
        for img in imgs:
            f = _image_frame(img, p)
            # key: the image's own file name ("<frame>.jpg"), plus its scene folder when present in the tree
            rel = p.relative_to(folder).parent
            name = Path(f.key).name
            f.key = str(rel / name) if str(rel) != "." else name
            frames[f.key] = f
    return GtSet(frames, "openadas", folder)
