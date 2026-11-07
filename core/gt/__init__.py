"""Ground-truth readers and the common GT model.

    gt = load_gt(path, fmt="auto")          -> GtSet (raw labels, raw attributes)
    frame = gt.find(name=..., index=...)    -> GtFrame or None
    r = frame.resolve(mapper, normalizer, (w, h))  -> ResolvedGt (model class names, normalised attributes)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


class GtError(ValueError):
    pass


class NeedsClassNames(GtError):
    """YOLO txt without classes.txt / data.yaml / obj.names: the UI asks for the names."""

    def __init__(self, folder: Path):
        super().__init__(f"no class names found for YOLO labels in {folder} (classes.txt, data.yaml or obj.names)")
        self.folder = folder


@dataclass
class GtBox:
    xyxy: tuple[float, float, float, float]
    label: str
    attributes: dict = field(default_factory=dict)


@dataclass
class GtFrame:
    key: str
    width: int | None = None
    height: int | None = None
    boxes: list[GtBox] = field(default_factory=list)
    polylines: list[tuple[str, list]] = field(default_factory=list)   # (label, [(x, y), ...])
    polygons: list[tuple[str, list]] = field(default_factory=list)
    normalized: bool = False   # YOLO: xyxy in 0..1, scaled by the frame size on resolve

    def resolve(self, mapper, normalizer, size: tuple[int, int]) -> "ResolvedGt":
        w, h = size
        sx, sy = (w, h) if self.normalized else (1.0, 1.0)
        out = ResolvedGt(self.key)
        for b in self.boxes:
            kind, tgt = mapper.target(b.label)
            if kind == "mask":
                continue                      # e.g. a COCO polygon's bbox: the polygon is used for the mask
            if kind is None:
                out.excluded[b.label] += 1
                continue
            xyxy = (b.xyxy[0] * sx, b.xyxy[1] * sy, b.xyxy[2] * sx, b.xyxy[3] * sy)
            out.boxes.append(ResolvedBox(xyxy, tgt, b.label, normalizer.normalize(tgt, b.attributes),
                                         dict(b.attributes)))
        for attr, store in (("polylines", out.polylines), ("polygons", out.polygons)):
            for lab, pts in getattr(self, attr):
                kind, tgt = mapper.target(lab)
                if kind == "mask":
                    store.append((tgt, [(x * sx, y * sy) for x, y in pts]))
                elif kind is None:
                    out.excluded_shapes[lab] += 1
        return out


@dataclass
class ResolvedBox:
    xyxy: tuple[float, float, float, float]
    cls: str                 # model class name
    label: str               # raw GT label
    attributes: dict         # head -> model value, or NOT_COMPARABLE
    raw_attributes: dict


@dataclass
class ResolvedGt:
    key: str
    boxes: list[ResolvedBox] = field(default_factory=list)
    polylines: list[tuple[str, list]] = field(default_factory=list)   # (mask target, points)
    polygons: list[tuple[str, list]] = field(default_factory=list)
    excluded: Counter = field(default_factory=Counter)        # raw box labels without a model class
    excluded_shapes: Counter = field(default_factory=Counter)


def _norm_key(s: str) -> str:
    return str(s).replace("\\", "/").strip().lower()


def _strip_ext(s: str) -> str:
    return re.sub(r"\.(jpe?g|png|bmp|tiff?|webp)$", "", s, flags=re.I)


class GtSet:
    def __init__(self, frames: dict[str, GtFrame], fmt: str, source: Path, by_index: dict[int, GtFrame] | None = None):
        self.frames, self.format, self.source = frames, fmt, Path(source)
        self.by_index = by_index or {}
        self._rel, self._noext, self._base, self._stem = {}, {}, {}, {}
        dup_base = Counter(Path(_norm_key(k)).name for k in frames)
        dup_stem = Counter(_strip_ext(Path(_norm_key(k)).name) for k in frames)
        for k, f in frames.items():
            nk = _norm_key(k)
            self._rel[nk] = f
            self._noext[_strip_ext(nk)] = f
            b = Path(nk).name
            if dup_base[b] == 1:
                self._base[b] = f
            st = _strip_ext(b)
            if dup_stem[st] == 1:
                self._stem[st] = f

    def __len__(self):
        return len(self.frames) or len(self.by_index)

    def labels(self) -> dict[str, Counter]:
        """Raw label counts per shape kind: {"box": Counter, "line": Counter, "polygon": Counter}."""
        c = {"box": Counter(), "line": Counter(), "polygon": Counter()}
        for f in list(self.frames.values()) + list(self.by_index.values()):
            c["box"].update(b.label for b in f.boxes)
            c["line"].update(lab for lab, _ in f.polylines)
            c["polygon"].update(lab for lab, _ in f.polygons)
        return c

    def find(self, name: str | None = None, index: int | None = None) -> GtFrame | None:
        if index is not None and self.by_index:
            return self.by_index.get(int(index))
        if not name:
            return None
        nk = _norm_key(name)
        for table, key in ((self._rel, nk), (self._noext, _strip_ext(nk)), (self._base, Path(nk).name),
                           (self._stem, _strip_ext(Path(nk).name))):
            f = table.get(key)
            if f is not None:
                return f
        # GT keys that are longer paths ending with the frame's relative name (e.g. CVAT export prefixes)
        for k, f in self._rel.items():
            if k.endswith("/" + nk):
                return f
        return None


FORMATS = {"auto": "Auto-detect", "cvat": "CVAT XML 1.1", "openadas": "Open ADAS XML (per image)",
           "coco": "COCO JSON", "yolo": "YOLO txt"}


def detect_format(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".xml":
            return "cvat"
        if path.suffix.lower() == ".json":
            import json
            try:
                with open(path, encoding="utf-8") as f:
                    head = json.load(f)
            except (OSError, ValueError) as e:
                raise GtError(f"{path.name}: not valid JSON ({e})") from e
            if isinstance(head, dict) and "images" in head and "annotations" in head:
                return "coco"
            raise GtError(f"{path.name}: JSON without 'images' and 'annotations' (not COCO)")
        raise GtError(f"{path.name}: unknown annotation file type")
    if path.is_dir():
        xml = next(path.rglob("*.xml"), None)
        txt = next((p for p in path.rglob("*.txt") if p.name not in ("classes.txt", "obj.names")), None)
        if xml is not None:
            return "openadas"
        if txt is not None:
            return "yolo"
        js = list(path.glob("*.json"))
        if len(js) == 1:
            return detect_format(js[0])
        raise GtError(f"{path}: no .xml, .txt or COCO .json annotations found")
    raise GtError(f"ground truth not found: {path}")


def load_gt(path: str | Path, fmt: str = "auto", class_names: list[str] | None = None) -> GtSet:
    path = Path(path).expanduser()
    if fmt in ("auto", "", None):
        fmt = detect_format(path)
    if fmt == "cvat":
        from .cvat_xml import read_cvat_xml
        if path.is_dir():
            files = sorted(path.glob("*.xml"))
            if len(files) != 1:
                raise GtError(f"{path}: pick the CVAT XML file (found {len(files)} .xml files)")
            path = files[0]
        return read_cvat_xml(path)
    if fmt == "openadas":
        from .cvat_xml import read_openadas_tree
        return read_openadas_tree(path)
    if fmt == "coco":
        from .coco import read_coco
        if path.is_dir():
            js = list(path.glob("*.json"))
            if len(js) != 1:
                raise GtError(f"{path}: pick the COCO JSON file")
            path = js[0]
        return read_coco(path)
    if fmt == "yolo":
        from .yolo import read_yolo
        return read_yolo(path, class_names)
    raise GtError(f"unknown GT format '{fmt}'")
