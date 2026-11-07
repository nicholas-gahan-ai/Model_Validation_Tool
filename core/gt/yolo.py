"""YOLO txt: one txt per image ('cls cx cy w h', normalised). Names from classes.txt, data.yaml or obj.names."""
from __future__ import annotations

from pathlib import Path

import yaml

from . import GtBox, GtError, GtFrame, GtSet, NeedsClassNames


def find_class_names(folder: Path) -> list[str] | None:
    for d in (folder, folder.parent, folder.parent.parent):
        for n in ("classes.txt", "obj.names"):
            p = d / n
            if p.is_file():
                return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for n in ("data.yaml", "dataset.yaml"):
            p = d / n
            if p.is_file():
                names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names)]
                if isinstance(names, list):
                    return [str(x) for x in names]
    return None


def read_names_file(p: Path) -> list[str]:
    p = Path(p)
    if p.suffix.lower() in (".yaml", ".yml"):
        names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names")
        return [names[k] for k in sorted(names)] if isinstance(names, dict) else [str(x) for x in names or []]
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_yolo(folder: Path, class_names: list[str] | None = None) -> GtSet:
    folder = Path(folder)
    names = class_names or find_class_names(folder)
    if not names:
        raise NeedsClassNames(folder)
    frames: dict[str, GtFrame] = {}
    for p in sorted(folder.rglob("*.txt")):
        if p.name in ("classes.txt", "obj.names"):
            continue
        key = str(p.relative_to(folder).with_suffix(""))
        f = GtFrame(key, normalized=True)
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise GtError(f"{p.name}: line {ln}: expected 'cls cx cy w h'")
            try:
                c = int(float(parts[0])); cx, cy, w, h = map(float, parts[1:5])
            except ValueError as e:
                raise GtError(f"{p.name}: line {ln}: {e}") from e
            if not 0 <= c < len(names):
                raise GtError(f"{p.name}: line {ln}: class id {c} but only {len(names)} names")
            f.boxes.append(GtBox((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), names[c], {}))
        frames[key] = f
    if not frames:
        raise GtError(f"{folder}: no YOLO .txt label files")
    return GtSet(frames, "yolo", folder)
