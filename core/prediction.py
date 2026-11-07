"""The app's internal and export format for one frame's decoded model output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Attr:
    value: str
    prob: float
    top3: list[tuple[str, float]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"value": self.value, "prob": round(self.prob, 4), "top3": [[v, round(p, 4)] for v, p in self.top3]}


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]  # original image px
    cls: str
    conf: float
    slot: int  # detection slot index; -1 for boxes from an extra box source
    attributes: dict[str, Attr] = field(default_factory=dict)
    source: str = "detection"  # detection | <extra_boxes output name>

    def to_json(self) -> dict:
        return {"cls": self.cls, "conf": round(self.conf, 4), "xyxy": [round(v, 2) for v in self.xyxy],
                "slot": self.slot, "source": self.source,
                "attributes": {k: a.to_json() for k, a in self.attributes.items()}}


@dataclass
class Mask:
    name: str
    type: Literal["binary", "classes"]
    data: np.ndarray  # HxW bool (binary) or HxW uint8 class map (classes), original px
    class_names: list[str] | None = None

    def areas(self) -> dict[str, int]:
        if self.type == "binary":
            return {self.name: int(self.data.sum())}
        counts = np.bincount(self.data.ravel(), minlength=len(self.class_names or []))
        return {n: int(counts[i]) for i, n in enumerate(self.class_names or [])}


def rle_encode(mask: np.ndarray) -> dict:
    """Column-major uncompressed RLE (COCO-style counts, starting with zeros)."""
    flat = np.asarray(mask, np.uint8).ravel(order="F")
    change = np.flatnonzero(np.diff(np.concatenate([[0], flat, [1 - flat[-1] if flat.size else 0]])))
    counts = np.diff(np.concatenate([[0], change])).tolist() if flat.size else []
    return {"size": list(mask.shape), "counts": counts}


@dataclass
class Prediction:
    frame_index: int
    timestamp_s: float | None
    image_size: tuple[int, int]  # (w, h)
    boxes: list[Box] = field(default_factory=list)
    masks: dict[str, Mask] = field(default_factory=dict)
    tags: dict[str, Attr] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)
    name: str | None = None  # file name / frame label

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for b in self.boxes:
            c[b.cls] = c.get(b.cls, 0) + 1
        return c

    def to_json(self, include_masks: bool = False) -> dict:
        masks = {}
        for k, m in self.masks.items():
            entry = {"type": m.type, "area_px": m.areas()}
            if include_masks:
                if m.type == "binary":
                    entry["rle"] = rle_encode(m.data)
                else:
                    entry["rle"] = {n: rle_encode(m.data == i) for i, n in enumerate(m.class_names or [])}
            masks[k] = entry
        return {"frame": self.frame_index, "timestamp_s": None if self.timestamp_s is None else round(self.timestamp_s, 4),
                "name": self.name, "image_size": list(self.image_size), "boxes": [b.to_json() for b in self.boxes],
                "masks": masks, "tags": {k: a.to_json() for k, a in self.tags.items()},
                "timing_ms": {k: round(v, 2) for k, v in self.timing_ms.items()}}
