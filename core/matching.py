"""Per-frame GT matching: per class, predictions sorted by confidence are greedily matched to the unmatched GT
box with the highest IoU >= 0.5 (the same rule flags() applies at IoU 0.5)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gt.normalize_attrs import NOT_COMPARABLE
from .metrics import iou

IOU_MATCH = 0.5


@dataclass
class Pair:
    pred: int
    gt: int
    iou: float


@dataclass
class FrameMatch:
    pairs: list[Pair] = field(default_factory=list)
    fp: list[int] = field(default_factory=list)      # prediction indices
    fn: list[int] = field(default_factory=list)      # GT indices (misses)
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)  # cls -> {tp, fp, fn}
    attr: dict[str, dict[str, int]] = field(default_factory=dict)       # head -> {n, agree, not_comparable}

    def pred_status(self) -> dict[int, str]:
        s = {p.pred: "tp" for p in self.pairs}
        s.update({i: "fp" for i in self.fp})
        return s

    def gt_of_pred(self) -> dict[int, Pair]:
        return {p.pred: p for p in self.pairs}


def match_frame(pred_boxes, gt_boxes, iou_thr: float = IOU_MATCH) -> FrameMatch:
    """pred_boxes: objects with .xyxy .cls .conf .attributes(head -> Attr); gt_boxes: ResolvedBox."""
    m = FrameMatch()
    classes = {b.cls for b in pred_boxes} | {g.cls for g in gt_boxes}
    for c in sorted(classes):
        pi = sorted((i for i, b in enumerate(pred_boxes) if b.cls == c), key=lambda i: -pred_boxes[i].conf)
        gi = [j for j, g in enumerate(gt_boxes) if g.cls == c]
        M = iou(np.array([pred_boxes[i].xyxy for i in pi], np.float64).reshape(-1, 4),
                np.array([gt_boxes[j].xyxy for j in gi], np.float64).reshape(-1, 4))
        used = set()
        tp = 0
        for r, i in enumerate(pi):
            best, bj = iou_thr, -1
            for k in range(len(gi)):
                if k not in used and M[r, k] >= best:
                    best, bj = M[r, k], k
            if bj >= 0:
                used.add(bj); tp += 1
                m.pairs.append(Pair(i, gi[bj], float(M[r, bj])))
            else:
                m.fp.append(i)
        miss = [gi[k] for k in range(len(gi)) if k not in used]
        m.fn += miss
        m.per_class[c] = {"tp": tp, "fp": len(pi) - tp, "fn": len(miss)}
    for p in m.pairs:
        pa, ga = pred_boxes[p.pred].attributes, gt_boxes[p.gt].attributes
        for head, gv in ga.items():
            if head not in pa:
                continue
            st = m.attr.setdefault(head, {"n": 0, "agree": 0, "not_comparable": 0})
            if gv is NOT_COMPARABLE:
                st["not_comparable"] += 1
                continue
            st["n"] += 1
            st["agree"] += int(pa[head].value == gv)
    return m
