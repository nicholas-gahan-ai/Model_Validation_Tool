"""Folder-level detection metrics. flags()/ap() are ported verbatim from v15/scripts/scorecard.py
(via the bundle's ap_reference.py); only the class list became a parameter and ap() also returns
the per-IoU-threshold values so AP50 comes out of the same code.

Scoring rules (COCO-style, as the project scores):
  * AP uses every prediction (threshold 0); the UI threshold applies only to precision/recall
  * per image and class keep the top 100 by confidence, then sort the image's kept boxes by confidence
  * greedy match per IoU threshold 0.50:0.05:0.95, best-IoU unused GT with IoU >= threshold
  * 101-point interpolated precision; mean over the 10 thresholds; mean over the classes
  * a class with GT but no prediction at all is absent from flags() (not counted as 0)
"""
from __future__ import annotations

import numpy as np

THR = np.linspace(0.5, 0.95, 10)
REC = np.linspace(0, 1, 101)
TOP_K = 100


def iou(a, b):  # a [n,4], b [m,4] -> [n,m]
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def prepare_image(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_scores, n_classes: int, top_k: int = TOP_K) -> dict:
    """One image's arrays as flags() expects them: predictions top-k per class, sorted by score."""
    gb = np.asarray(gt_boxes, np.float64).reshape(-1, 4); gc = np.asarray(gt_cls, int).reshape(-1)
    pb = np.asarray(pred_boxes, np.float64).reshape(-1, 4); pc = np.asarray(pred_cls, int).reshape(-1)
    ps = np.asarray(pred_scores, np.float64).reshape(-1)
    o = np.argsort(-ps, kind="stable"); pb, pc, ps = pb[o], pc[o], ps[o]
    k = np.concatenate([np.where(pc == c)[0][:top_k] for c in range(n_classes)] + [np.zeros(0, int)]).astype(int)
    k = k[np.argsort(-ps[k], kind="stable")]
    return {"gb": gb, "gc": gc, "pb": pb[k], "pc": pc[k], "ps": ps[k]}


def flags(imgs, CL):
    """per class: (scores [n], image index [n], tp [T, n] bool, n_gt per image [N_img])"""
    out = {}
    for c in range(len(CL)):
        S, I, TP, ngt = [], [], [], np.zeros(len(imgs))
        for i, im in enumerate(imgs):
            gi = np.where(im["gc"] == c)[0]; pi = np.where(im["pc"] == c)[0]; ngt[i] = len(gi)
            if not len(pi):
                continue
            M = iou(im["pb"][pi], im["gb"][gi]); tp = np.zeros((len(THR), len(pi)), bool)
            for t, th in enumerate(THR):
                used = set()
                for k in range(len(pi)):  # pi is already score-sorted
                    best, bj = th, -1
                    for gj in range(len(gi)):
                        if gj not in used and M[k, gj] >= best:
                            best, bj = M[k, gj], gj
                    if bj >= 0:
                        used.add(bj); tp[t, k] = True
            S.append(im["ps"][pi]); I.append(np.full(len(pi), i)); TP.append(tp)
        if S:
            S, I, TP = np.concatenate(S), np.concatenate(I), np.concatenate(TP, 1)
            o = np.argsort(-S, kind="stable"); out[CL[c]] = (S[o], I[o], TP[:, o], ngt)
    return out


def ap(fl, w, per_threshold: bool = False):
    """per-class AP50-95 with image weights w (all ones = the plain AP).
    per_threshold=True returns {class: array of the 10 per-IoU APs} instead of their mean."""
    res = {}
    for c, (S, I, TP, ngt) in fl.items():
        n = float((ngt * w).sum())
        if n == 0:
            res[c] = np.full(len(THR), np.nan) if per_threshold else np.nan; continue
        ww = w[I]; aps = []
        for t in range(len(THR)):
            ctp = np.cumsum(TP[t] * ww); cfp = np.cumsum(~TP[t] * ww)
            rc, pr = ctp / n, ctp / np.maximum(ctp + cfp, 1e-12)
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, REC, side="left")
            aps.append(np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0).mean())
        res[c] = np.asarray(aps) if per_threshold else float(np.mean(aps))
    return res


# ------------------------------------------------------------------------------------------------ folder metrics
class GtEvaluator:
    """Accumulates GT vs prediction over a run and computes the folder metrics on demand.

    Stores, per image, the GT boxes and every prediction (decoded at threshold 0; extra box sources at
    >= 0.01), so AP uses all predictions and precision / recall can be recomputed at any threshold.
    Attribute agreement and mask IoU are accumulated as frames arrive.
    """

    AP_EXTRA_FLOOR = 0.01   # extra box sources are scored down to this cutoff

    def __init__(self, desc, mapper, gt_source: str = "", gt_format: str = ""):
        self.desc = desc
        self.mapper = mapper
        self.classes = list(desc.class_names)
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        # which threshold applies to each class's predictions: "conf", or an extra box source output
        self.thr_key = {c: "conf" for c in (desc.detection.classes if desc.detection else [])}
        for e in desc.extra_boxes:
            self.thr_key[e.cls] = e.output
        self.images: list[dict] = []
        self.keys: list[str] = []
        self.attr = {}
        self.mask_acc = {}       # target -> [inter, union, frames]
        self.excluded = {}
        self.excluded_shapes = {}
        self.frames_without_gt = 0
        self.gt_source, self.gt_format = gt_source, gt_format

    def ap_thresholds(self) -> dict:
        t = {"conf": 0.0}
        for e in self.desc.extra_boxes:
            t[e.output] = self.AP_EXTRA_FLOOR
        return t

    def add(self, key: str, gt, all_boxes, match=None, masks: dict | None = None, size=None):
        """gt: ResolvedGt; all_boxes: boxes decoded at ap_thresholds(); match: FrameMatch at the UI threshold."""
        gb = [g.xyxy for g in gt.boxes if g.cls in self.cidx]
        gc = [self.cidx[g.cls] for g in gt.boxes if g.cls in self.cidx]
        keep = [b for b in all_boxes if b.cls in self.cidx]
        self.images.append({
            "gb": np.asarray(gb, np.float64).reshape(-1, 4), "gc": np.asarray(gc, int),
            "pb": np.asarray([b.xyxy for b in keep], np.float64).reshape(-1, 4),
            "pc": np.asarray([self.cidx[b.cls] for b in keep], int),
            "ps": np.asarray([b.conf for b in keep], np.float64),
            "pk": [self.thr_key.get(b.cls, "conf") for b in keep]})
        self.keys.append(key)
        for k, v in gt.excluded.items():
            self.excluded[k] = self.excluded.get(k, 0) + v
        for k, v in gt.excluded_shapes.items():
            self.excluded_shapes[k] = self.excluded_shapes.get(k, 0) + v
        if match is not None:
            for head, st in match.attr.items():
                a = self.attr.setdefault(head, {"n": 0, "agree": 0, "not_comparable": 0})
                for k in a:
                    a[k] += st[k]
        if masks and size:
            self._add_masks(gt, masks, size)

    def _add_masks(self, gt, masks, size):
        import cv2
        w, h = size
        by_target: dict[str, dict[str, list]] = {}
        for tgt, pts in gt.polylines:
            by_target.setdefault(tgt, {"lines": [], "polys": []})["lines"].append(pts)
        for tgt, pts in gt.polygons:
            by_target.setdefault(tgt, {"lines": [], "polys": []})["polys"].append(pts)
        for tgt, shapes in by_target.items():
            name, _, cls = tgt.partition("/")
            m = masks.get(name)
            if m is None:
                continue
            g = np.zeros((h, w), np.uint8)
            if shapes["lines"]:
                cv2.polylines(g, [np.round(np.asarray(p, np.float64)).astype(np.int32).reshape(-1, 1, 2)
                                  for p in shapes["lines"] if len(p) >= 2], False, 1, thickness=POLYLINE_WIDTH)
            if shapes["polys"]:
                cv2.fillPoly(g, [np.round(np.asarray(p, np.float64)).astype(np.int32).reshape(-1, 1, 2)
                                 for p in shapes["polys"] if len(p) >= 3], 1)
            if m.type == "binary":
                pm = m.data
            elif cls:
                pm = m.data == (m.class_names.index(cls) if cls in m.class_names else -1)
            else:
                ign = [m.class_names.index(c) for c in self._ignore(name)]
                pm = ~np.isin(m.data, ign)
            if pm.shape != g.shape:
                continue
            gb = g.astype(bool)
            acc = self.mask_acc.setdefault(tgt, [0, 0, 0])
            acc[0] += int(np.logical_and(gb, pm).sum()); acc[1] += int(np.logical_or(gb, pm).sum()); acc[2] += 1

    def _ignore(self, mask_name):
        for m in self.desc.masks:
            if m.name == mask_name:
                return [c for c in m.ignore if c in (m.class_names or [])]
        return []

    def compute(self, thresholds: dict) -> dict:
        n = len(self.images)
        CL = self.classes
        fl = flags(self.images, CL)
        aps = ap(fl, np.ones(n), per_threshold=True) if n else {}
        # precision / recall at the UI thresholds, IoU 0.5
        thr_imgs = []
        for im in self.images:
            keep = np.array([s >= thresholds.get(k, 0.0) for s, k in zip(im["ps"], im["pk"])], bool).reshape(-1)
            thr_imgs.append({"gb": im["gb"], "gc": im["gc"], "pb": im["pb"][keep], "pc": im["pc"][keep],
                             "ps": im["ps"][keep]})
        fl_t = flags(thr_imgs, CL)
        per = {}
        for ci, c in enumerate(CL):
            ngt = int(sum((im["gc"] == ci).sum() for im in self.images))
            npred = int(sum((im["pc"] == ci).sum() for im in thr_imgs))
            if ngt == 0 and npred == 0:
                continue
            tp = int(fl_t[c][2][0].sum()) if c in fl_t else 0
            a = aps.get(c)
            per[c] = {"gt": ngt, "pred": npred, "tp": tp, "fp": npred - tp, "fn": ngt - tp,
                      "precision": tp / npred if npred else float("nan"),
                      "recall": tp / ngt if ngt else float("nan"),
                      "ap50": float(a[0]) if a is not None and ngt else float("nan"),
                      "ap50_95": float(np.mean(a)) if a is not None and ngt else float("nan"),
                      "no_predictions": c not in fl and ngt > 0}
        scored = [c for c, r in per.items() if r["gt"] > 0 and not r["no_predictions"]]
        mean = {"n_classes": len(scored),
                "ap50": float(np.mean([per[c]["ap50"] for c in scored])) if scored else float("nan"),
                "ap50_95": float(np.mean([per[c]["ap50_95"] for c in scored])) if scored else float("nan"),
                "precision": float(np.nanmean([per[c]["precision"] for c in scored])) if scored else float("nan"),
                "recall": float(np.nanmean([per[c]["recall"] for c in scored])) if scored else float("nan")}
        masks = {t: {"iou": a[0] / a[1] if a[1] else float("nan"), "frames": a[2],
                     "note": "GT polylines rasterised at %d px" % POLYLINE_WIDTH if "/" not in t else "GT polygons"}
                 for t, a in self.mask_acc.items()}
        return {"gt_source": self.gt_source, "gt_format": self.gt_format, "images_matched": n,
                "frames_without_gt": self.frames_without_gt, "thresholds": dict(thresholds),
                "thresholds_text": self.desc.format_thresholds(thresholds),
                "per_class": per, "mean": mean, "attributes": dict(self.attr),
                "attributes_na": "" if self.attr else "the GT has no comparable attributes",
                "masks": masks, "excluded": dict(sorted(self.excluded.items())),
                "excluded_shapes": dict(sorted(self.excluded_shapes.items())),
                "no_prediction_classes": [c for c, r in per.items() if r["no_predictions"]]}


POLYLINE_WIDTH = 3   # GT polylines are rasterised at 3 px (original resolution)


def _clean(x):
    """NaN -> None (valid JSON), numpy scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if x != x else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def metrics_to_files(m: dict, out_dir, provenance: dict) -> list:
    import csv
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    pj = out_dir / "metrics.json"
    pj.write_text(json.dumps(_clean({"header": provenance, **m}), indent=2), encoding="utf-8")
    pc = out_dir / "metrics.csv"
    with open(pc, "w", newline="", encoding="utf-8") as f:
        f.write(f"# model={provenance.get('model_name')} version={provenance.get('model_version')} "
                f"sha256={provenance.get('model_sha256')} app={provenance.get('app_version')}\n")
        w = csv.writer(f)
        w.writerow(["class", "gt", "pred_at_thr", "tp", "fp", "fn", "precision", "recall", "ap50", "ap50_95"])
        for c, r in m["per_class"].items():
            w.writerow([c, r["gt"], r["pred"], r["tp"], r["fp"], r["fn"]] +
                       [("" if v != v else round(v, 6)) for v in (r["precision"], r["recall"], r["ap50"], r["ap50_95"])])
        mm = m["mean"]
        w.writerow([f"mean ({mm['n_classes']} classes)", "", "", "", "", ""] +
                   [("" if v != v else round(v, 6)) for v in (mm["precision"], mm["recall"], mm["ap50"], mm["ap50_95"])])
    return [pj, pc]
