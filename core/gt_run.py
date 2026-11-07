"""Ground truth for one run: load, map labels, match each frame, accumulate folder metrics.
Shared by the CLI and the UI worker."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .gt import GtSet, load_gt
from .gt.aliases import LabelMapper
from .gt.normalize_attrs import AttrNormalizer
from .matching import match_frame
from .metrics import GtEvaluator

log = logging.getLogger(__name__)


class GtRun:
    def __init__(self, desc, gt: GtSet, mapper: LabelMapper):
        self.desc, self.gt, self.mapper = desc, gt, mapper
        self.normalizer = AttrNormalizer(desc)
        self.evaluator = GtEvaluator(desc, mapper, str(gt.source), gt.format)
        self.lock = threading.Lock()
        self.last = None            # (frame index, ResolvedGt, FrameMatch) of the last processed frame
        self._resolved = {}         # frame index -> ResolvedGt (small cache for the UI)
        self.running = {"tp": 0, "fp": 0, "fn": 0}   # at the thresholds in effect when each frame was processed

    @classmethod
    def open(cls, desc, path: str, fmt: str = "auto", class_names=None) -> "GtRun":
        gt = load_gt(path, fmt, class_names)
        p = Path(path).expanduser()
        return cls(desc, gt, LabelMapper(desc, p if p.is_dir() else p.parent))

    def unresolved_labels(self) -> list[str]:
        labels = self.gt.labels()
        return self.mapper.unresolved(set(labels["box"]) | set(labels["line"]) | set(labels["polygon"]))

    def frame_gt(self, frame, size):
        """ResolvedGt for a source frame, or None when the GT has no entry for it."""
        f = self.gt.find(name=frame.name, index=frame.index if self.gt.by_index else None)
        if f is None and frame.timestamp_s is None:
            return None
        if f is None:
            return None
        return f.resolve(self.mapper, self.normalizer, size)

    def process(self, frame, raw, pred, decoder):
        """Worker thread: called once per processed frame (pred at the UI thresholds)."""
        size = pred.image_size
        r = self.frame_gt(frame, size)
        if r is None:
            with self.lock:
                self.evaluator.frames_without_gt += 1
            return None, None
        m = match_frame(pred.boxes, r.boxes)
        all_boxes = decoder.decode_boxes(raw, self.evaluator.ap_thresholds(), attributes=False)
        with self.lock:
            self.evaluator.add(frame.name, r, all_boxes, m, pred.masks, size)
            self.last = (frame.index, r, m)
            for st in m.per_class.values():
                for k in self.running:
                    self.running[k] += st[k]
            self._resolved[frame.index] = r
            if len(self._resolved) > 64:
                self._resolved.pop(next(iter(self._resolved)))
        return r, m

    def resolved_for(self, frame, size):
        with self.lock:
            r = self._resolved.get(frame.index)
        return r if r is not None else self.frame_gt(frame, size)

    def metrics(self, thresholds: dict) -> dict:
        with self.lock:
            return self.evaluator.compute(thresholds)
