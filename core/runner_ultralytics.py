"""Ultralytics .pt runner (optional dependency, imported lazily). Detection and segmentation tasks.

The description is built from the model itself (class names from model.names). Inference runs at a low
confidence so thresholds can be changed afterwards (and AP sees every box); decode() filters.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .description import parse
from .model_file import ModelFile
from .prediction import Box, Mask, Prediction

log = logging.getLogger(__name__)

INFER_CONF = 0.001
DEFAULT_CONF = 0.25


@dataclass
class UltraRaw:
    boxes: np.ndarray            # [n, 4] xyxy original px
    conf: np.ndarray
    cls: np.ndarray
    w: int
    h: int
    masks: np.ndarray | None = None   # [n, H, W] bool, original px (segmentation)
    frame_index: int = 0
    timestamp_s: float | None = None
    name: str | None = None
    timing_ms: dict = field(default_factory=dict)
    _masks: dict | None = None


class UltraDecoder:
    def __init__(self, desc, names: list[str], seg: bool):
        self.desc, self.names, self.seg = desc, names, seg

    def default_thresholds(self):
        return {"conf": self.desc.detection.conf_threshold}

    def decode_boxes(self, raw: UltraRaw, thresholds=None, attributes=True):
        th = {**self.default_thresholds(), **(thresholds or {})}
        keep = np.where(raw.conf >= th["conf"])[0]
        return [Box(tuple(float(v) for v in raw.boxes[k]), self.names[int(raw.cls[k])], float(raw.conf[k]), int(k), {})
                for k in keep]

    def decode_masks(self, raw: UltraRaw, thresholds=None):
        if not self.seg or raw.masks is None:
            return {}
        th = {**self.default_thresholds(), **(thresholds or {})}
        cm = np.zeros((raw.h, raw.w), np.uint8)    # class map: 0 background, class id + 1
        for k in np.argsort(raw.conf):              # higher confidence painted last
            if raw.conf[k] >= th["conf"]:
                cm[raw.masks[k]] = int(raw.cls[k]) + 1
        return {"instances": Mask("instances", "classes", cm, ["background"] + self.names)}

    def all_heads(self, raw, slot):
        return {}

    def decode(self, raw: UltraRaw, thresholds=None, masks=True) -> Prediction:
        t0 = time.perf_counter()
        boxes = self.decode_boxes(raw, thresholds)
        ms = self.decode_masks(raw, thresholds) if masks else {}
        timing = dict(raw.timing_ms)
        timing["postprocess"] = timing.get("postprocess", 0.0) + (time.perf_counter() - t0) * 1000
        return Prediction(raw.frame_index, raw.timestamp_s, (raw.w, raw.h), boxes, ms, {}, timing, raw.name)


class UltralyticsRunner:
    kind = "ultralytics"

    def __init__(self, model: ModelFile, device: str = "auto"):
        import os
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # never let Ultralytics pip-install packages (it can
        try:                                                  # replace onnxruntime-gpu with the CPU build)
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Ultralytics models need the optional 'ultralytics' package "
                              "(pip install ultralytics)") from e
        import torch
        self.model = model
        self.message = ""
        self.yolo = YOLO(str(model.path))
        task = getattr(self.yolo, "task", "detect")
        if task not in ("detect", "segment"):
            raise ValueError(f"Ultralytics task '{task}' is not supported (detect or segment)")
        names = self.yolo.names
        self.names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        cuda = torch.cuda.is_available()
        if device == "gpu" and not cuda:
            self.message = "GPU requested but PyTorch has no CUDA device; running on CPU."
        self.torch_device = 0 if cuda and device in ("auto", "gpu") else "cpu"
        self.device = "GPU" if self.torch_device == 0 else "CPU"
        imgsz = int((self.yolo.overrides or {}).get("imgsz") or 640)
        self.imgsz = imgsz
        seg = task == "segment"
        d = {"schema": "gahan-model-description/1", "name": model.path.stem, "version": f"ultralytics {task}",
             "input": {"name": "images", "width": imgsz, "height": imgsz, "resize": "letterbox"},
             "detection": {"output": "boxes", "classes": self.names, "conf_threshold": DEFAULT_CONF}}
        if seg:
            d["masks"] = [{"name": "instances", "output": "masks", "type": "classes",
                           "class_names": ["background"] + self.names, "ignore": ["background"]}]
        self.desc = parse(d)
        model.description = self.desc
        self.decoder = UltraDecoder(self.desc, self.names, seg)
        self.provider = "PyTorch CUDA" if self.device == "GPU" else "PyTorch"
        # warm-up so the first frame's timing is representative
        self.yolo.predict(np.zeros((imgsz, imgsz, 3), np.uint8), conf=INFER_CONF, device=self.torch_device,
                          verbose=False, imgsz=imgsz)

    @property
    def device_label(self) -> str:
        return f"{self.device} ({self.provider})"

    def infer_raw(self, frame_bgr, frame_index=0, timestamp_s=None, name=None) -> UltraRaw:
        t0 = time.perf_counter()
        r = self.yolo.predict(frame_bgr, conf=INFER_CONF, device=self.torch_device, verbose=False, imgsz=self.imgsz,
                              max_det=300)[0]
        h, w = frame_bgr.shape[:2]
        b = r.boxes
        raw = UltraRaw(b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                       b.cls.cpu().numpy().astype(int), w, h, frame_index=frame_index, timestamp_s=timestamp_s,
                       name=name)
        if r.masks is not None and len(b):
            m = r.masks.data.cpu().numpy()   # [n, mh, mw] in letterboxed model space
            from ultralytics.utils.ops import scale_image
            m = scale_image(m.transpose(1, 2, 0), (h, w))
            raw.masks = (m.transpose(2, 0, 1) > 0.5) if m.ndim == 3 else (m[None] > 0.5)
        sp = r.speed or {}
        raw.timing_ms = {"preprocess": float(sp.get("preprocess", 0.0)), "inference": float(sp.get("inference", 0.0)),
                         "postprocess": float(sp.get("postprocess", 0.0))}
        if not sp:
            raw.timing_ms["inference"] = (time.perf_counter() - t0) * 1000
        return raw

    def infer(self, frame_bgr, thresholds=None, **kw):
        return self.decoder.decode(self.infer_raw(frame_bgr, **kw), thresholds)
