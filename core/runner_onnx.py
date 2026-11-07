"""ONNX Runtime runner + generic decode driven only by the model description roles."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .description import AttrHead, ModelDescription
from .model_file import ModelFile
from .nms import nms
from .prediction import Attr, Box, Mask, Prediction

log = logging.getLogger(__name__)

_DLLS_PRELOADED = False


def _preload_cuda_dlls() -> None:
    """Load the CUDA/cuDNN runtime from the NVIDIA pip wheels (or torch's), so no system CUDA install is needed.
    Without it ONNX Runtime silently falls back to CPU when libcublasLt etc. are not on the library path."""
    global _DLLS_PRELOADED
    if _DLLS_PRELOADED:
        return
    _DLLS_PRELOADED = True
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001 - best effort, reported via the provider actually used
            log.warning("onnxruntime.preload_dlls failed: %s", e)


def gpu_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Geometry:
    """Map between original image px and model-input px."""
    w: int
    h: int
    mode: str  # letterbox | stretch
    s: float = 1.0
    l: int = 0
    t: int = 0
    nw: int = 0
    nh: int = 0
    sx: float = 1.0
    sy: float = 1.0

    def boxes_to_image(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, np.float32).reshape(-1, 4)
        if self.mode == "letterbox":
            b = (b - np.array([self.l, self.t, self.l, self.t], np.float32)) / self.s
        else:
            b = b / np.array([self.sx, self.sy, self.sx, self.sy], np.float32)
        return np.clip(b, 0, [self.w, self.h, self.w, self.h])

    def map_to_image(self, m: np.ndarray) -> np.ndarray:
        """Crop the padded region of a model-resolution map (HxW or HxWxC float) and resize to the original."""
        if self.mode == "letterbox":
            m = m[self.t:self.t + self.nh, self.l:self.l + self.nw]
        return cv2.resize(np.ascontiguousarray(m), (self.w, self.h), interpolation=cv2.INTER_LINEAR)


def preprocess(frame_bgr: np.ndarray, desc: ModelDescription) -> tuple[np.ndarray, Geometry]:
    i = desc.input
    h, w = frame_bgr.shape[:2]
    if i.resize == "letterbox":
        s = min(i.width / w, i.height / h)
        nw, nh = round(w * s), round(h * s)
        l, t = (i.width - nw) // 2, (i.height - nh) // 2
        canvas = np.full((i.height, i.width, 3), i.pad_value, np.uint8)
        canvas[t:t + nh, l:l + nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        geo = Geometry(w, h, "letterbox", s=s, l=l, t=t, nw=nw, nh=nh)
    else:
        canvas = cv2.resize(frame_bgr, (i.width, i.height), interpolation=cv2.INTER_LINEAR)
        geo = Geometry(w, h, "stretch", sx=i.width / w, sy=i.height / h)
    if i.color == "RGB":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = canvas.astype(np.float32) * np.float32(i.scale)
    if i.layout == "NCHW":
        x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(np.repeat(x[None], i.batch, axis=0))  # fixed batch: repeat the frame, use slot 0
    if i.dtype == "float16":
        x = x.astype(np.float16)
    return x, geo


def _activate(p: np.ndarray, activation: str) -> np.ndarray:
    p = np.asarray(p, np.float32)
    if activation == "softmax":
        e = np.exp(p - p.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-p))
    return p


def _attr(probs: np.ndarray, values: list[str]) -> Attr:
    order = np.argsort(-probs, kind="stable")
    return Attr(value=values[int(order[0])], prob=float(probs[order[0]]),
                top3=[(values[int(j)], float(probs[j])) for j in order[:3]])


@dataclass
class RawResult:
    """Slot-0 outputs of one forward pass + geometry. Re-decoded on every threshold/toggle change (no inference)."""
    outputs: dict[str, np.ndarray]
    geo: Geometry
    frame_index: int = 0
    timestamp_s: float | None = None
    name: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    _masks: dict[str, Mask] | None = None


class Decoder:
    def __init__(self, desc: ModelDescription):
        self.desc = desc

    def default_thresholds(self) -> dict[str, float]:
        t = {}
        if self.desc.detection:
            t["conf"] = self.desc.detection.conf_threshold
        for e in self.desc.extra_boxes:
            t[e.output] = e.threshold
        return t

    # ---- boxes -------------------------------------------------------------------------------------------------
    def _head_probs(self, o: dict, head: AttrHead, k: int, slot_maps: dict) -> np.ndarray | None:
        arr = o.get(head.output)
        if arr is None:
            return None
        if head.indexing == "slot":
            n = min(head.slots or len(arr), len(arr))
            return arr[k] if k < n else None
        j = slot_maps[head.slot_map].get(k)
        return None if j is None or j >= len(arr) else arr[j]

    def all_heads(self, raw: RawResult, slot: int) -> dict[str, Attr]:
        """Every attribute head's output for one detection slot (applicable or not), for the zoom dialog."""
        d = self.desc
        if not d.attributes or slot < 0:
            return {}
        slot_maps = {}
        for h in d.attributes.heads:
            if h.slot_map and h.slot_map not in slot_maps:
                sm = np.asarray(raw.outputs[h.slot_map]).reshape(-1)
                slot_maps[h.slot_map] = {int(round(float(v))): j for j, v in enumerate(sm)}
        out = {}
        for h in d.attributes.heads:
            p = self._head_probs(raw.outputs, h, slot, slot_maps)
            if p is not None:
                out[h.name] = _attr(_activate(p, d.attributes.activation), h.values)
        return out

    def decode_boxes(self, raw: RawResult, thresholds: dict[str, float] | None = None,
                     attributes: bool = True) -> list[Box]:
        d, o, geo = self.desc, raw.outputs, raw.geo
        th = {**self.default_thresholds(), **(thresholds or {})}
        boxes: list[Box] = []
        det_rows = None
        if d.detection:
            det = np.asarray(o[d.detection.output], np.float32).reshape(-1, 6)
            if d.detection.coords == "normalized":
                det = det.copy(); det[:, :4] *= [d.input.width, d.input.height, d.input.width, d.input.height]
            det_rows = det
            classes = d.detection.classes
            merged = {e.cls for e in d.extra_boxes if e.merge_with_detection_class}
            slot_maps = {}
            if d.attributes:
                for h in d.attributes.heads:
                    if h.slot_map and h.slot_map not in slot_maps:
                        sm = np.asarray(o[h.slot_map]).reshape(-1)
                        slot_maps[h.slot_map] = {int(round(float(v))): j for j, v in enumerate(sm)}
            keep = np.where(det[:, 4] >= th["conf"])[0]
            xyxy = geo.boxes_to_image(det[keep, :4])
            for row, k in enumerate(keep):
                c = int(round(float(det[k, 5])))
                if not 0 <= c < len(classes):
                    continue
                name = classes[c]
                if name in merged:
                    continue  # comes from the merged extra-box pipeline below
                attrs = {}
                if d.attributes and attributes:
                    for h in d.attributes.heads_for(name):
                        p = self._head_probs(o, h, int(k), slot_maps)
                        if p is not None:
                            attrs[h.name] = _attr(_activate(p, d.attributes.activation), h.values)
                boxes.append(Box(tuple(float(v) for v in xyxy[row]), name, float(det[k, 4]), int(k), attrs))
        for e in d.extra_boxes:
            cand = np.asarray(o[e.output], np.float32).reshape(-1, 5)
            if e.coords == "normalized":
                cand = cand.copy(); cand[:, :4] *= [d.input.width, d.input.height, d.input.width, d.input.height]
            if e.merge_with_detection_class and det_rows is not None:
                cid = d.detection.classes.index(e.cls)
                cand = np.concatenate([det_rows[np.round(det_rows[:, 5]) == cid, :5], cand])
            cand = cand[cand[:, 4] >= th.get(e.output, e.threshold)]
            if len(cand):
                cand = cand[nms(cand[:, :4], cand[:, 4], e.nms_iou)]
            for r, b in zip(cand, geo.boxes_to_image(cand[:, :4])):
                boxes.append(Box(tuple(float(v) for v in b), e.cls, float(r[4]), -1, {}, source=e.output))
        return boxes

    # ---- masks / tags ------------------------------------------------------------------------------------------
    def decode_masks(self, raw: RawResult) -> dict[str, Mask]:
        if raw._masks is not None:
            return raw._masks
        out = {}
        for m in self.desc.masks:
            a = np.asarray(raw.outputs[m.output], np.float32)
            if m.type == "binary":
                a = a.reshape(a.shape[-2:])
                up = raw.geo.map_to_image(a)
                if m.activation == "sigmoid":
                    t = float(np.log(m.threshold / (1 - m.threshold))) if 0 < m.threshold < 1 else 0.0
                    data = up > t  # threshold on logits == sigmoid > threshold
                else:
                    data = up > m.threshold
                out[m.name] = Mask(m.name, "binary", data)
            else:
                a = a.reshape(a.shape[-3:])  # C,H,W score maps -> per-channel resize, then a running argmax
                best = raw.geo.map_to_image(a[0])
                cls = np.zeros(best.shape, np.uint8)
                for c in range(1, a.shape[0]):
                    up = raw.geo.map_to_image(a[c])
                    better = up > best  # strict: ties keep the lower index, like argmax
                    cls[better] = c
                    np.maximum(best, up, out=best)
                out[m.name] = Mask(m.name, "classes", cls, list(m.class_names))
        raw._masks = out
        return out

    def decode_tags(self, raw: RawResult) -> dict[str, Attr]:
        return {t.name: _attr(_activate(np.asarray(raw.outputs[t.output]).reshape(-1), t.activation), t.values)
                for t in self.desc.tags}

    def decode(self, raw: RawResult, thresholds: dict[str, float] | None = None, masks: bool = True) -> Prediction:
        t0 = time.perf_counter()
        boxes = self.decode_boxes(raw, thresholds)
        ms = self.decode_masks(raw) if masks else {}
        tags = self.decode_tags(raw)
        timing = dict(raw.timing_ms)
        timing["postprocess"] = timing.get("postprocess", 0.0) + (time.perf_counter() - t0) * 1000
        return Prediction(raw.frame_index, raw.timestamp_s, (raw.geo.w, raw.geo.h), boxes, ms, tags, timing, raw.name)


class OnnxRunner:
    kind = "onnx"

    def __init__(self, model: ModelFile, device: str = "auto", threads: int = 0):
        import onnxruntime as ort
        self.model, self.desc = model, model.description
        self.decoder = Decoder(self.desc)
        self.requested_device = device
        self.message = ""
        want_gpu = device in ("auto", "gpu")
        if want_gpu:
            _preload_cuda_dlls()
        providers = ["CPUExecutionProvider"]
        if want_gpu and gpu_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "gpu":
            self.message = "GPU requested but ONNX Runtime has no CUDA provider; running on CPU."
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model.path), sess_options=so, providers=providers)
        self.provider = self.session.get_providers()[0]
        if want_gpu and providers[0] == "CUDAExecutionProvider" and self.provider != "CUDAExecutionProvider":
            self.message = ("The CUDA provider could not start (no usable NVIDIA GPU, or the CUDA 12 / cuDNN 9 runtime "
                            "is missing); running on CPU.")
        if self.message:
            log.warning(self.message)
        self.device = "GPU" if self.provider == "CUDAExecutionProvider" else "CPU"
        self.output_names = [o.name for o in self.session.get_outputs()]
        needed = set(self.desc.output_names())
        self._fetch = [n for n in self.output_names if n in needed]
        self._ranks = self.desc.output_ranks()
        self._warm_up()

    def _warm_up(self):
        """One inference at load time, so the first real frame (and its timing) is not paying for GPU setup."""
        import numpy as np
        try:
            i = self.desc.input
            self.infer_raw(np.full((i.height, i.width, 3), i.pad_value, np.uint8))
        except Exception as e:  # noqa: BLE001 - a failing warm-up shows up again (with context) on the first frame
            log.warning("warm-up inference failed: %s", e)

    @property
    def device_label(self) -> str:
        p = self.provider.replace("ExecutionProvider", "")
        return self.device if p == self.device else f"{self.device} ({p})"

    def infer_raw(self, frame_bgr: np.ndarray, frame_index: int = 0, timestamp_s: float | None = None,
                  name: str | None = None) -> RawResult:
        t0 = time.perf_counter()
        x, geo = preprocess(frame_bgr, self.desc)
        t1 = time.perf_counter()
        outs = self.session.run(self._fetch, {self.desc.input.name: x})
        t2 = time.perf_counter()
        o = {}
        for n, v in zip(self._fetch, outs):
            r = self._ranks.get(n, v.ndim)
            while v.ndim > r:
                v = v[0]  # batch axes: the frame is in slot 0
            o[n] = v
        return RawResult(o, geo, frame_index, timestamp_s, name,
                         {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000})

    def infer(self, frame_bgr: np.ndarray, thresholds: dict | None = None, **kw) -> Prediction:
        return self.decoder.decode(self.infer_raw(frame_bgr, **kw), thresholds)
