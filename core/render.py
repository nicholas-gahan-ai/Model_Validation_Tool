"""Draw a Prediction onto a frame. Everything comes from the model description; nothing model-specific here.

Draw order: class-map masks -> binary masks -> boxes -> labels -> HUD strip -> watermark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .description import ModelDescription
from .prediction import Box, Prediction
from .settings import APP_ROOT

# 20 high-contrast colours, assigned in class-id order. No maroon: that is the UI accent.
PALETTE_HEX = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#42D4F4", "#F032E6", "#BFEF45",
               "#FABED4", "#469990", "#DCBEFF", "#9A6324", "#FFFAC8", "#AAFFC3", "#808000", "#FFD8B1", "#7FDBFF",
               "#A9A9A9", "#FFFFFF"]
MASK_ALPHA = 0.35
FP_COLOR = (60, 60, 255)      # BGR red, dashed: false positive (GT mode)
MISS_COLOR = (0, 140, 255)    # BGR orange: missed GT box (GT mode)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def bgr_to_hex(c) -> str:
    return "#{:02X}{:02X}{:02X}".format(c[2], c[1], c[0])


def load_color_overrides(path: Path | None = None) -> dict[str, str]:
    path = path or APP_ROOT / "config/colors.yaml"
    try:
        d = yaml.safe_load(Path(path).read_text()) or {}
        return {str(k): str(v) for k, v in (d.get("classes") or d).items() if isinstance(v, str)}
    except (OSError, yaml.YAMLError):
        return {}


def load_wordmark_bgra():
    """assets/logo_wordmark_light.png with its transparent margins cropped (BGRA), or False if missing."""
    p = APP_ROOT / "assets/logo_wordmark_light.png"
    wm = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.is_file() else None
    if wm is None or wm.ndim != 3 or wm.shape[2] != 4:
        return False
    ys, xs = np.where(wm[..., 3] > 10)
    return wm[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy() if len(ys) else False


def norm_name(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


class Palette:
    """Stable class -> colour map: class-id order, overridable by class name in config/colors.yaml."""

    def __init__(self, class_names: list[str], overrides: dict[str, str] | None = None):
        ov = {norm_name(k): v for k, v in (overrides if overrides is not None else load_color_overrides()).items()}
        self.colors = {}
        for i, c in enumerate(class_names):
            self.colors[c] = hex_to_bgr(ov.get(norm_name(c), PALETTE_HEX[i % len(PALETTE_HEX)]))

    def __getitem__(self, cls: str) -> tuple[int, int, int]:
        if cls not in self.colors:  # GT-only class etc.
            self.colors[cls] = hex_to_bgr(PALETTE_HEX[len(self.colors) % len(PALETTE_HEX)])
        return self.colors[cls]


def mask_colors(desc: ModelDescription) -> dict[str, dict[str, tuple]]:
    """{mask name: {class name (or mask name for binary): BGR}}"""
    out, k = {}, 0
    fallback = ["#FFD400", "#22C55E", "#3B82F6", "#F97316", "#A855F7", "#14B8A6", "#EC4899"]
    for m in desc.masks:
        if m.type == "binary":
            out[m.name] = {m.name: hex_to_bgr(m.color or fallback[k % len(fallback)])}; k += 1
        else:
            cm = {}
            for c in m.class_names:
                if c in m.ignore:
                    continue
                cm[c] = hex_to_bgr(m.colors.get(c) or fallback[k % len(fallback)]); k += 1
            out[m.name] = cm
    return out


@dataclass
class RenderOptions:
    boxes: bool = True
    attributes: bool = True
    always_attrs: bool = False             # the description's `always` attribute heads
    masks: dict[str, bool] = field(default_factory=dict)   # mask name -> visible (default visible)
    hidden_classes: set[str] = field(default_factory=set)
    counts_strip: bool = False
    watermark: bool = False

    def mask_on(self, name: str) -> bool:
        return self.masks.get(name, True)


def visible_boxes(pred: Prediction, opts: RenderOptions) -> list[Box]:
    return [b for b in pred.boxes if b.cls not in opts.hidden_classes] if opts.boxes else []


def box_at(pred: Prediction, x: float, y: float, opts: RenderOptions) -> Box | None:
    """Smallest visible box containing the point (original px)."""
    hits = [b for b in visible_boxes(pred, opts) if b.xyxy[0] <= x <= b.xyxy[2] and b.xyxy[1] <= y <= b.xyxy[3]]
    return min(hits, key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])) if hits else None


def _dashed_rect(img, p1, p2, color, th, dash):
    x1, y1 = p1; x2, y2 = p2
    for xa, ya, xb, yb in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(1, int(np.hypot(xb - xa, yb - ya) // dash))
        for i in range(0, n, 2):
            a, b = i / n, min(1, (i + 1) / n)
            cv2.line(img, (int(xa + (xb - xa) * a), int(ya + (yb - ya) * a)),
                     (int(xa + (xb - xa) * b), int(ya + (yb - ya) * b)), color, th, cv2.LINE_AA)


def _text_color(bg) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)


class Renderer:
    def __init__(self, desc: ModelDescription, overrides: dict[str, str] | None = None):
        self.desc = desc
        self.palette = Palette(desc.class_names, overrides)
        self.mask_colors = mask_colors(desc)
        self._watermark = None

    # geometry helpers -----------------------------------------------------------------------------------------
    @staticmethod
    def scale(img) -> float:
        return max(0.5, min(img.shape[0], img.shape[1]) / 1080)

    def label_lines(self, b: Box, opts: RenderOptions) -> tuple[str, list[tuple[str, bool]]]:
        """(first line, [(attribute line, dimmed)])."""
        a = self.desc.attributes
        first = f"{b.cls} {b.conf:.2f}"
        lines: list[tuple[str, bool]] = []
        if not (a and opts.attributes and b.attributes):
            return first, lines
        thr = a.display_prob_threshold
        heads = {h.name: h for h in a.heads}
        for name, at in b.attributes.items():
            if name in a.always and name not in a.applies_to.get(b.cls, []):
                if not opts.always_attrs:
                    continue
            low = at.prob < thr
            if heads.get(name) is not None and heads[name].indexing == "slot_map":
                first += f" | {at.value}{' ?' if low else ''}"   # crop-classifier heads: on the first line
            else:
                lines.append((f"{name}: {at.value} {at.prob:.2f}{' ?' if low else ''}", low))
        return first, lines

    # drawing --------------------------------------------------------------------------------------------------
    def draw_masks(self, img: np.ndarray, pred: Prediction, opts: RenderOptions) -> None:
        """Translucent fill in one pass: paint a colour layer (class maps first, binary masks on top),
        blend it with the frame and copy the blend back only where some mask is set."""
        order = sorted(pred.masks.values(), key=lambda m: 0 if m.type == "classes" else 1)
        layer, any_sel = None, None
        for m in order:
            if not opts.mask_on(m.name) or m.data.shape[:2] != img.shape[:2]:
                continue
            cols = self.mask_colors.get(m.name, {})
            if m.type == "binary":
                parts = [(m.data, cols.get(m.name, (0, 212, 255)))]
            else:
                parts = [(m.data == i, cols[c]) for i, c in enumerate(m.class_names or []) if c in cols]
            for sel, color in parts:
                if layer is None:
                    layer, any_sel = img.copy(), np.zeros(img.shape[:2], np.uint8)
                sel8 = sel.view(np.uint8)  # bool -> uint8 without a copy
                cv2.copyTo(self._solid(img.shape, color), sel8, layer)
                cv2.bitwise_or(any_sel, sel8, any_sel)
        if layer is not None:
            blended = cv2.addWeighted(img, 1 - MASK_ALPHA, layer, MASK_ALPHA, 0)
            cv2.copyTo(blended, any_sel, img)

    def _solid(self, shape, color):
        key = (shape, tuple(color))
        cache = self.__dict__.setdefault("_solids", {})
        if key not in cache:
            if len(cache) > 32:
                cache.clear()
            cache[key] = np.full(shape, color, np.uint8)
        return cache[key]

    def draw_box(self, img, b: Box, opts: RenderOptions, color=None, dashed=False, label=True, prefix=""):
        sc = self.scale(img)
        th = max(1, int(round(2 * sc)))
        color = color or self.palette[b.cls]
        x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy)
        if dashed:
            _dashed_rect(img, (x1, y1), (x2, y2), color, th, max(4, int(8 * sc)))
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)
        if not label:
            return
        first, lines = self.label_lines(b, opts)
        if b.slot == -1 and b.conf == 1.0 and b.source == "gt":
            first = b.cls
        if prefix:
            first = prefix + first
        fs = 0.5 * sc
        ft = max(1, int(round(sc)))
        (tw, tht), base = cv2.getTextSize(first, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        pad = max(2, int(3 * sc))
        lh = tht + base + pad
        ty = y1 - lh if y1 - lh >= 0 else y1
        tx = min(max(0, x1), max(0, img.shape[1] - tw - 2 * pad))
        cv2.rectangle(img, (tx, ty), (tx + tw + 2 * pad, ty + lh), color, -1)
        cv2.putText(img, first, (tx + pad, ty + tht + pad // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, _text_color(color), ft,
                    cv2.LINE_AA)
        y = ty + lh
        afs = 0.42 * sc
        for text, dim in lines:
            (aw, ah), ab = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, afs, ft)
            h = ah + ab + pad
            if y + h > img.shape[0]:
                break
            sub = img[y:y + h, tx:tx + aw + 2 * pad]
            sub[:] = (sub * 0.35).astype(np.uint8)
            cv2.putText(img, text, (tx + pad, y + ah + pad // 2), cv2.FONT_HERSHEY_SIMPLEX, afs,
                        (150, 150, 150) if dim else (240, 240, 240), ft, cv2.LINE_AA)
            y += h

    def draw_counts_strip(self, img, pred: Prediction, opts: RenderOptions):
        counts = {}
        for b in visible_boxes(pred, opts):
            counts[b.cls] = counts.get(b.cls, 0) + 1
        sc = self.scale(img)
        fs, ft = 0.55 * sc, max(1, int(round(sc)))
        h = int(30 * sc)
        band = img[:h]
        band[:] = (band * 0.4).astype(np.uint8)
        x = int(8 * sc)
        items = [(c, n) for c, n in sorted(counts.items(), key=lambda kv: self.desc.class_names.index(kv[0])
                                          if kv[0] in self.desc.class_names else 999)]
        for c, n in items + [("total", sum(counts.values()))]:
            text = f"{c} {n}"
            if c != "total":
                cv2.rectangle(img, (x, int(9 * sc)), (x + int(12 * sc), int(21 * sc)), self.palette[c], -1)
                x += int(16 * sc)
            cv2.putText(img, text, (x, int(21 * sc)), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), ft, cv2.LINE_AA)
            x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][0] + int(14 * sc)

    def draw_watermark(self, img):
        """Gahan AI logo (wordmark) in the bottom-right corner, on a translucent dark plate so it reads on any scene."""
        if self._watermark is None:
            self._watermark = load_wordmark_bgra()
        if self._watermark is False:
            return
        H, W = img.shape[:2]
        tw = max(140, int(W * 0.16))
        lh, lw = self._watermark.shape[:2]
        th = max(1, int(lh * tw / lw))
        logo = cv2.resize(self._watermark, (tw, th), interpolation=cv2.INTER_AREA)
        pad, m = max(6, th // 4), max(10, int(min(H, W) * 0.02))
        x1, y1 = W - m - tw - 2 * pad, H - m - th - 2 * pad
        if x1 < 0 or y1 < 0:
            return
        plate = img[y1:H - m, x1:W - m]
        plate[:] = (plate.astype(np.float32) * 0.55).astype(np.uint8)
        a = logo[..., 3:4].astype(np.float32) / 255 * 0.9
        roi = img[y1 + pad:y1 + pad + th, x1 + pad:x1 + pad + tw].astype(np.float32)
        img[y1 + pad:y1 + pad + th, x1 + pad:x1 + pad + tw] = (roi * (1 - a) + logo[..., :3] * a).astype(np.uint8)

    def render(self, frame_bgr: np.ndarray, pred: Prediction | None, opts: RenderOptions,
               box_style=None) -> np.ndarray:
        """box_style(box) -> dict(color=, dashed=, prefix=) or None, for GT-mode TP/FP styling."""
        img = frame_bgr.copy()
        if pred is not None:
            self.draw_masks(img, pred, opts)
            for b in sorted(visible_boxes(pred, opts), key=lambda b: b.conf):
                st = box_style(b) if box_style else None
                self.draw_box(img, b, opts, **(st or {}))
            if opts.counts_strip:
                self.draw_counts_strip(img, pred, opts)
        if opts.watermark:
            self.draw_watermark(img)
        return img


def render_gt(renderer: "Renderer", frame_bgr: np.ndarray, gt, opts: RenderOptions) -> np.ndarray:
    """Left pane in GT mode: GT polygons translucent, polylines at 3 px, boxes in class colours."""
    img = frame_bgr.copy()
    if gt is None:
        sc = renderer.scale(img)
        cv2.putText(img, "no GT for this frame", (int(20 * sc), int(50 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * sc,
                    (80, 80, 255), max(1, int(2 * sc)), cv2.LINE_AA)
        return img
    layer, any_sel = img.copy(), np.zeros(img.shape[:2], np.uint8)
    for tgt, pts in gt.polygons:
        name, _, cls = tgt.partition("/")
        if not opts.mask_on(name) or len(pts) < 3:
            continue
        cols = renderer.mask_colors.get(name, {})
        col = cols.get(cls) or cols.get(name) or next(iter(cols.values()), (0, 212, 255))
        p = np.round(np.asarray(pts, np.float64)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(layer, [p], col)
        cv2.fillPoly(any_sel, [p], 1)
    if any_sel.any():
        blended = cv2.addWeighted(img, 1 - MASK_ALPHA, layer, MASK_ALPHA, 0)
        cv2.copyTo(blended, any_sel, img)
    for tgt, pts in gt.polylines:
        name = tgt.partition("/")[0]
        if not opts.mask_on(name) or len(pts) < 2:
            continue
        cols = renderer.mask_colors.get(name, {})
        col = cols.get(name) or next(iter(cols.values()), (0, 212, 255))
        cv2.polylines(img, [np.round(np.asarray(pts, np.float64)).astype(np.int32).reshape(-1, 1, 2)], False, col, 3,
                      cv2.LINE_AA)
    if opts.boxes:
        for g in gt.boxes:
            if g.cls in opts.hidden_classes:
                continue
            b = Box(g.xyxy, g.cls, 1.0, -1, {}, source="gt")
            renderer.draw_box(img, b, RenderOptions(attributes=False), label=True,
                              prefix="" if norm_name(g.label) == norm_name(g.cls) else f"{g.label}→")
    return img
