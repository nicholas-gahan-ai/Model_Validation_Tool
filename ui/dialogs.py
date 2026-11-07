"""Dialogs: about, model info, log panel. (Flag, zoom, label mapping, metrics and description dialogs are
added by later build steps.)"""
from __future__ import annotations

import html
import logging

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QTextBrowser, QVBoxLayout)

from core.settings import app_version

from .theme import wordmark_pixmap


def _wordmark_label(height=56) -> QLabel:
    lab = QLabel()
    pm = wordmark_pixmap(height)
    if not pm.isNull():
        lab.setPixmap(pm)
    lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet("background:#111317; padding:14px; border-radius:6px")
    return lab


class AboutDialog(QDialog):
    def __init__(self, model=None, runner=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Validation Viewer")
        v = QVBoxLayout(self)
        v.addWidget(_wordmark_label())
        d = model.description if model else None
        rows = [("Application", f"Gahan AI Validation Viewer {app_version()}")]
        if model:
            rows += [("Model", d.name if d else model.path.name), ("Version", d.version if d else "—"),
                     ("File", model.path.name), ("sha256", model.sha256)]
        if runner:
            rows.append(("Device", runner.device_label))
        t = QLabel("<table cellspacing=4>" + "".join(
            f"<tr><td style='color:#9AA0AA;padding-right:10px'>{html.escape(k)}</td>"
            f"<td><code>{html.escape(str(val))}</code></td></tr>" for k, val in rows) + "</table>")
        t.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(t)
        c = QLabel("© Gahan AI Pvt. Ltd."); c.setAlignment(Qt.AlignCenter); c.setStyleSheet("color:#9AA0AA")
        v.addWidget(c)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(self.reject); v.addWidget(bb)


class ModelInfoDialog(QDialog):
    def __init__(self, model, runner=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model info")
        self.resize(640, 560)
        v = QVBoxLayout(self)
        tb = QTextBrowser(); v.addWidget(tb)
        d = model.description
        e = html.escape
        h = [f"<h3>{e(d.name if d else model.path.name)}</h3>"]
        meta = [("File", str(model.path)), ("Kind", model.kind), ("Description", model.description_source),
                ("Version", d.version if d else ""), ("Created", d.created if d else ""),
                ("Precision", d.precision if d else ""), ("sha256 (file)", model.sha256),
                ("sha256 (graph)", d.sha256_graph if d else "")]
        if runner:
            meta.append(("Device", runner.device_label))
            if runner.message:
                meta.append(("Note", runner.message))
        h.append("<table>" + "".join(f"<tr><td style='color:#9AA0AA'>{e(k)}</td><td>{e(str(x))}</td></tr>"
                                     for k, x in meta if x) + "</table>")
        if d:
            i = d.input
            h.append(f"<h4>Input</h4><p>{e(i.name)}: {i.batch}×{i.layout} {i.width}×{i.height}, {i.resize}, "
                     f"{i.color}, scale {i.scale:.6g}, {i.dtype}</p>")
            if d.detection:
                h.append(f"<h4>Detection ({len(d.detection.classes)} classes, conf {d.detection.conf_threshold})</h4>"
                         f"<p>{e(', '.join(d.detection.classes))}</p>")
            for x in d.extra_boxes:
                h.append(f"<p>Extra boxes <b>{e(x.output)}</b> → {e(x.cls)}, threshold {x.threshold}, NMS {x.nms_iou}"
                         f"{', merged with the detector' if x.merge_with_detection_class else ''}</p>")
            if d.attributes:
                h.append("<h4>Attribute heads</h4><ul>")
                for a in d.attributes.heads:
                    used = [c for c, hs in d.attributes.applies_to.items() if a.name in hs]
                    who = "all classes" if a.name in d.attributes.always else ", ".join(used) or "—"
                    h.append(f"<li><b>{e(a.name)}</b> ({len(a.values)} values; {e(who)})</li>")
                h.append("</ul>")
            if d.masks:
                h.append("<h4>Masks</h4><ul>" + "".join(
                    f"<li><b>{e(m.name)}</b>: {m.type}"
                    f"{' (' + e(', '.join(m.class_names)) + ')' if m.class_names else ''}</li>" for m in d.masks) + "</ul>")
            if d.tags:
                h.append("<h4>Frame tags</h4><ul>" + "".join(f"<li>{e(t.name)}</li>" for t in d.tags) + "</ul>")
        tb.setHtml("".join(h))
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(self.reject); v.addWidget(bb)


class _LogEmitter(QObject):
    line = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to the log panel (thread-safe via a queued signal)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.emitter = _LogEmitter()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.emitter.line.emit(self.format(record))
        except RuntimeError:
            pass


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setStyleSheet("font-family: monospace; font-size: 9pt")


# ------------------------------------------------------------------------------------------------ flag dialog
class FlagDialog(QDialog):
    """Category + note + optional box drawn on the frame (drag on the image)."""

    def __init__(self, frame_bgr, annotated_bgr, categories: list[str], title: str, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton
        from .view import LiveView
        self.setWindowTitle(f"Flag frame · {title}")
        self.resize(980, 700)
        v = QVBoxLayout(self)
        self.view = LiveView()
        self.view.box_mode = True
        self.view.set_images([annotated_bgr])
        self.view.dragged_box.connect(self._box)
        self.view.placeholder_text = ""
        v.addWidget(self.view, 1)
        self.box = None
        self._frame = annotated_bgr
        row = QHBoxLayout()
        row.addWidget(QLabel("Category"))
        self.category = QComboBox(); self.category.addItems(categories)
        row.addWidget(self.category)
        self.box_label = QLabel("Optional: drag on the image to mark the object.")
        self.box_label.setStyleSheet("color:#9AA0AA")
        row.addWidget(self.box_label, 1)
        clear = QPushButton("Clear box"); clear.clicked.connect(self._clear_box); row.addWidget(clear)
        v.addLayout(row)
        v.addWidget(QLabel("Note"))
        self.note = QPlainTextEdit(); self.note.setFixedHeight(70)
        self.note.setPlaceholderText("What is wrong? (free text)")
        v.addWidget(self.note)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.note.setFocus()

    def _box(self, _pane, x1, y1, x2, y2):
        import cv2
        h, w = self._frame.shape[:2]
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        self.box = (x1, y1, x2, y2)
        img = self._frame.copy()
        th = max(2, int(min(h, w) / 360))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (60, 60, 255), th)
        self.view.set_images([img])
        self.box_label.setText(f"Box: {x1:.0f}, {y1:.0f} – {x2:.0f}, {y2:.0f}")

    def _clear_box(self):
        self.box = None
        self.view.set_images([self._frame])
        self.box_label.setText("Optional: drag on the image to mark the object.")

    def values(self):
        return self.category.currentText(), self.note.toPlainText().strip(), self.box


# ------------------------------------------------------------------------------------------------ zoom dialog
class ZoomDialog(QDialog):
    """3× crop with context, class / confidence / slot, every attribute head with its top 3, GT match if any."""

    def __init__(self, frame_bgr, box, renderer, opts, gt_match=None, parent=None, all_heads=None):
        super().__init__(parent)
        import cv2
        import numpy as np
        from PySide6.QtWidgets import QHBoxLayout
        from .view import bgr_to_qimage
        self.setWindowTitle(f"{box.cls} {box.conf:.2f}")
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box.xyxy
        bw, bh = max(8.0, x2 - x1), max(8.0, y2 - y1)
        cx1, cy1 = int(max(0, x1 - bw * 0.5)), int(max(0, y1 - bh * 0.5))
        cx2, cy2 = int(min(w, x2 + bw * 0.5)), int(min(h, y2 + bh * 0.5))
        crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
        scale = 3.0
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is not None:
            scale = min(scale, screen.width() * 0.55 / max(1, crop.shape[1]), screen.height() * 0.7 / max(1, crop.shape[0]))
        scale = max(scale, 1.0)
        big = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                         interpolation=cv2.INTER_CUBIC)
        col = renderer.palette[box.cls]
        th = max(2, int(scale))
        def r(b, c, dashed=False):
            p1 = (int((b[0] - cx1) * scale), int((b[1] - cy1) * scale)); p2 = (int((b[2] - cx1) * scale), int((b[3] - cy1) * scale))
            if dashed:
                from core.render import _dashed_rect
                _dashed_rect(big, p1, p2, c, th, 10)
            else:
                cv2.rectangle(big, p1, p2, c, th, cv2.LINE_AA)
        if gt_match is not None:
            r(gt_match.xyxy, (255, 255, 255), dashed=True)
        r(box.xyxy, col)
        lay = QHBoxLayout(self)
        img = QLabel(); img.setPixmap(QPixmap.fromImage(bgr_to_qimage(big)))
        img.setStyleSheet("background:#0C0D10"); img.setAlignment(Qt.AlignCenter)
        lay.addWidget(img, 1)
        e = html.escape
        hx = "#{:02X}{:02X}{:02X}".format(col[2], col[1], col[0])
        info = [f"<h3><span style='color:{hx}'>■</span> {e(box.cls)}</h3>",
                "<table cellspacing=3>"
                f"<tr><td style='color:#9AA0AA'>confidence</td><td>{box.conf:.4f}</td></tr>"
                f"<tr><td style='color:#9AA0AA'>slot</td><td>{box.slot if box.slot >= 0 else '—'}</td></tr>"
                f"<tr><td style='color:#9AA0AA'>source</td><td>{e(box.source)}</td></tr>"
                f"<tr><td style='color:#9AA0AA'>box (px)</td><td>{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}"
                f"<br>{x2 - x1:.1f} × {y2 - y1:.1f}</td></tr></table>"]
        heads = all_heads if all_heads else box.attributes
        if heads:
            info.append("<h4>Attribute heads (top 3)</h4><table cellspacing=3>")
            for name, a in heads.items():
                applies = name in box.attributes
                style = "" if applies else " style='color:#6B717C'"
                top = "<br>".join(f"{e(v)} <span style='color:#9AA0AA'>{p:.3f}</span>" for v, p in a.top3)
                gt = ""
                if gt_match is not None and name in (gt_match.attributes or {}):
                    gv = gt_match.attributes[name]
                    ok = gv == a.value
                    gt = f"<br><span style='color:{'#6EE7A0' if ok else '#F87171'}'>GT: {e(str(gv))}</span>"
                info.append(f"<tr><td{style} valign=top><b>{e(name)}</b>{'' if applies else '<br>(n/a for class)'}</td>"
                            f"<td{style}>{top}{gt}</td></tr>")
            info.append("</table>")
        if gt_match is not None:
            info.append(f"<h4>Matched GT</h4><p>{e(gt_match.label)} · IoU {gt_match.iou:.3f}"
                        f"<br><span style='color:#9AA0AA'>dashed white box</span></p>")
            extra = {k: v for k, v in (gt_match.attributes or {}).items() if k not in heads}
            if extra:
                info.append("<p>" + "<br>".join(f"{e(k)}: {e(str(v))}" for k, v in extra.items()) + "</p>")
        t = QLabel("".join(info)); t.setTextFormat(Qt.RichText); t.setAlignment(Qt.AlignTop)
        t.setTextInteractionFlags(Qt.TextSelectableByMouse); t.setMinimumWidth(340)
        lay.addWidget(t)
        self.adjustSize()


# ------------------------------------------------------------------------------------------------ label mapping
class LabelMappingDialog(QDialog):
    """Unmatched GT labels -> a model class, a mask, or ignore. Saved to <gt folder>/.gahan_label_map.yaml."""

    def __init__(self, labels: list[str], counts: dict, targets: list[str], gt_dir, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QComboBox, QTableWidget
        self.setWindowTitle("Map ground-truth labels")
        self.resize(620, 420)
        v = QVBoxLayout(self)
        t = QLabel("These GT labels have no matching model class. Map each to a model class or mask, or ignore it.\n"
                   f"The choice is saved to {gt_dir}/.gahan_label_map.yaml. Ignored labels are listed as "
                   "excluded in the report.")
        t.setWordWrap(True)
        v.addWidget(t)
        self.table = QTableWidget(len(labels), 3)
        self.table.setHorizontalHeaderLabels(["GT label", "count", "map to"])
        self.table.verticalHeader().hide()
        self.combos = {}
        for i, lab in enumerate(labels):
            from PySide6.QtWidgets import QTableWidgetItem
            self.table.setItem(i, 0, QTableWidgetItem(lab))
            self.table.setItem(i, 1, QTableWidgetItem(str(counts.get(lab, ""))))
            cb = QComboBox(); cb.addItem("ignore (excluded)", "ignore")
            for tg in targets:
                cb.addItem(tg, tg)
            self.table.setCellWidget(i, 2, cb)
            self.combos[lab] = cb
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def choices(self) -> dict:
        return {lab: cb.currentData() for lab, cb in self.combos.items()}


# ------------------------------------------------------------------------------------------------ metrics
def _f(x, d=4):
    return "—" if x is None or x != x else f"{x:.{d}f}"


class MetricsDialog(QDialog):
    def __init__(self, metrics: dict, palette, export_cb=None, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem
        from ui.panels import swatch
        self.setWindowTitle("Ground-truth metrics")
        self.resize(900, 640)
        v = QVBoxLayout(self)
        m = metrics
        th = ", ".join(f"{k}={val:.2f}" for k, val in m.get("thresholds", {}).items())
        hdr = QLabel(f"<b>{m['images_matched']}</b> frames with GT ({m['frames_without_gt']} without) · "
                     f"{html.escape(str(m['gt_format']))}: {html.escape(str(m['gt_source']))}<br>"
                     f"<span style='color:#9AA0AA'>Precision / recall at the thresholds ({html.escape(th)}), IoU 0.5. "
                     "AP over all predictions, COCO-style (101-point, IoU 0.50:0.05:0.95, top 100 per class per image)."
                     "</span>")
        hdr.setWordWrap(True)
        v.addWidget(hdr)
        cols = ["Class", "GT", "Pred@thr", "TP", "FP", "Miss", "Precision", "Recall", "AP50", "AP50-95"]
        per = m["per_class"]
        t = QTableWidget(len(per) + 1, len(cols))
        t.setHorizontalHeaderLabels(cols); t.verticalHeader().hide()
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        for i, (c, r) in enumerate(per.items()):
            it = QTableWidgetItem(c + ("  (no predictions)" if r["no_predictions"] else ""))
            it.setIcon(swatch(palette[c]))
            t.setItem(i, 0, it)
            for j, val in enumerate([r["gt"], r["pred"], r["tp"], r["fp"], r["fn"], _f(r["precision"]),
                                     _f(r["recall"]), _f(r["ap50"]), _f(r["ap50_95"])], 1):
                x = QTableWidgetItem(str(val)); x.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter); t.setItem(i, j, x)
        mm = m["mean"]
        i = len(per)
        it = QTableWidgetItem(f"mean ({mm['n_classes']} classes)"); f = it.font(); f.setBold(True); it.setFont(f)
        t.setItem(i, 0, it)
        for j, val in ((6, mm["precision"]), (7, mm["recall"]), (8, mm["ap50"]), (9, mm["ap50_95"])):
            x = QTableWidgetItem(_f(val)); x.setFont(f); x.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(i, j, x)
        t.resizeColumnsToContents()
        from PySide6.QtWidgets import QHeaderView
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        v.addWidget(t, 1)
        extra = []
        if m.get("attributes"):
            extra.append("<b>Attribute accuracy on matched boxes</b> (IoU ≥ 0.5): " + " · ".join(
                f"{html.escape(h)} {r['agree']}/{r['n']} = {_f(r['agree'] / r['n'] if r['n'] else None, 3)}"
                + (f" ({r['not_comparable']} n/c)" if r["not_comparable"] else "") for h, r in m["attributes"].items()))
        elif m.get("attributes_na"):
            extra.append(f"<b>Attributes:</b> N/A ({html.escape(m['attributes_na'])})")
        if m.get("masks"):
            extra.append("<b>Mask IoU:</b> " + " · ".join(f"{html.escape(k)} {_f(r['iou'])} ({r['frames']} frames, "
                                                          f"{html.escape(r['note'])})" for k, r in m["masks"].items()))
        if m.get("excluded"):
            extra.append("<b>GT box labels excluded</b> (no model class): " + ", ".join(
                f"{html.escape(k)} ({n})" for k, n in m["excluded"].items()))
        if m.get("excluded_shapes"):
            extra.append("<b>GT shapes excluded</b>: " + ", ".join(
                f"{html.escape(k)} ({n})" for k, n in m["excluded_shapes"].items()))
        lab = QLabel("<br>".join(extra)); lab.setWordWrap(True); lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(lab)
        row = QHBoxLayout(); row.addStretch(1)
        if export_cb:
            b = QPushButton("Export metrics.json + .csv"); b.clicked.connect(lambda: export_cb()); row.addWidget(b)
        c = QPushButton("Close"); c.clicked.connect(self.accept); row.addWidget(c)
        v.addLayout(row)


# ------------------------------------------------------------------------------------------------ description
class DescriptionDialog(QDialog):
    """A metadata-less ONNX: write or load its description once; saved as <model>.description.yaml beside it."""

    def __init__(self, err, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton
        from core.model_file import description_template, sidecar_path
        self.err, self.result_model = err, None
        self.setWindowTitle(f"Model description · {err.path.name}")
        self.resize(820, 680)
        v = QVBoxLayout(self)
        t = QLabel(f"<b>{html.escape(err.path.name)}</b> has no embedded model description. Describe its inputs and "
                   f"outputs once (or load a YAML). It is saved as <code>{html.escape(sidecar_path(err.path).name)}"
                   "</code> beside the model.")
        t.setWordWrap(True)
        v.addWidget(t)
        self.edit = QPlainTextEdit()
        self.edit.setStyleSheet("font-family: monospace; font-size: 9.5pt")
        self.edit.setPlainText(description_template(err.path, err.inputs, err.outputs))
        v.addWidget(self.edit, 1)
        self.msg = QLabel(""); self.msg.setWordWrap(True)
        v.addWidget(self.msg)
        row = QHBoxLayout()
        load = QPushButton("Load YAML…")
        load.clicked.connect(lambda: self._load(QFileDialog.getOpenFileName(
            self, "Description YAML", str(err.path.parent), "YAML (*.yaml *.yml)")[0]))
        check = QPushButton("Validate"); check.clicked.connect(self._validate)
        row.addWidget(load); row.addWidget(check); row.addStretch(1)
        save = QPushButton("Save and load model"); save.setObjectName("Primary"); save.clicked.connect(self._save)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        row.addWidget(save); row.addWidget(cancel)
        v.addLayout(row)

    def _load(self, p):
        if p:
            from pathlib import Path
            self.edit.setPlainText(Path(p).read_text(encoding="utf-8"))
            self._validate()

    def _parsed(self):
        import yaml
        from core.description import DescriptionError
        from core.model_file import open_model
        try:
            d = yaml.safe_load(self.edit.toPlainText())
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            raise DescriptionError(f"YAML error{f' at line {mark.line + 1}' if mark else ''}: {e}") from e
        return d, open_model(self.err.path, description_override=d)

    def _validate(self) -> bool:
        from core.description import DescriptionError
        try:
            _, m = self._parsed()
        except DescriptionError as e:
            self.msg.setText(f"<span style='color:#F87171'>{html.escape(str(e))}</span>".replace("\n", "<br>"))
            return False
        d = m.description
        self.msg.setText(f"<span style='color:#6EE7A0'>OK: {len(d.class_names)} classes, "
                         f"{len(d.attributes.heads) if d.attributes else 0} attribute heads, {len(d.masks)} masks.</span>")
        return True

    def _save(self):
        from core.description import DescriptionError
        from core.model_file import open_model, save_sidecar
        try:
            d, _ = self._parsed()
            save_sidecar(self.err.path, d)
            self.result_model = open_model(self.err.path)
        except (DescriptionError, OSError) as e:
            self.msg.setText(f"<span style='color:#F87171'>{html.escape(str(e))}</span>".replace("\n", "<br>"))
            return
        self.accept()
