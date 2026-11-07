"""Left (setup), right (stats) and bottom (toggles + transport) panels. Toggles are generated from the model."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLayout, QLineEdit, QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QSlider, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from core.render import bgr_to_hex
from core.worker import SPEEDS


def section(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("SectionTitle")
    return lab


def swatch(bgr, size=12) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(QColor(bgr_to_hex(bgr)))
    return pm


class FlowLayout(QLayout):
    """Wraps child widgets onto new lines (for the generated class toggles)."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self.items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, i):
        return self.items[i] if 0 <= i < len(self.items) else None

    def takeAt(self, i):
        return self.items.pop(i) if 0 <= i < len(self.items) else None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._layout(QRect(0, 0, w, 0), True)

    def setGeometry(self, r):
        super().setGeometry(r)
        self._layout(r, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self.items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _layout(self, r, test):
        x, y, line = r.x(), r.y(), 0
        for it in self.items:
            w = it.sizeHint()
            if x + w.width() > r.right() and line > 0:
                x, y, line = r.x(), y + line + self.spacing(), 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), w))
            x += w.width() + self.spacing()
            line = max(line, w.height())
        return y + line - r.y()


class PathRow(QWidget):
    changed = Signal()

    def __init__(self, placeholder="", browse=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        lay.addWidget(self.edit, 1)
        self.btn = QPushButton("Browse…")
        self.btn.clicked.connect(lambda: browse(self) if browse else None)
        lay.addWidget(self.btn)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t)


class LeftPanel(QWidget):
    start_clicked = Signal()
    stop_clicked = Signal()
    open_model_clicked = Signal()
    model_selected = Signal(str)
    model_info_clicked = Signal()
    device_changed = Signal(str)

    INPUT_KINDS = ("video", "folder", "camera", "rtsp")
    GT_FORMATS = ["Auto-detect", "CVAT XML 1.1", "Open ADAS XML (per image)", "COCO JSON", "YOLO txt"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        v = QVBoxLayout(self); v.setContentsMargins(10, 4, 10, 10); v.setSpacing(5)

        v.addWidget(section("Model"))
        row = QHBoxLayout(); row.setSpacing(4)
        self.model_combo = QComboBox(); self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(14)
        self.model_combo.activated.connect(lambda i: self.model_selected.emit(self.model_combo.itemData(i) or ""))
        row.addWidget(self.model_combo, 1)
        b = QPushButton("Open…"); b.clicked.connect(self.open_model_clicked); row.addWidget(b)
        self.info_btn = QToolButton(); self.info_btn.setText("ⓘ"); self.info_btn.setToolTip("Model info")
        self.info_btn.clicked.connect(self.model_info_clicked); row.addWidget(self.info_btn)
        v.addLayout(row)

        v.addWidget(section("Input"))
        self.kind_group = QButtonGroup(self)
        g = QGridLayout(); g.setHorizontalSpacing(6); g.setVerticalSpacing(4)
        self.radio = {}
        for i, (k, label) in enumerate(zip(self.INPUT_KINDS, ("Video", "Image folder", "Camera", "RTSP"))):
            r = QRadioButton(label); self.radio[k] = r; self.kind_group.addButton(r, i)
            g.addWidget(r, i, 0)
        self.video_path = PathRow("video file", self._browse_video)
        self.folder_path = PathRow("folder of images", self._browse_folder)
        cam = QWidget(); cl = QHBoxLayout(cam); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(4)
        self.camera_combo = QComboBox(); self.camera_combo.addItems([str(i) for i in range(6)])
        self.camera_combo.setEditable(True)
        cl.addWidget(self.camera_combo, 1)
        self.probe_btn = QPushButton("Probe"); cl.addWidget(self.probe_btn)
        self.rtsp_url = QLineEdit(); self.rtsp_url.setPlaceholderText("rtsp://host:8554/stream")
        for i, w in enumerate((self.video_path, self.folder_path, cam, self.rtsp_url)):
            g.addWidget(w, i, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        self.folder_recursive = QCheckBox("Image folder: include subfolders")
        v.addWidget(self.folder_recursive)
        self.radio["video"].setChecked(True)

        v.addWidget(section("Ground truth"))
        self.gt_path = PathRow("annotation file or folder", self._browse_gt)
        v.addWidget(self.gt_path)
        self.gt_format = QComboBox(); self.gt_format.addItems(self.GT_FORMATS)
        v.addWidget(self.gt_format)

        v.addWidget(section("Outputs"))
        self.save_video = QCheckBox("Save annotated video / images"); self.save_video.setChecked(True)
        self.side_by_side_save = QCheckBox("Save side by side (GT / input | prediction)")
        self.folder_video = QCheckBox("Image folder: also save an mp4")
        self.counts_strip = QCheckBox("Counts overlay strip")
        self.watermark = QCheckBox("Gahan AI logo watermark")
        self.export_dets = QCheckBox("Export detections (JSONL + CSV)")
        self.include_masks = QCheckBox("Include masks (RLE)"); self.include_masks.setEnabled(False)
        self.export_dets.toggled.connect(self.include_masks.setEnabled)
        self.summary = QCheckBox("Session summary report (HTML)")
        for w in (self.save_video, self.folder_video, self.side_by_side_save, self.counts_strip, self.watermark,
                  self.export_dets, self.include_masks, self.summary):
            v.addWidget(w)
        self.include_masks.setContentsMargins(18, 0, 0, 0)
        self.out_dir = PathRow("output folder", self._browse_out)
        v.addWidget(QLabel("Output folder")); v.addWidget(self.out_dir)

        v.addWidget(section("Device"))
        self.device = QComboBox(); self.device.addItems(["Auto", "GPU", "CPU"])
        self.device.currentTextChanged.connect(lambda t: self.device_changed.emit(t.lower()))
        v.addWidget(self.device)

        v.addSpacing(8)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.setObjectName("Primary")
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_clicked); self.stop_btn.clicked.connect(self.stop_clicked)
        row.addWidget(self.start_btn, 1); row.addWidget(self.stop_btn, 1)
        v.addLayout(row)
        v.addStretch(1)

    # ---- browse helpers ------------------------------------------------------------------------------------------
    def _browse_video(self, row):
        p, _ = QFileDialog.getOpenFileName(self, "Open video", row.text(),
                                           "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.mpg *.mpeg *.wmv *.webm *.ts);;All files (*)")
        if p:
            row.setText(p); self.radio["video"].setChecked(True)

    def _browse_folder(self, row):
        p = QFileDialog.getExistingDirectory(self, "Open image folder", row.text())
        if p:
            row.setText(p); self.radio["folder"].setChecked(True)

    def _browse_gt(self, row):
        p, _ = QFileDialog.getOpenFileName(self, "Ground truth file (Cancel to pick a folder)", row.text(),
                                           "Annotations (*.xml *.json);;All files (*)")
        if not p:
            p = QFileDialog.getExistingDirectory(self, "Ground truth folder (YOLO txt / Open ADAS XML tree)", row.text())
        if p:
            row.setText(p)

    def _browse_out(self, row):
        p = QFileDialog.getExistingDirectory(self, "Output folder", row.text())
        if p:
            row.setText(p)

    # ---- state ---------------------------------------------------------------------------------------------------
    def input_kind(self) -> str:
        return self.INPUT_KINDS[max(0, self.kind_group.checkedId())]

    def input_spec(self) -> str:
        k = self.input_kind()
        return {"video": self.video_path.text(), "folder": self.folder_path.text(),
                "camera": f"camera:{self.camera_combo.currentText().strip() or '0'}",
                "rtsp": self.rtsp_url.text().strip()}[k]

    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.model_combo, self.video_path, self.folder_path, self.camera_combo, self.rtsp_url, self.gt_path,
                  self.gt_format, self.save_video, self.side_by_side_save, self.export_dets, self.include_masks,
                  self.folder_recursive, self.folder_video, self.counts_strip, self.watermark,
                  self.summary, self.out_dir, self.device, self.probe_btn, *self.radio.values()):
            w.setEnabled(not running)
        if not running:
            self.include_masks.setEnabled(self.export_dets.isChecked())


class CountsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Class", "Count"])
        self.verticalHeader().hide()
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch); h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.verticalHeader().setDefaultSectionSize(20)

    def set_rows(self, rows):
        """rows: [(name, count, bgr or None)]"""
        self.setRowCount(len(rows))
        for i, (name, n, col) in enumerate(rows):
            it = QTableWidgetItem(name)
            if col is not None:
                it.setIcon(swatch(col))
            self.setItem(i, 0, it)
            c = QTableWidgetItem(str(n)); c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(i, 1, c)
        self.setFixedHeight(min(400, 24 + 20 * max(1, len(rows))))


class RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        v = QVBoxLayout(self); v.setContentsMargins(10, 4, 10, 10); v.setSpacing(5)
        v.addWidget(section("Detections (frame)"))
        self.table = CountsTable()
        v.addWidget(self.table)
        row = QHBoxLayout()
        self.total = QLabel("total 0"); self.total.setStyleSheet("font-weight:600")
        self.show_all = QCheckBox("all classes")
        row.addWidget(self.total); row.addStretch(1); row.addWidget(self.show_all)
        v.addLayout(row)
        self.totals_btn = QToolButton(); self.totals_btn.setText("Session totals ▸"); self.totals_btn.setCheckable(True)
        self.totals_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        v.addWidget(self.totals_btn)
        self.totals = CountsTable(); self.totals.hide()
        self.totals_note = QLabel(""); self.totals_note.hide(); self.totals_note.setStyleSheet("color:#9AA0AA")
        v.addWidget(self.totals); v.addWidget(self.totals_note)
        self.totals_btn.toggled.connect(self._toggle_totals)

        v.addWidget(section("Frame"))
        self.frame_info = QLabel("—"); self.frame_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.frame_info.setWordWrap(True)
        self.frame_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # long file names must not widen the panel
        v.addWidget(self.frame_info)

        self.gt_title = section("Ground truth")
        self.gt_info = QLabel(""); self.gt_info.setWordWrap(True); self.gt_info.setTextFormat(Qt.RichText)
        self.gt_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.metrics_btn = QPushButton("Folder metrics…")
        for w in (self.gt_title, self.gt_info, self.metrics_btn):
            v.addWidget(w); w.hide()
        v.addStretch(1)

    def _toggle_totals(self, on):
        self.totals.setVisible(on); self.totals_note.setVisible(on)
        self.totals_btn.setText("Session totals ▾" if on else "Session totals ▸")

    def show_gt(self, on: bool):
        for w in (self.gt_title, self.gt_info, self.metrics_btn):
            w.setVisible(on)


class Slider(QWidget):
    changed = Signal(float)

    def __init__(self, label: str, value: float, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lay.addWidget(QLabel(label))
        self.s = QSlider(Qt.Horizontal); self.s.setRange(0, 100); self.s.setFixedWidth(140)
        self.val = QLabel(); self.val.setFixedWidth(34)
        lay.addWidget(self.s); lay.addWidget(self.val)
        self.s.valueChanged.connect(self._ch)
        self.set_value(value)

    def _ch(self, v):
        self.val.setText(f"{v / 100:.2f}")
        self.changed.emit(v / 100)

    def set_value(self, x: float):
        self.s.setValue(int(round(x * 100)))
        self.val.setText(f"{x:.2f}")

    def value(self) -> float:
        return self.s.value() / 100


class ControlsBar(QWidget):
    """Generated layer + class toggles, threshold sliders, and the transport row."""
    render_changed = Signal()
    play_toggled = Signal()
    step = Signal(int)
    home = Signal()
    end = Signal()
    seek = Signal(int)
    speed_changed = Signal(object)
    flag_clicked = Signal()
    snapshot_clicked = Signal()
    sbs_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        v = QVBoxLayout(self); v.setContentsMargins(10, 6, 10, 6); v.setSpacing(4)
        self.layers_row = QHBoxLayout(); self.layers_row.setSpacing(12)
        v.addLayout(self.layers_row)
        mid = QHBoxLayout(); mid.setSpacing(16)
        self.classes_host = QWidget(); self.classes_flow = FlowLayout(self.classes_host, spacing=10)
        sc = QScrollArea(); sc.setWidget(self.classes_host); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        sc.setMaximumHeight(52); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.classes_label = QLabel("CLASSES"); self.classes_label.setObjectName("SectionTitle")
        mid.addWidget(self.classes_label, 0, Qt.AlignTop)
        mid.addWidget(sc, 1)
        self.sliders_box = QHBoxLayout(); self.sliders_box.setSpacing(16)
        mid.addLayout(self.sliders_box)
        v.addLayout(mid)

        t = QHBoxLayout(); t.setSpacing(4)
        def tb(text, tip, sig):
            b = QToolButton(); b.setText(text); b.setToolTip(tip); b.clicked.connect(sig); t.addWidget(b); return b
        self.home_btn = tb("⏮", "First frame (Home)", self.home.emit)
        self.prev_btn = tb("◀", "Step back (←)", lambda: self.step.emit(-1))
        self.play_btn = tb("❚❚", "Play / pause (Space)", self.play_toggled.emit)
        self.next_btn = tb("▶", "Step forward (→)", lambda: self.step.emit(1))
        self.end_btn = tb("⏭", "Last frame (End)", self.end.emit)
        self.play_btn.setFixedWidth(36)
        from .view import SeekBar
        self.seekbar = SeekBar(); self.seekbar.seek_requested.connect(self.seek)
        t.addWidget(self.seekbar, 1)
        self.time_label = QLabel("--:-- / --:--"); self.time_label.setMinimumWidth(130)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        t.addWidget(self.time_label)
        t.addSpacing(8)
        t.addWidget(QLabel("Speed"))
        self.speed = QComboBox(); self.speed.addItems(list(SPEEDS)); self.speed.setCurrentText("1× (real time)")
        self.speed.currentTextChanged.connect(lambda s: self.speed_changed.emit(SPEEDS[s]))
        t.addWidget(self.speed)
        self.sbs_btn = QToolButton(); self.sbs_btn.setText("Side by side"); self.sbs_btn.setCheckable(True)
        self.sbs_btn.setToolTip("Side-by-side view (T)"); self.sbs_btn.toggled.connect(self.sbs_toggled)
        t.addWidget(self.sbs_btn)
        self.snap_btn = QPushButton("📷 Snapshot"); self.snap_btn.setToolTip("Save raw + annotated PNG (S)")
        self.snap_btn.clicked.connect(self.snapshot_clicked); t.addWidget(self.snap_btn)
        self.flag_btn = QPushButton("⚑ Flag"); self.flag_btn.setObjectName("Flag"); self.flag_btn.setToolTip("Flag this frame (F)")
        self.flag_btn.clicked.connect(self.flag_clicked); t.addWidget(self.flag_btn)
        v.addLayout(t)
        self.layer_boxes: dict[str, QCheckBox] = {}
        self.class_boxes: dict[str, QCheckBox] = {}
        self.sliders: dict[str, Slider] = {}
        self.set_transport_enabled(False)

    def _clear(self, lay):
        while lay.count():
            it = lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def build(self, desc, palette, thresholds: dict, saved: dict | None = None):
        """(Re)generate toggles and sliders from a model description."""
        saved = saved or {}
        self._clear(self.layers_row); self._clear(self.classes_flow); self._clear(self.sliders_box)
        self.layer_boxes, self.class_boxes, self.sliders = {}, {}, {}
        lab = QLabel("LAYERS"); lab.setObjectName("SectionTitle"); self.layers_row.addWidget(lab)
        layers = []
        if desc.detection or desc.extra_boxes:
            layers.append(("boxes", "Boxes", True))
        if desc.attributes:
            layers.append(("attributes", "Attributes", True))
            if desc.attributes.always:
                layers.append(("always_attrs", "/".join(a.capitalize() for a in desc.attributes.always)[:40], False))
        for m in desc.masks:
            layers.append((f"mask:{m.name}", m.name.replace("_", " ").capitalize(), True))
        for i, (key, text, default) in enumerate(layers):
            cb = QCheckBox(f"{text}" + (f"  [{i + 1}]" if i < 6 else ""))
            cb.setChecked(saved.get("layers", {}).get(key, default))
            cb.toggled.connect(self.render_changed)
            self.layer_boxes[key] = cb
            self.layers_row.addWidget(cb)
        self.layers_row.addStretch(1)
        hidden = set(saved.get("hidden_classes", []))
        for c in desc.class_names:
            cb = QCheckBox(c); cb.setIcon(swatch(palette[c])); cb.setChecked(c not in hidden)
            cb.toggled.connect(self.render_changed)
            self.class_boxes[c] = cb
            self.classes_flow.addWidget(cb)
        self.classes_label.setVisible(bool(self.class_boxes))
        if "conf" in thresholds:
            s = Slider("Conf", thresholds["conf"]); s.changed.connect(self.render_changed)
            self.sliders["conf"] = s; self.sliders_box.addWidget(s)
        for e in desc.extra_boxes:
            s = Slider(e.threshold_label or f"{e.cls} cutoff", thresholds.get(e.output, e.threshold))
            s.changed.connect(self.render_changed)
            self.sliders[e.output] = s; self.sliders_box.addWidget(s)

    def thresholds(self) -> dict:
        return {k: s.value() for k, s in self.sliders.items()}

    def toggle_layer(self, n: int):
        keys = list(self.layer_boxes)
        if 0 <= n < len(keys):
            cb = self.layer_boxes[keys[n]]
            cb.setChecked(not cb.isChecked())

    def set_transport_enabled(self, on: bool, seekable: bool = True):
        for w in (self.home_btn, self.prev_btn, self.next_btn, self.end_btn, self.seekbar):
            w.setEnabled(on and seekable)
        for w in (self.play_btn, self.flag_btn, self.snap_btn):
            w.setEnabled(on)

    def set_playing(self, playing: bool):
        self.play_btn.setText("❚❚" if playing else "▶")
