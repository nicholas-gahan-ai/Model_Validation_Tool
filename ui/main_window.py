"""Main window: header, setup panel (left), live view + controls (centre), stats panel (right)."""
from __future__ import annotations

import logging
import os
from html import escape as html_escape
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QFont, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QScrollArea, QSplashScreen, QVBoxLayout, QWidget)

from core.description import DescriptionError
from core.model_file import NeedsDescription, list_models, open_model
from core.render import Renderer, RenderOptions
from core.runners import make_runner
from core.session import ImageFolderSaver, LiveVideoSaver, Session, VideoSaver
from core.settings import APP_ROOT, app_version
from core.sources import FolderSource, SourceError, make_source
from core.worker import SPEEDS, PipelineWorker

from .dialogs import AboutDialog, LogView, ModelInfoDialog, QtLogHandler
from .panels import ControlsBar, LeftPanel, RightPanel
from .theme import ICON_PNG, apply_theme, wordmark_pixmap
from .view import LiveView

log = logging.getLogger("viewer")
ORG, APP = "Gahan AI", "Validation Viewer"
FOLDER_VIDEO_FPS = 10.0   # image folder -> optional mp4


def fmt_time(s: float | None) -> str:
    if s is None:
        return "--:--"
    m, s = divmod(max(0.0, s), 60)
    return f"{int(m):02d}:{s:05.2f}"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:60] or "run"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings()   # org/app names come from the QApplication (run_app sets ORG / APP)
        self.setWindowTitle("Gahan AI · Validation Viewer")
        self.model = self.runner = self.renderer = None
        self.worker: PipelineWorker | None = None
        self.session: Session | None = None
        self.source = None
        self.current = None            # FrameResult on screen
        self.current_pred = None       # its Prediction at the current thresholds
        self.run_dir: Path | None = None
        self.click_zoom = True
        self.gt_run = None
        self.current_gt = self.current_match = None
        self.sbs = False
        self._fps_ema = None
        self._last_result_t = None
        self._building = False

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.log_handler = QtLogHandler()
        self.log_handler.emitter.line.connect(self.log_view.appendPlainText)
        logging.getLogger().addHandler(self.log_handler)
        self._restore()
        self._refresh_model_list()

    # ---------------------------------------------------------------------------------------------------- layout
    def _build_ui(self):
        header = QWidget(); header.setObjectName("Header")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 8, 14, 8); hl.setSpacing(18)
        wm = QLabel()
        pm = wordmark_pixmap(46)
        if not pm.isNull():
            wm.setPixmap(pm)
        hl.addWidget(wm)
        title = QLabel("Validation Viewer"); title.setObjectName("HeaderTitle"); hl.addWidget(title)
        hl.addStretch(1)
        self.header_model = QLabel("No model loaded"); self.header_model.setObjectName("HeaderModel")
        self.header_device = QLabel(""); self.header_device.setObjectName("HeaderDevice")
        hl.addWidget(self.header_model); hl.addWidget(self.header_device)

        central = QWidget(); cv = QVBoxLayout(central); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(0)
        cv.addWidget(header)
        self.view = LiveView()
        if ICON_PNG.is_file():
            self.view.placeholder = QPixmap(str(ICON_PNG)).scaledToHeight(160, Qt.SmoothTransformation)
        self.view.placeholder_text = "Pick a model and an input, then press Start"
        self.view.clicked.connect(self._on_view_click)
        cv.addWidget(self.view, 1)
        self.controls = ControlsBar()
        cv.addWidget(self.controls)
        self.setCentralWidget(central)

        self.left = LeftPanel()
        sa = QScrollArea(); sa.setWidget(self.left); sa.setWidgetResizable(True); sa.setMinimumWidth(290)
        self.left_dock = QDockWidget("Setup", self); self.left_dock.setObjectName("SetupDock")
        self.left_dock.setWidget(sa)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.right = RightPanel()
        sa2 = QScrollArea(); sa2.setWidget(self.right); sa2.setWidgetResizable(True); sa2.setMinimumWidth(250); sa2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_dock = QDockWidget("Statistics", self); self.right_dock.setObjectName("StatsDock")
        self.right_dock.setWidget(sa2)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        self.log_view = LogView()
        self.log_dock = QDockWidget("Log", self); self.log_dock.setObjectName("LogDock")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

        L, C, R = self.left, self.controls, self.right
        L.start_clicked.connect(self.start); L.stop_clicked.connect(self.stop)
        L.open_model_clicked.connect(self.open_model_dialog)
        L.model_selected.connect(lambda p: p and self.load_model(p))
        L.model_info_clicked.connect(self.show_model_info)
        L.device_changed.connect(self._device_changed)
        L.probe_btn.clicked.connect(self._probe_cameras)
        C.render_changed.connect(self._render_changed)
        C.play_toggled.connect(self.toggle_play)
        C.step.connect(self.step)
        C.home.connect(lambda: self.seek(0))
        C.end.connect(self.seek_end)
        C.seek.connect(self.seek)
        C.speed_changed.connect(self._speed_changed)
        C.sbs_toggled.connect(self._set_sbs)
        C.flag_clicked.connect(self.flag_frame)
        C.snapshot_clicked.connect(self.snapshot)
        R.show_all.toggled.connect(lambda _: self._update_stats())
        R.metrics_btn.clicked.connect(lambda: self.show_metrics())
        R.totals_btn.toggled.connect(lambda _: self._update_stats())
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        a = QAction("Open input…", self, shortcut=QKeySequence("Ctrl+O")); a.triggered.connect(self.open_input_dialog)
        f.addAction(a)
        a = QAction("Open model…", self, shortcut=QKeySequence("Ctrl+M")); a.triggered.connect(self.open_model_dialog)
        f.addAction(a)
        a = QAction("Open output folder", self); a.triggered.connect(self.open_run_dir); f.addAction(a)
        f.addSeparator()
        a = QAction("Quit", self, shortcut=QKeySequence.Quit); a.triggered.connect(self.close); f.addAction(a)
        v = mb.addMenu("&View")
        for dock in (self.left_dock, self.right_dock, self.log_dock):
            v.addAction(dock.toggleViewAction())
        a = QAction("Fit view", self); a.triggered.connect(self.view.reset_view); v.addAction(a)
        h = mb.addMenu("&Help")
        a = QAction("Keyboard shortcuts", self, shortcut=QKeySequence("F1")); a.triggered.connect(self.show_help)
        h.addAction(a)
        a = QAction("Model info", self); a.triggered.connect(self.show_model_info); h.addAction(a)
        a = QAction("About", self); a.triggered.connect(self.show_about); h.addAction(a)

    def _build_shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self); s.setContext(Qt.WindowShortcut); s.activated.connect(fn)
        sc(Qt.Key_Space, self.toggle_play)
        sc(Qt.Key_Left, lambda: self.step(-1))
        sc(Qt.Key_Right, lambda: self.step(1))
        sc(Qt.Key_Home, lambda: self.seek(0))
        sc(Qt.Key_End, self.seek_end)
        sc(Qt.Key_F, self.flag_frame)
        sc(Qt.Key_Z, self._toggle_click_zoom)
        sc(Qt.Key_S, self.snapshot)
        sc(Qt.Key_T, lambda: self.controls.sbs_btn.toggle())
        for n in range(6):
            sc(str(n + 1), lambda n=n: self.controls.toggle_layer(n))

    # ---------------------------------------------------------------------------------------------------- settings
    def _restore(self):
        s, L = self.settings, self.left
        if s.value("geometry"):
            self.restoreGeometry(s.value("geometry"))
        else:
            self.resize(1500, 900)
        if s.value("state"):
            self.restoreState(s.value("state"))
        L.video_path.setText(s.value("input/video", ""))
        L.folder_path.setText(s.value("input/folder", ""))
        L.rtsp_url.setText(s.value("input/rtsp", ""))
        L.camera_combo.setCurrentText(s.value("input/camera", "0"))
        kind = s.value("input/kind", "video")
        if kind in L.radio:
            L.radio[kind].setChecked(True)
        L.gt_path.setText(s.value("gt/path", ""))
        L.gt_format.setCurrentText(s.value("gt/format", "Auto-detect"))
        for k in ("save_video", "side_by_side_save", "counts_strip", "watermark", "export_dets", "include_masks",
                  "summary", "folder_recursive", "folder_video"):
            w = getattr(L, k)
            w.setChecked(s.value(f"out/{k}", w.isChecked(), type=bool))
        default_out = Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or Path.home()) / "GahanValidation"
        L.out_dir.setText(s.value("out/dir", str(default_out)))
        L.device.setCurrentText(s.value("device", "Auto"))
        self.controls.speed.setCurrentText(s.value("speed", "1× (real time)"))

    def _save_settings(self):
        s, L = self.settings, self.left
        s.setValue("geometry", self.saveGeometry()); s.setValue("state", self.saveState())
        s.setValue("input/video", L.video_path.text()); s.setValue("input/folder", L.folder_path.text())
        s.setValue("input/rtsp", L.rtsp_url.text()); s.setValue("input/camera", L.camera_combo.currentText())
        s.setValue("input/kind", L.input_kind())
        s.setValue("gt/path", L.gt_path.text()); s.setValue("gt/format", L.gt_format.currentText())
        for k in ("save_video", "side_by_side_save", "counts_strip", "watermark", "export_dets", "include_masks",
                  "summary", "folder_recursive", "folder_video"):
            s.setValue(f"out/{k}", getattr(L, k).isChecked())
        s.setValue("out/dir", L.out_dir.text()); s.setValue("device", L.device.currentText())
        s.setValue("speed", self.controls.speed.currentText())
        if self.model:
            s.setValue("model/last", str(self.model.path))
            self._save_model_view_state()

    def _model_key(self) -> str:
        d = self.model.description if self.model else None
        return "view/" + safe_name(d.name if d else (self.model.path.stem if self.model else "none"))

    def _save_model_view_state(self):
        if not self.model:
            return
        C = self.controls
        st = {"layers": {k: cb.isChecked() for k, cb in C.layer_boxes.items()},
              "hidden_classes": [c for c, cb in C.class_boxes.items() if not cb.isChecked()],
              "thresholds": C.thresholds()}
        import json
        self.settings.setValue(self._model_key(), json.dumps(st))

    def _load_model_view_state(self) -> dict:
        import json
        try:
            return json.loads(self.settings.value(self._model_key(), "") or "{}")
        except ValueError:
            return {}

    # ---------------------------------------------------------------------------------------------------- model
    def _refresh_model_list(self):
        L = self.left
        L.model_combo.blockSignals(True)
        L.model_combo.clear()
        from core.model_file import pick_variant
        from core.runner_onnx import gpu_available
        dev = self.left.device.currentText().lower()
        paths = pick_variant(list_models(APP_ROOT / "models"),
                             "cpu" if dev == "cpu" or (dev == "auto" and not gpu_available()) else "gpu")
        recent = [p for p in self._recent_models() if Path(p).is_file()]
        seen = set()
        for p in [*map(str, paths), *recent]:
            if p in seen:
                continue
            seen.add(p)
            L.model_combo.addItem(Path(p).name, p)
        if self.model:
            i = L.model_combo.findData(str(self.model.path))
            if i < 0:
                L.model_combo.addItem(self.model.path.name, str(self.model.path)); i = L.model_combo.count() - 1
            L.model_combo.setCurrentIndex(i)
        L.model_combo.blockSignals(False)

    def _recent_models(self) -> list[str]:
        v = self.settings.value("model/recent", [])
        if isinstance(v, str):          # QSettings returns a one-item list as a plain string
            v = [v]
        return [str(p) for p in (v or []) if len(str(p)) > 1]

    def open_model_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open model", str(APP_ROOT / "models"),
                                           "Models (*.onnx *.pt);;All files (*)")
        if p:
            self.load_model(p)

    def load_model(self, path: str, splash: QSplashScreen | None = None) -> bool:
        if self.worker is not None:
            return False
        device = self.left.device.currentText().lower()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if splash:
                splash.showMessage(f"Loading {Path(path).name}…", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
                QApplication.processEvents()
            try:
                model = open_model(path)
            except NeedsDescription as e:
                QApplication.restoreOverrideCursor()
                model = self._ask_description(e)
                QApplication.setOverrideCursor(Qt.WaitCursor)
                if model is None:
                    return False
            runner = make_runner(model, device)
        except (DescriptionError, FileNotFoundError, ImportError, RuntimeError, OSError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot load model", f"{Path(path).name}\n\n{e}")
            self._refresh_model_list()
            return False
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            log.exception("model load failed")
            QMessageBox.critical(self, "Cannot load model", f"{Path(path).name}\n\n{type(e).__name__}: {e}")
            self._refresh_model_list()
            return False
        QApplication.restoreOverrideCursor()
        if self.model:
            self._save_model_view_state()
        self.model, self.runner = model, runner
        self.renderer = Renderer(runner.desc)
        recent = [str(model.path)] + [p for p in self._recent_models() if p != str(model.path)]
        self.settings.setValue("model/recent", recent[:10])
        self.settings.setValue("model/last", str(model.path))
        self._rebuild_controls()
        self._update_header()
        self._refresh_model_list()
        if runner.message:
            QMessageBox.warning(self, "Device", runner.message)
        log.info("model loaded: %s (%s) on %s", model.display_name, model.path.name, runner.device_label)
        self.statusBar().showMessage(f"Model loaded: {model.display_name} · {runner.device_label}", 8000)
        self.current = None
        self.current_pred = None
        self.view.clear()
        self._update_stats()
        return True

    def _ask_description(self, e: NeedsDescription):
        try:
            from .dialogs import DescriptionDialog
        except ImportError:
            QMessageBox.critical(self, "Model description missing", str(e))
            return None
        dlg = DescriptionDialog(e, self)
        return dlg.result_model if dlg.exec() else None

    def _rebuild_controls(self):
        self._building = True
        saved = self._load_model_view_state()
        th = self.runner.decoder.default_thresholds()
        th.update({k: v for k, v in saved.get("thresholds", {}).items() if k in th})
        self.controls.build(self.runner.desc, self.renderer.palette, th, saved)
        self._building = False

    def _update_header(self):
        if not self.model:
            self.header_model.setText("No model loaded"); self.header_device.setText(""); return
        self.header_model.setText(f"Model: {self.model.display_name}")
        self.header_device.setText(self.runner.device_label)
        self.header_device.setProperty("gpu", self.runner.device == "GPU")
        self.header_device.style().unpolish(self.header_device); self.header_device.style().polish(self.header_device)

    def _device_changed(self, _):
        self._refresh_model_list()
        if self.model and self.worker is None:
            self.load_model(str(self.model.path))

    def show_model_info(self):
        if self.model:
            ModelInfoDialog(self.model, self.runner, self).exec()

    def show_about(self):
        AboutDialog(self.model, self.runner, self).exec()

    def show_help(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "Space\tplay / pause\n←  →\tstep one frame (pauses)\nHome / End\tfirst / last frame\n"
            "F\tflag the frame\nZ\tclick-to-zoom on / off (click a box)\nS\tsnapshot (raw + annotated PNG)\n"
            "T\tside-by-side view\n1 – 6\ttoggle layers\nCtrl+O\topen input\nCtrl+M\topen model\nF1\tthis help\n\n"
            "Mouse: wheel zooms, drag pans, double-click fits."))

    # ---------------------------------------------------------------------------------------------------- inputs
    def open_input_dialog(self):
        """Ctrl+O: browse for the input type that is selected (video file or image folder)."""
        L = self.left
        kind = L.input_kind()
        if kind == "folder":
            L._browse_folder(L.folder_path)
        elif kind == "camera":
            L.camera_combo.setFocus()
        elif kind == "rtsp":
            L.rtsp_url.setFocus()
        else:
            L._browse_video(L.video_path)

    def _probe_cameras(self):
        from core.sources import CameraSource
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = CameraSource.probe()
        finally:
            QApplication.restoreOverrideCursor()
        L = self.left
        L.camera_combo.clear()
        L.camera_combo.addItems([str(i) for i in found] or ["0"])
        self.statusBar().showMessage(f"Cameras found: {', '.join(map(str, found)) or 'none'}", 6000)

    # ---------------------------------------------------------------------------------------------------- run
    def _new_run_dir(self, src) -> Path:
        base = Path(self.left.out_dir.text() or Path.home() / "GahanValidation").expanduser()
        d = base / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(Path(src.label).stem if not src.is_live else src.kind)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start(self):
        if self.worker is not None:
            return
        if not self.model:
            QMessageBox.warning(self, "No model", "Load a model first (Model → Open…)."); return
        spec = self.left.input_spec()
        if not spec or spec.endswith(":"):
            QMessageBox.warning(self, "No input", "Choose a video, an image folder, a camera or an RTSP URL."); return
        try:
            src = make_source(spec)
            if isinstance(src, FolderSource):
                src.recursive = self.left.folder_recursive.isChecked()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                src.open()
            finally:
                QApplication.restoreOverrideCursor()
        except SourceError as e:
            QMessageBox.critical(self, "Cannot open input", str(e)); return
        if not self._setup_gt(src):   # before the run folder exists: a GT problem must not leave an empty folder
            src.close()
            return
        try:
            self.run_dir = self._new_run_dir(src)
        except OSError as e:
            src.close(); QMessageBox.critical(self, "Output folder", f"Cannot create the output folder:\n{e}"); return
        self._file_log = logging.FileHandler(self.run_dir / "viewer.log", encoding="utf-8")
        self._file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self._file_log.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._file_log)
        L = self.left
        th = self.controls.thresholds()
        self.session = Session(self.run_dir, self.model, self.runner, src, th, export_dets=L.export_dets.isChecked(),
                               include_masks=L.include_masks.isChecked(), summary=L.summary.isChecked())
        self.session.flags_dir = self.run_dir / "flags"
        savers = []
        if L.save_video.isChecked():
            if isinstance(src, FolderSource):
                savers.append(("folder", ImageFolderSaver(self.run_dir / f"{safe_name(src.path.name)}_annotated")))
                if L.folder_video.isChecked():
                    savers.append(("video", VideoSaver(self.run_dir / f"{safe_name(src.path.name)}_annotated.mp4",
                                                       FOLDER_VIDEO_FPS)))
            else:
                fps = src.fps or (25.0 if not src.is_live else 15.0)
                stem = safe_name(Path(src.label).stem if not src.is_live else src.kind)
                savers.append(("video", LiveVideoSaver(self.run_dir / f"{stem}_annotated.mp4", fps, src)
                               if src.is_live else VideoSaver(self.run_dir / f"{stem}_annotated.mp4", fps)))
        self.source = src
        w = PipelineWorker(self.runner, src, self.renderer, self.session, savers)
        w.speed = SPEEDS[self.controls.speed.currentText()]
        w.set_render_state(th, self._render_options())
        w.overlay = self._save_overlay() if L.side_by_side_save.isChecked() else None
        w.gt_run = self.gt_run
        w.result.connect(self._on_result)
        w.error.connect(self._on_worker_error)
        w.status.connect(lambda m: self.statusBar().showMessage(m, 5000))
        w.finished_run.connect(self._on_finished)
        self.worker = w
        self.left.set_running(True)
        self.controls.set_transport_enabled(True, seekable=src.seekable)
        self.controls.set_playing(True)
        sb = self.controls.seekbar
        sb.setRange(0, max(0, (src.count or 1) - 1)); sb.setValue(0)
        sb.set_markers([])
        self._fps_ema = None; self._last_result_t = None
        log.info("run started: input %s -> %s", src.label, self.run_dir)
        self.statusBar().showMessage(f"Running · output: {self.run_dir}")
        w.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.statusBar().showMessage("Stopping…")

    def _on_worker_error(self, msg: str):
        log.error(msg)
        QMessageBox.critical(self, "Input error", msg)

    def _on_finished(self, sess):
        w, self.worker = self.worker, None
        if w:
            w.wait(3000)
        self.left.set_running(False)
        self.controls.set_playing(False)
        self.controls.set_transport_enabled(self.current is not None, seekable=False)
        if getattr(self, "_file_log", None):
            logging.getLogger().removeHandler(self._file_log); self._file_log.close(); self._file_log = None
        n = sess.frames if sess else 0
        msg = f"Finished: {n} frames processed"
        if sess and sess.bad_frames:
            msg += f", {len(sess.bad_frames)} skipped (see viewer.log)"
        if sess and sess.written:
            msg += " · wrote " + ", ".join(p.name for p in sess.written)
        log.info(msg)
        self.statusBar().showMessage(msg + f" · {self.run_dir}")
        self._update_stats()
        self._after_finish(sess)

    def _after_finish(self, sess):
        if self.gt_run is not None and sess is not None and sess.gt_metrics and not getattr(self, "quiet", False):
            self.show_metrics(sess.gt_metrics)

    def open_run_dir(self):
        d = self.run_dir or Path(self.left.out_dir.text()).expanduser()
        if d and Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # ---------------------------------------------------------------------------------------------------- transport
    def toggle_play(self):
        w = self.worker
        if not w:
            return
        w.set_paused(not w.paused)
        self.controls.set_playing(not w.paused)

    def step(self, d: int):
        if self.worker and self.source and self.source.seekable:
            self.worker.step(d)
            self.controls.set_playing(False)

    def seek(self, i: int):
        if self.worker and self.source and self.source.seekable:
            self.worker.seek(i)

    def seek_end(self):
        if self.source and self.source.count:
            self.seek(self.source.count - 1)

    def _speed_changed(self, speed):
        if self.worker:
            self.worker.speed = speed

    # ---------------------------------------------------------------------------------------------------- render
    def _render_options(self) -> RenderOptions:
        C, L = self.controls, self.left
        lb = C.layer_boxes
        return RenderOptions(
            boxes=lb["boxes"].isChecked() if "boxes" in lb else True,
            attributes=lb["attributes"].isChecked() if "attributes" in lb else True,
            always_attrs=lb["always_attrs"].isChecked() if "always_attrs" in lb else False,
            masks={k[5:]: cb.isChecked() for k, cb in lb.items() if k.startswith("mask:")},
            hidden_classes={c for c, cb in C.class_boxes.items() if not cb.isChecked()},
            counts_strip=L.counts_strip.isChecked(), watermark=L.watermark.isChecked())

    def _render_changed(self):
        if self._building:
            return
        if self.worker:
            self.worker.set_render_state(self.controls.thresholds(), self._render_options())
        self.refresh()

    def _save_overlay(self):
        """Side-by-side image for saving: left = GT (if loaded) or the raw input, right = prediction."""
        def overlay(frame, pred):
            import numpy as np
            opts = self.worker.options if self.worker else self._render_options()
            right = self.renderer.render(frame.image, pred, opts)
            left = self._left_pane(frame, pred, opts)
            return np.hstack([left, right])
        return overlay

    def _left_pane(self, frame, pred, opts):
        """Left pane in side-by-side mode: the GT drawing when GT is loaded, else the raw input."""
        if self.gt_run is None:
            return frame.image
        from core.render import render_gt
        return render_gt(self.renderer, frame.image, self.gt_run.resolved_for(frame, pred.image_size), opts)

    def _left_title(self) -> str:
        return "Ground truth" if self.gt_run is not None else "Input"

    def _set_sbs(self, on: bool):
        self.sbs = on
        self.view.reset_view()
        self.refresh()

    def refresh(self):
        res = self.current
        if res is None or self.runner is None:
            return
        th = self.controls.thresholds()
        try:
            pred = self.runner.decoder.decode(res.raw, th)
        except Exception:  # noqa: BLE001
            log.exception("decode failed"); return
        pred.timing_ms = dict(res.pred.timing_ms)
        self.current_pred = pred
        opts = self._render_options()
        t0 = time.perf_counter()
        self._match_current(res, pred)
        img = self._render_pred(res.frame, pred, opts)
        self._render_ms = (time.perf_counter() - t0) * 1000
        if self.sbs:
            self.view.set_images([self._left_pane(res.frame, pred, opts), img], [self._left_title(), "Prediction"])
        else:
            self.view.set_images([img])
        self._update_stats()

    # ---------------------------------------------------------------------------------------------------- GT
    def _match_current(self, res, pred):
        self.current_gt = self.current_match = None
        if self.gt_run is None:
            return
        from core.matching import match_frame
        r = self.gt_run.resolved_for(res.frame, pred.image_size)
        if r is not None:
            self.current_gt, self.current_match = r, match_frame(pred.boxes, r.boxes)

    def _render_pred(self, frame, pred, opts):
        """Prediction pane; in GT mode TP solid, FP dashed red, misses as orange GT boxes."""
        from core.render import FP_COLOR, MISS_COLOR
        from core.prediction import Box
        m = getattr(self, "current_match", None) if pred is self.current_pred else None
        if self.gt_run is None or m is None:
            return self.renderer.render(frame.image, pred, opts)
        status = {id(pred.boxes[i]): s for i, s in m.pred_status().items()}

        def style(b):
            return {"color": FP_COLOR, "dashed": True, "prefix": "FP "} if status.get(id(b)) == "fp" else None
        img = self.renderer.render(frame.image, pred, opts, box_style=style)
        if opts.boxes:
            for j in m.fn:
                g = self.current_gt.boxes[j]
                if g.cls not in opts.hidden_classes:
                    self.renderer.draw_box(img, Box(g.xyxy, g.cls, 1.0, -1, {}, source="gt"),
                                           RenderOptions(attributes=False), color=MISS_COLOR, prefix="miss ")
        return img

    def _setup_gt(self, src) -> bool:
        from core.gt import GtError, NeedsClassNames
        from core.gt.yolo import read_names_file
        from core.gt_run import GtRun
        self.gt_run = None
        path = self.left.gt_path.text()
        if not path:
            self.right.show_gt(False)
            return True
        fmt = {"Auto-detect": "auto", "CVAT XML 1.1": "cvat",
               "COCO JSON": "coco", "YOLO txt": "yolo"}.get(self.left.gt_format.currentText(), "auto")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                gt_run = GtRun.open(self.runner.desc, path, fmt)
            except NeedsClassNames as e:
                QApplication.restoreOverrideCursor()
                p, _ = QFileDialog.getOpenFileName(self, f"Class names for the YOLO labels in {e.folder.name}",
                                                   str(e.folder), "Names (*.txt *.names *.yaml *.yml);;All files (*)")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                if not p:
                    return False
                gt_run = GtRun.open(self.runner.desc, path, fmt, read_names_file(Path(p)))
        except GtError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Ground truth", str(e))
            return False
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            log.exception("GT load")
            QMessageBox.critical(self, "Ground truth", f"{type(e).__name__}: {e}")
            return False
        QApplication.restoreOverrideCursor()
        missing = gt_run.unresolved_labels()
        if missing:
            from core.gt.aliases import mask_targets
            from .dialogs import LabelMappingDialog
            labels = gt_run.gt.labels()
            counts = {k: labels["box"][k] + labels["line"][k] + labels["polygon"][k] for k in missing}
            dlg = LabelMappingDialog(missing, counts, self.runner.desc.class_names + mask_targets(self.runner.desc),
                                     gt_run.mapper.gt_dir, self)
            if not dlg.exec():
                return False
            gt_run.mapper.set_user(dlg.choices())
        self.gt_run = gt_run
        self.right.show_gt(True)
        log.info("ground truth: %s, %d frames (%s)", gt_run.gt.format, len(gt_run.gt), path)
        return True

    def _update_gt_stats(self):
        R = self.right
        if self.gt_run is None:
            return
        m, r = getattr(self, "current_match", None), getattr(self, "current_gt", None)
        if r is None:
            txt = "<span style='color:#9AA0AA'>no GT for this frame</span>"
        else:
            tp = len(m.pairs); fp = len(m.fp); fn = len(m.fn)
            txt = (f"<b>TP {tp} &nbsp; <span style='color:#F87171'>FP {fp}</span> &nbsp; "
                   f"<span style='color:#FB923C'>miss {fn}</span></b>")
            rows = [f"{html_escape(c)}: {st['tp']} / {st['fp']} / {st['fn']}" for c, st in m.per_class.items()
                    if st["tp"] or st["fp"] or st["fn"]]
            if rows:
                txt += "<br><span style='color:#9AA0AA'>class: TP / FP / miss</span><br>" + "<br>".join(rows)
            n = sum(a["n"] for a in m.attr.values()); ag = sum(a["agree"] for a in m.attr.values())
            if n:
                txt += f"<br>attr agree {ag}/{n} = {ag / n:.0%}"
            if r.excluded:
                txt += "<br><span style='color:#9AA0AA'>excluded: " + ", ".join(
                    f"{html_escape(k)} {v}" for k, v in r.excluded.items()) + "</span>"
        run = self.gt_run.running
        p = run["tp"] / (run["tp"] + run["fp"]) if run["tp"] + run["fp"] else float("nan")
        rc = run["tp"] / (run["tp"] + run["fn"]) if run["tp"] + run["fn"] else float("nan")
        txt += (f"<br><br>running (all classes): precision {p:.3f} · recall {rc:.3f}"
                f"<br><span style='color:#9AA0AA'>{len(self.gt_run.evaluator.images)} frames with GT</span>")
        R.gt_info.setText(txt)

    def _gt_match_for(self, box):
        m = getattr(self, "current_match", None)
        if m is None or self.current_pred is None:
            return None
        from types import SimpleNamespace
        from core.gt.normalize_attrs import NOT_COMPARABLE
        idx = next((i for i, b in enumerate(self.current_pred.boxes) if b is box), None)
        pair = m.gt_of_pred().get(idx)
        if pair is None:
            return None
        g = self.current_gt.boxes[pair.gt]
        attrs = {k: (f"n/c ({g.raw_attributes})" if v is NOT_COMPARABLE else v) for k, v in g.attributes.items()}
        return SimpleNamespace(xyxy=g.xyxy, label=g.label, iou=pair.iou, attributes=attrs)

    def show_metrics(self, metrics: dict | None = None):
        if self.gt_run is None:
            return
        from .dialogs import MetricsDialog
        if metrics is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                metrics = self.gt_run.metrics(self.controls.thresholds())
            finally:
                QApplication.restoreOverrideCursor()

        def export():
            from core.metrics import metrics_to_files
            if self.session is None or self.run_dir is None:
                return
            files = metrics_to_files(metrics, self.run_dir, self.session.provenance())
            self.statusBar().showMessage("Wrote " + ", ".join(str(f) for f in files), 8000)
        MetricsDialog(metrics, self.renderer.palette, export, self).exec()

    def _on_result(self, res):
        self.current = res
        now = time.monotonic()
        if self._last_result_t is not None:
            dt = now - self._last_result_t
            self._fps_ema = 1 / dt if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 / max(dt, 1e-6)
        self._last_result_t = now
        self.refresh()
        sb = self.controls.seekbar
        if not sb.isSliderDown():
            sb.blockSignals(True); sb.setValue(res.position); sb.blockSignals(False)
        src = self.source
        if src is not None:
            if isinstance(src, FolderSource):
                self.controls.time_label.setText(f"{res.position + 1} / {src.count}")
            elif src.is_live:
                self.controls.time_label.setText(f"{fmt_time(res.frame.timestamp_s)}  live")
            else:
                total = (src.count / src.fps) if src.count and src.fps else None
                self.controls.time_label.setText(f"{fmt_time(res.frame.timestamp_s)} / {fmt_time(total)}")
        if self.worker:
            self.worker.ui_ready.set()

    def _update_stats(self):
        R = self.right
        pred = self.current_pred
        opts = self._render_options() if self.model else RenderOptions()
        pal = self.renderer.palette if self.renderer else None
        counts = {}
        if pred:
            for b in pred.boxes:
                if b.cls not in opts.hidden_classes:
                    counts[b.cls] = counts.get(b.cls, 0) + 1
        classes = self.runner.desc.class_names if self.runner else []
        names = classes if R.show_all.isChecked() else [c for c in classes if c in counts]
        names += [c for c in counts if c not in classes]
        R.table.set_rows([(c, counts.get(c, 0), pal[c] if pal else None) for c in names])
        R.total.setText(f"total {sum(counts.values())}")
        sess = self.session
        if R.totals_btn.isChecked() and sess:
            tot = sess.class_totals
            R.totals.set_rows([(c, tot[c], pal[c] if pal else None) for c in sorted(
                tot, key=lambda c: classes.index(c) if c in classes else 999)])
            R.totals_note.setText(f"{sess.frames} frames · {sess.frames_with_det} with ≥1 detection")
        res = self.current
        if res is None:
            R.frame_info.setText("—"); return
        t = res.pred.timing_ms
        src = self.source
        idx = f"#{res.frame.index}" + (f" / {src.count}" if src is not None and src.count else "")
        where = res.frame.name if isinstance(src, FolderSource) else fmt_time(res.frame.timestamp_s)
        if len(where) > 34:
            where = "…" + where[-33:]
        lines = [f"<b>{idx}</b> &nbsp; {where}",
                 f"pre {t.get('preprocess', 0):.0f} · inf {t.get('inference', 0):.0f} · post "
                 f"{t.get('postprocess', 0):.0f} · render {getattr(self, '_render_ms', 0):.0f} ms",
                 f"{self._fps_ema or 0:.1f} fps (display) · {self.runner.device_label if self.runner else ''}"]
        if src is not None and src.is_live:
            lines.append(f"dropped {res.dropped} · {getattr(src, 'status', '')}"
                         + (f" · reconnects {src.reconnects}" if hasattr(src, "reconnects") else ""))
        if pred and pred.tags:
            lines.append(" · ".join(f"{k}: {a.value} {a.prob:.2f}" for k, a in pred.tags.items()))
        R.frame_info.setText("<br>".join(lines))
        self._update_gt_stats()

    # ---------------------------------------------------------------------------------------------------- clicks
    def _toggle_click_zoom(self):
        self.click_zoom = not self.click_zoom
        self.statusBar().showMessage(f"Click a box to zoom: {'on' if self.click_zoom else 'off'}", 3000)

    def _on_view_click(self, pane, x, y):
        if not self.click_zoom or self.current_pred is None:
            return
        from core.render import box_at
        b = box_at(self.current_pred, x, y, self._render_options())
        if b is None:
            return
        try:
            from .dialogs import ZoomDialog
        except ImportError:
            return
        if self.worker and not self.worker.paused:
            self.toggle_play()
        heads = self.runner.decoder.all_heads(self.current.raw, b.slot) if hasattr(self.runner.decoder, "all_heads") else None
        ZoomDialog(self.current.frame.image, b, self.renderer, self._render_options(), self._gt_match_for(b),
                   self, all_heads=heads).exec()

    # ---------------------------------------------------------------------------------------------------- flags etc.
    def _annotated_current(self):
        """What the user sees: the annotated frame (side by side when shown)."""
        import numpy as np
        res, pred = self.current, self.current_pred
        opts = self._render_options()
        img = self._render_pred(res.frame, pred, opts)
        if self.sbs:
            img = np.hstack([self._left_pane(res.frame, pred, opts), img])
        return img

    def flag_frame(self):
        if self.current is None or self.current_pred is None or self.run_dir is None:
            return
        from core.session import FlagStore, flag_categories
        from .dialogs import FlagDialog
        was_playing = self.worker is not None and not self.worker.paused
        if was_playing:
            self.toggle_play()
        res = self.current
        ann = self._render_pred(res.frame, self.current_pred, self._render_options())
        dlg = FlagDialog(res.frame.image, ann, flag_categories(self.runner.desc),
                         f"#{res.frame.index} {res.frame.name}", self)
        if dlg.exec():
            cat, note, box = dlg.values()
            if self.session is None:
                return
            if getattr(self.session, "flag_store", None) is None:
                self.session.flag_store = FlagStore(self.run_dir, self.session.provenance())
            item = self.session.flag_store.add(res.frame.index, res.frame.timestamp_s, res.frame.name, cat, note, box,
                                               res.frame.image, self._annotated_current())
            self.session.flag_items.append(item)
            self.controls.seekbar.set_markers([f["frame"] for f in self.session.flag_items])
            log.info("flagged frame %s: %s %s", res.frame.index, cat, note)
            self.view.flash(f"⚑  Flag saved · frame {res.frame.index} · {cat}")
            if self.worker is None and self.session.summary:   # run already finished: refresh the report
                from core.report import write_summary
                write_summary(self.session)
            self.statusBar().showMessage(f"Flag saved: frame {res.frame.index} · {cat} · {self.run_dir / 'flags'}", 6000)
        if was_playing and self.worker is not None:
            self.toggle_play()

    def snapshot(self):
        if self.current is None or self.current_pred is None:
            return
        import cv2
        d = (self.run_dir or Path(self.left.out_dir.text()).expanduser()) / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        res = self.current
        stem = f"frame_{res.frame.index:06d}"
        k = 1
        while (d / f"{stem}_raw.png").exists():
            k += 1; stem = f"frame_{res.frame.index:06d}_{k}"
        cv2.imwrite(str(d / f"{stem}_raw.png"), res.frame.image)
        cv2.imwrite(str(d / f"{stem}_annotated.png"), self._annotated_current())
        log.info("snapshot %s", d / stem)
        self.view.flash(f"📷  Snapshot saved · {stem}_raw.png / _annotated.png")
        self.statusBar().showMessage(f"Snapshot saved: {d / stem}_raw.png / _annotated.png", 6000)

    # ---------------------------------------------------------------------------------------------------- close
    def closeEvent(self, e):
        self._save_settings()
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
        logging.getLogger().removeHandler(self.log_handler)
        super().closeEvent(e)


def make_splash() -> QSplashScreen:
    pm = QPixmap(520, 300)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(Qt.GlobalColor.black)
    from PySide6.QtGui import QColor
    p.setBrush(QColor("#16181D")); p.setPen(QColor("#2A2E36"))
    p.drawRoundedRect(0, 0, 519, 299, 10, 10)
    icon = QPixmap(str(ICON_PNG))
    if not icon.isNull():
        p.drawPixmap(30, 60, icon.scaledToHeight(150, Qt.SmoothTransformation))
    p.setPen(QColor("#E6E6E6"))
    f = QFont(); f.setPointSize(22); f.setBold(True); p.setFont(f)
    p.drawText(200, 125, "Validation Viewer")
    f.setPointSize(12); f.setBold(False); p.setFont(f)
    p.setPen(QColor("#9AA0AA"))
    p.drawText(202, 158, "Gahan AI Pvt. Ltd.")
    f.setPointSize(9); p.setFont(f)
    p.drawText(202, 182, f"version {app_version()}")
    p.end()
    return QSplashScreen(pm)


def run_app(argv) -> int:
    if sys.platform == "win32":
        try:  # own taskbar group + icon instead of python.exe's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GahanAI.ValidationViewer")
        except Exception:  # noqa: BLE001
            pass
    if sys.platform.startswith("linux"):
        # the GTK3 platform theme crashes on some GNOME setups (missing xsettings key); we style everything ourselves
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(argv)
    app.setOrganizationName(ORG); app.setApplicationName(APP); app.setApplicationVersion(app_version())
    app.setDesktopFileName("gahan-validation-viewer")
    apply_theme(app)
    splash = make_splash()
    splash.show(); app.processEvents()
    w = MainWindow()
    last = w.settings.value("model/last", "")
    if not last:
        models = list_models(APP_ROOT / "models")
        last = str(models[0]) if models else ""
    if last and Path(last).is_file():
        w.load_model(last, splash)
    w.show()
    splash.finish(w)
    return app.exec()
