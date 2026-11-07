"""QThread pipeline: read -> infer -> decode -> (render + save) -> hand the result to the UI.

The UI never blocks: it receives FrameResult objects and re-renders them itself (so toggles and sliders act on
the cached raw outputs without re-running inference). Saving always processes every frame; display frames are
dropped when the UI is still busy with the previous one.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from .render import Renderer, RenderOptions
from .sources import Frame, Source, SourceError

log = logging.getLogger(__name__)

SPEEDS = {"0.25×": 0.25, "0.5×": 0.5, "1× (real time)": 1.0, "2×": 2.0, "4×": 4.0, "Max": None}


@dataclass
class FrameResult:
    frame: Frame
    raw: object          # RawResult (ONNX) or the Ultralytics equivalent; decoder.decode(raw, thresholds) works on it
    pred: object         # Prediction at the thresholds in effect when it was produced
    position: int        # index for the seek bar
    dropped: int = 0     # live sources: frames dropped so far


class RawCache:
    """The last N frames' raw outputs, so stepping back re-renders without inference."""

    def __init__(self, n: int = 16):
        self.n, self.d = n, OrderedDict()

    def get(self, i):
        if i in self.d:
            self.d.move_to_end(i)
            return self.d[i]
        return None

    def put(self, i, v):
        self.d[i] = v
        self.d.move_to_end(i)
        while len(self.d) > self.n:
            self.d.popitem(last=False)

    def clear(self):
        self.d.clear()


class PipelineWorker(QThread):
    result = Signal(object)          # FrameResult
    status = Signal(str)
    error = Signal(str)
    finished_run = Signal(object)    # Session (or None)

    def __init__(self, runner, source: Source, renderer: Renderer, session=None, savers=None, parent=None):
        super().__init__(parent)
        self.runner, self.source, self.renderer = runner, source, renderer
        self.session = session
        self.savers = savers or []           # [(kind, saver)]; kind "video" or "folder"
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop = False
        self._paused = False
        self._step = 0                       # +1 / -1 single steps while paused
        self._seek_to: int | None = None
        self.speed: float | None = 1.0      # None = as fast as possible
        self.thresholds = runner.decoder.default_thresholds()
        self.options = RenderOptions()
        self.overlay = None                  # callable(frame, pred) -> img for saving (e.g. side-by-side); else renderer
        self.ui_ready = threading.Event()
        self.ui_ready.set()
        self.cache = RawCache()
        self.position = 0
        self.processed = 0
        self.bad = 0
        self.on_seek = []                    # callbacks run in the worker thread after a seek
        self.gt_run = None                   # GtRun: per-frame matching + folder metrics

    # ---- control (called from the UI thread) ---------------------------------------------------------------------
    def set_paused(self, p: bool):
        with self._wake:
            self._paused = p; self._wake.notify_all()

    @property
    def paused(self) -> bool:
        return self._paused

    def step(self, d: int):
        with self._wake:
            self._paused = True
            self._step = d; self._wake.notify_all()

    def seek(self, i: int):
        with self._wake:
            self._seek_to = int(i); self._wake.notify_all()

    def stop(self):
        with self._wake:
            self._stop = True; self._wake.notify_all()

    def set_render_state(self, thresholds: dict, options: RenderOptions):
        with self._lock:
            self.thresholds = dict(thresholds)
            self.options = options

    # ---- thread ---------------------------------------------------------------------------------------------------
    def _process(self, fr: Frame, save: bool) -> FrameResult | None:
        cached = self.cache.get(fr.index)
        try:
            raw = cached if cached is not None else self.runner.infer_raw(
                fr.image, frame_index=fr.index, timestamp_s=fr.timestamp_s, name=fr.name)
            self.cache.put(fr.index, raw)
            with self._lock:
                th, opts, overlay = dict(self.thresholds), self.options, self.overlay
            pred = self.runner.decoder.decode(raw, th)
            if save and self.gt_run is not None:
                try:
                    self.gt_run.process(fr, raw, pred, self.runner.decoder)
                except Exception:  # noqa: BLE001 - a GT problem must not stop the run
                    log.exception("GT matching failed on %s", fr.name)
            if save and self.savers:
                t0 = time.perf_counter()
                img = overlay(fr, pred) if overlay else self.renderer.render(fr.image, pred, opts)
                pred.timing_ms["render"] = (time.perf_counter() - t0) * 1000
                for kind, s in self.savers:
                    s.write(img, fr.name) if kind == "folder" else s.write(img)
        except Exception as e:  # noqa: BLE001 - never crash on one bad frame
            self.bad += 1
            log.exception("frame %s skipped: %s", fr.name, e)
            if self.session:
                self.session.bad_frame(fr)
            return None
        if save and self.session:
            self.session.add(fr, pred)
        self.processed += 1
        return FrameResult(fr, raw, pred, fr.index, getattr(self.source, "dropped", 0))

    def _emit(self, res: FrameResult | None, force: bool = False):
        if res is None:
            return
        if force or self.ui_ready.is_set():
            self.ui_ready.clear()
            self.result.emit(res)

    def run(self):
        src = self.source
        saved_upto = -1                       # highest frame index saved: each frame saved once, in play order
        next_due = None
        try:
            while True:
                with self._wake:
                    while self._paused and not self._stop and self._step == 0 and self._seek_to is None:
                        self._wake.wait(0.1)
                    if self._stop:
                        break
                    step, seek = self._step, self._seek_to
                    self._step, self._seek_to = 0, None
                if seek is not None and src.seekable:
                    src.seek(seek)
                    for cb in self.on_seek:
                        cb()
                    next_due = None
                    fr = src.read()
                    if fr is None:
                        continue
                    save = fr.index > saved_upto
                    res = self._process(fr, save=save)
                    if save and res:
                        saved_upto = fr.index
                    self.position = fr.index
                    self._emit(res, force=True)
                    continue
                if step and src.seekable:
                    target = self.position + step
                    if step < 0 or target != getattr(src, "pos", target):
                        if target < 0 or (src.count and target >= src.count):
                            continue
                        src.seek(target)
                    fr = src.read()
                    if fr is None:
                        continue
                    save = fr.index > saved_upto
                    res = self._process(fr, save=save)
                    if save and res:
                        saved_upto = fr.index
                    self.position = fr.index
                    self._emit(res, force=True)
                    continue
                # ---- playing ----
                speed = self.speed
                saving = bool(self.savers) or bool(self.session and (self.session.export_dets or self.session.summary))
                if not src.is_live and speed is not None and next_due is not None and not saving:
                    # real-time pacing without saving: skip frames we are late for
                    late = time.monotonic() - next_due
                    fps = (src.fps or 25.0) * speed
                    skip = int(late * fps)
                    if skip > 0:
                        src.skip(skip)          # dropped display frames (nothing is being saved)
                        next_due += skip / fps
                fr = src.read()
                if fr is None:
                    self.status.emit("end of input")
                    break
                save = fr.index > saved_upto or src.is_live
                res = self._process(fr, save=save)
                if save and res:
                    saved_upto = fr.index
                self.position = fr.index
                self._emit(res)
                if not src.is_live and speed is not None and src.fps:
                    period = 1.0 / (src.fps * speed)
                    now = time.monotonic()
                    next_due = (next_due or now) + period
                    wait = next_due - now
                    if wait > 0:
                        with self._wake:
                            self._wake.wait(wait)
                    elif wait < -1.0:
                        next_due = now  # far behind (e.g. saving at full quality): stop trying to catch up
        except SourceError as e:
            self.error.emit(str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("pipeline stopped")
            self.error.emit(f"pipeline stopped: {e}")
        finally:
            try:
                src.close()
            finally:
                for _, s in self.savers:
                    try:
                        s.close()
                    except Exception:  # noqa: BLE001
                        log.exception("closing saver")
                sess = self.session
                if sess and self.gt_run is not None:
                    try:
                        from .metrics import metrics_to_files
                        with self._lock:
                            th = dict(self.thresholds)
                        sess.gt_metrics = self.gt_run.metrics(th)
                        if sess.export_dets or sess.summary:
                            sess.written += metrics_to_files(sess.gt_metrics, sess.out_dir, sess.provenance())
                    except Exception as e:  # noqa: BLE001
                        log.exception("GT metrics")
                        self.error.emit(f"computing GT metrics failed: {e}")
                if sess:
                    try:
                        sess.finish()
                    except Exception as e:  # noqa: BLE001
                        log.exception("session finish")
                        self.error.emit(f"writing session outputs failed: {e}")
                self.finished_run.emit(sess)
