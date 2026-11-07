"""Session outputs: annotated video / image folder (always available), detections export and summary (opt-in), flags."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class _quiet_opencv:
    """Silence OpenCV's own log for a moment (the expected H.264-encoder miss prints two ERROR lines)."""

    def __enter__(self):
        self.prev = None
        try:
            self.prev = cv2.getLogLevel()
            cv2.setLogLevel(0)  # 0 = silent
        except AttributeError:
            pass

    def __exit__(self, *a):
        if self.prev is not None:
            cv2.setLogLevel(self.prev)


class VideoSaver:
    """Annotated mp4 at the source fps and frame size; opened on the first frame. Tries avc1 (H.264), then mp4v."""

    def __init__(self, path: Path, fps: float, codecs=("avc1", "mp4v")):
        self.path, self.fps, self.codecs = Path(path), float(fps or 25.0), codecs
        self.writer = None
        self.codec = None
        self.size = None
        self.frames = 0

    def write(self, img: np.ndarray):
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.size = (img.shape[1], img.shape[0])
            for c in self.codecs:
                with _quiet_opencv():  # a missing H.264 encoder is expected; do not spam the console
                    w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*c), self.fps, self.size)
                if w.isOpened():
                    self.writer, self.codec = w, c
                    break
                w.release()
            if self.writer is None:
                raise RuntimeError(f"no working video codec for {self.path} (tried {', '.join(self.codecs)})")
            log.info("saving video %s (%s, %.3f fps, %dx%d)", self.path, self.codec, self.fps, *self.size)
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        self.writer.write(img)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release(); self.writer = None


class LiveVideoSaver(VideoSaver):
    """Camera/RTSP: the writer opens at the measured fps once enough frames have arrived."""

    def __init__(self, path, fps, source, warmup=30):
        super().__init__(path, fps)
        self.source, self.warmup, self.buffer = source, warmup, []

    def write(self, img):
        if self.writer is None and len(self.buffer) < self.warmup:
            self.buffer.append(img); return
        if self.writer is None:
            self.fps = getattr(self.source, "measured_fps", None) or self.fps
            for b in self.buffer:
                super().write(b)
            self.buffer = []
        super().write(img)

    def close(self):
        if self.buffer:
            self.fps = getattr(self.source, "measured_fps", None) or self.fps
            for b in self.buffer:
                super().write(b)
            self.buffer = []
        super().close()


class ImageFolderSaver:
    """Annotated images under out_dir with the same relative names as the input folder."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.frames = 0

    def write(self, img: np.ndarray, rel_name: str):
        p = self.out_dir / rel_name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(p), img):
            raise RuntimeError(f"could not write {p}")
        self.frames += 1

    def close(self):
        pass


class Session:
    """Per-run bookkeeping: latency stats, per-class totals, attribute distributions, opt-in exports."""

    def __init__(self, out_dir: Path, model, runner, source, thresholds: dict, export_dets: bool = False,
                 include_masks: bool = False, summary: bool = False):
        import time as _t
        from .settings import app_version
        self.out_dir = Path(out_dir)
        self.model, self.runner, self.source = model, runner, source
        self.thresholds = dict(thresholds)
        self.export_dets, self.include_masks, self.summary = export_dets, include_masks, summary
        self.started = _t.strftime("%Y-%m-%d %H:%M:%S")
        self.app_version = app_version()
        self.latency: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": [], "render": []}
        self.class_totals: dict[str, int] = {}
        self.frames_with_det = 0
        self.frames = 0
        self.bad_frames: list[str] = []
        self.attr_dist: dict[str, dict[str, dict[str, int]]] = {}  # class -> head -> value -> n
        self.flag_items: list[dict] = []
        self.gt_metrics: dict | None = None
        self.written: list[Path] = []
        self._jsonl = self._csv = None
        self._csv_heads: list[str] = []
        self._t0 = _t.time()
        self.wall_s = 0.0
        if export_dets:
            self._open_exports()
        self._write_run_info()

    # provenance written into every output file
    def provenance(self) -> dict:
        d = self.model.description
        return {"app": "Gahan AI Validation Viewer", "app_version": self.app_version,
                "model_name": d.name if d else self.model.path.stem, "model_version": d.version if d else "",
                "model_sha256": self.model.sha256, "model_file": self.model.path.name,
                "device": self.runner.device_label, "input": self.source.label, "started": self.started,
                "thresholds": self.thresholds}

    def _open_exports(self):
        import csv
        import json
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / "detections.jsonl"
        self._jsonl = open(p, "w", encoding="utf-8")
        self._jsonl.write(json.dumps({"header": self.provenance()}) + "\n")
        self.written.append(p)
        d = self.runner.desc
        self._csv_heads = [h.name for h in d.attributes.heads] if d and d.attributes else []
        p = self.out_dir / "detections.csv"
        self._csv_f = open(p, "w", newline="", encoding="utf-8")
        self._csv_f.write("# " + "; ".join(f"{k}={v}" for k, v in self.provenance().items()) + "\n")
        self._csv = csv.writer(self._csv_f)
        cols = ["frame", "timestamp_s", "name", "class", "conf", "x1", "y1", "x2", "y2", "slot", "source"]
        for h in self._csv_heads:
            cols += [h, f"{h}_prob"]
        self._csv.writerow(cols)
        self.written.append(p)

    def add(self, frame, pred):
        import json
        self.frames += 1
        for k in self.latency:
            if k in pred.timing_ms:
                self.latency[k].append(pred.timing_ms[k])
        if pred.boxes:
            self.frames_with_det += 1
        for b in pred.boxes:
            self.class_totals[b.cls] = self.class_totals.get(b.cls, 0) + 1
            cd = self.attr_dist.setdefault(b.cls, {})
            for h, a in b.attributes.items():
                hd = cd.setdefault(h, {})
                hd[a.value] = hd.get(a.value, 0) + 1
        if self._jsonl:
            self._jsonl.write(json.dumps(pred.to_json(self.include_masks)) + "\n")
            for b in pred.boxes:
                row = [pred.frame_index, "" if pred.timestamp_s is None else round(pred.timestamp_s, 4), pred.name,
                       b.cls, round(b.conf, 4), *(round(v, 2) for v in b.xyxy), b.slot, b.source]
                for h in self._csv_heads:
                    a = b.attributes.get(h)
                    row += [a.value, round(a.prob, 4)] if a else ["", ""]
                self._csv.writerow(row)

    def bad_frame(self, frame):
        self.bad_frames.append(str(getattr(frame, "name", frame)))

    def _write_run_info(self, final: bool = False):
        """run_info.json in every run folder: provenance for all outputs there (videos and images included)."""
        import json
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            info = {**self.provenance(), "outputs": {"export_detections": self.export_dets,
                                                      "include_masks": self.include_masks, "summary": self.summary}}
            if final:
                info.update({"frames_processed": self.frames, "frames_skipped": len(self.bad_frames),
                             "wall_s": round(self.wall_s, 2), "flags": len(self.flag_items),
                             "files": [p.name for p in self.written]})
            (self.out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        except OSError:
            log.exception("run_info.json")

    def finish(self):
        import time as _t
        self.wall_s = _t.time() - self._t0
        if self._jsonl:
            self._jsonl.close(); self._csv_f.close(); self._jsonl = None
        if self.summary:
            from .report import write_summary
            self.written.append(write_summary(self))
        self._write_run_info(final=True)


FLAG_BASE_CATEGORIES = ["Missed object", "False detection", "Wrong class", "Wrong attribute"]


def flag_categories(desc) -> list[str]:
    """Generic categories plus one per mask task of the loaded model, then Other."""
    cats = list(FLAG_BASE_CATEGORIES) if desc is None or desc.detection or desc.extra_boxes else []
    if desc is not None:
        cats += [m.name.replace("_", " ").capitalize() for m in desc.masks]
        cats += [t.name.replace("_", " ").capitalize() for t in desc.tags]
    return cats + ["Other"]


class FlagStore:
    """<run_dir>/flags/flags.csv + frame_<n>_raw.png / frame_<n>_annotated.png."""
    COLUMNS = ["frame", "timestamp_s", "file", "category", "note", "box_x1", "box_y1", "box_x2", "box_y2",
               "model_name", "model_version", "model_sha256", "app_version", "flagged_at"]

    def __init__(self, run_dir: Path, provenance: dict):
        self.dir = Path(run_dir) / "flags"
        self.prov = provenance
        self.items: list[dict] = []

    def add(self, frame_index: int, timestamp_s, name: str, category: str, note: str, box, raw_img, annotated_img) -> dict:
        import csv
        import time as _t
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "flags.csv"
        new = not p.exists()
        stem = f"frame_{frame_index:06d}"
        k = 1
        while (self.dir / f"{stem}_raw.png").exists():  # several flags on one frame
            k += 1
            stem = f"frame_{frame_index:06d}_{k}"
        cv2.imwrite(str(self.dir / f"{stem}_raw.png"), raw_img)
        cv2.imwrite(str(self.dir / f"{stem}_annotated.png"), annotated_img)
        row = {"frame": frame_index, "timestamp_s": "" if timestamp_s is None else round(timestamp_s, 3), "file": name,
               "category": category, "note": note,
               **{f"box_{c}": ("" if box is None else round(v, 1)) for c, v in zip(("x1", "y1", "x2", "y2"), box or (0,) * 4)},
               "model_name": self.prov.get("model_name"), "model_version": self.prov.get("model_version"),
               "model_sha256": self.prov.get("model_sha256"), "app_version": self.prov.get("app_version"),
               "flagged_at": _t.strftime("%Y-%m-%d %H:%M:%S")}
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        item = dict(row, raw_png=self.dir / f"{stem}_raw.png", annotated_png=self.dir / f"{stem}_annotated.png")
        self.items.append(item)
        return item
