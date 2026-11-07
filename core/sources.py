"""Frame sources with one interface: video file, image folder, USB camera, RTSP stream.

    src.open(); frame = src.read()  ->  Frame(image_bgr, index, timestamp_s, name) or None at the end
    src.seek(i); src.count; src.fps; src.is_live; src.close()

Live sources (camera, RTSP) read in a background thread and always hand out the newest frame, so frames are
dropped when inference is slower than the source.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm", ".ts"}


class SourceError(RuntimeError):
    pass


@dataclass
class Frame:
    image: np.ndarray
    index: int
    timestamp_s: float | None
    name: str


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class Source:
    is_live = False
    seekable = False
    count: int | None = None
    fps: float | None = None
    kind = "source"

    def open(self): ...

    def read(self) -> Frame | None: ...

    def seek(self, i: int):
        raise SourceError(f"{self.kind} source is not seekable")

    def skip(self, n: int):
        """Drop the next n frames (real-time catch-up)."""
        for _ in range(max(0, n)):
            if self.read() is None:
                break

    def close(self): ...

    @property
    def label(self) -> str:
        return self.kind

    def __enter__(self):
        self.open(); return self

    def __exit__(self, *a):
        self.close()


class VideoSource(Source):
    kind, seekable = "video", True

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = None
        self.pos = 0

    @property
    def label(self):
        return self.path.name

    def open(self):
        if not self.path.is_file():
            raise SourceError(f"video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise SourceError(f"cannot open video (unsupported codec or corrupt file): {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.count = n if n > 0 else None
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, f = self.cap.read()
        if not ok:
            raise SourceError(f"video has no readable frames: {self.path}")
        self._pending = f
        self.pos = 0

    def read(self):
        if self._pending is not None:
            f, self._pending = self._pending, None
        else:
            ok, f = self.cap.read()
            if not ok:
                return None
        i = self.pos
        self.pos += 1
        return Frame(f, i, i / self.fps, f"{self.path.name}#{i}")

    def seek(self, i: int):
        i = max(0, min(int(i), (self.count or i + 1) - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        self._pending = None
        self.pos = i

    def skip(self, n: int):
        """Drop the next n frames: grab() for short gaps (no colour conversion), a seek for long ones."""
        if n <= 0:
            return
        if n > 30:
            self.seek(self.pos + n)
            return
        if self._pending is not None:
            self._pending = None
            self.pos += 1
            n -= 1
        for _ in range(n):
            if not self.cap.grab():
                break
            self.pos += 1

    def close(self):
        if self.cap is not None:
            self.cap.release(); self.cap = None


class FolderSource(Source):
    kind, seekable = "folder", True

    def __init__(self, path: str | Path, recursive: bool = False):
        self.path = Path(path)
        self.recursive = recursive
        self.files: list[Path] = []
        self.pos = 0

    @property
    def label(self):
        return self.path.name

    def open(self):
        if not self.path.is_dir():
            raise SourceError(f"folder not found: {self.path}")
        it = self.path.rglob("*") if self.recursive else self.path.iterdir()
        self.files = sorted((p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                            key=lambda p: natural_key(str(p.relative_to(self.path))))
        if not self.files:
            raise SourceError(f"no images (jpg, jpeg, png, bmp) in {self.path}")
        self.count = len(self.files)
        self.fps = None
        self.pos = 0

    def rel_name(self, i: int) -> str:
        return str(self.files[i].relative_to(self.path))

    def read(self):
        while self.pos < len(self.files):
            i = self.pos
            self.pos += 1
            img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("unreadable image skipped: %s", self.files[i])
                self.skipped = getattr(self, "skipped", 0) + 1
                continue
            return Frame(img, i, None, self.rel_name(i))
        return None

    def seek(self, i: int):
        self.pos = max(0, min(int(i), len(self.files) - 1))

    def skip(self, n: int):
        self.pos += max(0, n)


class _NullCap:
    def release(self):
        pass


class _LiveSource(Source):
    """Background reader that keeps only the newest frame."""
    is_live = True

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest = None
        self._stop = threading.Event()
        self._thread = None
        self._index = 0
        self._t0 = None
        self.dropped = 0
        self.status = "idle"
        self._fps_times: list[float] = []

    def _capture(self) -> cv2.VideoCapture: ...

    def _reader(self, cap):
        while not self._stop.is_set():
            ok, f = cap.read()
            if not ok:
                if not self._on_read_fail():
                    break
                cap.release()
                cap = self._capture_or_none()
                if cap is None:
                    cap = _NullCap()
                    break
                continue
            now = time.monotonic()
            with self._cond:
                if self._latest is not None:
                    self.dropped += 1
                self._t0 = self._t0 or now
                self._latest = Frame(f, self._index, now - self._t0, f"{self.label}#{self._index}")
                self._index += 1
                self._fps_times = (self._fps_times + [now])[-60:]
                self._cond.notify_all()
        cap.release()
        with self._cond:
            self.status = "stopped" if self._stop.is_set() else "ended"
            self._cond.notify_all()

    def _capture_or_none(self):
        return None

    def _on_read_fail(self) -> bool:
        return False

    @property
    def measured_fps(self) -> float | None:
        t = self._fps_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 10 and t[-1] > t[0] else None

    def open(self):
        cap = self._capture()
        self.status = "live"
        self.fps = cap.get(cv2.CAP_PROP_FPS) or None
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, args=(cap,), daemon=True)
        self._thread.start()

    def read(self, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None:
                if self.status in ("ended", "stopped"):
                    return None
                left = deadline - time.monotonic()
                if left <= 0:
                    raise SourceError(f"{self.label}: no frame for {timeout:.0f} s ({self.status})")
                self._cond.wait(left)
            f, self._latest = self._latest, None
            return f

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


class CameraSource(_LiveSource):
    kind = "camera"

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = int(index)

    @property
    def label(self):
        return f"camera {self.index}"

    def _capture(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise SourceError(f"cannot open camera {self.index}")
        return cap

    @staticmethod
    def probe(max_index: int = 5) -> list[int]:
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                found.append(i)
            cap.release()
        return found


class RtspSource(_LiveSource):
    kind = "rtsp"

    def __init__(self, url: str, open_timeout_s: float = 10.0, max_backoff_s: float = 8.0):
        super().__init__()
        self.url = url
        self.open_timeout_s, self.max_backoff_s = open_timeout_s, max_backoff_s
        self.reconnects = 0

    @property
    def label(self):
        return re.sub(r"//[^@/]+@", "//***@", self.url)  # never show credentials

    def _open_cap(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout_s * 1000),
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.open_timeout_s * 1000)])
        return cap if cap.isOpened() else None

    def _capture(self):
        cap = self._open_cap()
        if cap is None:
            raise SourceError(f"RTSP timeout or refused: {self.label}")
        return cap

    def _on_read_fail(self) -> bool:
        return not self._stop.is_set()

    def _capture_or_none(self):
        delay = 0.5
        while not self._stop.is_set():
            self.status = f"reconnecting (retry in {delay:.1f} s)"
            log.warning("%s: stream lost, reconnecting in %.1f s", self.label, delay)
            if self._stop.wait(delay):
                return None
            cap = self._open_cap()
            if cap is not None:
                self.reconnects += 1
                self.status = "live"
                return cap
            delay = min(delay * 2, self.max_backoff_s)
        return None


def make_source(spec: str) -> Source:
    """'camera:0', 'rtsp://…', a video path or an image folder."""
    s = str(spec)
    if s.startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        return RtspSource(s)
    if s.startswith("camera:") or s.isdigit():
        return CameraSource(int(s.split(":")[-1]))
    p = Path(s).expanduser()
    if p.is_dir():
        return FolderSource(p)
    if p.suffix.lower() in IMAGE_EXTS:
        raise SourceError("pick the folder that contains the images, not a single image")
    return VideoSource(p)
