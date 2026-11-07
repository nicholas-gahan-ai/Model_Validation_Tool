import time

import cv2
import numpy as np
import pytest

from core.sources import FolderSource, RtspSource, SourceError, VideoSource, make_source, natural_key


class FakeCap:
    """Delivers `n` frames at `fps`, then fails reads (stream lost)."""

    def __init__(self, n, fps=100.0, start=0):
        self.n, self.fps, self.i, self.start = n, fps, 0, start

    def read(self):
        if self.i >= self.n:
            time.sleep(0.01)
            return False, None
        time.sleep(1 / self.fps)
        self.i += 1
        return True, np.full((4, 4, 3), (self.start + self.i) % 255, np.uint8)

    def get(self, _):
        return self.fps

    def isOpened(self):
        return True

    def release(self):
        pass


def test_live_source_drops_stale_frames_and_reconnects(monkeypatch):
    caps = [FakeCap(30, 200), None, None, FakeCap(1000, 200, start=100)]   # lost after 30, two failed retries
    src = RtspSource("rtsp://user:secret@127.0.0.1:8554/test", max_backoff_s=0.05)
    monkeypatch.setattr(src, "_open_cap", lambda: caps.pop(0) if caps else None)
    assert "secret" not in src.label
    src.open()
    got = []
    t_end = time.monotonic() + 3
    while time.monotonic() < t_end and src.reconnects == 0:
        f = src.read(timeout=2)
        got.append(f.index)
        time.sleep(0.03)                       # slow consumer: the reader must drop frames
    f2 = src.read(timeout=2)                    # frames flow again after reconnecting
    src.close()
    assert src.reconnects == 1
    assert src.dropped > 0                      # newest frame only
    assert got == sorted(got) and len(set(got)) == len(got)
    assert f2 is not None and f2.index > got[-1]


def test_rtsp_open_timeout(monkeypatch):
    src = RtspSource("rtsp://127.0.0.1:1/none")
    monkeypatch.setattr(src, "_open_cap", lambda: None)
    with pytest.raises(SourceError, match="timeout"):
        src.open()


def test_folder_natural_sort_and_case(tmp_path):
    for n in ["img10.JPG", "img2.jpg", "img1.png", "notes.txt", "x.Bmp"]:
        (tmp_path / n).write_bytes(b"")
    img = np.zeros((8, 8, 3), np.uint8)
    for n in ["img10.JPG", "img2.jpg", "img1.png", "x.Bmp"]:
        cv2.imwrite(str(tmp_path / n), img) if not n.endswith("JPG") else cv2.imencode(".jpg", img)[1].tofile(str(tmp_path / n))
    s = FolderSource(tmp_path); s.open()
    assert [p.name for p in s.files] == ["img1.png", "img2.jpg", "img10.JPG", "x.Bmp"]
    assert natural_key("a2") < natural_key("a10")


def test_empty_folder_and_bad_video(tmp_path):
    with pytest.raises(SourceError, match="no images"):
        FolderSource(tmp_path).open()
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a video")
    with pytest.raises(SourceError):
        VideoSource(bad).open()
    assert isinstance(make_source(str(tmp_path)), FolderSource)
    assert isinstance(make_source("rtsp://h/x"), RtspSource)


def _make_video(path, n=80, fps=20.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 3, np.uint8))   # the frame's content encodes its index
    w.release()


def _idx(frame):
    return int(round(float(frame.image.mean()) / 3))


def test_video_read_seek_skip_land_on_the_right_frames(tmp_path):
    p = tmp_path / "v.avi"; _make_video(p)
    s = VideoSource(p); s.open()
    assert s.count == 80 and s.fps == pytest.approx(20.0)
    f = s.read(); assert f.index == 0 and _idx(f) == 0 and f.timestamp_s == 0
    s.skip(4); f = s.read(); assert f.index == 5 and _idx(f) == 5          # short skip: grab()
    s.skip(40); f = s.read(); assert f.index == 46 and abs(_idx(f) - 46) <= 1  # long skip: seek
    s.seek(10); f = s.read(); assert f.index == 10 and _idx(f) == 10
    s.seek(999); f = s.read(); assert f.index == 79
    assert s.read() is None
    s.close()


def test_skip_right_after_open_drops_the_buffered_first_frame(tmp_path):
    p = tmp_path / "v.avi"; _make_video(p, 10)
    s = VideoSource(p); s.open()
    s.skip(2); f = s.read()
    assert f.index == 2 and _idx(f) == 2
