"""Developer tool: regenerate the README screenshots (docs/screenshots/*.png) by driving the real window.
  GAHAN_BUNDLE=~/validation_tool_bundle python tools/dev/screenshots.py --out-runs /tmp/vv_runs
Not shipped in the release zip."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.dialogs as D  # noqa: E402
from core.settings import bundle_path  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.theme import apply_theme  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out-runs", required=True, type=Path)
a = ap.parse_args()
B = bundle_path()
SHOTS = ROOT / "docs/screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv)
app.setOrganizationName("Gahan AI Screens"); app.setApplicationName("Validation Viewer Screens")
apply_theme(app)
w = MainWindow(); w.resize(1600, 940); w.show(); w.quiet = True
w.load_model(str(ROOT / "models/open_adas_v13s_single.onnx"))
L = w.left
L.out_dir.setText(str(a.out_runs)); L.save_video.setChecked(False)


def grab(widget, name):
    app.processEvents()
    widget.grab().save(str(SHOTS / f"{name}.png"))
    print("saved", SHOTS / f"{name}.png", flush=True)


def dialog_grab(cls, name, before=None):
    def ex(self):
        if before:
            before(self)
        self.show(); grab(self, name); return 0
    cls.exec = ex


steps = []


def step(ms):
    def deco(fn):
        steps.append((ms, fn)); return fn
    return deco


# 1. video, paused on a busy frame
@step(300)
def s1():
    L.video_path.setText(str(B / "data/videos/day1.mp4")); L.radio["video"].setChecked(True)
    L.gt_path.setText("")
    w.controls.speed.setCurrentText("1× (real time)")
    w.start()


@step(2500)
def s2():
    w.toggle_play(); w.seek(2410)


@step(1500)
def s3():
    grab(w, "main_video")
    p = w.current_pred
    if p and p.boxes:
        b = max(p.boxes, key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]))
        dialog_grab(D.ZoomDialog, "zoom")
        w._on_view_click(0, (b.xyxy[0] + b.xyxy[2]) / 2, (b.xyxy[1] + b.xyxy[3]) / 2)
    dialog_grab(D.FlagDialog, "flag", before=lambda d: (d._box(0, 700, 420, 1000, 640),
                                                        d.note.setPlainText("Example: object in the marked region not detected")))
    w.flag_frame()
    dialog_grab(D.AboutDialog, "about")
    w.show_about()
    w.stop()


# 2. image folder + GT, side by side
@step(2000)
def s4():
    L.folder_path.setText(str(B / "data/oa_val/images/scene_0047")); L.radio["folder"].setChecked(True)
    L.gt_path.setText(str(B / "data/oa_val/annotations_xml"))
    w.controls.speed.setCurrentText("Max")
    w.start()


@step(1200)
def s5():
    w.toggle_play(); w.seek(24); w.controls.sbs_btn.setChecked(True)


@step(1500)
def s6():
    grab(w, "gt_side_by_side")
    dialog_grab(D.MetricsDialog, "metrics")
    w.show_metrics()
    w.controls.sbs_btn.setChecked(False)
    w.stop()


@step(1500)
def s7():
    w.close(); app.quit()


t = 0
for ms, fn in steps:
    t += ms
    QTimer.singleShot(t, fn)
QTimer.singleShot(t + 60000, app.quit)
app.exec()
