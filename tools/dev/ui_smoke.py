"""Developer smoke test: drive the real window, take screenshots. Not part of the app.
  python tools/dev/ui_smoke.py --input VIDEO_OR_FOLDER --out SHOT_DIR [--seconds 6] [--sbs] [--gt PATH]"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402
from ui.theme import apply_theme  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "models/open_adas_v13s_single.onnx"))
ap.add_argument("--seconds", type=float, default=6); ap.add_argument("--sbs", action="store_true")
ap.add_argument("--gt", default=""); ap.add_argument("--run-out", default="")
ap.add_argument("--speed", default="1× (real time)"); ap.add_argument("--no-save", action="store_true")
ap.add_argument("--summary", action="store_true"); ap.add_argument("--export", action="store_true")
ap.add_argument("--wait-end", action="store_true")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv); app.setOrganizationName("Gahan AI Smoke"); app.setApplicationName("Validation Viewer Smoke")
apply_theme(app)
w = MainWindow(); w.resize(1600, 950); w.show()
assert w.load_model(a.model)
L = w.left
if Path(a.input).is_dir():
    L.folder_path.setText(a.input); L.radio["folder"].setChecked(True)
else:
    L.video_path.setText(a.input); L.radio["video"].setChecked(True)
L.gt_path.setText(a.gt)
L.save_video.setChecked(not a.no_save); L.summary.setChecked(a.summary); L.export_dets.setChecked(a.export)
if a.run_out:
    L.out_dir.setText(a.run_out)
w.controls.speed.setCurrentText(a.speed)
shots = []


def shot(name):
    p = a.out / f"{name}.png"; w.grab().save(str(p)); shots.append(p); print("shot", p, flush=True)


def go():
    w.start()
    if a.sbs:
        QTimer.singleShot(1500, lambda: w.controls.sbs_btn.setChecked(True))
    QTimer.singleShot(int(a.seconds * 1000), lambda: (shot("running"), w.toggle_play()))
    QTimer.singleShot(int(a.seconds * 1000) + 800, lambda: (w.step(1), None))
    QTimer.singleShot(int(a.seconds * 1000) + 1800, lambda: (shot("paused_step"), None if a.wait_end else w.stop()))
    if a.wait_end:
        QTimer.singleShot(int(a.seconds * 1000) + 2000, w.toggle_play)


def finished(_):
    QTimer.singleShot(500, lambda: (shot("finished"), w.close(), app.quit()))


QTimer.singleShot(500, go)
orig = w._on_finished
w._on_finished = lambda s: (orig(s), finished(s))
QTimer.singleShot(int(max(a.seconds, 1) * 1000) + 600000, app.quit)
app.exec()
print("status:", w.statusBar().currentMessage())
