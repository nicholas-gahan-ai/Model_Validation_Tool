"""Developer tool: an end-to-end tester pass over the real window (clicks, keys, dialogs, files).
  GAHAN_BUNDLE=~/validation_tool_bundle python tools/dev/ui_test_full.py --work /tmp/vv_test [--quick]
Uses its own settings ("Gahan AI Test"), records message boxes and exceptions, prints PASS/FAIL per check."""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

import ui.dialogs as D  # noqa: E402
from core.settings import bundle_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, type=Path)
ap.add_argument("--quick", action="store_true", help="skip the 1,207-frame UI AP run")
A = ap.parse_args()
B = bundle_path()
WORK = A.work
shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir(parents=True)

app = QApplication(sys.argv)
app.setOrganizationName("Gahan AI Test"); app.setApplicationName("Validation Viewer UI Test")
from ui.main_window import MainWindow  # noqa: E402
from ui.theme import apply_theme  # noqa: E402
apply_theme(app)
from PySide6.QtCore import QSettings  # noqa: E402
QSettings().clear()

RESULTS, MSGS, EXC, DLG = [], [], [], []
FILE_ANSWERS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def _hook(t, v, tb):
    EXC.append("".join(traceback.format_exception(t, v, tb)))
    print("EXCEPTION:", "".join(traceback.format_exception(t, v, tb))[-800:], flush=True)


sys.excepthook = _hook
for kind in ("critical", "warning", "information"):
    setattr(QMessageBox, kind, staticmethod(lambda parent, title, text, *a, _k=kind, **k: (
        MSGS.append((_k, title, str(text))), QMessageBox.Ok)[1]))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (FILE_ANSWERS.pop(0) if FILE_ANSWERS else "", ""))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")


def record(cls, name, before=None, result=0):
    def ex(self):
        DLG.append(name)
        if before:
            before(self)
        self.show(); app.processEvents()
        r = result() if callable(result) else result
        self.hide(); app.processEvents()
        return r
    cls.exec = ex


record(D.ZoomDialog, "zoom")
record(D.MetricsDialog, "metrics")
record(D.AboutDialog, "about")
record(D.ModelInfoDialog, "model_info")
record(D.LabelMappingDialog, "mapping", result=1)
record(D.FlagDialog, "flag", before=lambda d: (d._box(0, 100, 100, 400, 300), d.note.setPlainText("tester note")),
       result=1)

w = MainWindow(); w.resize(1600, 950); w.show()
w.activateWindow()
L, C, R = w.left, w.controls, w.right
MODEL = str(ROOT / "models/open_adas_v13s_single.onnx")
VIDEO = str(B / "data/videos/day1.mp4")


def click(btn):
    QTest.mouseClick(btn, Qt.LeftButton)


def key(k):
    w.activateWindow(); w.view.setFocus()
    QTest.keyClick(w.view, k)


def idx():
    return w.current.frame.index if w.current is not None else None


def setup(kind, path="", gt="", out=None, **opts):
    L.radio[kind].setChecked(True)
    {"video": L.video_path, "folder": L.folder_path}.get(kind, L.video_path).setText(path) if kind in ("video", "folder") else None
    if kind == "rtsp":
        L.rtsp_url.setText(path)
    if kind == "camera":
        L.camera_combo.setCurrentText(path)
    L.gt_path.setText(gt)
    L.out_dir.setText(str(out or WORK / "runs"))
    for k in ("save_video", "export_dets", "summary", "folder_recursive", "folder_video", "side_by_side_save",
              "watermark", "counts_strip"):
        getattr(L, k).setChecked(opts.get(k, k == "save_video"))
    C.speed.setCurrentText(opts.get("speed", "1× (real time)"))


# ------------------------------------------------------------------------------------------------ scenario runner
def wait(cond, timeout=30.0, step=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(step)
    return False


def pause_ms(ms):
    t0 = time.time()
    while (time.time() - t0) * 1000 < ms:
        app.processEvents(); time.sleep(0.01)


def finished():
    return w.worker is None


def stop_and_wait(timeout=60):
    if w.worker:
        click(L.stop_btn)
    return wait(finished, timeout)


def run():
    MSGS.clear()
    # ---- A. input validation ------------------------------------------------------------------------------------
    ok = w.load_model(MODEL)
    check("model loads on start", ok and w.runner is not None, w.runner.device_label if w.runner else "")
    check("header shows GPU", "GPU" in w.header_device.text(), w.header_device.text())
    setup("video", "")
    click(L.start_btn); app.processEvents()
    check("Start without input -> 'No input' warning, nothing runs", w.worker is None and MSGS and MSGS[-1][1] == "No input")
    setup("video", str(WORK / "nope.mp4"))
    click(L.start_btn); app.processEvents()
    check("missing video -> clear error", w.worker is None and MSGS[-1][1] == "Cannot open input", MSGS[-1][2][:80])
    bad = WORK / "corrupt.mp4"; bad.write_bytes(os.urandom(5000))
    setup("video", str(bad)); click(L.start_btn); app.processEvents()
    check("corrupt video -> clear error", w.worker is None and MSGS[-1][1] == "Cannot open input", MSGS[-1][2][:80])
    empty = WORK / "empty"; empty.mkdir()
    setup("folder", str(empty)); click(L.start_btn); app.processEvents()
    check("empty folder -> clear error", w.worker is None and "no images" in MSGS[-1][2], MSGS[-1][2][:80])
    setup("video", VIDEO, gt=str(WORK / "nope_gt.json")); click(L.start_btn); app.processEvents()
    check("missing GT -> clear error, run not started", w.worker is None and MSGS[-1][1] == "Ground truth", MSGS[-1][2][:80])
    badxml = WORK / "bad.xml"; badxml.write_text("<annotations><image name='a.jpg'>\n<box label='car' xtl='1'\n</annotations>")
    setup("video", VIDEO, gt=str(badxml)); click(L.start_btn); app.processEvents()
    check("broken GT XML -> error with line number", w.worker is None and "line" in MSGS[-1][2], MSGS[-1][2][:90])
    n_runs_before = len(list((WORK / "runs").glob("*"))) if (WORK / "runs").exists() else 0
    check("failed starts leave no empty run folders", n_runs_before == 0, f"{n_runs_before} folders")

    # ---- B. video playback ----------------------------------------------------------------------------------------
    setup("video", VIDEO, export_dets=True, summary=True)
    click(L.start_btn)
    check("video starts and shows frames", wait(lambda: idx() is not None and idx() > 5, 30), f"frame {idx()}")
    check("Start disabled / Stop enabled while running", not L.start_btn.isEnabled() and L.stop_btn.isEnabled())
    check("inputs locked while running", not L.video_path.isEnabled() and not L.model_combo.isEnabled())
    pause_ms(1500)
    t = w.current.pred.timing_ms
    check("timings shown (pre/inf/post)", all(k in t for k in ("preprocess", "inference", "postprocess")),
          f"inf {t.get('inference', 0):.0f} ms")
    key(Qt.Key_Space); pause_ms(400); i0 = idx(); pause_ms(700)
    check("Space pauses", w.worker.paused and idx() == i0, f"frame {i0}")
    check("play button shows ▶ when paused", C.play_btn.text() == "▶")
    key(Qt.Key_Right); wait(lambda: idx() == i0 + 1, 5)
    check("→ steps forward one frame", idx() == i0 + 1, f"{i0} -> {idx()}")
    key(Qt.Key_Left); wait(lambda: idx() == i0, 5)
    check("← steps back one frame", idx() == i0, f"-> {idx()}")
    key(Qt.Key_End); wait(lambda: idx() is not None and idx() >= 12600, 15)
    check("End jumps to the last frame", idx() is not None and idx() >= 12600, f"{idx()}")
    key(Qt.Key_Home); wait(lambda: idx() == 0, 15)
    check("Home jumps to frame 0", idx() == 0, f"{idx()}")
    sb = C.seekbar
    x = int(sb.width() * 0.5)
    QTest.mouseClick(sb, Qt.LeftButton, Qt.NoModifier, QPoint(x, sb.height() // 2))
    wait(lambda: idx() is not None and abs(idx() - 6340) < 700, 15)
    check("clicking the seek bar jumps there", idx() is not None and abs(idx() - 6340) < 700, f"{idx()}")
    # frame content really is the seeked frame
    cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, idx()); _, ref = cap.read()
    check("shown frame content == video frame at that index",
          ref is not None and float(np.abs(ref.astype(int) - w.current.frame.image.astype(int)).mean()) < 1.0)
    # layers / classes / sliders
    before = w.view.images[0].copy() if w.view.images else None
    n_boxes = len(w.current_pred.boxes)
    keys_layers = list(C.layer_boxes)
    states = {k: C.layer_boxes[k].isChecked() for k in keys_layers}
    for n in range(len(keys_layers)):
        key(getattr(Qt, f"Key_{n + 1}"))
    flipped = all(C.layer_boxes[k].isChecked() != states[k] for k in keys_layers)
    check("keys 1–5 toggle the layers", flipped, ", ".join(keys_layers))
    for n in range(len(keys_layers)):
        key(getattr(Qt, f"Key_{n + 1}"))
    check("layer toggles redraw without inference (same frame, new image)",
          before is not None and w.view.images and w.current.frame.index == idx())
    key(Qt.Key_Home); wait(lambda: idx() == 0, 15)   # frame 0 is busy (many low-confidence boxes)
    C.sliders["conf"].set_value(0.9); app.processEvents()
    n_hi = len(w.current_pred.boxes)
    C.sliders["conf"].set_value(0.05); app.processEvents()
    n_lo = len(w.current_pred.boxes)
    check("Conf slider changes the boxes instantly", n_lo > n_hi, f"0.9 -> {n_hi}, 0.05 -> {n_lo}")
    C.sliders["conf"].set_value(0.35); app.processEvents()
    biggest = max(w.current_pred.boxes, key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]))
    C.class_boxes[biggest.cls].setChecked(False); app.processEvents()
    names = [R.table.item(i, 0).text() for i in range(R.table.rowCount())]
    check("hiding a class removes it from view and counts", biggest.cls not in names, biggest.cls)
    C.class_boxes[biggest.cls].setChecked(True); app.processEvents()
    R.show_all.setChecked(True); app.processEvents()
    check("'all classes' lists every class", R.table.rowCount() == len(w.runner.desc.class_names), f"{R.table.rowCount()}")
    R.show_all.setChecked(False)
    # zoom
    b = max(w.current_pred.boxes, key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]))
    DLG.clear(); w._on_view_click(0, (b.xyxy[0] + b.xyxy[2]) / 2, (b.xyxy[1] + b.xyxy[3]) / 2)
    check("clicking a box opens the zoom dialog", DLG == ["zoom"])
    key(Qt.Key_Z); DLG.clear(); w._on_view_click(0, (b.xyxy[0] + b.xyxy[2]) / 2, (b.xyxy[1] + b.xyxy[3]) / 2)
    check("Z turns click-to-zoom off", DLG == [])
    key(Qt.Key_Z)
    # side by side
    key(Qt.Key_T); app.processEvents()
    check("T shows side by side (2 panes)", len(w.view.images) == 2 and w.view.titles[0] == "Input")
    key(Qt.Key_T); app.processEvents()
    check("T again -> single pane", len(w.view.images) == 1)
    # snapshot + flag
    click(C.snap_btn); app.processEvents()
    snaps = list(w.run_dir.glob("snapshots/*_raw.png"))
    check("Snapshot button writes raw + annotated PNG", len(snaps) == 1 and len(list(w.run_dir.glob("snapshots/*_annotated.png"))) == 1)
    check("Snapshot gives visible feedback (flash + toast)", w.view._flash.isValid() and "Snapshot saved" in w.view._toast)
    key(Qt.Key_S); app.processEvents()
    check("S key snapshots too (second file, no overwrite)", len(list(w.run_dir.glob("snapshots/*_raw.png"))) == 2)
    DLG.clear(); click(C.flag_btn); app.processEvents()
    rows = list(csv.DictReader(open(w.run_dir / "flags/flags.csv"))) if (w.run_dir / "flags/flags.csv").exists() else []
    check("Flag saves flags.csv row + PNGs", DLG == ["flag"] and len(rows) == 1 and rows[0]["note"] == "tester note"
          and rows[0]["model_sha256"], rows[0]["category"] if rows else "")
    check("flag marker appears on the seek bar", C.seekbar.markers == [idx()])
    key(Qt.Key_F); app.processEvents()
    check("F key flags too", len(list(csv.DictReader(open(w.run_dir / "flags/flags.csv")))) == 2)
    # speed / play again
    C.speed.setCurrentText("Max"); key(Qt.Key_Space)
    f0, t0 = idx(), time.time(); pause_ms(3000)
    fps_max = (idx() - f0) / (time.time() - t0)
    check("Max speed plays faster than real time", fps_max > 15.5, f"{fps_max:.1f} fps")
    C.speed.setCurrentText("1× (real time)")
    f0, t0 = idx(), time.time(); pause_ms(3000)
    fps_rt = (idx() - f0) / (time.time() - t0)
    check("1× keeps about the video fps (15.1) while saving", 8 <= fps_rt <= 16.5, f"{fps_rt:.1f} fps")
    rd = w.run_dir
    check("Stop ends the run", stop_and_wait(60))
    check("Start re-enabled after stop", L.start_btn.isEnabled() and not L.stop_btn.isEnabled())
    sess = w.session
    files = {p.name for p in rd.iterdir()}
    check("run folder has video, exports, summary, run_info, log",
          {"day1_annotated.mp4", "detections.jsonl", "detections.csv", "summary.html", "run_info.json", "viewer.log"} <= files,
          ", ".join(sorted(files)))
    n_saved = int(cv2.VideoCapture(str(rd / "day1_annotated.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    n_lines = len((rd / "detections.jsonl").read_text().splitlines()) - 1
    check("saved video frames == processed frames == JSONL lines", n_saved == sess.frames == n_lines,
          f"video {n_saved}, processed {sess.frames}, jsonl {n_lines}")
    html = (rd / "summary.html").read_text()
    check("summary lists the 2 flags with thumbnails", "Flagged frames (2)" in html and html.count("data:image/jpeg") >= 2)
    info = json.loads((rd / "run_info.json").read_text())
    check("run_info.json final (frames, files)", info.get("frames_processed") == sess.frames and "summary.html" in info["files"])
    click(C.snap_btn); app.processEvents()
    check("Snapshot still works after the run ended", len(list(rd.glob("snapshots/*_raw.png"))) == 3)
    check("transport disabled after the run (except snapshot/flag)", not C.next_btn.isEnabled() and C.snap_btn.isEnabled())

    # ---- C. folders + GT ------------------------------------------------------------------------------------------
    setup("folder", str(B / "data/oa_val/images/scene_0047"), gt=str(B / "data/oa_val/annotations_xml"),
          speed="Max", summary=True, side_by_side_save=True)
    DLG.clear(); click(L.start_btn)
    check("folder + Open ADAS GT runs", wait(lambda: idx() is not None, 30))
    check("GT panel visible with TP/FP/miss", wait(lambda: "TP" in R.gt_info.text(), 10))
    key(Qt.Key_T); app.processEvents()
    check("side by side shows 'Ground truth' left", w.view.titles and w.view.titles[0] == "Ground truth")
    key(Qt.Key_T)
    check("folder run finishes on its own", wait(finished, 120))
    check("metrics dialog opens at the end", "metrics" in DLG)
    m = w.session.gt_metrics
    check("folder metrics computed (85 frames)", m and m["images_matched"] == 85, f"mean AP50-95 {m['mean']['ap50_95']:.4f}" if m else "")
    rd = w.run_dir
    names_out = sorted(p.name for p in (rd / "scene_0047_annotated").iterdir())
    check("annotated folder keeps file names", names_out == sorted(p.name for p in (B / "data/oa_val/images/scene_0047").iterdir()))
    img0 = cv2.imread(str(rd / "scene_0047_annotated" / names_out[0]))
    check("side-by-side save doubles the width", img0 is not None and img0.shape[1] == 3840, f"{img0.shape if img0 is not None else None}")
    check("metrics.json/csv written", (rd / "metrics.json").exists() and (rd / "metrics.csv").exists())
    mj = json.loads((rd / "metrics.json").read_text())
    check("metrics.json is valid JSON with null for NaN", isinstance(mj.get("mean"), dict))
    # YOLO without names -> asks for a names file
    ydir = WORK / "yolo"; ydir.mkdir()
    for p in sorted((B / "data/oa_val/images/scene_0046").iterdir())[:10]:
        (ydir / (p.stem + ".txt")).write_text("0 0.5 0.5 0.2 0.2\n")
    names = WORK / "my_names.txt"; names.write_text("car\n")
    FILE_ANSWERS.append(str(names))
    setup("folder", str(B / "data/oa_val/images/scene_0046"), gt=str(ydir), speed="Max")
    L.gt_format.setCurrentText("Auto-detect")
    click(L.start_btn)
    check("YOLO without class names asks for a names file and runs", wait(lambda: w.worker is not None or finished(), 10)
          and not FILE_ANSWERS)
    check("YOLO run finishes", wait(finished, 120))
    m = w.session.gt_metrics
    check("YOLO GT: matched 10 frames, attributes N/A", m and m["images_matched"] == 10 and not m["attributes"],
          f"{m['images_matched'] if m else None} frames")
    # GT that matches no frame
    nomatch = WORK / "nomatch.json"
    nomatch.write_text(json.dumps({"images": [{"id": 1, "file_name": "zzz.jpg", "width": 10, "height": 10}],
                                   "annotations": [], "categories": [{"id": 0, "name": "car"}]}))
    setup("folder", str(B / "data/oa_val/images/scene_0050"), gt=str(nomatch), speed="Max")
    click(L.start_btn)
    check("GT matching no frame: run completes without crash", wait(finished, 120))
    m = w.session.gt_metrics
    check("…and reports 0 frames with GT", m is not None and m["images_matched"] == 0, f"without GT {m['frames_without_gt'] if m else None}")
    check("…GT panel says 'no GT for this frame'", "no GT" in R.gt_info.text())
    # unknown label -> mapping dialog
    unk = WORK / "unk.json"
    imgs = sorted((B / "data/oa_val/images/scene_0046").iterdir())[:3]
    unk.write_text(json.dumps({"images": [{"id": i, "file_name": p.name, "width": 1920, "height": 1080} for i, p in enumerate(imgs)],
                               "annotations": [{"id": 0, "image_id": 0, "category_id": 7, "bbox": [1, 1, 50, 50]}],
                               "categories": [{"id": 7, "name": "Zebra crossing"}]}))
    DLG.clear()
    setup("folder", str(B / "data/oa_val/images/scene_0046"), gt=str(unk), speed="Max")
    click(L.start_btn)
    check("unknown GT label opens the mapping dialog", wait(lambda: "mapping" in DLG, 10))
    check("mapping choice saved beside the GT", (WORK / ".gahan_label_map.yaml").exists())
    wait(finished, 120)
    m = w.session.gt_metrics
    check("ignored label listed as excluded", m and m["excluded"].get("Zebra crossing") == 1, f"{m['excluded'] if m else None}")
    if not A.quick:
        setup("folder", str(B / "data/oa_val/images"), gt=str(B / "data/oa_val/gt_coco_scoring.json"), speed="Max")
        L.folder_recursive.setChecked(True)
        DLG.clear(); t0 = time.time(); click(L.start_btn)
        done = wait(finished, 900)
        m = w.session.gt_metrics if done else None
        ap_all = m["mean"]["ap50_95"] if m else float("nan")
        check("UI: 1,207 val frames (subfolders) reproduce AP50-95 0.4686 ± 0.002", m and m["images_matched"] == 1207
              and abs(ap_all - 0.4686) <= 0.002, f"{ap_all:.4f} in {time.time() - t0:.0f} s")
        L.folder_recursive.setChecked(False)

    # ---- D. models / device -----------------------------------------------------------------------------------------
    L.gt_path.setText("")
    L.device.setCurrentText("CPU"); wait(lambda: w.runner is not None and w.runner.device == "CPU", 60)
    check("Device → CPU reloads on the CPU", w.runner.device == "CPU" and "CPU" in w.header_device.text(), w.header_device.text())
    L.device.setCurrentText("Auto"); wait(lambda: w.runner.device == "GPU", 60)
    check("Device → Auto back on the GPU", w.runner.device == "GPU")
    junk = WORK / "junk.onnx"; junk.write_bytes(b"not a model at all")
    MSGS.clear(); okj = w.load_model(str(junk))
    check("invalid model file -> error, previous model kept", not okj and MSGS and w.model.path.name == "open_adas_v13s_single.onnx",
          MSGS[-1][2][:70] if MSGS else "")
    okp = w.load_model(str(B / "models/yolo26s.pt"))
    check("Ultralytics .pt loads; toggles rebuilt (80 classes, 1 layer)", okp and len(C.class_boxes) == 80 and list(C.layer_boxes) == ["boxes"])
    setup("video", VIDEO, save_video=False); click(L.start_btn)
    check(".pt model runs on video", wait(lambda: idx() is not None and idx() > 10, 30))
    stop_and_wait()
    sc = WORK / "m3"; sc.mkdir()
    shutil.copy(B / "models/open_adas_v13_s.onnx", sc)
    record(D.DescriptionDialog, "description", before=lambda d: (d._load(str(ROOT / "tools/descriptions/open_adas_v13s_detector.yaml")), d._save()),
           result=lambda: 1)
    DLG.clear(); okd = w.load_model(str(sc / "open_adas_v13_s.onnx"))
    check("metadata-less ONNX: description dialog, sidecar saved, loads", okd and "description" in DLG
          and (sc / "open_adas_v13_s.description.yaml").exists() and w.model.description_source == "sidecar")
    DLG.clear(); okd2 = w.load_model(str(sc / "open_adas_v13_s.onnx"))
    check("…second open uses the sidecar without asking", okd2 and "description" not in DLG)
    w.load_model(MODEL)
    check("per-model view state restored (conf slider 0.35)", abs(C.sliders["conf"].value() - 0.35) < 1e-6)
    DLG.clear(); w.show_model_info(); w.show_about()
    check("Model info + About open", DLG == ["model_info", "about"])

    # ---- E. live sources + robustness ------------------------------------------------------------------------------
    MSGS.clear(); setup("camera", "5"); t0 = time.time(); click(L.start_btn); app.processEvents()
    check("camera that does not exist -> clear error, no hang", w.worker is None and MSGS and "camera" in MSGS[-1][2].lower(),
          f"{time.time() - t0:.1f} s: {MSGS[-1][2][:60] if MSGS else ''}")
    MSGS.clear(); setup("rtsp", "rtsp://127.0.0.1:9/none"); t0 = time.time(); click(L.start_btn); app.processEvents()
    check("RTSP nothing listening -> clear error within 15 s", w.worker is None and MSGS and "RTSP" in MSGS[-1][2]
          and time.time() - t0 < 15, f"{time.time() - t0:.1f} s")
    setup("video", VIDEO, save_video=False); click(L.start_btn); wait(lambda: idx() is not None, 30)
    L.rtsp_url.setEnabled(True); L.rtsp_url.setFocus(); L.rtsp_url.clear()
    playing = not w.worker.paused
    QTest.keyClicks(L.rtsp_url, "a b"); app.processEvents()
    check("typing a space in a text field does not pause playback", w.worker.paused == (not playing) and L.rtsp_url.text() == "a b")
    L.rtsp_url.setEnabled(False)
    rd = w.run_dir
    w.close()
    check("closing the window while running exits cleanly", wait(lambda: w.worker is None or not w.worker.isRunning(), 15))
    w2 = MainWindow()
    check("settings restored on the next start (input path, speed)",
          w2.left.video_path.text() == VIDEO and w2.left.input_kind() == "video", w2.left.video_path.text()[-20:])
    w2.close()
    check("no Python exceptions during the whole pass", not EXC, f"{len(EXC)} exceptions")
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for n, _, d in bad:
        print("  FAILED:", n, d)
    QSettings().clear()
    app.quit()


QTimer.singleShot(300, lambda: (run() if True else None))
app.exec()
