# Gahan AI Validation Viewer

A desktop tool to validate perception models. Load a model, run it on a **video, an image folder, a USB camera or an
RTSP stream**, see every output live, save the annotated result and, if you have annotations, **measure accuracy
against ground truth**.

Runs on Linux and Windows, on an NVIDIA GPU when there is one and on the CPU otherwise.

![Live view](docs/screenshots/main_video.png)

**Contents:** [Install](#install) · [Quick start](#quick-start) · [Features](#features) ·
[Shortcuts](#keyboard-shortcuts) · [Output files](#output-files) · [Ground truth](#ground-truth) ·
[Command line](#command-line) · [Troubleshooting](#troubleshooting) · [Adding a model](#adding-a-model) ·
[Build from source](#build-from-source)

---

## Install

Needs **Python 3.9+**. For GPU use, an NVIDIA driver with CUDA 12 support (525 or newer); CUDA itself is installed by
the setup script.

* **Release zip** (`validation_tool_<version>.zip`): the app plus the packaged models. Use this on validation machines.
* **This repository**: the app only. Model files are not stored in git; put yours in `models/`.

**Linux**
```bash
unzip validation_tool_1.0.0.zip && cd validation_tool_1.0.0
./setup_linux.sh        # GPU build if an NVIDIA GPU is found, else CPU   (--cpu to force CPU)
./run_linux.sh          # or "Validation Viewer" in the application menu
```

**Windows:** install Python from <https://www.python.org/downloads/> (tick *Add python.exe to PATH*), unzip, run
**`setup_windows.bat`**, then start from the **Validation Viewer** desktop shortcut or `run_windows.bat`.

Add `--with-ultralytics` to the setup command to also open Ultralytics `.pt` models (adds about 2 GB).

---

## Quick start

1. **Model:** pick one from the list (all files in `models/`) or **Open…** one from anywhere. **ⓘ** shows what it outputs.
2. **Input:** Video, Image folder, Camera or RTSP.
3. **Ground truth** *(if you have annotations)*: pick the annotation file or folder.
4. **Outputs:** choose what to save, and the output folder.
5. Press **Start**. Space pauses, ←/→ step, click a box to inspect it.

The device in use is shown top right (green = GPU).

---

## Features

### Models
Any model can be used without code changes. A model is one file:
* **ONNX with an embedded description**: inputs, class names, attribute heads, masks, colours and thresholds
  (see [Adding a model](#adding-a-model)).
* **ONNX without a description**: the app asks for one once (edit the template it fills in, or load a YAML) and saves
  it beside the model.
* **Ultralytics `.pt`** (detection or segmentation).

### Inputs
| Input | |
|---|---|
| Video | mp4, avi, mov, mkv … with seek and step. |
| Image folder | jpg, png, bmp. Tick **include subfolders** to process a whole folder tree (e.g. one folder per scene) in one run. |
| Camera | USB camera by index; **Probe** lists the cameras found. |
| RTSP | A live network camera (IP/CCTV camera, dashcam, in-vehicle device), e.g. `rtsp://user:pass@192.168.1.20:554/stream1`. Reconnects on its own; the password is never shown. |

Live sources always show the newest frame and drop the rest when the model is slower than the camera.

### Live view
* Boxes in a distinct colour per class, with confidence and attributes. Uncertain attribute values are dimmed with `?`.
* Masks (e.g. lane, drivable area) drawn translucent.
* **Layers** and **classes** can be switched on and off (keys 1–6), and the **confidence slider(s)** adjust the
  thresholds. All of this redraws instantly, without re-running the model.
* Mouse wheel zooms, drag pans, double-click fits.

| | |
|---|---|
| **Zoom on a box** (click it) | 3× crop, confidence, and every attribute with its top-3 values. In GT mode, also the matched GT box. |
| **Side by side** (T) | Ground truth (or the raw input) on the left, prediction on the right. |
| **Flag** (F) | Mark a frame with a category, a note and an optional box. Saved with images; markers appear on the seek bar. |
| **Snapshot** (S) | Saves the raw and the annotated frame as PNG. |

![Zoom](docs/screenshots/zoom.png)

### Statistics panel
Detections per class in the current frame and for the whole session; timings and FPS; and, with ground truth,
TP / FP / miss for the frame, running precision and recall, and **Folder metrics…**.

### Playback
Play/pause, step, seek, and speed from 0.25× to 4×, or Max. While saving, every frame is processed, at any speed.

### Output options
| Option | |
|---|---|
| Save annotated video / images | Video for video/camera/RTSP input; an annotated image folder (same names) for image input. On by default. |
| Image folder: also save an mp4 | Also turns the annotated images into one video. |
| Save side by side | Saves the side-by-side view (double width). |
| Counts overlay strip | Per-class counts across the top of each saved frame. |
| Gahan AI logo watermark | Company logo in the bottom-right corner of saved frames. |
| Export detections | Every box of every frame as JSONL and CSV. |
| Include masks (RLE) | Adds the masks to the JSONL export. |
| Session summary report | One `summary.html` to share: model, input, speed, counts, flagged frames, GT metrics. |

### Menus
**File:** open input, open model, open output folder. **View:** show/hide the Setup, Statistics and Log panels
(panels can be moved or undocked). **Help:** shortcuts, **Model info** (details of the loaded model), About.

The app remembers your last model, inputs, settings and window layout.

---

## Keyboard shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| Space | Play / pause | F | Flag frame |
| ← / → | Step back / forward | S | Snapshot |
| Home / End | First / last frame | T | Side by side |
| 1 – 6 | Toggle layers | Z | Click-to-zoom on/off |
| Ctrl+O | Open input | Ctrl+M | Open model |
| F1 | Help | | |

---

## Output files

Each run gets its own folder, `<output folder>/<date>_<time>_<input>/`:

| File | |
|---|---|
| `<input>_annotated.mp4` / `<folder>_annotated/` | Annotated video / images (same size and fps / file names as the input). |
| `detections.jsonl`, `detections.csv` | One line per frame / one row per box, with class, confidence, box (pixels) and attributes. |
| `summary.html` | Self-contained report; opens offline. |
| `metrics.json`, `metrics.csv` | Ground-truth metrics (when GT is loaded and an export or the summary is on). |
| `flags/` | `flags.csv` + raw and annotated PNG per flag. |
| `snapshots/` | Raw and annotated PNG per snapshot. |
| `run_info.json`, `viewer.log` | Model name, version and checksum, settings, and the run log. |

---

## Ground truth

| Format | Pick |
|---|---|
| CVAT XML 1.1 (images or video) | the `.xml` file, or a folder of `.xml` files (searched recursively) |
| COCO JSON | the `.json` file |
| YOLO txt | the label folder (class names from `classes.txt`, `data.yaml` or `obj.names`, else the app asks) |

* The format is detected automatically. Frames are matched to GT by file name (video GT by frame number).
* **Labels** are matched to the model's classes ignoring case, spaces, `_` and `-`, plus the aliases in
  `config/label_aliases.yaml`. Unknown labels open a dialog: map them or ignore them. The choice is remembered, and
  ignored labels are listed in the report.
* **Attributes** are compared using the rules in `config/gt_attributes.yaml`.
* **Per frame:** predictions are matched to GT at IoU ≥ 0.5. TP is drawn solid, FP dashed red, misses in orange.
* **Folder metrics:** per class GT count, precision, recall, **AP50** and **AP50-95** (COCO-style, using all
  predictions), the mean over classes, attribute accuracy, and mask IoU.

![Ground truth side by side](docs/screenshots/gt_side_by_side.png)

---

## Command line

Everything except the interactive parts runs headless:
```bash
./run_linux.sh --cli --model models/<model>.onnx --input video.mp4 --out runs/ --export-dets --summary
./run_linux.sh --cli --model models/<model>.onnx --input images/ --recursive --gt annotations.json --out runs/
```
`./run_linux.sh --cli --help` lists all options: device, thresholds, layers, GT format, label mapping and more.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Header shows **CPU** on a GPU machine | Run the setup script again. Check with `.venv/bin/python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"`; it must list `CUDAExecutionProvider`. Never install both `onnxruntime` and `onnxruntime-gpu`. |
| Slow on CPU | Expected for large models. Use speed **Max** for files. |
| RTSP does not connect | Test the URL in VLC first. For unstable streams set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`. |
| Saved video does not play in a browser or PowerPoint | It is MPEG-4 (plays in VLC). Convert: `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4` |
| "description does not match the model" | The message names the output whose name or shape is wrong in the description. |

---

## Adding a model

1. Export the model to ONNX. Boxes must come out after NMS as `[N, 6]` = `x1, y1, x2, y2, confidence, class_id`.
2. Write a description YAML (format below).
3. Package it into one file and drop it into `models/`:
   ```bash
   python tools/package_model.py --onnx exported.onnx --description my_model.yaml --out models/my_model.onnx
   ```
   This checks the description against the model's outputs and embeds it in the file. Weights are unchanged.

**Description format** (every section except `input` is optional):
```yaml
schema: gahan-model-description/1
name: My model
version: "1.0"

input:   {name: images, width: 640, height: 640, batch: 1, layout: NCHW, resize: letterbox,   # or stretch
          pad_value: 114, color: RGB, scale: 0.00392156862745098, dtype: float32}

detection:
  output: detections                    # [N, 6]
  classes: [car, person, sign, plate]   # in class-id order
  conf_threshold: 0.3

attributes:                             # per-box classification heads
  heads:
    - {name: occlusion, output: attr_occlusion, values: [low, medium, high], indexing: slot}
    - {name: sign_type, output: sign_cls, values: [stop, yield], indexing: slot_map, slot_map: sign_slots}
  applies_to: {sign: [sign_type]}       # heads shown for specific classes
  always: [occlusion]                   # heads shown for every class

extra_boxes:                            # an extra box source merged into one class (optional)
  - {output: plate_boxes, class: plate, merge_with_detection_class: true, nms_iou: 0.5, threshold: 0.45,
     threshold_label: Plate cutoff}

masks:
  - {name: lane, output: lane_logits, type: binary, activation: sigmoid, threshold: 0.5, color: "#FFD400"}
  - {name: drivable, output: drivable_logits, type: classes, class_names: [background, road],
     ignore: [background], colors: {road: "#22C55E"}}

tags:                                   # frame-level classification (optional)
  - {name: weather, output: weather_logits, values: [clear, rain], activation: softmax}
```
* `indexing: slot`: row *k* of the head belongs to detection *k*. `indexing: slot_map`: row *j* belongs to the
  detection given by `<slot_map output>[j]`.
* A leading batch axis on any output is fine; the frame goes in slot 0.
* A model opened without a description shows a template filled from its outputs, as a starting point.

---

## Build from source

```bash
git clone git@github.com:nicholas-gahan-ai/Model_Validation_Tool.git && cd Model_Validation_Tool
cp /path/to/<model>.onnx models/                  # models are not in git
./setup_linux.sh                                  # or by hand, see below
./run_linux.sh
```

By hand:
```bash
python3 -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "onnxruntime-gpu[cuda,cudnn]>=1.20"   # or, CPU only: pip install "onnxruntime>=1.20"
python app.py                                     # add --cli for headless mode
```

**Tests:** `pip install -r requirements-dev.txt`, then `python -m pytest tests`.

**Release zip:** set the version in `VERSION`, then run `python tools/make_release.py` → `dist/validation_tool_<version>.zip`
(app + `models/` + scripts). Before handing it out, unzip it into an empty folder and run the setup script.

Windows release test: [docs/windows_checklist.md](docs/windows_checklist.md).
