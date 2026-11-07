# Gahan AI Validation Viewer

A desktop tool for the validation team. It loads **any of our perception models from one model file**, runs it on a
**video, an image folder, a USB camera or an RTSP stream**, draws **every output live**, saves the annotated result, and
optionally **compares the model with ground-truth annotations** and computes accuracy.

Runs on Linux and Windows, on an NVIDIA GPU when one is present (ONNX Runtime CUDA) and on the CPU otherwise.

![Live view](docs/screenshots/main_video.png)

**Contents:** [Install](#install) · [Quick start](#quick-start) · [Features](#features) ·
[Keyboard shortcuts](#keyboard-shortcuts) · [Output files](#output-files) · [Ground truth](#ground-truth) ·
[Command line](#command-line) · [Troubleshooting](#troubleshooting) · [Packaging a new model](#packaging-a-new-model) ·
[Validation](#validation) · [Windows checklist](#windows-checklist) · [Build from source](#build-from-source) ·
[For developers](#for-developers)

---

## Install

You need **Python 3.9 or newer** (tested with 3.10). For the GPU build you need an NVIDIA driver that supports CUDA 12
(driver 525 or newer). You do **not** need to install CUDA or cuDNN: the setup installs them from NVIDIA's pip wheels.

There are two ways to get the app:
* **Release zip** (`validation_tool_<version>.zip`): the app *and* the model file, ready to install. Use this on the
  validation machines.
* **This GitHub repository**: the app only. **Model files are not stored in git.** After cloning, copy the model file
  (`open_adas_v13s_single.onnx`, from the release zip or the team's model share) into `models/`. See
  [Build from source](#build-from-source).

### Linux

```bash
unzip validation_tool_1.0.0.zip
cd validation_tool_1.0.0
./setup_linux.sh          # GPU build if nvidia-smi finds a GPU, otherwise the CPU build
./run_linux.sh            # or "Validation Viewer" in the application menu
```

| Option | Effect |
|---|---|
| `./setup_linux.sh --cpu` | Force the CPU build (for example on a machine whose GPU you do not want to use) |
| `./setup_linux.sh --with-ultralytics` | Also install Ultralytics so `.pt` models open (adds about 2 GB, PyTorch) |

`setup_linux.sh` creates a virtual environment in `.venv/`, installs everything into it, prints the ONNX Runtime
providers it found (look for `CUDAExecutionProvider` on a GPU machine) and adds a launcher with the Gahan AI icon to
the application menu. If it says `python3-venv is missing`, run `sudo apt install python3-venv` and run it again.

### Windows

1. Install Python 3.10+ from <https://www.python.org/downloads/> and tick **Add python.exe to PATH**.
2. Unzip `validation_tool_1.0.0.zip`.
3. Double-click **`setup_windows.bat`** (same options as Linux: `--cpu`, `--with-ultralytics`).
   It creates a **Validation Viewer** shortcut on the desktop.
4. Start with the shortcut or **`run_windows.bat`**.

---

## Quick start

1. **Model.** The model list shows every model file in `models/`. `open_adas_v13s_single.onnx` (the V13-S single
   model) is pre-selected. **Open…** loads a model from anywhere, and **ⓘ** shows what the model outputs.
2. **Input.** Pick *Video*, *Image folder*, *Camera* or *RTSP* and choose the file, folder, camera index or URL.
3. **Ground truth.** If you have annotations, pick the file or folder to compare the model with GT
   (see [Ground truth](#ground-truth)).
4. **Outputs.** *Save annotated video / images* is ticked by default. *Export detections* and *Session summary report*
   are written only when you tick them. Every run writes into its own sub-folder of the *Output folder*.
5. **Device.** *Auto* uses the GPU when ONNX Runtime can, otherwise the CPU. The device in use is shown top right
   (green = GPU).
6. Press **Start**. **Space** pauses, **←/→** step one frame, and the seek bar jumps.

---

## Features

### One model file, any model
A model is **one file**. For an ONNX model, the file carries a *description* of its inputs and outputs inside its
metadata, so the app needs no model-specific code. Class names, attribute heads, masks, colours and thresholds all
come from that description. Today's **V13-S single model** (detector, sign/light crop classifier and plate stage in one
graph) ships in `models/`. To add a future model, see [Packaging a new model](#packaging-a-new-model).

* **ONNX with an embedded description**: the normal case.
* **Ultralytics `.pt`** (detection or segmentation): class names are read from the model. Needs `--with-ultralytics`.
* **ONNX without a description**: the app asks once. You can edit a template it fills from the graph, or **Load YAML…**.
  The description is then saved as `<model>.description.yaml` beside the model file, and the app does not ask again.

If an fp16 and an fp32 file of the same model (same `name` and `version`, different `precision`) are both in `models/`,
the list shows the fp32 file when running on the CPU and the fp16 file otherwise.

### Inputs
| Input | Notes |
|---|---|
| Video file | mp4, avi, mov, mkv, … Seek, step and pacing at the video's fps. |
| Image folder | jpg, jpeg, png, bmp (any case), natural sort order (`img2` before `img10`). Tick *include subfolders* for a folder tree. The file name is shown per frame. |
| Camera | Device index 0–5. **Probe** lists the cameras found. Live input always processes the newest frame and drops the stale ones. |
| RTSP | A live network camera stream: IP / CCTV cameras, dashcams, a Jetson in a test vehicle. Type the camera's address, `rtsp://[user:pass@]host:port/path` (e.g. `rtsp://admin:pass@192.168.1.20:554/stream1`). Credentials are never shown or logged. Live, with frame dropping. The stream reconnects on its own (backoff 0.5 s up to 8 s). |

**Image folder: include subfolders.** Tick it when the images sit in sub-folders, for example a dataset with one folder
per scene (`images/scene_0046/`, `images/scene_0047/`, …). All images in all sub-folders are then processed in one run
(one GT comparison over the whole dataset), and the annotated output keeps the same folder layout. Unticked, only the
images directly inside the chosen folder are used.

### Live view
* Draw order: class-map masks (for example drivable area) → binary masks (for example lane) → boxes → labels.
  Masks are drawn translucent. Their colours come from the model description, or from a built-in palette.
* **A distinct colour per class**, from a stable 20-colour palette assigned in class-id order. To override it, edit
  `config/colors.yaml`, keyed by class name.
* **Labels**: `class conf`. For crop-classifier heads (sign type, light state) the value goes on the first line, e.g.
  `traffic sign 0.88 | Speed Limit 40`. Then one line per attribute that applies to the class, e.g.
  `person_work: Walking 0.91`. An attribute below the model's display threshold is dimmed and marked `?`.
* **Layers** (keys 1–6): Boxes, Attributes, the "always" attributes (occlusion/truncation for the V13-S model, off by
  default), and every mask. The layer list is built from the model.
* **Classes**: one toggle per class.
* **Thresholds**: a confidence slider, plus one slider per extra box source (for the V13-S model, *Plate cutoff* for
  the merged detector + plate-stage number plates).
  **Toggles and sliders re-draw the current frame instantly from the cached model outputs. Inference is not re-run.**
* **Zoom and pan**: mouse wheel zooms, drag pans, double-click fits. In side-by-side mode both panes zoom together.

### Zoom on a box
Click a box (click-to-zoom is on by default; **Z** turns it off). The dialog shows:
* a 3× crop with some context around the box;
* class, confidence, detection slot, source and box size;
* **every attribute head with its top-3 values and probabilities**. Heads that do not apply to the class are shown greyed;
* in GT mode, the matched GT box (dashed white), its IoU and its attributes (green where they agree, red where they do not).

![Zoom](docs/screenshots/zoom.png)

### Side by side (T)
Left pane: the **ground truth** when GT is loaded, otherwise the raw input. Right pane: the prediction.
Tick *Save side by side* to save the doubled-width view.

### Flag frames (F)
Choose a category, write a note and optionally drag a box on the frame. Categories: Missed object / False detection /
Wrong class / Wrong attribute / one per mask of the model / Other. Each flag is saved to `flags/flags.csv` with a raw and
an annotated PNG. Flags appear as red markers on the seek bar; click a marker to jump to that frame.

![Flag](docs/screenshots/flag.png)

### Snapshot (S)
Saves the raw frame and the annotated view (as shown, including side by side) as PNGs to `snapshots/`. The view flashes
and a "Snapshot saved" message confirms it. Snapshot and Flag also work on the last frame after a run has ended.

### Output options
| Option | What it does |
|---|---|
| Save annotated video / images | The annotated video (video, camera, RTSP) or an annotated image folder (image input). On by default. |
| Image folder: also save an mp4 | For image-folder input, also writes the annotated images as one mp4 (10 fps). |
| Save side by side (GT / input \| prediction) | Saves the doubled-width view: GT (or the raw input) on the left, prediction on the right. |
| Counts overlay strip | Draws the per-class counts of each frame as a strip across the top of the saved frames. |
| Gahan AI logo watermark | Puts the Gahan AI logo in the bottom-right corner of the saved frames. Off by default. |
| Export detections (JSONL + CSV) | Every box of every frame, with attributes and probabilities (see [Output files](#output-files)). |
| Include masks (RLE) | Adds the lane / drivable masks to `detections.jsonl` as run-length encoding (bigger files). |
| Session summary report (HTML) | One self-contained `summary.html` per run, to share or attach to a report. |
| Output folder | Each run gets its own sub-folder here, named `<date>_<time>_<input>`. |

### Menus and panels
* **File**: Open input (Ctrl+O), Open model (Ctrl+M), Open output folder (the current run's folder), Quit.
* **View**: show or hide the Setup, Statistics and Log panels, and Fit view. The panels can also be dragged, undocked
  or closed; the layout is remembered.
* **Help**: Keyboard shortcuts (F1), **Model info** (everything about the *currently loaded* model, read from the model
  file itself: classes, attribute heads, masks, thresholds, version and sha256; the same as the **ⓘ** button), About.

### Statistics panel
* **Detections (frame)**: colour swatch, class and count for the classes present (tick *all classes* to list every
  class), plus the total. **Session totals** gives per-class totals and the number of frames with at least one detection.
* **Frame**: index, timestamp or file name, pre-process / inference / post-process / render times in ms, display FPS,
  device and provider. For live input it also shows dropped frames, the stream state and reconnects.
* **Ground truth** (when loaded): this frame's TP / FP / miss per class, attribute agreement on the matched boxes,
  running precision and recall, and the **Folder metrics…** button.

### Playback
Space play/pause, ←/→ step (pauses), Home/End, seek bar, and speed 0.25×, 0.5×, 1× (real time), 2×, 4×, Max.
* **Real time** paces playback to the video's fps. When the computer is slower, it drops *display* frames.
* **When anything is being saved, every frame is processed and saved**, whatever the speed setting.
* Each frame is saved once, in the order it was played. If you seek forward while saving, the saved video skips the
  frames you jumped over.

### Robustness
The app remembers the last model, folders, toggles, thresholds (per model) and window layout. Errors are explained
in dialogs: an unreadable video, an empty folder, an RTSP timeout, a description that does not match the model, or a
GT parse error with its line number. A frame that fails is skipped, logged and counted in the summary; it never stops
the run. Logs go to `viewer.log` in the run folder and to **View → Log**.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | Step one frame back / forward (pauses) |
| Home / End | First / last frame |
| F | Flag the frame |
| Z | Click-to-zoom on / off |
| S | Snapshot (raw + annotated PNG) |
| T | Side-by-side view |
| 1 – 6 | Toggle layers (Boxes, Attributes, Occlusion/Truncation, Lane, Drivable for the V13-S model) |
| Ctrl+O | Open input |
| Ctrl+M | Open model |
| F1 | Shortcut help |
| Mouse | Wheel: zoom · drag: pan · double-click: fit · click a box: zoom dialog |

---

## Output files

Each run writes to `<output folder>/<YYYYmmdd_HHMMSS>_<input name>/`:

| File | When | Content |
|---|---|---|
| `run_info.json` | always | Model name, version and sha256, app version, device, input, thresholds, frames processed, files written. This gives the videos and images in the folder their provenance. |
| `viewer.log` | always | Log of the run, including skipped frames. |
| `<video>_annotated.mp4` | *Save annotated video* | Same frame size and fps as the input (doubled width when saving side by side). Camera/RTSP: saved at the measured fps. *Gahan AI logo watermark* adds the company logo in the bottom-right corner (off by default). |
| `<folder>_annotated/` | *Save* + image folder | Annotated images with the same file names (and sub-folders); optionally an mp4 as well, at 10 fps. |
| `detections.jsonl`, `detections.csv` | *Export detections* | All boxes per frame (see below). |
| `summary.html` | *Session summary report* | One self-contained file (images embedded, opens offline): model, input, latency (mean / p95), FPS, per-class totals, attribute value distributions, flagged frames with thumbnails, GT metrics. |
| `metrics.json`, `metrics.csv` | GT loaded + *Export* or *Summary* | Folder metrics (see [Ground truth](#ground-truth)). |
| `flags/` | after a flag | `flags.csv` and `frame_<n>_raw.png` / `frame_<n>_annotated.png`. |
| `snapshots/` | after S | `frame_<n>_raw.png` / `frame_<n>_annotated.png`. |

Every export records the model name, version and sha256 and the app version: as the first JSON line, the CSV
comment line or the `flags.csv` columns.

**`detections.jsonl`**: line 1 is `{"header": {...provenance...}}`, then one line per frame:
```json
{"frame": 0, "timestamp_s": 0.0, "name": "day1.mp4#0", "image_size": [1920, 1080],
 "boxes": [{"cls": "car", "conf": 0.9619, "xyxy": [1172.31, 475.96, 1538.08, 702.69], "slot": 0, "source": "detection",
            "attributes": {"occlusion": {"value": "<10%", "prob": 0.9072, "top3": [["<10%", 0.9072], ["10-50%", 0.08], ["50-75%", 0.01]]}}}],
 "masks": {"lane": {"type": "binary", "area_px": {"lane": 51234}},
           "drivable": {"type": "classes", "area_px": {"background": 1500000, "direct": 400000, "alternative": 173600}}},
 "tags": {}, "timing_ms": {"preprocess": 4.2, "inference": 23.1, "postprocess": 11.8, "render": 5.6}}
```
Coordinates are in original image pixels. `slot` is the model's detection slot (−1 for boxes from an extra box source,
for example the plate stage). With *Include masks (RLE)*, each mask also carries a column-major COCO-style
uncompressed RLE.

**`detections.csv`**: one row per box: `frame, timestamp_s, name, class, conf, x1, y1, x2, y2, slot, source`, then
`<head>, <head>_prob` for every attribute head of the model (empty when the head does not apply to the class).

**`flags.csv`**: `frame, timestamp_s, file, category, note, box_x1..box_y2, model_name, model_version, model_sha256,
app_version, flagged_at`.

---

## Ground truth

### Formats
| Format | What to pick | Notes |
|---|---|---|
| **CVAT XML 1.1**, "CVAT for images" | the `.xml` file | `<image name>` with `<box>`, `<polyline>`, `<polygon>` and `<attribute>` children. |
| **CVAT XML 1.1**, "CVAT for video" | the `.xml` file | `<track>` / `<box frame=…>` (`outside="1"` skipped). Matched to video frames by frame number. |
| **Open ADAS original XML** | the annotation folder | The same CVAT XML, one file per image (`scene_XXXX/<frame>.jpg.xml`), searched recursively. |
| **COCO JSON** | the `.json` file | `images`, `annotations` (`bbox` xywh, optional `attributes`, polygon `segmentation`), `categories`. |
| **YOLO txt** | the label folder | One `.txt` per image (`cls cx cy w h`, normalised). Names from `classes.txt`, `data.yaml` or `obj.names` (searched in the folder and two parents); otherwise the app asks for a names file. Boxes only. |

**Auto-detect**: a `.xml` file means CVAT; a folder of `.xml` files means Open ADAS; a `.json` with `images` and
`annotations` means COCO; a folder of `.txt` files means YOLO.

**Frames are matched to GT by file name**: the relative path first, then the bare file name (when it is unique),
ignoring the extension for YOLO. CVAT video GT is matched by frame number. A frame without GT is counted and shown as
"no GT for this frame".

### Label mapping
GT labels are mapped to model classes, and to masks for polylines and polygons.
* Matching is case-insensitive, and space, `_` and `-` count as the same character.
* `config/label_aliases.yaml` lists known aliases, e.g. `number_plate|plate → Number plate`,
  `lane markings → lane`, `drivable_area → drivable/direct`. It also lists labels to ignore (`road boundaries`, the
  `Image` tag).
* A known label whose target the loaded model does not have (for example `lane markings` with a detection-only model)
  is **excluded automatically**.
* Any other label opens the **mapping dialog**: map it to a model class or mask, or ignore it. The choice is saved to
  `<GT folder>/.gahan_label_map.yaml`. A saved choice never overrides a class the loaded model actually has.
* **Excluded GT labels are listed in the metrics and the report with their counts; they are never dropped silently.**

### Attribute comparison
Raw GT attribute values are mapped to the model's attribute values by the rules in `config/gt_attributes.yaml`.
These were ported from the training data loader: key variants (`Work`/`work`, `Truncation`/`truncation`,
`Vehicle type`), the occlusion/truncation formats (`<10%`, `10–50%`, `10% – 50%`, `> 75%`) and the sign-type synonyms.
A value that does not map is counted as **not comparable**, separately from disagreements. An attribute that is not
annotated is not compared. YOLO (and COCO without `attributes`) give "N/A".

### Matching and metrics
* **Per frame** (shown live): for each class, predictions sorted by confidence are greedily matched to the unmatched GT
  box with the highest IoU ≥ 0.5.
  The prediction pane then shows **TP** as a solid box, **FP** as a dashed red box, and **misses** as orange GT boxes.
  Attribute agreement is counted on the matched pairs.
* **Folder metrics** (**Folder metrics…**, `metrics.json/csv`, `summary.html`), per class:
  * GT count, predictions at the threshold, and TP / FP / miss;
  * precision and recall at the current thresholds (IoU 0.5);
  * **AP50** and **AP50-95**, the same code as the project scorecard (`v15/scripts/scorecard.py`): top 100 detections
    per image per class, greedy matching per IoU threshold 0.50:0.05:0.95, 101-point interpolation;
  * **AP uses every prediction** (confidence threshold 0; extra box sources down to 0.01), as COCO does.
    The UI thresholds affect only precision and recall;
  * the mean over the classes present in both GT and model. A class with GT but no prediction at all is marked
    *no predictions* and left out of the mean, as in the scorecard;
  * attribute accuracy per head on matched boxes (IoU ≥ 0.5);
  * **mask IoU**. Lane: GT polylines drawn with `cv2.polylines` at **3 px** at original resolution (as
    `scripts/rasterize_open_adas_polylines.py`), against the predicted lane mask. Class masks: GT polygons per class.
    IoU is summed over the frames that have GT shapes for that mask.

![GT side by side](docs/screenshots/gt_side_by_side.png)
![Metrics](docs/screenshots/metrics.png)

---

## Command line

Everything except the interactive parts also runs headless:

```bash
./run_linux.sh --cli --model models/open_adas_v13s_single.onnx --input day1.mp4 --out runs/day1 --export-dets --summary
./run_linux.sh --cli --model models/open_adas_v13s_single.onnx --input val/images --recursive \
               --gt val/gt_coco.json --out runs/val --summary            # prints the GT metrics table
```

| Option | |
|---|---|
| `--model M --input I --out DIR` | Required. `I` is a video, an image folder, `camera:N` or an `rtsp://` URL. |
| `--device auto\|gpu\|cpu` | Default `auto`. |
| `--conf X`, `--threshold OUTPUT=X` | Thresholds (an extra box source by its output name, e.g. `plates=0.45`). |
| `--no-video`, `--folder-video` | Do not save the annotated video/images; for image folders also write an mp4. |
| `--export-dets`, `--include-masks`, `--summary` | The opt-in outputs. |
| `--hide LAYER`, `--hide-class C`, `--always-attrs`, `--counts-strip`, `--watermark` | Same as the UI toggles. |
| `--gt PATH`, `--gt-format auto\|cvat\|openadas\|coco\|yolo` | Compare with GT. |
| `--map 'LABEL=TARGET'`, `--save-gt-map` | Map an unknown GT label (`TARGET` = model class, mask, or `ignore`). |
| `--recursive`, `--max-frames N` | Folder recursion; stop after N frames. |

---

## Troubleshooting

**The GPU is not used** (the header says `CPU`, or a message says the CUDA provider failed to load)
* Check: `.venv/bin/python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"`
  (Windows: `.venv\Scripts\python`). It must list `CUDAExecutionProvider`.
* If only CPU providers are listed, the CPU build of ONNX Runtime is installed, or both builds are (they share one
  package folder). Run `./setup_linux.sh` again: it removes the wrong build and reinstalls the right one.
* Update the NVIDIA driver (CUDA 12 needs driver ≥ 525). `nvidia-smi` must work.
* `CUDA_VISIBLE_DEVICES=""` hides the GPU: the app then runs on the CPU (that is how the CPU fallback is tested).
* Do not `pip install onnxruntime` into the same environment by hand. Ultralytics' *export* command can do it on its
  own; the app disables that (`YOLO_AUTOINSTALL=false`).

**It is slow on the CPU.** The V13-S single model takes about 0.3 s per frame on a 16-core CPU (its graph has a fixed
batch of 2). Use *Max* speed for files. Live sources drop frames by design.

**RTSP**
* Test the URL in VLC first. Credentials go in the URL (`rtsp://user:pass@host:554/path`); the app hides them.
* If the stream stutters or never starts, force TCP: set the environment variable
  `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` before starting the app.
* "RTSP timeout or refused" means nothing answered within 10 s. After the stream drops, the app reconnects on its own;
  the Frame panel shows `reconnecting` and the reconnect count.

**Saved video codec.** The pip build of OpenCV has no H.264 encoder, so videos are written as MPEG-4 Part 2 (`mp4v`).
VLC and most players play it; some browsers and PowerPoint do not. To convert:
`ffmpeg -i day1_annotated.mp4 -c:v libx264 -crf 20 -pix_fmt yuv420p day1_h264.mp4`.
The OpenCV writer stores the fps with two decimals (14.661 → 14.66).

**The app crashes at start-up on Linux with a GLib "Settings schema" error.** This is the GTK3 Qt theme on some GNOME
setups. The app sets `QT_QPA_PLATFORMTHEME=generic` itself. If you start it some other way, set that variable.

**Qt "xcb" plugin errors** usually come from having `opencv-python` (non-headless) and PySide6 in the same
environment. The app's environment uses `opencv-python-headless`.

**"description does not match the model".** An output named in the description is missing from the graph, or its
shape disagrees (for example the number of values of a head). The message names the output.

---

## Packaging a new model

The app never needs a code change for a new model. You need:

1. **Export the model to ONNX.** It must output post-NMS boxes as `[N, 6]` = `x1, y1, x2, y2, conf, class_id`
   (all slots, sorted by confidence), plus any attribute heads, masks and frame-level tags. A leading batch axis is fine:
   the frame is placed in every batch slot and slot 0 is read.
2. **Write the description YAML** (schema below). Start from `tools/descriptions/open_adas_v13s_single.yaml` (everything),
   `tools/descriptions/yolo26s_coco_example.yaml` (detection only), or the template the app fills from the graph when you
   open the bare ONNX.
3. **Package it into one file:**
   ```bash
   python tools/package_model.py --onnx exported.onnx --description my_model.yaml --out models/my_model.onnx
   ```
   This checks that every output named in the description exists in the graph with a compatible shape (columns, number
   of values per head, mask channels, slot-map rows). It records the `sha256_graph` of the exported file and embeds the
   description as JSON in the ONNX metadata (`gahan_model_description`). The graph and weights are not changed.
4. **Drop the file into `models/`** (or use **Open…**). Check **ⓘ Model info**, then run it on a few frames.
5. If your GT uses new label names or attribute spellings, add them to `config/label_aliases.yaml` /
   `config/gt_attributes.yaml`.

### Description schema (`gahan-model-description/1`)

```yaml
schema: gahan-model-description/1        # checked; other versions are refused
name: Open ADAS V13-S single model        # shown in the header, reports and exports
version: v13s-single-2026-11-06
created: 2026-11-06
precision: fp16                           # optional; fp16 / fp32 twins of one name+version: fp32 is used on CPU
sha256_graph: <written by package_model.py>

input:                                    # REQUIRED
  name: image                             # graph input name
  batch: 2                                # fixed batch: the frame is repeated, slot 0 is used
  layout: NCHW                            # NCHW | NHWC
  width: 832
  height: 480
  resize: letterbox                       # letterbox (aspect kept, centred, padded) | stretch
  pad_value: 114
  color: RGB                              # RGB | BGR
  scale: 0.00392156862745098              # pixel * scale (1/255)
  dtype: float32                          # float32 | float16

detection:                                # optional
  output: raw_detections                  # [.., N, 6]
  format: xyxy_conf_cls
  coords: input_px                        # input_px (model-input pixels) | normalized (0..1)
  classes: [Number plate, bus, car, ...]  # class names in id order
  conf_threshold: 0.35                    # default of the Conf slider

attributes:                               # optional; needs detection
  display_prob_threshold: 0.5             # below it a value is shown dimmed with "?"
  activation: none                        # none (outputs are probabilities) | softmax
  heads:
    - {name: person_work, output: attr_person_work, values: [Standing, Walking, ...],
       indexing: slot, slots: 32}         # row k belongs to detection slot k (k < slots; omit slots = all rows)
    - {name: sign_type, output: sign_type, values: [...],
       indexing: slot_map, slot_map: sign_light_slots}   # row j belongs to slot <slot_map output>[j]
  applies_to: {person: [person_work], traffic sign: [sign_type], ...}   # heads shown per class
  always: [occlusion, truncation]         # heads for every class (a layer toggle of their own)

extra_boxes:                              # optional: more box sources, e.g. a plate stage
  - output: plates                        # [.., M, 5] = x1, y1, x2, y2, score
    format: xyxy_score
    coords: input_px
    class: Number plate
    merge_with_detection_class: true      # pool with the detector's boxes of that class, then NMS
    nms_iou: 0.5
    threshold: 0.45                       # default of its own slider
    threshold_label: Plate cutoff

masks:                                    # optional, any number
  - {name: lane, output: lane_logits, type: binary, activation: sigmoid, threshold: 0.5, color: "#FFD400"}
  - {name: drivable, output: drivable_logits, type: classes, class_names: [background, direct, alternative],
     ignore: [background], colors: {direct: "#22C55E", alternative: "#3B82F6"}}

tags:                                     # optional frame-level classification outputs
  - {name: road_type, output: road_type_logits, values: [urban, highway, rural], activation: softmax}
```

Rules:
* Every section except `input` is optional: a detection-only model has `detection`, a segmentation-only model `masks`.
* Unknown keys are ignored, so a description can carry notes.
* Each output's per-image rank is fixed by its role: boxes, heads and extra boxes 2, slot maps and tags 1, binary
  masks 2 (`[H, W]`), class masks 3 (`[C, H, W]`). Extra leading axes are batch axes and slot 0 is read.
  (`output_batch_dim` in older descriptions is accepted and ignored.)
* Masks are decoded at full resolution: the padded region is cropped, the **logits/scores** are resized to the original
  frame, and then thresholded (binary) or argmax-ed (classes).
* Attribute heads are evaluated for the slots that have rows. A box beyond the last row simply has no attributes.

### Example: a model the app has never seen

Ultralytics YOLO26-S (COCO, 80 classes) exported with `yolo export model=yolo26s.pt format=onnx imgsz=640 opset=17` has
`images [1,3,640,640] → output0 [1,300,6]` (NMS-free, sorted). Its whole description is
`tools/descriptions/yolo26s_coco_example.yaml`:

```yaml
schema: gahan-model-description/1
name: YOLO26-S COCO (example)
version: yolo26s-8.4.27
input: {name: images, batch: 1, layout: NCHW, width: 640, height: 640, resize: letterbox, pad_value: 114,
        color: RGB, scale: 0.00392156862745098, dtype: float32}
detection: {output: output0, format: xyxy_conf_cls, coords: input_px, classes: [person, bicycle, car, ...],
            conf_threshold: 0.25}
```

Packaged with `package_model.py` and run through the app, it reproduces the original `.pt` model: on 50 frames of
`day1.mp4`, 160 of 164 boxes match (same class, IoU > 0.9, mean IoU 0.996).

A model with attributes, slot maps, an extra box source and two masks: `tools/descriptions/open_adas_v13s_single.yaml`.
The metadata-less V13-S detector-only ONNX, whose attribute rows follow its kept boxes:
`tools/descriptions/open_adas_v13s_detector.yaml`.

---

## Validation

Measured on 2026-11-06 on a laptop with an **NVIDIA RTX 4060 Laptop GPU** (8 GB, driver 580) and a 16-thread CPU,
ONNX Runtime 1.23.2, against the reference bundle (`GAHAN_BUNDLE`). The tools are in `tools/` (see
[For developers](#for-developers)). The reference data (models, PyTorch reference outputs, Open ADAS val GT) is the
separate validation bundle, found through the `GAHAN_BUNDLE` environment variable.

### 1. The one model file reproduces the verified model
`models/open_adas_v13s_single.onnx` = the exported `open_adas_v13_s_single_lb832_b2.onnx` plus embedded metadata only.
The graph bytes are identical (graph sha256 `c6e4f752…`).

**Raw tensors vs the PyTorch reference (`.ref.npz`)**: the confident boxes match within 0.125 px, the attribute argmax
agrees 100 %, `sign_light_slots` is identical, and the lane / drivable masks agree on 99.999 % / 99.97 % of pixels.
The large differences in `raw_detections` are low-confidence slots swapping places where scores tie, as expected for
fp16.

**50 bench frames vs the stored PyTorch handler output** (deployed cutoffs 0.35 / plate 0.45; `tools/verify_decode.py bench`):

| | Boxes within 0.5 letterbox px (same class) | Same attribute values | Lane IoU (min) | Drivable IoU (min) |
|---|---|---|---|---|
| GPU, ONNX Runtime CUDA vs GPU fp16 reference | 371 / 372 = **99.73 %** | 371 / 372 = **99.73 %** | **0.9978** (0.9952) | **0.9988** (0.9941) |
| CPU, ONNX Runtime CPU vs CPU fp32 reference | 370 / 371 = **99.73 %** | 370 / 371 = **99.73 %** | **0.9970** (0.9926) | **0.9977** (0.9724) |

Required: ≥ 99 %, ≥ 99 %, IoU ≥ 0.99. The reference stores masks as polygons, so both sides are turned into polygons the
handler's way before the IoU. The one box outside 0.5 px on the GPU is a truck at 0.52 px. The one attribute
difference is an occlusion value on a 0.36-confidence truck (GPU) and a rider-pillion value on a 0.51 rider (CPU).

**All 1,207 Open ADAS val frames, AP50-95** (every slot scored, plate cutoff 0.01, GT `gt_coco_scoring.json`):

| | all 11 | car | Number plate |
|---|---|---|---|
| Required (reference) | 0.4686 ± 0.002 | 0.706 | 0.328 |
| `verify_decode.py ap`, GPU | **0.4685** | 0.7053 | 0.3277 |
| `verify_decode.py ap`, CPU | **0.4684** | 0.7057 | 0.3277 |
| **The app's own GT metrics** (`--gt gt_coco_scoring.json`, GPU) | **0.4686** | 0.7053 | 0.3277 |

Per class (app, GPU): Number plate 0.3277 · bus 0.4733 · car 0.7053 · motorcycle 0.3993 · other 0.4954 ·
person 0.3022 · rider 0.3953 · traffic light 0.2548 · traffic sign 0.6529 · truck 0.6443 · auto rickshaw 0.5041.
The reference values are 0.3276, 0.4728, 0.7060, 0.3977, 0.4957, 0.3018, 0.3957, 0.2552, 0.6536, 0.6446 and 0.5040.
Differences up to 0.0016 come from fp16 versus PyTorch, and from which of two equal scores comes first when images are
processed in a different order.

**Speed** (RTX 4060 Laptop): ONNX Runtime CUDA inference **22–24 ms** per frame (median; the graph's fixed batch of 2
runs both slots). CPU (16 threads): **316 ms** median. That is under the plan's 400 ms limit, so **no fp32 twin was
made** (it would also need the training checkpoints, which are not in the bundle). End to end on a 1080p video with
saving (1080p, video written to disk): **20 fps** on the GPU, **2.9 fps** on the CPU.

Important: without `onnxruntime.preload_dlls()`, ONNX Runtime silently falls back to the CPU when the CUDA libraries are
not on the library path. The app calls it and always shows the provider it actually got.

### 2. Generic, with no model-specific code
* `tests/test_generic.py` reads every class, attribute-head and mask name from the shipped descriptions and fails if any
  appears anywhere in `core/` or `ui/` (code or comments). It passes.
* In one session, three different models were loaded and run:
  * the packaged **V13-S single model**: 11 classes, layers Boxes / Attributes / Occlusion-Truncation / Lane / Drivable,
    sliders Conf + Plate cutoff;
  * **`yolo26s.pt`** (Ultralytics, 80 COCO classes): only the Boxes layer and Conf;
  * the **metadata-less V13-S detector ONNX**, through the description dialog (sidecar saved): 11 classes, 7 heads, 2 masks.

  The toggles, palette, counts and GT mapping rebuilt for each. With the Open ADAS GT, YOLO's missing classes
  (`other`, `Number plate`) were excluded automatically, and only the unknown `rider` was asked about.
* A brand-new model (YOLO26-S ONNX) was packaged with a 10-line description and reproduces its `.pt` model
  (see the [example](#example-a-model-the-app-has-never-seen)).

### 3. Ego lane / LDW
Not built: removed from the scope on 2026-11-06.

### 4. GT readers and metrics
`tools/convert_gt.py` converted the 837 Open ADAS val frames that have raw XML (scenes 0046–0056, 6,351 boxes) to COCO
JSON, YOLO txt + `classes.txt` and one CVAT XML 1.1. Run through the app (V13-S single model, thresholds 0.35 / 0.45),
**all four readers give identical box metrics**:

| Class | GT | Pred | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| Number plate | 451 | 526 | 0.5817 | 0.6785 | 0.5967 | 0.3174 |
| bus | 11 | 12 | 0.5833 | 0.6364 | 0.6643 | 0.6039 |
| car | 1531 | 1397 | 0.8618 | 0.7864 | 0.8875 | 0.7133 |
| motorcycle | 615 | 570 | 0.8439 | 0.7821 | 0.8682 | 0.5172 |
| other | 1079 | 656 | 0.8354 | 0.5079 | 0.6704 | 0.5303 |
| person | 35 | 12 | 0.9167 | 0.3143 | 0.5400 | 0.3692 |
| rider | 627 | 538 | 0.8327 | 0.7145 | 0.8417 | 0.4794 |
| traffic light | 82 | 48 | 0.7500 | 0.4390 | 0.5244 | 0.2871 |
| traffic sign | 104 | 76 | 0.9342 | 0.6827 | 0.8179 | 0.5712 |
| truck | 1816 | 1939 | 0.7390 | 0.7891 | 0.8312 | 0.6623 |
| **mean (10)** | | | **0.7879** | **0.6331** | **0.7242** | **0.5051** |

Attribute accuracy is identical for the three formats that carry attributes (Open ADAS XML, CVAT, COCO): occlusion
0.928 · truncation 0.983 · other_vehicle_type 0.971 · rider_pillion 0.788 · light_state 0.861 (31/36) ·
person_work 0.364 (4/11) · sign_type 0.929 (26/28; 43 GT values such as the generic `speed_limit` are not comparable).
Lane IoU against the 3-px rasterised GT polylines is 0.427 over 836 frames. For YOLO, attributes are N/A.
Excluded shapes: `road boundaries` (1,042).

The raw XML and `gt_coco_scoring.json` hold **the same 6,351 boxes** on those 837 frames: every box matched at
IoU > 0.999 with the same label. These frames contain no `Emergency vehicle` or `animal` boxes. The other 370 val
frames (`tsl_val`) have canonical JSON only, so the 1,207-frame check uses the COCO file.

Unit tests (`tests/`, 38 in all) cover matching (duplicates, IoU exactly 0.5, class mismatch, empty GT, empty prediction),
the AP port against the bundle's copy of the scorecard (5 random sets, equal to 1e-12), alias mapping and the
unknown-label path, saved mapping choices, attribute normalisation, the sources and the description validation.
IoU exactly 0.5 is not a match, as in the scorecard (its IoU adds 1e-9 to the union).

### 5. Outputs and inputs
`tools/verify_outputs.py`: **12 / 12 checks pass**.
* The saved video of the full night clip has 2,639 / 2,639 frames, 1920×1080 and 14.66 fps (input 14.661; the OpenCV
  writer keeps 2 decimals).
* Five frames re-rendered from scratch are bit-identical to each other, and match the saved video at PSNR ≥ 37.3 dB.
  mp4 is lossy, so the saved frames cannot be bit-equal.
* An image folder (76 images) gives an annotated folder with the same file names.
* The JSONL / CSV schemas, the provenance lines and the box counts check out (10,304 boxes in both), and `summary.html`
  is self-contained (no external links; images embedded).

| Check | Result |
|---|---|
| CPU fallback, `CUDA_VISIBLE_DEVICES=""`, device *Auto* | **Pass** on the first 300 frames of `day1.mp4`: the app reports the fallback, runs on the CPU (321 ms inference, 2.9 fps with saving), and the saved video has 300 / 300 frames, 15.096 fps, 1920×1080. The full 12,681-frame run (about 75 min) is still to do. |
| RTSP (play, frame dropping, reconnect) | **Not yet tested on a real stream.** The newest-frame / drop / reconnect-with-backoff logic is tested with a simulated stream (`tests/test_sources.py`), and a URL with nothing listening fails at once with a clear message. |
| USB camera | **Not yet tested with a camera.** A camera index that does not exist fails at once with a clear message. |
| Fresh install from the release zip (Linux, GPU) | **Pass.** `unzip` + `./setup_linux.sh` in an empty folder: `.venv` created, `CUDAExecutionProvider` found, desktop launcher written. The installed copy (NumPy 2.2, OpenCV 4.14, PySide6 6.9.3) reproduces **0.4686** on the 1,207 val frames (car 0.7056, plate 0.3276). |

### 6. UI walkthrough
`tools/dev/ui_test_full.py` drives the real window like a tester: it clicks the buttons, presses the keys, drags the
seek bar and checks the files written. **81 / 81 checks pass**:
* bad inputs (missing or corrupt video, empty folder, broken GT XML, a camera or RTSP URL that does not exist);
* every playback control, layer, class, slider and shortcut, snapshot, flag and zoom;
* folder and GT modes (Open ADAS XML, COCO, YOLO without names, GT that matches no frame, an unknown label →
  mapping dialog), including the full 1,207-frame val set through the window (AP50-95 **0.4686** in 69 s);
* model and device switching (.pt, metadata-less ONNX through the description dialog, an invalid file);
* settings persistence, typing in text fields, and closing the window during a run.

There were no Python exceptions. The screenshots in this README come from `tools/dev/screenshots.py`.

---

## Windows checklist

There was no Windows machine available for this build. Please run through this once and report anything that fails:

- [ ] `setup_windows.bat` finishes and prints `onnxruntime … providers: […CUDAExecutionProvider…]` on a GPU machine
      (`CPUExecutionProvider` only on a CPU machine), then `PySide6 / OpenCV / onnx OK`.
- [ ] A **Validation Viewer** shortcut with the π icon appears on the desktop. The app starts from it without a console
      window, and the **taskbar shows the π icon**.
- [ ] The header shows `GPU (CUDA)` on a GPU machine. **Device → CPU** switches to `CPU (CPU)` and back.
- [ ] Video: `day1.mp4` plays in real time. Space / ← / → / Home / End and the seek bar work.
- [ ] Image folder (with and without *include subfolders*) and the annotated output folder keep the file names.
- [ ] Camera: **Probe** finds the webcam, and live view works (if the machine has a camera).
- [ ] Saved `…_annotated.mp4` plays in VLC, with the same frame count as the input.
- [ ] Flag (F) with a drawn box → `flags\flags.csv` + PNGs; the marker appears on the seek bar and jumps there on click.
- [ ] Export detections + Summary report → `detections.jsonl/.csv`, `summary.html` opens offline with images.
- [ ] GT: `gt_coco_scoring.json` on the val images (*include subfolders*) → **Folder metrics…** shows mean AP50-95 ≈ 0.4686.
- [ ] Paths with spaces and non-ASCII characters (input, output folder) work.
- [ ] `setup_windows.bat --with-ultralytics`, then `yolo26s.pt` opens and runs.

---

## Build from source

Use this to work on the app, or to build the release zip yourself.

**1. Get the code and the model.**
```bash
git clone git@github.com:nicholas-gahan-ai/Model_Validation_Tool.git
cd Model_Validation_Tool
cp /path/to/open_adas_v13s_single.onnx models/      # model files are not in git (release zip or model share)
```

**2. Install.** The easiest way is the setup script: `./setup_linux.sh` (Linux) or `setup_windows.bat` (Windows), as
in [Install](#install). By hand, in a virtual environment:
```bash
python3 -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "onnxruntime-gpu[cuda,cudnn]>=1.20"        # NVIDIA GPU;  CPU only: pip install "onnxruntime>=1.20"
pip install -r requirements-optional.txt               # optional: Ultralytics .pt models (install BEFORE onnxruntime)
pip install -r requirements-dev.txt                    # tests + icon tool
```
Install only **one** of `onnxruntime` / `onnxruntime-gpu`. They share a package folder, and having both silently
disables the GPU.

**3. Run.**
```bash
python app.py                      # desktop app
python app.py --cli --help         # headless mode
```

**4. Test.**
```bash
python -m pytest tests                                             # unit tests (bundle tests are skipped)
export GAHAN_BUNDLE=~/validation_tool_bundle                         # the validation data bundle
python -m pytest tests                                             # + the tests that need the bundle
python tools/verify_decode.py bench --device gpu                   # decode parity vs the PyTorch reference
python tools/verify_decode.py ap --device gpu                      # 1,207-frame AP (expects 0.4686)
python tools/verify_outputs.py --out /tmp/vv_verify                # saved video / exports checks
python tools/dev/ui_test_full.py --work /tmp/vv_ui --quick         # end-to-end UI pass (opens the window)
```

**5. Package a model** (only when a new model is exported): see [Packaging a new model](#packaging-a-new-model).
```bash
python tools/package_model.py --onnx exported.onnx --description tools/descriptions/<model>.yaml --out models/<model>.onnx
```

**6. Build the release zip.**
```bash
echo 1.0.1 > VERSION                       # bump the version
python tools/make_release.py               # -> dist/validation_tool_1.0.1.zip (app + models/ + scripts + docs)
```
The zip includes whatever model files are in `models/`, and leaves out `.git`, `.venv`, caches and `tools/dev/`.
Test it like a user: unzip it into an empty folder and run `./setup_linux.sh`, then `./run_linux.sh`.

**Other tools**
* `python tools/make_icons.py`: rebuilds the π icon set (`assets/logo_icon*.png`, `.ico`) from `assets/logo_icon_src.jpg`.
* `python tools/dev/screenshots.py --out-runs <folder>`: re-takes the README screenshots in `docs/screenshots/`.
* `python tools/convert_gt.py --xml-root <annotations_xml> --out <dir>`: Open ADAS XML → COCO / YOLO / CVAT XML 1.1
  (used to check that the GT readers agree).

---

## For developers

```
app.py                     entry point (desktop app, or --cli)
core/                      model file + description, ONNX/Ultralytics runners, decode, sources, render,
                           session outputs, report, GT readers / mapping / matching / metrics, worker thread
ui/                        main window, panels, view, dialogs, theme (theme_dark.qss, icons/)
config/                    colors.yaml, label_aliases.yaml, gt_attributes.yaml
models/                    packaged model files (one file per model; not in git)
tools/package_model.py     ONNX + description YAML -> one model file
tools/descriptions/        description YAMLs (V13-S single, V13-S detector sidecar, YOLO26 example)
tools/verify_decode.py     decode parity (bench50) + 1,207-frame AP    needs GAHAN_BUNDLE
tools/verify_outputs.py    saved video + export checks                    needs GAHAN_BUNDLE
tools/convert_gt.py        Open ADAS XML -> COCO / YOLO / CVAT XML 1.1
tools/make_icons.py        transparent π icon set + .ico from assets/logo_icon_src.jpg
tools/make_release.py      dist/validation_tool_<version>.zip
tests/                     pytest
tools/dev/                 developer-only: screenshots.py, ui_test_full.py, ui_smoke.py
docs/screenshots/          README images
```

* **Reference data**: the checks read the validation bundle from the `GAHAN_BUNDLE` environment variable
  (`export GAHAN_BUNDLE=~/validation_tool_bundle`). Nothing in the app depends on it.
* **Tests**: `pip install -r requirements-dev.txt`, then `GAHAN_BUNDLE=… python -m pytest tests`. The tests that need the
  bundle are skipped without it.
* **Release**: put the version in `VERSION`, then run `python tools/make_release.py`. It leaves out
  caches and `tools/dev/`.
* **No PyTorch and no model source code ship with the app.** PyTorch comes in only with the optional Ultralytics install.
