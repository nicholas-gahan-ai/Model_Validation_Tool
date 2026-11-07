# Windows checklist

Run through this once on a Windows machine and report anything that fails.

- [ ] `setup_windows.bat` finishes and prints `onnxruntime … providers: […CUDAExecutionProvider…]` on a GPU machine
      (`CPUExecutionProvider` only on a CPU machine), then `PySide6 / OpenCV / onnx OK`.
- [ ] A **Validation Viewer** shortcut with the Gahan AI icon appears on the desktop. The app starts from it without a
      console window, and the taskbar shows the same icon.
- [ ] The header shows `GPU (CUDA)` on a GPU machine. **Device → CPU** switches to `CPU (CPU)` and back.
- [ ] A video plays in real time. Space / ← / → / Home / End and the seek bar work.
- [ ] Image folder (with and without *include subfolders*): the annotated output folder keeps the file names.
- [ ] Camera: **Probe** finds the webcam, and live view works (if the machine has a camera).
- [ ] The saved `…_annotated.mp4` plays in VLC, with the same frame count as the input.
- [ ] Flag (F) with a drawn box → `flags\flags.csv` + PNGs; the marker appears on the seek bar and jumps there on click.
- [ ] Export detections + Summary report → `detections.jsonl/.csv`; `summary.html` opens offline with its images.
- [ ] Ground truth on an image folder → **Folder metrics…** shows the same numbers as on Linux.
- [ ] Paths with spaces and non-ASCII characters (input, output folder) work.
- [ ] `setup_windows.bat --with-ultralytics`, then a `.pt` model opens and runs.
