"""Output check: saved outputs match the input and the live render; export files are well-formed.

  GAHAN_BUNDLE=~/validation_tool_bundle python tools/verify_outputs.py --out /tmp/vv_verify [--device gpu]

1. video:  full night1 clip through the CLI (save video + exports + summary); read back frame count, fps, size;
           5 frames re-rendered from scratch and compared with the saved video (mp4 is lossy: PSNR, not bit-equal;
           the live render itself is compared bit-exact with a second render of the same frame).
2. folder: one Open ADAS scene -> annotated folder with the same file names.
3. exports: detections.jsonl / .csv schema, summary.html self-contained (no external URLs, images embedded).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.render import Renderer, RenderOptions  # noqa: E402
from core.runners import make_runner  # noqa: E402
from core.settings import bundle_path  # noqa: E402

MODEL = ROOT / "models/open_adas_v13s_single.onnx"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def cli(*args):
    r = subprocess.run([sys.executable, str(ROOT / "app.py"), "--cli", *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--video", default=None, help="default: the bundle's night1_first3min.mp4")
    a = ap.parse_args()
    B = bundle_path()
    video = Path(a.video) if a.video else B / "data/videos/night1_first3min.mp4"

    # ---- 1. video
    out = a.out / "video"
    r = cli("--model", MODEL, "--input", video, "--out", out, "--device", a.device, "--export-dets", "--summary")
    check("video run exits 0", r.returncode == 0)
    saved = out / f"{video.stem}_annotated.mp4"
    ci, cs = cv2.VideoCapture(str(video)), cv2.VideoCapture(str(saved))
    n_in, n_out = int(ci.get(cv2.CAP_PROP_FRAME_COUNT)), int(cs.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in, fps_out = ci.get(cv2.CAP_PROP_FPS), cs.get(cv2.CAP_PROP_FPS)
    size_in = (int(ci.get(3)), int(ci.get(4))); size_out = (int(cs.get(3)), int(cs.get(4)))
    check("saved frame count == input", n_in == n_out, f"{n_out} vs {n_in}")
    check("saved fps == input (OpenCV writer keeps 2 decimals)", abs(fps_in - fps_out) <= 0.01, f"{fps_out} vs {fps_in}")
    check("saved size == input", size_in == size_out, f"{size_out} vs {size_in}")
    runner = make_runner(MODEL, a.device)
    ren = Renderer(runner.desc)
    opts = RenderOptions()
    worst, exact = 99.0, True
    for i in np.linspace(0, n_in - 1, 5).astype(int):
        ci.set(cv2.CAP_PROP_POS_FRAMES, int(i)); cs.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f = ci.read(); ok2, s = cs.read()
        if not (ok1 and ok2):
            worst = 0; continue
        live = ren.render(f, runner.infer(f), opts)
        again = ren.render(f, runner.infer(f), opts)
        exact &= bool(np.array_equal(live, again))
        worst = min(worst, psnr(live, s))
    check("live render is deterministic (bit-exact on re-render, 5 frames)", exact)
    check("saved frames == live render up to mp4 compression (PSNR >= 30 dB, 5 frames)", worst >= 30, f"min PSNR {worst:.1f} dB")

    # ---- 3. exports from the video run
    lines = (out / "detections.jsonl").read_text().splitlines()
    hdr = json.loads(lines[0]).get("header", {})
    frames = [json.loads(l) for l in lines[1:]]
    keys = {"frame", "timestamp_s", "name", "image_size", "boxes", "masks", "tags", "timing_ms"}
    check("detections.jsonl: header with model name/version/sha256 + app version",
          all(hdr.get(k) for k in ("model_name", "model_version", "model_sha256", "app_version")))
    check("detections.jsonl: one line per frame, schema", len(frames) == n_in and all(keys <= set(f) for f in frames),
          f"{len(frames)} lines")
    bx = [b for f in frames for b in f["boxes"]]
    check("detections.jsonl: box schema", all({"cls", "conf", "xyxy", "slot", "attributes"} <= set(b) for b in bx),
          f"{len(bx)} boxes")
    with open(out / "detections.csv", encoding="utf-8") as fh:
        first = fh.readline()
        rows = list(csv.DictReader(fh))
    check("detections.csv: provenance line + one row per box", first.startswith("# app=") and len(rows) == len(bx),
          f"{len(rows)} rows")
    html = (out / "summary.html").read_text(encoding="utf-8")
    ext = re.findall(r"""(?:src|href)\s*=\s*['"](https?:)?//""", html)
    check("summary.html self-contained (no external src/href), wordmark embedded",
          not ext and "data:image/png;base64," in html, f"{len(html) / 1e3:.0f} kB")

    # ---- 2. folder
    scene = B / "data/oa_val/images/scene_0046"
    out2 = a.out / "folder"
    r = cli("--model", MODEL, "--input", scene, "--out", out2, "--device", a.device)
    src_names = sorted(p.name for p in scene.iterdir())
    dst_names = sorted(p.name for p in (out2 / "scene_0046_annotated").iterdir())
    check("image folder -> annotated folder with the same file names", r.returncode == 0 and src_names == dst_names,
          f"{len(dst_names)} files")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
