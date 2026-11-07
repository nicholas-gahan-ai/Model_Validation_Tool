"""Headless run: python app.py --cli --model M --input I --out DIR [options]"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .model_file import NeedsDescription, open_model
from .render import Renderer, RenderOptions
from .runners import make_runner
from .session import ImageFolderSaver, LiveVideoSaver, VideoSaver
from .settings import app_version
from .sources import FolderSource, SourceError, make_source

log = logging.getLogger("viewer")


def build_parser():
    p = argparse.ArgumentParser(prog="app.py --cli", description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="video file, image folder, camera:N or rtsp://…")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--conf", type=float, help="detection confidence threshold (default: from the model)")
    p.add_argument("--threshold", action="append", default=[], metavar="OUTPUT=VALUE",
                   help="threshold for an extra box source, by its output name (OUTPUT=0.45)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-video", action="store_true", help="do not save the annotated video / images")
    p.add_argument("--folder-video", action="store_true", help="image folder input: also write an mp4")
    p.add_argument("--hide", action="append", default=[], metavar="LAYER",
                   help="hide a layer: boxes, attributes, or a mask name")
    p.add_argument("--hide-class", action="append", default=[])
    p.add_argument("--always-attrs", action="store_true", help="show the always-on attribute heads")
    p.add_argument("--counts-strip", action="store_true")
    p.add_argument("--watermark", action="store_true")
    p.add_argument("--export-dets", action="store_true", help="write detections.jsonl + detections.csv")
    p.add_argument("--include-masks", action="store_true", help="RLE masks in detections.jsonl")
    p.add_argument("--summary", action="store_true", help="write summary.html")
    p.add_argument("--recursive", action="store_true", help="image folder input: include subfolders")
    p.add_argument("--gt", help="ground truth: CVAT XML file or folder of .xml files, COCO JSON, or YOLO label folder")
    p.add_argument("--gt-format", default="auto", choices=["auto", "cvat", "coco", "yolo"])
    p.add_argument("--map", action="append", default=[], metavar="GT_LABEL=TARGET",
                   help="map a GT label to a model class / mask, or =ignore")
    p.add_argument("--save-gt-map", action="store_true", help="remember --map choices in the GT folder")
    return p


def run(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(a.out / "viewer.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    try:
        model = open_model(a.model)
    except NeedsDescription as e:
        print(f"ERROR: {e}. Open it once in the app to add a description, or write {Path(a.model).stem}"
              ".description.yaml beside it.", file=sys.stderr)
        return 2
    runner = make_runner(model, a.device)
    if runner.message:
        print("WARNING:", runner.message, file=sys.stderr)
    desc = runner.desc
    print(f"model: {model.display_name}  sha256 {model.sha256[:12]}…  device: {runner.device_label}  app {app_version()}")
    th = runner.decoder.default_thresholds()
    if a.conf is not None:
        th["conf"] = a.conf
    for kv in a.threshold:
        k, v = kv.split("=", 1); th[k] = float(v)
    opts = RenderOptions(boxes="boxes" not in a.hide, attributes="attributes" not in a.hide,
                         always_attrs=a.always_attrs, masks={m.name: m.name not in a.hide for m in desc.masks},
                         hidden_classes=set(a.hide_class), counts_strip=a.counts_strip, watermark=a.watermark)
    renderer = Renderer(desc)
    src = make_source(a.input)
    if a.recursive and isinstance(src, FolderSource):
        src.recursive = True
    gt_run = None
    if a.gt:
        from .gt import GtError
        from .gt_run import GtRun
        try:
            gt_run = GtRun.open(desc, a.gt, a.gt_format)
        except GtError as e:
            print("ERROR (ground truth):", e, file=sys.stderr)
            return 2
        if a.map:
            gt_run.mapper.set_user(dict(kv.split("=", 1) for kv in a.map), save=a.save_gt_map)
        missing = gt_run.unresolved_labels()
        if missing:
            print("ERROR: GT labels without a mapping: " + ", ".join(missing) +
                  "\n  add --map 'LABEL=<model class|mask|ignore>' (model classes: " + ", ".join(desc.class_names) + ")",
                  file=sys.stderr)
            return 2
        print(f"ground truth: {gt_run.gt.format}, {len(gt_run.gt)} frames from {a.gt}")
    try:
        src.open()
    except SourceError as e:
        print("ERROR:", e, file=sys.stderr)
        return 2
    from .session import Session
    sess = Session(a.out, model, runner, src, th, export_dets=a.export_dets, include_masks=a.include_masks,
                   summary=a.summary)
    savers = []
    if not a.no_video:
        if isinstance(src, FolderSource):
            savers.append(("folder", ImageFolderSaver(a.out / f"{src.path.name}_annotated")))
            if a.folder_video:
                savers.append(("video", VideoSaver(a.out / f"{src.path.name}_annotated.mp4", 10.0)))
        else:
            stem = Path(a.input).stem if not src.is_live else src.kind
            path = a.out / f"{stem}_annotated.mp4"
            savers.append(("video", LiveVideoSaver(path, src.fps or 15.0, src) if src.is_live
                           else VideoSaver(path, src.fps or 25.0)))
    n = bad = 0
    t_start = time.time()
    try:
        while a.max_frames is None or n < a.max_frames:
            fr = src.read()
            if fr is None:
                break
            try:
                raw = runner.infer_raw(fr.image, frame_index=fr.index, timestamp_s=fr.timestamp_s, name=fr.name)
                pred = runner.decoder.decode(raw, th)
                if gt_run is not None:
                    gt_run.process(fr, raw, pred, runner.decoder)
                t0 = time.perf_counter()
                img = renderer.render(fr.image, pred, opts)
                pred.timing_ms["render"] = (time.perf_counter() - t0) * 1000
            except Exception as e:  # noqa: BLE001 - never stop on one bad frame
                bad += 1
                log.exception("frame %s skipped: %s", fr.name, e)
                sess.bad_frame(fr)
                continue
            for kind, s in savers:
                s.write(img, fr.name) if kind == "folder" else s.write(img)
            sess.add(fr, pred)
            n += 1
            if n % 100 == 0:
                el = time.time() - t_start
                print(f"  {n}{'/' + str(src.count) if src.count else ''} frames  {n / el:.1f} fps", flush=True)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        src.close()
        for _, s in savers:
            s.close()
    el = time.time() - t_start
    if gt_run is not None:
        from .metrics import metrics_to_files
        m = gt_run.metrics(th)
        sess.gt_metrics = m
        print_metrics(m)
        if a.export_dets or a.summary:
            sess.written += metrics_to_files(m, a.out, sess.provenance())
    sess.finish()
    inf = [p for p in sess.latency["inference"]]
    print(f"done: {n} frames in {el:.1f} s ({n / max(el, 1e-9):.1f} fps)  skipped {bad}  "
          f"inference avg {np.mean(inf) if inf else 0:.1f} ms  p95 {np.percentile(inf, 95) if inf else 0:.1f} ms")
    for kind, s in savers:
        print(f"  saved {kind}: {getattr(s, 'path', getattr(s, 'out_dir', ''))}  ({s.frames} frames"
              f"{', codec ' + s.codec if getattr(s, 'codec', None) else ''})")
    for p in sess.written:
        print("  wrote", p)
    return 0


def print_metrics(m: dict):
    fmt = lambda x: "  —   " if x != x else f"{x:.4f}"  # noqa: E731
    print(f"GT metrics: {m['images_matched']} frames with GT ({m['frames_without_gt']} without)  "
          f"thresholds {m['thresholds']}")
    print(f"  {'class':16s} {'GT':>6s} {'pred':>6s} {'prec':>7s} {'recall':>7s} {'AP50':>7s} {'AP50-95':>7s}")
    for c, r in m["per_class"].items():
        print(f"  {c:16s} {r['gt']:6d} {r['pred']:6d} {fmt(r['precision']):>7s} {fmt(r['recall']):>7s} "
              f"{fmt(r['ap50']):>7s} {fmt(r['ap50_95']):>7s}{'  (no predictions)' if r['no_predictions'] else ''}")
    mm = m["mean"]
    print(f"  {'mean (' + str(mm['n_classes']) + ' cls)':16s} {'':6s} {'':6s} {fmt(mm['precision']):>7s} "
          f"{fmt(mm['recall']):>7s} {fmt(mm['ap50']):>7s} {fmt(mm['ap50_95']):>7s}")
    for h, r in m["attributes"].items():
        print(f"  attr {h:20s} {r['agree']}/{r['n']} = {r['agree'] / r['n']:.4f}" if r["n"] else f"  attr {h}: n/a",
              f"(not comparable {r['not_comparable']})" if r["not_comparable"] else "")
    for k, r in m["masks"].items():
        print(f"  mask {k:20s} IoU {r['iou']:.4f} over {r['frames']} frames ({r['note']})")
    if m["excluded"]:
        print("  excluded GT box labels:", m["excluded"])
    if m["excluded_shapes"]:
        print("  excluded GT shapes:", m["excluded_shapes"])


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    for h in logging.getLogger().handlers:
        h.setLevel(logging.WARNING)
    return run(build_parser().parse_args(argv))
