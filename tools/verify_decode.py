"""Decode check: the packaged model file + the app's generic decode reproduce the verified model.

  GAHAN_BUNDLE=~/validation_tool_bundle python tools/verify_decode.py bench --device gpu
  GAHAN_BUNDLE=~/validation_tool_bundle python tools/verify_decode.py bench --device cpu
  GAHAN_BUNDLE=~/validation_tool_bundle python tools/verify_decode.py ap --device gpu

bench: 50 bench frames vs the PyTorch handler's stored shapes (gpu fp16 / cpu fp32), deployed cutoffs.
       boxes matched by class, max corner diff < 0.5 letterbox px on >= 99%; same attribute values on >= 99%;
       lane / drivable mask IoU >= 0.99 (both sides turned into polygons the handler's way, then rasterised).
ap:    all 1,207 val frames, every slot (threshold 0) + plate cutoff 0.01, scored against gt_coco_scoring.json.
       must give all 11 = 0.4686 +- 0.002.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.metrics import ap, flags, prepare_image  # noqa: E402
from core.model_file import open_model  # noqa: E402
from core.runner_onnx import OnnxRunner  # noqa: E402
from core.settings import APP_ROOT, bundle_path  # noqa: E402

MIN_POLY_AREA = 50
POLY_EPS = {"lane_marking": 0.5, "drivable_area": 1.0}


def cvat(name: str) -> str:
    return name.lower().replace(" ", "_")


def mask_polys_raster(mask: np.ndarray, label: str) -> np.ndarray:
    """mask -> handler-style polygons -> filled raster (so both sides go through the same polygonisation)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    polys = [cv2.approxPolyDP(c, POLY_EPS[label], True) for c in cs if cv2.contourArea(c) >= MIN_POLY_AREA]
    polys = [p for p in polys if len(p) >= 3]
    if polys:
        cv2.fillPoly(out, polys, 1)
    return out.astype(bool)


def ref_raster(shapes: list, label: str, hw) -> np.ndarray:
    out = np.zeros(hw, np.uint8)
    polys = [np.round(np.array(s["points"], np.float64).reshape(-1, 2)).astype(np.int32)
             for s in shapes if s["type"] == "polygon" and s["label"] == label]
    if polys:
        cv2.fillPoly(out, polys, 1)
    return out.astype(bool)


def m_iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u


def bench(runner: OnnxRunner, B: Path, device: str) -> bool:
    ref_file = B / "reference/outputs" / ("bench50_gpu_fp16_shapes.jsonl" if device == "gpu" else "bench50_cpu_fp32_shapes.jsonl")
    refs = [json.loads(l) for l in open(ref_file)]
    n_ref = n_ok = n_unmatched_pred = 0
    attr_n = attr_ok = 0
    ious = {"lane_marking": [], "drivable_area": []}
    worst = []
    for r in refs:
        img = cv2.imread(str(B / "data/oa_val/images" / r["image"]))
        raw = runner.infer_raw(img)
        pred = runner.decoder.decode(raw)
        s = raw.geo.s
        rects = [x for x in r["shapes"] if x["type"] == "rectangle"]
        pb = list(pred.boxes)
        used = set()
        # greedy one-to-one by corner distance within each class
        pairs = []
        for i, x in enumerate(rects):
            for j, b in enumerate(pb):
                if cvat(b.cls) == x["label"]:
                    pairs.append((np.abs(np.array(b.xyxy) - np.array(x["points"])).max() * s, i, j))
        pairs.sort()
        got = {}
        for d, i, j in pairs:
            if i not in got and j not in used:
                got[i] = (d, j); used.add(j)
        for i, x in enumerate(rects):
            n_ref += 1
            if i not in got:
                worst.append((r["image"], x["label"], x["confidence"], "no match"))
                continue
            d, j = got[i]
            if d < 0.5:
                n_ok += 1
            else:
                worst.append((r["image"], x["label"], x["confidence"], f"{d:.2f}px"))
            ra = {a["name"]: a["value"] for a in x["attributes"]}
            pa = {k: v.value for k, v in pb[j].attributes.items()}
            attr_n += 1
            attr_ok += ra == pa
            if ra != pa:
                worst.append((r["image"], x["label"], x["confidence"], f"attr {ra} vs {pa}"))
        n_unmatched_pred += len(pb) - len(used)
        hw = (raw.geo.h, raw.geo.w)
        lane = pred.masks["lane"].data
        drv = pred.masks["drivable"].data == pred.masks["drivable"].class_names.index("direct")
        ious["lane_marking"].append(m_iou(mask_polys_raster(lane, "lane_marking"), ref_raster(r["shapes"], "lane_marking", hw)))
        ious["drivable_area"].append(m_iou(mask_polys_raster(drv, "drivable_area"), ref_raster(r["shapes"], "drivable_area", hw)))
    box_rate, attr_rate = n_ok / max(n_ref, 1), attr_ok / max(attr_n, 1)
    lane_iou, drv_iou = np.mean(ious["lane_marking"]), np.mean(ious["drivable_area"])
    print(f"[bench50 {device}] {runner.device_label}")
    print(f"  boxes within 0.5 lb px: {n_ok}/{n_ref} = {box_rate:.4f}   extra predicted boxes: {n_unmatched_pred}")
    print(f"  attribute values identical: {attr_ok}/{attr_n} = {attr_rate:.4f}")
    print(f"  mask IoU  lane {lane_iou:.4f} (min {min(ious['lane_marking']):.4f})  "
          f"drivable {drv_iou:.4f} (min {min(ious['drivable_area']):.4f})")
    for w in worst[:15]:
        print("   diff:", *w)
    ok = box_rate >= 0.99 and attr_rate >= 0.99 and lane_iou >= 0.99 and drv_iou >= 0.99
    print("  PASS" if ok else "  FAIL")
    return ok


def ap_check(runner: OnnxRunner, B: Path, out_jsonl: Path | None) -> bool:
    coco = json.load(open(B / "data/oa_val/gt_coco_scoring.json"))
    CL = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    cid = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    idx = {n: i for i, n in enumerate(CL)}
    missing = [c for c in CL if c not in runner.desc.class_names]
    if missing:
        print("GT classes not in the model:", missing)
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    th = {"conf": 0.0, **{e.output: 0.01 for e in runner.desc.extra_boxes}}
    imgs, t0, lat = [], time.time(), []
    fo = open(out_jsonl, "w") if out_jsonl else None
    for n, im in enumerate(coco["images"]):
        frame = cv2.imread(str(B / "data/oa_val/images" / im["file_name"]))
        raw = runner.infer_raw(frame, frame_index=n, name=im["file_name"])
        lat.append(raw.timing_ms["inference"])
        pred = runner.decoder.decode(raw, th, masks=False)
        if fo:
            fo.write(json.dumps(pred.to_json()) + "\n")
        A = by_img.get(im["id"], [])
        gb = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in A]
        keep = [b for b in pred.boxes if b.cls in idx]
        imgs.append(prepare_image(gb, [cid[a["category_id"]] for a in A], [b.xyxy for b in keep],
                                  [idx[b.cls] for b in keep], [b.conf for b in keep], len(CL)))
        if n % 200 == 0:
            print(f"  {n}/{len(coco['images'])}  {time.time() - t0:.0f}s", flush=True)
    res = ap(flags(imgs, CL), np.ones(len(imgs)), per_threshold=True)
    m = float(np.mean([v.mean() for v in res.values()]))
    print(f"[AP {runner.device_label}] images {len(imgs)}  all {len(res)} = {m:.4f}  "
          f"(AP50 {np.mean([v[0] for v in res.values()]):.4f})   inference median {np.median(lat):.1f} ms")
    for c, v in res.items():
        print(f"  {c:15s} AP50-95 {v.mean():.4f}  AP50 {v[0]:.4f}")
    ok = abs(m - 0.4686) <= 0.002
    print("  PASS" if ok else "  FAIL")
    return ok


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("check", choices=["bench", "ap"])
    ap_.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap_.add_argument("--model", default=str(APP_ROOT / "models/open_adas_v13s_single.onnx"))
    ap_.add_argument("--save-jsonl", type=Path)
    a = ap_.parse_args()
    B = bundle_path()
    runner = OnnxRunner(open_model(a.model), device=a.device)
    if a.device == "gpu" and runner.device != "GPU":
        sys.exit(f"GPU requested but running on {runner.device_label}: {runner.message}")
    ok = bench(runner, B, a.device) if a.check == "bench" else ap_check(runner, B, a.save_jsonl)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
