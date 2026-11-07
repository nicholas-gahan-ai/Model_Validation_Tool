"""summary.html: one self-contained file (images embedded as base64), written when the user ticks it."""
from __future__ import annotations

import base64
import html
import time
from pathlib import Path

import cv2
import numpy as np

from .render import bgr_to_hex
from .settings import APP_ROOT

E_ = html.escape


def _b64_file(p: Path, mime="image/png") -> str:
    try:
        return f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()
    except OSError:
        return ""


def _thumb(p: Path, width=360) -> str:
    img = cv2.imread(str(p))
    if img is None:
        return ""
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _stats(v):
    if not v:
        return "—", "—"
    return f"{np.mean(v):.1f}", f"{np.percentile(v, 95):.1f}"


CSS = """
:root{--bg:#16181D;--panel:#1E2127;--text:#E6E6E6;--muted:#9AA0AA;--line:#2A2E36;--accent:#7A1F1F}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{background:#0E1013;border-bottom:3px solid var(--accent);padding:18px 28px;display:flex;align-items:center;gap:22px;flex-wrap:wrap}
header img{height:44px}header h1{font-size:20px;margin:0;font-weight:600}
main{max-width:1200px;margin:0 auto;padding:20px 28px 40px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin:16px 0}
h2{font-size:15px;margin:0 0 10px;color:#fff;letter-spacing:.3px}h3{font-size:13px;margin:14px 0 6px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}td.n,th.n{text-align:right}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px}.kv div:nth-child(odd){color:var(--muted)}
code{font-size:12px;word-break:break-all}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:7px;vertical-align:-1px}
.bar{height:8px;background:#7A1F1F;border-radius:2px;display:inline-block;vertical-align:middle}
.flags{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.flag{background:#16181D;border:1px solid var(--line);border-radius:6px;overflow:hidden}.flag img{width:100%;display:block}
.flag div{padding:8px 10px}.tag{display:inline-block;background:var(--accent);border-radius:3px;padding:0 6px;font-size:12px;margin-right:6px}
.muted{color:var(--muted)}footer{color:var(--muted);text-align:center;font-size:12px;padding:20px}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
"""


def write_summary(sess) -> Path:
    out = Path(sess.out_dir) / "summary.html"
    prov = sess.provenance()
    pal = getattr(sess.runner, "palette", None)
    from .render import Palette
    palette = pal or Palette(sess.runner.desc.class_names)
    wm = _b64_file(APP_ROOT / "assets/logo_wordmark_light.png")
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Validation summary</title><style>{CSS}</style></head><body><header>"
         + (f"<img alt='Gahan AI' src='{wm}'>" if wm else "<b>Gahan AI Pvt. Ltd.</b>")
         + "<h1>Validation Viewer · session summary</h1></header><main>"]

    # ---- model + run ----
    d = sess.runner.desc
    proc = [a + b + c for a, b, c in zip(sess.latency["preprocess"], sess.latency["inference"], sess.latency["postprocess"])]
    fps = sess.frames / sess.wall_s if sess.wall_s else 0
    h.append("<section><h2>Model and run</h2><div class=kv>")
    for k, v in [("Model", prov["model_name"]), ("Version", prov["model_version"] or "—"), ("File", prov["model_file"]),
                 ("sha256", f"<code>{E_(prov['model_sha256'])}</code>"), ("Device", prov["device"]),
                 ("Input", prov["input"]), ("Started", prov["started"]),
                 ("Thresholds", d.format_thresholds(prov["thresholds"])),
                 ("Frames processed", f"{sess.frames}"), ("Frames with ≥1 detection", f"{sess.frames_with_det}"),
                 ("Skipped (bad) frames", f"{len(sess.bad_frames)}"),
                 ("Throughput", f"{fps:.1f} fps over {sess.wall_s:.1f} s (incl. saving and display)"),
                 ("App", f"Gahan AI Validation Viewer {prov['app_version']}")]:
        h.append(f"<div>{E_(k)}</div><div>{v if k == 'sha256' else E_(str(v))}</div>")
    h.append("</div><h3>Latency per frame (ms)</h3><table><tr><th>Stage</th><th class=n>mean</th><th class=n>p95</th></tr>")
    for k in ("preprocess", "inference", "postprocess", "render"):
        m, p = _stats(sess.latency.get(k, []))
        h.append(f"<tr><td>{k}</td><td class=n>{m}</td><td class=n>{p}</td></tr>")
    m, p = _stats(proc)
    h.append(f"<tr><td><b>model total</b> (pre + inf + post)</td><td class=n><b>{m}</b></td><td class=n><b>{p}</b></td></tr>"
             "</table></section>")

    # ---- per-class totals ----
    order = d.class_names
    tot = sess.class_totals
    names = [c for c in order if c in tot] + [c for c in tot if c not in order]
    mx = max(tot.values()) if tot else 1
    h.append("<section><h2>Detections per class</h2><table><tr><th>Class</th><th class=n>Boxes</th>"
             "<th class=n>per frame</th><th></th></tr>")
    for c in names:
        h.append(f"<tr><td><span class=sw style='background:{bgr_to_hex(palette[c])}'></span>{E_(c)}</td>"
                 f"<td class=n>{tot[c]}</td><td class=n>{tot[c] / max(sess.frames, 1):.2f}</td>"
                 f"<td style='width:40%'><span class=bar style='width:{100 * tot[c] / mx:.1f}%'></span></td></tr>")
    if not names:
        h.append("<tr><td colspan=4 class=muted>no detections</td></tr>")
    h.append(f"<tr><td><b>total</b></td><td class=n><b>{sum(tot.values())}</b></td><td></td><td></td></tr></table></section>")

    # ---- attribute distributions ----
    if sess.attr_dist and any(sess.attr_dist.values()):
        h.append("<section><h2>Attribute values per class</h2><div class=grid2>")
        for c in names:
            heads = sess.attr_dist.get(c) or {}
            if not heads:
                continue
            h.append(f"<div><h3><span class=sw style='background:{bgr_to_hex(palette[c])}'></span>{E_(c)}</h3><table>")
            for head, vals in heads.items():
                n = sum(vals.values())
                items = sorted(vals.items(), key=lambda kv: -kv[1])
                h.append(f"<tr><td class=muted>{E_(head)}</td><td>" + "<br>".join(
                    f"{E_(v)} <span class=muted>{k / n:.0%} ({k})</span>" for v, k in items[:8])
                    + (f"<br><span class=muted>+{len(items) - 8} more</span>" if len(items) > 8 else "") + "</td></tr>")
            h.append("</table></div>")
        h.append("</div></section>")

    # ---- GT metrics ----
    if sess.gt_metrics:
        h.append(gt_metrics_html(sess.gt_metrics, palette))

    # ---- flags ----
    flags = getattr(sess, "flag_items", []) or []
    h.append(f"<section><h2>Flagged frames ({len(flags)})</h2>")
    if flags:
        h.append("<div class=flags>")
        for f in flags:
            where = f["file"] if f.get("file") else ""
            h.append(f"<div class=flag><img alt='' src='{_thumb(f['annotated_png'])}'><div><span class=tag>"
                     f"{E_(f['category'])}</span> frame {f['frame']} <span class=muted>{E_(str(where))}</span>"
                     f"<br>{E_(f['note'] or '')}</div></div>")
        h.append("</div>")
    else:
        h.append("<p class=muted>No frames were flagged.</p>")
    h.append("</section>")
    if sess.bad_frames:
        h.append("<section><h2>Skipped frames</h2><p class=muted>" + E_(", ".join(sess.bad_frames[:200]))
                 + ("…" if len(sess.bad_frames) > 200 else "") + "</p></section>")
    h.append(f"</main><footer>Generated by Gahan AI Validation Viewer {E_(prov['app_version'])} · "
             f"{time.strftime('%Y-%m-%d %H:%M')}</footer></body></html>")
    out.write_text("".join(h), encoding="utf-8")
    return out


def gt_metrics_html(m: dict, palette) -> str:
    """GT section (filled in by the GT build step's metrics dict)."""
    h = ["<section><h2>Ground-truth comparison</h2>"]
    h.append(f"<p class=muted>GT: {E_(str(m.get('gt_source', '')))} ({E_(str(m.get('gt_format', '')))}) · "
             f"{m.get('images_matched', 0)} images with GT · IoU 0.5 for TP/precision/recall · "
             f"AP over all predictions (COCO-style, top 100 per class per image)</p>")
    h.append("<table><tr><th>Class</th><th class=n>GT</th><th class=n>Pred@thr</th><th class=n>Precision</th>"
             "<th class=n>Recall</th><th class=n>AP50</th><th class=n>AP50-95</th></tr>")
    for c, r in m.get("per_class", {}).items():
        fmt = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"  # noqa: E731
        h.append(f"<tr><td><span class=sw style='background:{bgr_to_hex(palette[c])}'></span>{E_(c)}</td>"
                 f"<td class=n>{r['gt']}</td><td class=n>{r['pred']}</td><td class=n>{fmt(r['precision'])}</td>"
                 f"<td class=n>{fmt(r['recall'])}</td><td class=n>{fmt(r['ap50'])}</td><td class=n>{fmt(r['ap50_95'])}</td></tr>")
    mean = m.get("mean", {})
    h.append(f"<tr><td><b>mean ({mean.get('n_classes', 0)} classes)</b></td><td></td><td></td>"
             f"<td class=n>{mean.get('precision', float('nan')):.4f}</td><td class=n>{mean.get('recall', float('nan')):.4f}</td>"
             f"<td class=n><b>{mean.get('ap50', float('nan')):.4f}</b></td><td class=n><b>{mean.get('ap50_95', float('nan')):.4f}</b></td></tr></table>")
    if m.get("attributes"):
        h.append("<h3>Attribute accuracy on matched boxes (IoU ≥ 0.5)</h3><table><tr><th>Head</th><th class=n>compared</th>"
                 "<th class=n>agree</th><th class=n>accuracy</th></tr>")
        for head, r in m["attributes"].items():
            acc = f"{r['agree'] / r['n']:.4f}" if r["n"] else "—"
            note = f" <span class=muted>({r['not_comparable']} not comparable)</span>" if r.get("not_comparable") else ""
            h.append(f"<tr><td>{E_(head)}{note}</td><td class=n>{r['n']}</td><td class=n>{r['agree']}</td><td class=n>{acc}</td></tr>")
        h.append("</table>")
    elif m.get("attributes_na"):
        h.append(f"<p class=muted>Attribute metrics: N/A ({E_(m['attributes_na'])}).</p>")
    if m.get("masks"):
        h.append("<h3>Mask IoU</h3><table><tr><th>Mask</th><th class=n>IoU</th><th class=n>frames</th><th>note</th></tr>")
        for k, r in m["masks"].items():
            h.append(f"<tr><td>{E_(k)}</td><td class=n>{r['iou']:.4f}</td><td class=n>{r['frames']}</td>"
                     f"<td class=muted>{E_(r.get('note', ''))}</td></tr>")
        h.append("</table>")
    if m.get("excluded"):
        h.append("<h3>GT labels excluded (no model class)</h3><table><tr><th>GT label</th><th class=n>boxes</th></tr>"
                 + "".join(f"<tr><td>{E_(k)}</td><td class=n>{v}</td></tr>" for k, v in m["excluded"].items())
                 + "</table>")
    h.append("</section>")
    return "".join(h)
