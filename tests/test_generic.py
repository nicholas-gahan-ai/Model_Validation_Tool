"""Genericity check: no class, attribute or mask names in core/ or ui/ (everything comes from the model description)."""
import re

import yaml

from conftest import ROOT


def model_names():
    names = set()
    for p in (ROOT / "tools/descriptions").glob("open_adas*.yaml"):   # our models (the COCO example is generic words)
        d = yaml.safe_load(p.read_text())
        names |= set((d.get("detection") or {}).get("classes", []))
        for h in (d.get("attributes") or {}).get("heads", []):
            names.add(h["name"])     # head values (e.g. a sign type "stop") are ordinary words: not checked
        names |= {m["name"] for m in d.get("masks", [])}
        for m in d.get("masks", []):
            names |= set(m.get("class_names") or [])
    names -= {"background", "other"}   # ordinary words that also happen to be class names
    return names


def test_no_model_specifics_in_app_code():
    names = model_names()
    assert len(names) > 15
    pats = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(n).replace(r"\ ", r"[ _-]?") + r"(?![A-Za-z0-9])", re.I)
            for n in names]
    hits = []
    for p in list((ROOT / "core").rglob("*.py")) + list((ROOT / "ui").rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for pat in pats:
                if pat.search(line):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not hits, "model-specific names in app code:\n" + "\n".join(hits)
