"""Raw GT attribute values -> the model's head values, driven by config/gt_attributes.yaml.
No class or attribute names are hard-coded here."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from ..settings import APP_ROOT
from .aliases import norm

NOT_COMPARABLE = object()


def bucket_percent_4(raw) -> int:
    """<10%, 10-50%, 50-75%, >75% -> 0..3, -1 if unrecognised."""
    if raw is None:
        return -1
    s = str(raw).strip()
    if "<" in s:
        return 0
    if ">" in s:
        return 3
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if len(nums) >= 2:
        lo, hi = nums[0], nums[1]
        if lo == 10 and hi == 50:
            return 1
        if lo == 50 and hi == 75:
            return 2
    return -1


BUCKETS = {"percent_4": bucket_percent_4}


class AttrNormalizer:
    def __init__(self, desc, rules_file: Path | None = None):
        self.desc = desc
        p = rules_file or APP_ROOT / "config/gt_attributes.yaml"
        try:
            cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            cfg = {}
        self.rules = {str(r["head"]): r for r in cfg.get("rules") or [] if "head" in r}

    def _raw_value(self, attrs: dict, keys: list[str]):
        for k in keys:
            if k in attrs and attrs[k] not in (None, ""):
                return attrs[k]
        low = {norm(k): v for k, v in attrs.items()}
        for k in keys:
            v = low.get(norm(k))
            if v not in (None, ""):
                return v
        return None

    def normalize(self, cls: str, attrs: dict) -> dict:
        """{head: model value | NOT_COMPARABLE} for the heads that apply to cls and are annotated."""
        a = self.desc.attributes
        if not a or not attrs:
            return {}
        out = {}
        for h in a.heads_for(cls):
            rule = self.rules.get(h.name)
            if rule and rule.get("classes") and norm(cls) not in {norm(c) for c in rule["classes"]}:
                continue
            keys = [str(k) for k in (rule.get("keys") if rule else None) or [h.name]]
            raw = self._raw_value(attrs, keys)
            if raw is None:
                continue  # not annotated for this object: not compared
            if rule and rule.get("bucket"):
                fn = BUCKETS.get(rule["bucket"])
                i = fn(raw) if fn else -1
                out[h.name] = h.values[i] if 0 <= i < len(h.values) else NOT_COMPARABLE
                continue
            v = str(raw).strip()
            syn = (rule or {}).get("synonyms") or {}
            v = str(syn.get(v, v))
            if v in h.values:
                out[h.name] = v
            else:
                low = {norm(x): x for x in h.values}
                out[h.name] = low.get(norm(v), NOT_COMPARABLE)
        return out
