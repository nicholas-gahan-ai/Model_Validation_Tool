"""GT label -> model target mapping: config/label_aliases.yaml + a per-GT-folder saved choice."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..settings import APP_ROOT

USER_MAP_NAME = ".gahan_label_map.yaml"
IGNORE = "ignore"


def norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def mask_targets(desc) -> list[str]:
    """Mask targets from the description: '<mask>' and, for class maps, '<mask>/<class>'."""
    out = []
    for m in desc.masks:
        out.append(m.name)
        if m.type == "classes":
            out += [f"{m.name}/{c}" for c in m.class_names if c not in m.ignore]
    return out


class LabelMapper:
    def __init__(self, desc, gt_dir: Path | None = None, aliases_file: Path | None = None):
        self.classes = list(desc.class_names)
        self.masks = mask_targets(desc)
        self._cls = {norm(c): c for c in self.classes}
        self._mask = {norm(m): m for m in self.masks}
        cfg = {}
        p = aliases_file or APP_ROOT / "config/label_aliases.yaml"
        try:
            cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            cfg = {}
        self.alias: dict[str, str] = {}
        for target, names in (cfg.get("aliases") or {}).items():
            for n in [target, *(names or [])]:
                self.alias[norm(n)] = str(target)
        self.ignore = {norm(x) for x in cfg.get("ignore") or []}
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.user: dict[str, str] = {}
        if self.gt_dir and (self.gt_dir / USER_MAP_NAME).is_file():
            try:
                self.user = {norm(k): str(v) for k, v in
                             (yaml.safe_load((self.gt_dir / USER_MAP_NAME).read_text(encoding="utf-8")) or {}).items()}
            except (OSError, yaml.YAMLError):
                self.user = {}
        self._cache: dict[str, tuple] = {}

    def _resolve_target(self, t: str):
        n = norm(t)
        if n in self._cls:
            return "class", self._cls[n]
        if n in self._mask:
            return "mask", self._mask[n]
        return None

    def lookup(self, label: str):
        """-> ("class", name) | ("mask", target) | ("ignore", None) | ("known", None) | None (unknown label)."""
        n = norm(label)
        hit = self._resolve_target(label)          # the model's own names always win
        if hit:
            return hit
        if n in self.ignore:
            return ("ignore", None)
        if n in self.alias:
            return self._resolve_target(self.alias[n]) or ("known", None)   # known label, the model lacks it
        if n in self.user:                          # a saved choice for an otherwise unmatched label
            v = self.user[n]
            return ("ignore", None) if norm(v) == IGNORE else (self._resolve_target(v) or None)
        return None

    def target(self, label: str):
        """(kind, target) with kind in {"class", "mask", None}; None = excluded (ignored or no model counterpart)."""
        if label not in self._cache:
            r = self.lookup(label)
            self._cache[label] = r if r and r[0] in ("class", "mask") else (None, None)
        return self._cache[label]

    def unresolved(self, labels) -> list[str]:
        return sorted({lab for lab in labels if self.lookup(lab) is None}, key=str.lower)

    def set_user(self, choices: dict[str, str], save: bool = True):
        for k, v in choices.items():
            self.user[norm(k)] = v
        self._cache.clear()
        if save and self.gt_dir:
            p = self.gt_dir / USER_MAP_NAME
            old = {}
            if p.is_file():
                old = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            old.update(choices)
            try:
                p.write_text("# GT label -> model target (or 'ignore'); written by Gahan AI Validation Viewer\n"
                             + yaml.safe_dump(old, sort_keys=True, allow_unicode=True), encoding="utf-8")
            except OSError:
                pass  # read-only GT folder: the choice still applies to this session
