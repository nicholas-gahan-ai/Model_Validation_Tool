"""Model description schema (gahan-model-description/1): dataclasses, parsing and validation.

The description is the only place model specifics live. Every section except `input` is optional;
unknown keys are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA = "gahan-model-description/1"
METADATA_KEY = "gahan_model_description"


class DescriptionError(ValueError):
    pass


@dataclass
class InputSpec:
    name: str
    width: int
    height: int
    batch: int = 1
    layout: str = "NCHW"
    resize: str = "letterbox"  # letterbox | stretch
    pad_value: int = 114
    color: str = "RGB"
    scale: float = 1 / 255
    dtype: str = "float32"
    output_batch_dim: bool = True  # informational: leading batch axes are detected per output by rank (slot 0 used)


@dataclass
class DetectionSpec:
    output: str
    classes: list[str]
    format: str = "xyxy_conf_cls"
    coords: str = "input_px"
    conf_threshold: float = 0.25


@dataclass
class AttrHead:
    name: str
    output: str
    values: list[str]
    indexing: str = "slot"  # slot | slot_map
    slots: int | None = None
    slot_map: str | None = None


@dataclass
class AttributesSpec:
    heads: list[AttrHead]
    applies_to: dict[str, list[str]] = field(default_factory=dict)
    always: list[str] = field(default_factory=list)
    display_prob_threshold: float = 0.5
    activation: str = "none"  # none | softmax

    def heads_for(self, cls: str) -> list[AttrHead]:
        names = list(self.applies_to.get(cls, [])) + [n for n in self.always if n not in self.applies_to.get(cls, [])]
        by = {h.name: h for h in self.heads}
        return [by[n] for n in names if n in by]


@dataclass
class ExtraBoxes:
    output: str
    cls: str
    format: str = "xyxy_score"
    coords: str = "input_px"
    merge_with_detection_class: bool = False
    nms_iou: float = 0.5
    threshold: float = 0.25
    threshold_label: str | None = None


@dataclass
class MaskSpec:
    name: str
    output: str
    type: str  # binary | classes
    activation: str = "none"  # binary: sigmoid | none
    threshold: float = 0.5
    class_names: list[str] | None = None
    ignore: list[str] = field(default_factory=list)
    color: str | None = None
    colors: dict[str, str] = field(default_factory=dict)


@dataclass
class TagSpec:
    name: str
    output: str
    values: list[str]
    activation: str = "none"


@dataclass
class ModelDescription:
    name: str
    input: InputSpec
    version: str = ""
    created: str = ""
    precision: str = ""
    sha256_graph: str = ""
    detection: DetectionSpec | None = None
    attributes: AttributesSpec | None = None
    extra_boxes: list[ExtraBoxes] = field(default_factory=list)
    masks: list[MaskSpec] = field(default_factory=list)
    tags: list[TagSpec] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def class_names(self) -> list[str]:
        names = list(self.detection.classes) if self.detection else []
        for e in self.extra_boxes:
            if e.cls not in names:
                names.append(e.cls)
        return names

    def output_ranks(self) -> dict[str, int]:
        """Rank of one image's output per role; extra leading axes are batch axes (slot 0 is used)."""
        r = {}
        if self.detection:
            r[self.detection.output] = 2
        if self.attributes:
            for h in self.attributes.heads:
                r[h.output] = 2
                if h.slot_map:
                    r[h.slot_map] = 1
        for e in self.extra_boxes:
            r[e.output] = 2
        for m in self.masks:
            r[m.output] = 2 if m.type == "binary" else 3
        for t in self.tags:
            r[t.output] = 1
        return r

    def output_names(self) -> list[str]:
        out = []
        if self.detection:
            out.append(self.detection.output)
        if self.attributes:
            for h in self.attributes.heads:
                out.append(h.output)
                if h.slot_map:
                    out.append(h.slot_map)
        out += [e.output for e in self.extra_boxes] + [m.output for m in self.masks] + [t.output for t in self.tags]
        return list(dict.fromkeys(out))


def _req(d: dict, key: str, where: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise DescriptionError(f"{where}: missing required key '{key}'")
    return d[key]


def _list_of_str(v: Any, where: str) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, (str, int, float)) for x in v):
        raise DescriptionError(f"{where}: expected a list of names")
    return [str(x) for x in v]


def parse(d: dict) -> ModelDescription:
    """Parse and validate a description dict. Raises DescriptionError with a readable message."""
    if not isinstance(d, dict):
        raise DescriptionError("description must be a mapping")
    schema = d.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise DescriptionError(f"unsupported schema '{schema}' (expected '{SCHEMA}')")

    i = _req(d, "input", "description")
    inp = InputSpec(
        name=str(_req(i, "name", "input")), width=int(_req(i, "width", "input")), height=int(_req(i, "height", "input")),
        batch=int(i.get("batch", 1)), layout=str(i.get("layout", "NCHW")), resize=str(i.get("resize", "letterbox")),
        pad_value=int(i.get("pad_value", 114)), color=str(i.get("color", "RGB")), scale=float(i.get("scale", 1 / 255)),
        dtype=str(i.get("dtype", "float32")), output_batch_dim=bool(i.get("output_batch_dim", True)))
    if inp.layout not in ("NCHW", "NHWC"):
        raise DescriptionError(f"input.layout '{inp.layout}' must be NCHW or NHWC")
    if inp.resize not in ("letterbox", "stretch"):
        raise DescriptionError(f"input.resize '{inp.resize}' must be letterbox or stretch")
    if inp.color not in ("RGB", "BGR"):
        raise DescriptionError(f"input.color '{inp.color}' must be RGB or BGR")
    if inp.batch < 1:
        raise DescriptionError("input.batch must be >= 1")

    det = None
    if d.get("detection"):
        x = d["detection"]
        det = DetectionSpec(output=str(_req(x, "output", "detection")),
                            classes=_list_of_str(_req(x, "classes", "detection"), "detection.classes"),
                            format=str(x.get("format", "xyxy_conf_cls")), coords=str(x.get("coords", "input_px")),
                            conf_threshold=float(x.get("conf_threshold", 0.25)))
        if det.format != "xyxy_conf_cls":
            raise DescriptionError(f"detection.format '{det.format}' not supported (xyxy_conf_cls)")
        if det.coords not in ("input_px", "normalized"):
            raise DescriptionError(f"detection.coords '{det.coords}' must be input_px or normalized")

    attrs = None
    if d.get("attributes"):
        a = d["attributes"]
        heads = []
        for k, h in enumerate(_req(a, "heads", "attributes")):
            w = f"attributes.heads[{k}]"
            head = AttrHead(name=str(_req(h, "name", w)), output=str(_req(h, "output", w)),
                            values=_list_of_str(_req(h, "values", w), w + ".values"),
                            indexing=str(h.get("indexing", "slot")), slots=h.get("slots"), slot_map=h.get("slot_map"))
            if head.indexing not in ("slot", "slot_map"):
                raise DescriptionError(f"{w}.indexing must be slot or slot_map")
            if head.indexing == "slot_map" and not head.slot_map:
                raise DescriptionError(f"{w}: indexing slot_map needs 'slot_map'")
            heads.append(head)
        names = {h.name for h in heads}
        if len(names) != len(heads):
            raise DescriptionError("attributes.heads: duplicate head names")
        applies = {str(c): _list_of_str(v, f"attributes.applies_to.{c}") for c, v in (a.get("applies_to") or {}).items()}
        always = _list_of_str(a.get("always") or [], "attributes.always")
        for ref in [n for v in applies.values() for n in v] + always:
            if ref not in names:
                raise DescriptionError(f"attributes: unknown head '{ref}' referenced")
        if det:
            for c in applies:
                if c not in det.classes:
                    raise DescriptionError(f"attributes.applies_to: '{c}' is not a detection class")
        attrs = AttributesSpec(heads=heads, applies_to=applies, always=always,
                               display_prob_threshold=float(a.get("display_prob_threshold", 0.5)),
                               activation=str(a.get("activation", "none")))
        if not det:
            raise DescriptionError("attributes need a detection section")

    extras = []
    for k, e in enumerate(d.get("extra_boxes") or []):
        w = f"extra_boxes[{k}]"
        extras.append(ExtraBoxes(output=str(_req(e, "output", w)), cls=str(_req(e, "class", w)),
                                 format=str(e.get("format", "xyxy_score")), coords=str(e.get("coords", "input_px")),
                                 merge_with_detection_class=bool(e.get("merge_with_detection_class", False)),
                                 nms_iou=float(e.get("nms_iou", 0.5)), threshold=float(e.get("threshold", 0.25)),
                                 threshold_label=e.get("threshold_label")))
        if extras[-1].merge_with_detection_class and (not det or extras[-1].cls not in det.classes):
            raise DescriptionError(f"{w}: merge_with_detection_class but '{extras[-1].cls}' is not a detection class")

    masks = []
    for k, m in enumerate(d.get("masks") or []):
        w = f"masks[{k}]"
        ms = MaskSpec(name=str(_req(m, "name", w)), output=str(_req(m, "output", w)), type=str(_req(m, "type", w)),
                      activation=str(m.get("activation", "none")), threshold=float(m.get("threshold", 0.5)),
                      class_names=_list_of_str(m["class_names"], w + ".class_names") if m.get("class_names") else None,
                      ignore=_list_of_str(m.get("ignore") or [], w + ".ignore"), color=m.get("color"),
                      colors={str(a): str(b) for a, b in (m.get("colors") or {}).items()})
        if ms.type not in ("binary", "classes"):
            raise DescriptionError(f"{w}.type must be binary or classes")
        if ms.type == "classes" and not ms.class_names:
            raise DescriptionError(f"{w}: type classes needs class_names")
        masks.append(ms)
    if len({m.name for m in masks}) != len(masks):
        raise DescriptionError("masks: duplicate names")

    tags = []
    for k, t in enumerate(d.get("tags") or []):
        w = f"tags[{k}]"
        tags.append(TagSpec(name=str(_req(t, "name", w)), output=str(_req(t, "output", w)),
                            values=_list_of_str(_req(t, "values", w), w + ".values"),
                            activation=str(t.get("activation", "none"))))

    return ModelDescription(name=str(_req(d, "name", "description")), input=inp, version=str(d.get("version", "")),
                            created=str(d.get("created", "")), precision=str(d.get("precision", "")),
                            sha256_graph=str(d.get("sha256_graph", "")), detection=det, attributes=attrs,
                            extra_boxes=extras, masks=masks, tags=tags, raw=d)


def check_against_outputs(desc: ModelDescription, shapes: dict[str, list]) -> list[str]:
    """Check description roles against graph output shapes ({name: [dims]}, dims int or None/str when dynamic).
    Returns a list of problems (empty = OK)."""
    errs = []
    ranks = desc.output_ranks()

    def dims(name):
        """The output's per-image dims (its last `rank` dims); extra leading dims must be batch-like."""
        if name not in shapes:
            errs.append(f"output '{name}' not found in the model (has: {', '.join(shapes)})")
            return None
        s, r = list(shapes[name]), ranks.get(name, len(shapes[name]))
        if len(s) < r and not (r == 2 and len(s) == 3):
            errs.append(f"output '{name}' has {len(s)} dims, expected at least {r}")
            return None
        return s[-r:] if len(s) >= r else s

    def last_is(name, n, what):
        s = dims(name)
        if s is not None and s and isinstance(s[-1], int) and s[-1] > 0 and s[-1] != n:
            errs.append(f"output '{name}': last dim {s[-1]} != {n} ({what})")
        return s

    if desc.detection:
        last_is(desc.detection.output, 6, "xyxy_conf_cls columns")
    if desc.attributes:
        for h in desc.attributes.heads:
            s = last_is(h.output, len(h.values), f"{len(h.values)} values of head '{h.name}'")
            if h.indexing == "slot" and s is not None and h.slots and isinstance(s[0], int) and s[0] > 0 and s[0] != h.slots:
                errs.append(f"output '{h.output}': {s[0]} rows != slots {h.slots}")
            if h.indexing == "slot_map":
                m = dims(h.slot_map)
                if s is not None and m is not None and m and s and isinstance(m[0], int) and isinstance(s[0], int) \
                        and m[0] > 0 and s[0] > 0 and m[0] != s[0]:
                    errs.append(f"slot_map '{h.slot_map}' has {m[0]} rows but '{h.output}' has {s[0]}")
    for e in desc.extra_boxes:
        last_is(e.output, 5, "xyxy_score columns")
    for m in desc.masks:
        s = dims(m.output)
        if s is not None and m.type == "classes" and len(s) == 3 and isinstance(s[0], int) and s[0] > 0 \
                and s[0] != len(m.class_names):
            errs.append(f"mask '{m.name}': {s[0]} channels != {len(m.class_names)} class_names")
    for t in desc.tags:
        last_is(t.output, len(t.values), f"values of tag '{t.name}'")
    return errs
