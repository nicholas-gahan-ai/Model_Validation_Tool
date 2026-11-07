import importlib.util
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from conftest import ROOT
from core.description import parse
from core.gt import GtBox, GtFrame
from core.gt.aliases import LabelMapper, norm
from core.gt.normalize_attrs import NOT_COMPARABLE, AttrNormalizer, bucket_percent_4
from core.matching import match_frame
from core.metrics import ap, flags, prepare_image

DESC = parse(yaml.safe_load((ROOT / "tools/descriptions/open_adas_v13s_single.yaml").read_text()))
C0, C1 = DESC.detection.classes[1], DESC.detection.classes[2]   # any two model classes


def P(xyxy, cls, conf, **attrs):
    return NS(xyxy=xyxy, cls=cls, conf=conf, attributes={k: NS(value=v) for k, v in attrs.items()})


def G(xyxy, cls, **attrs):
    return NS(xyxy=xyxy, cls=cls, attributes=attrs)


# ---------------------------------------------------------------- matching
def test_duplicates_one_tp_one_fp():
    m = match_frame([P((0, 0, 10, 10), C0, 0.9), P((0, 0, 10, 10), C0, 0.8)], [G((0, 0, 10, 10), C0)])
    assert [(p.pred, p.gt) for p in m.pairs] == [(0, 0)] and m.fp == [1] and m.fn == []


def test_higher_confidence_matches_first():
    m = match_frame([P((0, 0, 10, 10), C0, 0.3), P((1, 0, 11, 10), C0, 0.9)], [G((0, 0, 10, 10), C0)])
    assert m.pairs[0].pred == 1 and m.fp == [0]


def test_iou_exactly_half_follows_scorecard():
    # IoU = 100 / 200 = 0.5; the scorecard's iou() adds 1e-9 to the union, so exact 0.5 is NOT a match there
    # and must not be one here either (same rule as the AP code).
    m = match_frame([P((0, 0, 10, 10), C0, 0.9)], [G((0, 0, 10, 20), C0)])
    assert m.pairs == [] and m.fp == [0] and m.fn == [0]
    m = match_frame([P((0, 0, 10, 10.01), C0, 0.9)], [G((0, 0, 10, 20), C0)])  # just above 0.5
    assert len(m.pairs) == 1


def test_class_must_agree():
    m = match_frame([P((0, 0, 10, 10), C1, 0.9)], [G((0, 0, 10, 10), C0)])
    assert m.pairs == [] and m.per_class[C0] == {"tp": 0, "fp": 0, "fn": 1} and m.per_class[C1]["fp"] == 1


def test_empty_gt_and_empty_prediction():
    m = match_frame([P((0, 0, 10, 10), C0, 0.9)], [])
    assert m.fp == [0] and m.fn == []
    m = match_frame([], [G((0, 0, 10, 10), C0)])
    assert m.fn == [0] and m.fp == []
    m = match_frame([], [])
    assert m.pairs == [] and m.per_class == {}


def test_attribute_agreement_and_not_comparable():
    m = match_frame([P((0, 0, 10, 10), C0, 0.9, a="x", b="y", c="z")],
                    [G((0, 0, 10, 10), C0, a="x", b="q", c=NOT_COMPARABLE)])
    assert m.attr["a"] == {"n": 1, "agree": 1, "not_comparable": 0}
    assert m.attr["b"]["agree"] == 0 and m.attr["c"] == {"n": 0, "agree": 0, "not_comparable": 1}


# ---------------------------------------------------------------- AP vs the scorecard
def _random_set(seed, n_img=40, n_cls=3):
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n_img):
        k = rng.integers(0, 6)
        xy = rng.uniform(0, 200, (k, 2)); wh = rng.uniform(10, 60, (k, 2))
        gb = np.concatenate([xy, xy + wh], 1); gc = rng.integers(0, n_cls, k)
        # predictions: jittered GT + random false positives
        pb = gb + rng.normal(0, 4, gb.shape); pc = gc.copy()
        fp = rng.integers(0, 4); fxy = rng.uniform(0, 200, (fp, 2))
        pb = np.concatenate([pb, np.concatenate([fxy, fxy + 30], 1)]); pc = np.concatenate([pc, rng.integers(0, n_cls, fp)])
        ps = rng.uniform(0, 1, len(pb))
        imgs.append((gb, gc, pb, pc, ps))
    return imgs


def test_ap_matches_reference_scorecard(bundle):
    spec = importlib.util.spec_from_file_location("ap_reference", bundle / "reference/scripts/ap_reference.py")
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    CL = ["a", "b", "c"]
    for seed in range(5):
        imgs = [prepare_image(*x, n_classes=3) for x in _random_set(seed)]
        ours = ap(flags(imgs, CL), np.ones(len(imgs)))
        theirs = ref.ap(ref.flags(imgs, CL), np.ones(len(imgs)))
        assert ours.keys() == theirs.keys()
        for c in ours:
            assert ours[c] == pytest.approx(theirs[c], abs=1e-12)


def test_ap_perfect_predictions_is_one():
    imgs = [prepare_image([[0, 0, 10, 10]], [0], [[0, 0, 10, 10]], [0], [0.9], 1)]
    assert ap(flags(imgs, ["a"]), np.ones(1))["a"] == pytest.approx(1.0)


# ---------------------------------------------------------------- aliases + attribute normalisation
def test_alias_mapping(tmp_path):
    m = LabelMapper(DESC, tmp_path)
    assert norm("Traffic_Light") == norm("traffic-light") == "traffic light"
    cls_alias = yaml.safe_load((ROOT / "config/label_aliases.yaml").read_text())["aliases"]
    for target, names in cls_alias.items():
        for n in names:
            kind, t = m.target(n)
            assert t == target, (n, t, target)
    assert m.target("road boundaries") == (None, None)          # ignored by config
    assert m.unresolved(["car", "Zebra crossing", "lane markings"]) == ["Zebra crossing"]


def test_unknown_label_dialog_choice_is_saved(tmp_path):
    m = LabelMapper(DESC, tmp_path)
    m.set_user({"Zebra crossing": "ignore", "Lorry": DESC.detection.classes[-2]})
    assert (tmp_path / ".gahan_label_map.yaml").is_file()
    m2 = LabelMapper(DESC, tmp_path)                              # re-opened: choices remembered
    assert m2.unresolved(["Zebra crossing", "Lorry"]) == []
    assert m2.target("lorry") == ("class", DESC.detection.classes[-2])
    assert m2.target("Zebra crossing") == (None, None)


def test_known_label_without_model_counterpart_is_excluded_not_asked(tmp_path):
    det_only = parse({"name": "d", "input": {"name": "i", "width": 64, "height": 64},
                      "detection": {"output": "o", "classes": ["car"]}})
    m = LabelMapper(det_only, tmp_path)
    assert m.unresolved(["lane markings"]) == []                  # known alias, model has no lane mask
    assert m.target("lane markings") == (None, None)


def test_bucket_percent_4():
    for s, i in [("<10%", 0), ("< 10%", 0), ("10-50%", 1), ("10–50%", 1), ("10% – 50%", 1), ("50–75%", 2),
                 ("50% – 75%", 2), (">75%", 3), ("> 75%", 3), ("weird", -1), (None, -1)]:
        assert bucket_percent_4(s) == i, s


def test_attribute_normalisation_open_adas_values():
    n = AttrNormalizer(DESC)
    heads = {h.name: h for h in DESC.attributes.heads}
    always = DESC.attributes.always
    # per-class heads come from the description; check that every rule maps a raw value into the head's values
    for cls, hs in DESC.attributes.applies_to.items():
        for h in hs:
            rule = n.rules[h]
            raw = {rule["keys"][0]: heads[h].values[0]}
            assert n.normalize(cls, raw)[h] == heads[h].values[0]
    cls = DESC.detection.classes[2]
    out = n.normalize(cls, {"occlusion": "10% – 50%", "Truncation": "> 75%", "pose": "0°"})
    assert out[always[0]] == heads[always[0]].values[1] and out[always[1]] == heads[always[1]].values[3]
    out = n.normalize(cls, {"occlusion": "unknown"})
    assert out[always[0]] is NOT_COMPARABLE
    sign = [c for c, hs in DESC.attributes.applies_to.items() if any(heads[h].indexing == "slot_map" for h in hs)][0]
    h = [h for h in DESC.attributes.applies_to[sign] if heads[h].indexing == "slot_map"][0]
    syn = n.rules[h].get("synonyms") or {}
    if syn:
        k, v = next(iter(syn.items()))
        assert n.normalize(sign, {n.rules[h]["keys"][0]: k})[h] == v


def test_gtframe_resolve_yolo_normalized(tmp_path):
    f = GtFrame("x", normalized=True, boxes=[GtBox((0.1, 0.2, 0.3, 0.4), C0, {})])
    r = f.resolve(LabelMapper(DESC, tmp_path), AttrNormalizer(DESC), (1000, 500))
    assert r.boxes[0].xyxy == pytest.approx((100, 100, 300, 200))


def test_saved_choice_does_not_override_a_model_class(tmp_path):
    # a choice saved for a model without the class must not hide the class from a model that has it
    m = LabelMapper(DESC, tmp_path)
    m.set_user({C0: "ignore", "Zebra crossing": "ignore"})
    m2 = LabelMapper(DESC, tmp_path)
    assert m2.target(C0) == ("class", C0)
    assert m2.target("Zebra crossing") == (None, None)


def test_saved_choice_for_a_target_the_model_lacks_asks_again(tmp_path):
    m = LabelMapper(DESC, tmp_path)
    m.set_user({"Lorry": "nonexistent class"})
    assert LabelMapper(DESC, tmp_path).unresolved(["Lorry"]) == ["Lorry"]
