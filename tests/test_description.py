import copy

import pytest
import yaml

from conftest import ROOT
from core.description import DescriptionError, check_against_outputs, parse

DESC = yaml.safe_load((ROOT / "tools/descriptions/open_adas_v13s_single.yaml").read_text())


def test_shipped_description_parses():
    d = parse(DESC)
    assert d.detection and len(d.detection.classes) == len(DESC["detection"]["classes"])
    assert [m.name for m in d.masks] == ["lane", "drivable"]
    cls = d.attributes.applies_to and next(iter(d.attributes.applies_to))
    assert [h.name for h in d.attributes.heads_for(cls)][-2:] == d.attributes.always


def test_minimal_detection_only():
    d = parse({"name": "x", "input": {"name": "images", "width": 640, "height": 640},
               "detection": {"output": "out", "classes": ["a", "b"]}})
    assert d.masks == [] and d.attributes is None and d.input.batch == 1


def test_unknown_keys_ignored_and_schema_checked():
    d = copy.deepcopy(DESC); d["whatever"] = 1; d["input"]["foo"] = 2
    parse(d)
    d["schema"] = "gahan-model-description/2"
    with pytest.raises(DescriptionError, match="schema"):
        parse(d)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.pop("input"), "input"),
    (lambda d: d["attributes"]["always"].append("nope"), "unknown head"),
    (lambda d: d["attributes"]["applies_to"].update({"zebra": []}), "not a detection class"),
    (lambda d: d["masks"][1].pop("class_names"), "class_names"),
    (lambda d: d["extra_boxes"][0].update({"class": "zebra"}), "merge_with_detection_class"),
    (lambda d: d["input"].update({"resize": "crop"}), "resize"),
])
def test_invalid(mutate, msg):
    d = copy.deepcopy(DESC); mutate(d)
    with pytest.raises(DescriptionError, match=msg):
        parse(d)


def test_check_against_outputs():
    d = parse(DESC)
    shapes = {"raw_detections": [2, 300, 6], "lane_logits": [2, 1, 480, 832], "drivable_logits": [2, 3, 480, 832],
              "sign_light_slots": [2, 16], "plates": [2, 64, 5]}
    for h in d.attributes.heads:
        shapes[h.output] = [2, 16 if h.indexing == "slot_map" else 32, len(h.values)]
    assert check_against_outputs(d, shapes) == []
    shapes["sign_type"] = [2, 16, 45]
    shapes.pop("plates")
    errs = check_against_outputs(d, shapes)
    assert any("sign_type" in e for e in errs) and any("plates" in e for e in errs)
