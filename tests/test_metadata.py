"""Public producer metadata uses one owner without serving dependencies."""
from __future__ import annotations

import copy
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module("tessera.metadata")


def legacy():
    """Answers captured from the exact base owners in an admitted action."""
    reference = json.loads((ROOT / "tests/fixtures/metadata_reference_v1.json").read_text())
    namespace = dict(reference["constants"])
    calls = {}
    for case in reference["cases"]:
        key = json.dumps([case["args"], case["kwargs"]], sort_keys=True, separators=(",", ":"))
        calls.setdefault(case["function"], {})[key] = case["outcome"]
    for name, answers in calls.items():
        def answer(*args, _answers=answers, **kwargs):
            key = json.dumps([args, kwargs], sort_keys=True, separators=(",", ":"))
            kind, value = _answers[key]
            if kind == "ValueError":
                raise ValueError(value)
            assert kind == "return", (kind, value)
            return copy.deepcopy(value)
        namespace[name] = answer
    return SimpleNamespace(**namespace)



def record(owner, *, reference=None, statuses=("recorded",)):
    return {"instrument": "experiments/moe_greedy_smoke.py", "rule": "scored completion",
            "reference": reference,
            "rows": [{"prompt": f"prompt-{i}", "form": owner.EVIDENCE_SMOKE_FORMS[0],
                      "interface": owner.EVIDENCE_SMOKE_INTERFACES[0], "status": status,
                      "reference_status": None if reference is None else "recorded"}
                     for i, status in enumerate(statuses)]}


def outcome(call):
    try:
        return ("return", call())
    except Exception as error:
        return (type(error).__name__, str(error))


def test_metadata_import_and_published_api_are_dependency_free():
    script = '''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1] + "/src")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in
               ("tessera.serving", "torch", "vllm", "triton", "numpy", "tessera.encode")) or \
                fullname.startswith(("tessera.kernel", "tessera.encoder", "cuda", "cupy")):
            raise ImportError("metadata crossed dependency boundary: " + fullname)
sys.meta_path.insert(0, Block())
from tessera import metadata
assert metadata.METADATA_API_VERSION == "tessera.producer-metadata.v1"
assert metadata.parse_smoke_record(None, control=None, where="smoke.record") is None
with open(sys.argv[1] + "/src/tessera/serving/runtime_contract.json") as handle:
    table = json.load(handle)
for cell in table["lane_eligibility"]["cells"]:
    smoke = cell["evidence"]["smoke"]
    metadata.parse_smoke_record(smoke["record"], control=smoke["control"], where="smoke.record")
assert not any(name == "tessera.serving" or name.startswith("tessera.serving.") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_smoke_derivations_equal_exact_base_for_every_status_pair():
    new, old = api(), legacy()
    references = (None, *old.EVIDENCE_CONTROL_REFERENCES)
    for reference in references:
        against = (None,) if reference is None else old.EVIDENCE_SMOKE_STATUSES
        row_states = tuple(itertools.product(old.EVIDENCE_SMOKE_STATUSES, against))
        for rows in itertools.chain(((row,) for row in row_states), itertools.product(row_states, repeat=2)):
            payload = record(old, reference=reference, statuses=tuple(row[0] for row in rows))
            for item, (_, ref_status) in zip(payload["rows"], rows):
                item["reference_status"] = ref_status
            before = copy.deepcopy(payload)
            assert new.parse_smoke_record(payload, control=None, where="smoke.record") == \
                old._evidence_smoke_record(payload, "smoke.record", None, "smoke")
            smoke = {"record": payload, "control": None, "status": "not_recorded"}
            for name in ("smoke_status_is_derived", "derive_smoke_status", "derive_smoke_attribution"):
                assert getattr(new, name)(smoke) == getattr(old, name)(smoke)
            assert payload == before
    for form, interface in itertools.product(old.EVIDENCE_SMOKE_FORMS, old.EVIDENCE_SMOKE_INTERFACES):
        payload = record(old)
        payload["rows"][0].update(form=form, interface=interface)
        assert new.parse_smoke_record(payload, control=None, where="custom.path") == \
            old._evidence_smoke_record(payload, "custom.path", None, "custom.path")
    for control in (None, {"outcome": "identical_completion"}, {"outcome": "different_completion"}):
        for status in (*old.EVIDENCE_SMOKE_STATUSES, None, 17):
            smoke = {"record": None, "control": control, "status": status}
            assert new.parse_smoke_record(None, control=control, where="smoke.record") is None
            for name in ("smoke_status_is_derived", "derive_smoke_status", "derive_smoke_attribution"):
                assert getattr(new, name)(smoke) == getattr(old, name)(smoke)


@pytest.mark.parametrize("defect", ["object", "missing", "unknown", "rule", "reference", "rows",
    "row-object", "row-key", "prompt", "form", "interface", "status", "reference-status",
    "nullable-reference", "duplicate", "absolute", "traversal", "backslash", "whitespace", "control"])
def test_smoke_refusals_equal_exact_base(defect):
    new, old = api(), legacy()
    payload, control = record(old), None
    if defect == "object": payload = []
    elif defect == "missing": del payload["rule"]
    elif defect == "unknown": payload["extra"] = True
    elif defect == "rule": payload["rule"] = " "
    elif defect == "reference": payload["reference"] = "unknown"
    elif defect == "rows": payload["rows"] = []
    elif defect == "row-object": payload["rows"] = [None]
    elif defect == "row-key": del payload["rows"][0]["status"]
    elif defect in ("prompt", "form", "interface", "status"):
        payload["rows"][0][defect] = " " if defect == "prompt" else "unknown"
    elif defect == "reference-status":
        payload["reference"] = old.EVIDENCE_CONTROL_REFERENCES[0]
        payload["rows"][0]["reference_status"] = None
    elif defect == "nullable-reference": payload["rows"][0]["reference_status"] = "recorded"
    elif defect == "duplicate": payload["rows"] *= 2
    elif defect == "control": control = {"outcome": "identical_completion"}
    else: payload["instrument"] = {"absolute": "/tmp/rule.py", "traversal": "../rule.py",
        "backslash": "code\\rule.py", "whitespace": "code/my rule.py"}[defect]
    before = copy.deepcopy(payload)
    expected = outcome(lambda: old._evidence_smoke_record(payload, "cell.smoke.record", control, "cell.smoke"))
    assert expected[0] == "ValueError"
    assert outcome(lambda: new.parse_smoke_record(payload, control=control, where="cell.smoke.record")) == expected
    assert payload == before


def test_parsed_collections_belong_to_caller():
    new = api()
    payload = record(new)
    before = copy.deepcopy(payload)
    first = new.parse_smoke_record(payload, control=None, where="smoke.record")
    second = new.parse_smoke_record(payload, control=None, where="smoke.record")
    first["rows"][0]["prompt"] = "changed"
    first["rows"].append({})
    first["rule"] = "changed"
    assert payload == before == second


def test_every_lane_requirement_equals_exact_base_and_retains_order():
    new, old = api(), legacy()
    cases = {
        "every_in": ([1, 2], [(1, 2), (1, 3), (), None]),
        "every_in_routed_moe": ([1, 2], [(1, 2), (1, 3), (), None]),
        "one_of": ([2, 4], [2, 3, None]),
        "wire_spelling": (None, []), "state_in": (["none"], ["NONE", "R_IN_ONLY", None]),
        "carries": (False, [False, True, 0, 3, None]),
    }
    for name, (fact_name, mode) in old._LANE_WIRE_CHECKS.items():
        allowed, values = cases[mode]
        if mode == "wire_spelling":
            dialect = getattr(old, "_ATTESTED_WIRE_" + name.upper())
            for spelling, fact in dialect.items():
                for value in (fact, "UNKNOWN", None):
                    assert new.decide_lane_requirements("lane", {name: spelling}, {fact_name: value}) == \
                        old.decide_lane_requirements("lane", {name: spelling}, {fact_name: value})
            for spelling in ("unknown", None):
                assert outcome(lambda: new.decide_lane_requirements("lane", {name: spelling}, {fact_name: "UNKNOWN"})) == \
                    outcome(lambda: old.decide_lane_requirements("lane", {name: spelling}, {fact_name: "UNKNOWN"}))
            continue
        for value in values:
            for structure in (None, "unknown", *old.STRUCTURES):
                for requires_value in ([False, True] if mode == "carries" else [allowed]):
                    requires, facts = {name: requires_value}, {fact_name: value, "structure": structure}
                    before = copy.deepcopy((requires, facts))
                    assert new.decide_lane_requirements("lane", requires, facts) == \
                        old.decide_lane_requirements("lane", requires, facts)
                    assert (requires, facts) == before
    requires = {name: cases[mode][0] for name, (_, mode) in old._LANE_WIRE_CHECKS.items()
                if mode != "wire_spelling"}
    assert new.decide_lane_requirements("lane", requires, {}) == old.decide_lane_requirements("lane", requires, {})
    expected = outcome(lambda: old.decide_lane_requirements("lane", {"unknown": []}, {}))
    assert expected[0] == "ValueError"
    assert outcome(lambda: new.decide_lane_requirements("lane", {"unknown": []}, {})) == expected


def test_serving_callers_use_public_owners():
    new = api()
    from tessera.serving import contract, scheme
    for name in ("smoke_status_is_derived", "derive_smoke_status", "derive_smoke_attribution", "parse_smoke_record"):
        assert getattr(contract, name) is getattr(new, name)
    assert scheme.decide_lane_requirements is new.decide_lane_requirements
    assert contract.decide_lane_requirements is new.decide_lane_requirements
    assert not hasattr(contract, "_evidence_smoke_record")
