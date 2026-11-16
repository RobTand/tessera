"""Per-key module coverage has an independent signed-wire expectation set."""
import copy

import pytest

from experiments.numerics_rows import row
from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.join import join, rule_unit_id
from tests.test_numerics_rows_receipts import joined_inputs


@pytest.mark.parametrize("known,expected", [("FAIL", "FAIL"), ("STOPPED", "STOPPED"), ("INVALID", "INVALID"), ("PASS", "UNATTESTED")])
@pytest.mark.parametrize("missing", ["manifest", "row", "both", "key"])
def test_missing_required_pair_preserves_a_worse_known_outcome(monkeypatch, known, expected, missing):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    positive = join(population, digest, manifests, receipts)
    assert positive["verdict"] == "PASS"
    key = rule_unit_id(receipts[0][1]["units"][0])
    receipts[0][1]["units"][0]["outcome"] = known
    if missing in ("manifest", "both"):
        manifests.pop()
    if missing in ("row", "both"):
        receipts.pop()
    if missing == "key":
        receipts[1][1]["units"] = [u for u in receipts[1][1]["units"] if rule_unit_id(u) != key]
    result = join(population, digest, manifests, receipts)
    assert result["verdict"] == expected
    unit, = [u for u in result["units"] if u["unit_id"] == key]
    assert unit["outcome"] == expected
    assert len(unit["modules"]) == 2 and unit["modules"][1]["missing_evidence"]


def test_missing_c1_cannot_remove_its_expected_keys(monkeypatch):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    keys = join(population, digest, manifests, receipts)["units"]
    result = join(population, digest, [], [])
    assert [u["unit_id"] for u in result["units"]] == [u["unit_id"] for u in keys]
    assert all(u["outcome"] == "UNATTESTED" for u in result["units"])


def test_an_unrelated_modules_absent_key_does_not_weaken_pass(monkeypatch):
    from experiments.numerics_rows import scope

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    # Signed module B has a distinct rung and therefore no obligation for A's key.
    population["modules"][1]["q256"] = 960
    contract, _ = scope.contract_document()
    for cell in contract["lane_eligibility"]["cells"]:
        cell["rungs_q256"].append(960)
    manifests[1][1]["modules"][0]["q256"] = 960
    for unit in receipts[1][1]["units"]:
        unit["rung"] = 960
        unit["unit_id"] = row.unit_identity(unit)
    result = join(population, digest, manifests, receipts[:1])
    assert all(u["outcome"] == ("PASS" if u["unit_id"][3] == 1024 else "UNATTESTED") for u in result["units"])


@pytest.mark.parametrize("field,value", [("role", "observation"), ("admitted_by_v68", False)])
def test_observation_pass_never_supplies_attestation(monkeypatch, field, value):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    for unit in receipts[1][1]["units"]:
        unit[field] = value
    assert join(population, digest, manifests, receipts)["verdict"] == "UNATTESTED"


@pytest.mark.parametrize("duplicate", ["document", "unit", "module"])
def test_duplicate_evidence_has_no_accepted_join(monkeypatch, duplicate):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    if duplicate == "document":
        receipts.append(receipts[0])
    elif duplicate == "unit":
        receipts[0][1]["units"].append(copy.deepcopy(receipts[0][1]["units"][0]))
    else:
        receipts.append(("other-digest", copy.deepcopy(receipts[0][1])))
    with pytest.raises(Invalid, match="duplicate"):
        join(population, digest, manifests, receipts)


def test_c1_cannot_define_the_geometry_it_is_judged_against(monkeypatch):
    from experiments.numerics_rows.cases import configurations

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    cut = manifests[0][1]["modules"][0]["cuts"][0]
    cut["launch_geometry"]["rows"] *= 2
    cut["m_configs"] = configurations(**cut["launch_geometry"], library="e4m3", sms=1, structure="dense")
    with pytest.raises(Invalid, match="signed wires"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("structure,decoder,config", [("routed_moe", "native_routed_window_classes_e4m3mma", {"superblock_width": 64}),
                                                        ("dense", "native_window_gemm", {"triton_shape": [32, 64, 32]})])
def test_rule_key_projection_for_routed_and_triton(structure, decoder, config):
    unit = {"cell": "cell", "symbol": "symbol", "rung": 1024, "structure": structure, "decoder": decoder,
            "m_config": {"M": 8, **config}, "tp": {"size": 2, "rank": 0, "cut_range": [0, 32]}}
    assert rule_unit_id(unit) == ["cell", "symbol", decoder, 1024, config, {"size": 2, "rank": 0}]


def test_a_canary_is_never_joinable(monkeypatch):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    receipts[0][1]["canary"] = True
    with pytest.raises(Invalid, match="canary"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("mutation", ["frozen", "unmarked", "counted"])
def test_canary_freeze_record_is_explicit_and_cannot_be_counted(tmp_path, mutation):
    from experiments.numerics_rows.common import sha256

    rule = tmp_path / "rule.md"
    rule.write_text("draft rule")
    freeze = {"sha256": sha256(rule), "reviewer": "lead", "canary": True, "frozen_utc": None}
    row.check_rule_freeze(rule, freeze, canary=True)
    if mutation == "frozen":
        freeze["frozen_utc"] = "2026-11-16T20:00:00Z"
    elif mutation == "unmarked":
        freeze.pop("canary")
    with pytest.raises(Invalid):
        row.check_rule_freeze(rule, freeze, canary=mutation != "counted")


def test_tp_selection_requires_a_valid_cut_and_retains_only_that_cut():
    groups = [(None, None, {"size": size, "rank": rank}, []) for size, rank in ((1, 0), (2, 0), (2, 1))]
    assert row.selected_groups(groups) == groups
    assert row.selected_groups(groups, 2, 0) == [groups[1]]
    for size, rank in ((1, 1), (2, None), (None, 0)):
        with pytest.raises(Invalid):
            row.selected_groups(groups, size, rank)


def test_gpu_minutes_integrate_measured_resident_time_and_record_coverage():
    from experiments.numerics_rows.gpu_time import summarize

    samples = [{"t": t, "busy_percent": busy, "memory_mib": 1024} for t, busy in ((10, 0), (40, 100), (70, 0))]
    result = summarize(samples, 0, 120)
    assert result["wall_minutes"] == 2 and result["gpu_busy_minutes"] == 0.5
    assert result["sampled_seconds"] == 60 and result["peak_device_memory_mib"] == 1024
    assert summarize([], 0, 60, ["no device"])["gpu_busy_minutes"] is None
    assert summarize([], 0, 60, ["no device"])["errors"] == ["no device"]


def test_runtime_version_mismatch_is_an_invalid_precondition(tmp_path, monkeypatch):
    import json
    from experiments.numerics_rows import common

    expected = common.IMAGES["N"]
    inspect = tmp_path / "inspect.json"
    inspect.write_text(json.dumps([{"Id": expected["local_id"], "Digest": expected["image_digest"]}]))
    monkeypatch.setenv("TESSERA_NUMERICS_IMAGE_INSPECT", str(inspect))
    monkeypatch.setattr(common.importlib.metadata, "version", lambda name: expected[name])
    assert common.check_runtime("N")["torch"] == expected["torch"]
    monkeypatch.setattr(common.importlib.metadata, "version", lambda name: "wrong")
    with pytest.raises(Invalid, match="runtime differs"):
        common.check_runtime("N")
