"""Per-key module coverage has an independent signed-wire expectation set."""
import copy

import pytest

from experiments.numerics_rows import row
from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.join import join, rule_unit_id
from tests.test_numerics_rows_receipts import joined_inputs


def test_join_keeps_stopped_per_key_below_fail(monkeypatch):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    stopped_key = rule_unit_id(receipts[0][1]["units"][0])
    failed = next(u for u in receipts[0][1]["units"] if rule_unit_id(u) != stopped_key and u["role"] == "unit" and u["admitted_by_v68"])
    for _, receipt in receipts:
        assert receipt["row"]["finalization_finished"]
        for unit in receipt["units"]:
            if rule_unit_id(unit) == stopped_key:
                unit.update(outcome="STOPPED", completed=False, reasons=["open STOPPED attempt unit/1"])
    result = join(population, digest, manifests, receipts)
    assert result["verdict"] == "STOPPED"
    stopped, = [u for u in result["units"] if u["unit_id"] == stopped_key]
    assert stopped["outcome"] == "STOPPED" and all(m["outcome"] == "STOPPED" for m in stopped["modules"])
    failed.update(outcome="FAIL", findings=["known mismatch"])
    mixed = join(population, digest, manifests, receipts)
    assert mixed["verdict"] == "FAIL"
    assert next(u for u in mixed["units"] if u["unit_id"] == stopped_key) == stopped
    assert next(u for u in mixed["units"] if u["unit_id"] == rule_unit_id(failed))["outcome"] == "FAIL"


@pytest.mark.parametrize("target", ("manifest", "runtime", "unit"))
def test_join_refuses_missing_dependency_math_binding(monkeypatch, target):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    record = manifests[0][1] if target == "manifest" else receipts[0][1]["runtime"] if target == "runtime" else receipts[0][1]["units"][0]
    del record["dependency_math_sha256"]
    with pytest.raises(Invalid, match="math settings"):
        join(population, digest, manifests, receipts)


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
    assert [u["unit_id"][:-1] for u in result["units"]] == [u["unit_id"][:-1] for u in keys]
    assert result["required_scope"]["launch"] is None
    assert all(u["unit_id"][-1] is None and u["outcome"] == "UNATTESTED" for u in result["units"])


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
def test_join_refuses_observation_labels_on_required_fail_keys(monkeypatch, field, value):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    positive = join(population, digest, manifests, receipts)
    assert positive["verdict"] == "PASS"
    unit = receipts[0][1]["units"][0]
    observation = copy.deepcopy(unit)
    observation.update(outcome="FAIL", role="observation", admitted_by_v68=False, rung=unit["rung"] + 1)
    observation["unit_id"] = row.unit_identity(observation)
    receipts[0][1]["units"].append(observation)
    assert join(population, digest, manifests, receipts) == positive
    unit.update(outcome="FAIL", **{field: value})
    with pytest.raises(Invalid, match="observation label.*required signed scope"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("field,value", [("role", "observation"), ("admitted_by_v68", False)])
def test_observation_pass_never_supplies_attestation(monkeypatch, field, value):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    for unit in receipts[1][1]["units"]:
        unit[field] = value
    with pytest.raises(Invalid, match="observation label conflicts with the required signed scope"):
        join(population, digest, manifests, receipts)


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
    assert rule_unit_id(unit) == ["cell", "symbol", decoder, 1024, config, {"size": 2, "rank": 0}, None]


@pytest.mark.parametrize("owner", [None, "row", "rule"])
def test_a_canary_is_never_joinable(monkeypatch, owner):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    target = receipts[0][1] if owner is None else receipts[0][1][owner]
    target["canary"] = True
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
    assert row.selected_groups(groups, 1, 0) == [groups[0]]
    assert row.selected_groups(groups, 2, None) == groups[1:]
    for rank in (0, 1):
        with pytest.raises(Invalid, match="rule P5 requires both TP2 ranks in one process"):
            row.selected_groups(groups, 2, rank)
    for size, rank in ((1, 1), (None, 0)):
        with pytest.raises(Invalid):
            row.selected_groups(groups, size, rank)
    with pytest.raises(Invalid, match="selected TP cut is absent"):
        row.selected_groups(groups[:1], 2)


def test_gpu_minutes_integrate_measured_resident_time_and_record_coverage():
    from experiments.numerics_rows.gpu_time import summarize

    samples = [{"t": t, "busy_percent": busy, "compute_process_count": 1, "compute_apps_memory_mib": 1024} for t, busy in ((10, 0), (40, 100), (70, 0))]
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
    monkeypatch.setenv("NUMERICS_IMAGE_INSPECT", str(inspect))
    monkeypatch.setattr(common.importlib.metadata, "version", lambda name: expected[name])
    assert common.check_runtime("N")["torch"] == expected["torch"]
    monkeypatch.setattr(common.importlib.metadata, "version", lambda name: "wrong")
    with pytest.raises(Invalid, match="runtime differs"):
        common.check_runtime("N")


def test_missing_m_evidence_is_unattested_even_when_another_m_has_the_same_key(monkeypatch):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    receipts[1][1]["units"] = [u for u in receipts[1][1]["units"] if u["m_config"]["M"] != 8]
    result = join(population, digest, manifests, receipts)
    assert result["verdict"] == "UNATTESTED"
    assert any(u["outcome"] == "UNATTESTED" and u["unit_id"][0] == "batch" for u in result["units"])


@pytest.mark.parametrize("frozen", [None, "absent", "", "  "])
def test_join_refuses_an_unfrozen_rule(monkeypatch, frozen):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    rule = receipts[1][1]["rule"]
    if frozen == "absent":
        rule.pop("frozen_utc")
    else:
        rule["frozen_utc"] = frozen
    with pytest.raises(Invalid, match="frozen rule"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("split", ["projected_key", "empty", "observations"])
def test_join_refuses_overlapping_row_modules_even_without_overlapping_units(monkeypatch, split):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    original = receipts[0][1]
    other = copy.deepcopy(original)
    if split == "projected_key":
        key = rule_unit_id(original["units"][0])
        other["units"] = [u for u in other["units"] if rule_unit_id(u) != key]
        original["units"] = [u for u in original["units"] if rule_unit_id(u) == key]
        assert other["units"] and original["units"]
    elif split == "empty":
        other["units"] = []
    else:
        for unit in other["units"]:
            unit.update(role="observation", admitted_by_v68=False)
    receipts.insert(1, ("distinct-receipt-hash", other))
    with pytest.raises(Invalid, match="duplicate row module selection"):
        join(population, digest, manifests, receipts)


def test_one_missing_c1_keeps_that_modules_distinct_keys_unattested(monkeypatch):
    from experiments.numerics_rows import scope

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    population["modules"][1]["q256"] = 960
    contract, _ = scope.contract_document()
    for cell in contract["lane_eligibility"]["cells"]:
        cell["rungs_q256"].append(960)
    manifests[1][1]["modules"][0]["q256"] = 960
    for unit in receipts[1][1]["units"]:
        unit["rung"] = 960
        unit["unit_id"] = row.unit_identity(unit)
    positive = join(population, digest, manifests, receipts)
    assert positive["verdict"] == "PASS"
    manifests.pop()
    result = join(population, digest, manifests, receipts)
    assert [u["unit_id"] for u in result["units"]] == [u["unit_id"] for u in positive["units"]]
    assert result["verdict"] == "UNATTESTED"
    for unit in result["units"]:
        missing = unit["unit_id"][3] == 960
        assert unit["outcome"] == ("UNATTESTED" if missing else "PASS")
        assert len(unit["modules"]) == 1 and unit["modules"][0]["missing_evidence"] is missing


@pytest.mark.parametrize("where", ("row", "unit", "manifest", "object", "policy"))
def test_join_refuses_mixed_launch_state_and_lever_policy(monkeypatch, where):
    from experiments.numerics_rows.launch_state import digest as state_digest

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    receipt = receipts[1][1]
    if where == "row":
        receipt["runtime"]["launch_state_names"] = ["TESSERA_NEW"]
        receipt["runtime"]["launch_state_sha256"] = state_digest({"TESSERA_NEW": "1"})
    elif where == "unit":
        unit = receipt["units"][0]
        unit["launch_state_sha256"] = "b" * 64
        unit["unit_id"] = row.unit_identity(unit)
    elif where == "manifest":
        manifests[1][1]["launch_state_sha256"] = "b" * 64
    elif where == "object":
        receipt["runtime"]["launch_state_names"].append("TESSERA_NEW")
    else:
        receipt["inputs"]["launch_levers_sha256"] = "b" * 64
    with pytest.raises(Invalid, match="launch state|lever|execution identities"):
        join(population, digest, manifests, receipts)
