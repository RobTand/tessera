"""Synthetic CPU protocol tests, never GPU/placement admission evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.full_engine_resource_partition import assemble_full_engine_resource_report
from experiments.full_engine_resources import analyze_engine_resource_ledger


DIGESTS = ("configuration_sha256", "model_sha256", "runtime_manifest_sha256",
           "workload_sha256", "assignment_sha256", "canonical_units_sha256")


def ledger():
    raw = json.loads((Path(__file__).parent / "fixtures/full_engine_resource_ledger.json").read_text())
    return analyze_engine_resource_ledger(raw)


def members(world=1):
    return {
        "reference": {"canonical_census": "synthetic", "runtime_binding": "synthetic",
                      "selected_rows": ["synthetic"]},
        "workload": {"calibration": "synthetic", "prompt_ids": ["synthetic"], "sampling": "synthetic"},
        "execution": {"graph_mode": "eager", "residency": "resident", "topology": f"tp{world}"},
    }


def rank_ledger(rank):
    result = ledger()
    result["identity"].update(rank=rank, world_size=2, device_id=0,
                              device_uuid=f"GPU-rank-{rank}", host={"ip": f"192.168.1.{107 + rank}"})
    result["rank_world"] = {
        "schema": "tessera.full_engine_rank_world.v1", "world_size": 2,
        "raw_run": {"path": "/fixture/run.json", "sha256": "a" * 64},
        "raw_plan": {"path": "/fixture/plan.json", "sha256": "b" * 64},
        "run_identity": {name: result["identity"][name] for name in DIGESTS},
        "ranks": [{"rank": i, "world_size": 2, "device_id": 0, "device_uuid": f"GPU-rank-{i}",
                   "host": {"ip": f"192.168.1.{107 + i}"}, "process_id": 100 + i,
                   "capture": {"path": f"/fixture/rank-{i}.json", "sha256": "c" * 64},
                   "runtime_evidence": {"path": f"/fixture/runtime-{i}.json", "sha256": "d" * 64}}
                  for i in (0, 1)],
    }
    return result


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_real_assembler_emits_v3_observation_not_placement(rank):
    source = rank_ledger(rank)
    report = assemble_full_engine_resource_report(source, **members(2))
    assert report["schema"] == "tessera.full_engine_resource_report.v3"
    assert report["observations"]["rank_world"] == source["rank_world"]
    assert report["derived"]["placement_obligation"] is None
    assert report["derived"]["certifies_placement"] is False
    assert "not a reusable fixed byte price" in report["derived"]["proposal_placement_rule"]


def test_tp1_v2_baseline_bytes():
    report = assemble_full_engine_resource_report(ledger(), **members())
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    # Captured on pre-change master by PB ee1071d55ec9; this fixes the exact
    # serialized TP1 v2 report, not merely its schema or a subset of fields.
    assert hashlib.sha256(encoded).hexdigest() == (
        "67a4af9371e51fddb5838f33eef56349af27c3ce70e4e66f3c81263bd951ca1f")
    assert report["schema"] == "tessera.full_engine_resource_report.v2"
    assert "rank_world" not in report["observations"]
    assert "certifies_placement" not in report["derived"]


@pytest.mark.parametrize("world", [0, -1, 3, True, 2.0])
def test_unsupported_world_is_not_projected_to_tp1(world):
    source = ledger()
    source["identity"]["world_size"] = world
    with pytest.raises(ValueError):
        assemble_full_engine_resource_report(source, **members())


@pytest.mark.parametrize("mutation", ["rank", "world", "own_device", "digest", "missing_ref"])
def test_tp2_roster_corruption_refuses(mutation):
    source = rank_ledger(1)
    roster = copy.deepcopy(source["rank_world"])
    if mutation == "rank":
        roster["ranks"][1]["rank"] = 0
    elif mutation == "world":
        roster["ranks"][0]["world_size"] = 1
    elif mutation == "own_device":
        roster["ranks"][1]["device_uuid"] = "GPU-other"
    elif mutation == "digest":
        roster["run_identity"]["configuration_sha256"] = "0" * 64
    else:
        del roster["ranks"][1]["capture"]
    source["rank_world"] = roster
    with pytest.raises(ValueError):
        assemble_full_engine_resource_report(source, **members(2))
