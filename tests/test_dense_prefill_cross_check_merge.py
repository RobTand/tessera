"""The dense prefill crosscheck's per-rank join, pinned on fabricated tables.

``native_dense_prefill_cross_check.py`` drives one RANK per process and joins
the reports with ``--merge``.  The join is what decides whether a device run's
table is complete, and its failure modes are exactly the ones a green per-rank
report cannot show: a rank that stopped mid-table, and a world that was never
launched.  Both look like "every arm this report contains passed".

CPU-only, no device and no vLLM: these are fabricated tables in a tmp dir, and
the properties are the join's, not the route's.
"""
from __future__ import annotations

import json

import pytest

import native_dense_prefill_cross_check as cc

#: The fixtures' own group names, so a fabricated table has the right rows.
GROUPS = {
    "TESSERA_FP8": ("tessera_model_language_model_layers_3_mlp_shared_experts_gate_up_proj",
                    "tessera_model_language_model_layers_3_mlp_shared_experts_down_proj"),
    "TESSERA_BF16": ("tessera_model_layers_0_mlp_gate_up_proj",
                     "tessera_model_layers_0_mlp_down_proj"),
}
M = (0, 1)
SELECTED = {family: {"modules": [{"group": group} for group in groups]}
            for family, groups in GROUPS.items()}


def _arm(family, group, world, rank, m, passed=True):
    return {"family": family, "group": group, "tp_size": world, "tp_rank": rank, "m": m,
            "passed": passed, "native_calls": 1, "materialiser_calls": [],
            "shape": [m, 4096]}


def _report(world, rank, drop=(), passed=True):
    arms = [_arm(family, group, world, rank, m, passed=passed)
            for family, groups in GROUPS.items() for group in groups for m in M
            if (family, group, m) not in drop]
    expected = [[family, group, world, rank, m]
                for family, groups in GROUPS.items() for group in groups for m in M]
    refusals = [{"family": family, "group": group, "refusal": label, "refused": True,
                 "named_the_mismatch": True, "message": "sidecar scheme declares ..."}
                for family, groups in GROUPS.items() for group in groups
                for label in ("family", "rung")]
    return {"device": "device", "torch": "torch", "vllm": "vllm", "mode": "streamed",
            "world_size": world, "tp_rank": rank, "arms": arms,
            "expected_arms": expected, "refusals": refusals}


def _write(tmp_path, reports):
    paths = []
    for label, report in reports.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    return paths


def test_the_full_three_rank_table_merges(tmp_path):
    """Both worlds, every rank: the join keeps every arm and dedupes refusals."""
    paths = _write(tmp_path, {"tp1": _report(1, 0), "tp2r0": _report(2, 0),
                              "tp2r1": _report(2, 1)})
    merged = cc._merged(SELECTED, paths, M)
    assert merged["arms_total"] == 2 * 2 * (1 + 2) * len(M)
    assert merged["worlds"] == [1, 2]
    assert len(merged["refusals"]) == 8, "one refusal row per module per family"
    assert merged["all_arms_passed"] and merged["all_refusals_passed"]


def test_a_rank_that_stopped_mid_table_is_refused(tmp_path):
    """Every rank declares what it will drive, so a short table is a failure."""
    dropped = ("TESSERA_FP8", GROUPS["TESSERA_FP8"][0], 1)
    paths = _write(tmp_path, {"tp1": _report(1, 0, drop=(dropped,)),
                              "tp2r0": _report(2, 0), "tp2r1": _report(2, 1)})
    with pytest.raises(SystemExit, match="ran 7 of 8 arms it declared"):
        cc._merged(SELECTED, paths, M)


def test_a_world_that_was_never_launched_is_refused(tmp_path):
    """A TP1-only set is not a smaller table: rank 1's row is missing."""
    paths = _write(tmp_path, {"tp1": _report(1, 0), "tp2r0": _report(2, 0)})
    with pytest.raises(SystemExit, match="never run"):
        cc._merged(SELECTED, paths, M)


def test_a_preflight_report_is_not_a_device_table(tmp_path):
    """The same flag merges a table; a mode report is refused as one."""
    paths = _write(tmp_path, {"preflight": {"ok": True, "fixtures": [], "refusals": []}})
    with pytest.raises(SystemExit, match="not a per-rank device report"):
        cc._merged(SELECTED, paths, M)


def test_a_red_arm_does_not_merge_green(tmp_path):
    """Arm-level verdicts survive the join; they are not recomputed here."""
    paths = _write(tmp_path, {"tp1": _report(1, 0), "tp2r0": _report(2, 0, passed=False),
                              "tp2r1": _report(2, 1)})
    merged = cc._merged(SELECTED, paths, M)
    assert merged["all_arms_passed"] is False
    assert any(not arm["passed"] for arm in merged["arms"])
