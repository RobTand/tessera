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
#: Keyed by FIXTURE, with the payload family beside it: the arm identity is the
#: tree, because two trees serve one family now.
GROUPS = {
    "TESSERA_FP8": ("TESSERA_FP8",
                    ("tessera_model_language_model_layers_3_mlp_shared_experts_gate_up_proj",
                     "tessera_model_language_model_layers_3_mlp_shared_experts_down_proj")),
    "TESSERA_BF16": ("TESSERA_BF16",
                     ("tessera_model_layers_0_mlp_gate_up_proj",
                      "tessera_model_layers_0_mlp_down_proj")),
}
M = (0, 1)
SELECTED = {fixture: {"family": family,
                      "modules": [{"group": group} for group in groups]}
            for fixture, (family, groups) in GROUPS.items()}


def _arm(fixture, family, group, world, rank, m, passed=True):
    return {"fixture": fixture, "family": family, "group": group,
            "tp_size": world, "tp_rank": rank, "m": m,
            "passed": passed, "native_calls": 1, "materialiser_calls": [],
            "shape": [m, 4096]}


def _report(world, rank, drop=(), passed=True):
    arms = [_arm(fixture, family, group, world, rank, m, passed=passed)
            for fixture, (family, groups) in GROUPS.items() for group in groups for m in M
            if (fixture, group, m) not in drop]
    expected = [[fixture, group, world, rank, m]
                for fixture, (_family, groups) in GROUPS.items()
                for group in groups for m in M]
    refusals = [{"fixture": fixture, "family": family, "group": group,
                 "refusal": label, "refused": True,
                 "named_the_mismatch": True, "message": "sidecar scheme declares ..."}
                for fixture, (family, groups) in GROUPS.items() for group in groups
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
    dropped = ("TESSERA_FP8", GROUPS["TESSERA_FP8"][1][0], 1)
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


def test_a_fixture_key_names_a_tree_and_the_family_names_a_route():
    """Two artifacts, one family: the key cannot be the family any more.

    The small modules the lane landed against and the GLM layer-0 export are
    both served by their family's route, so a table keyed by family could hold
    only one of them at a time -- and the one it dropped would be the artifact
    the lane is held to now.
    """
    from collections import Counter

    counts = Counter(spec["family"] for spec in cc.FIXTURES.values())
    assert counts["TESSERA_BF16"] >= 2 and counts["TESSERA_FP8"] >= 2, counts
    assert any(key != spec["family"] for key, spec in cc.FIXTURES.items())
    assert set(cc.FAMILIES) == set(counts)
    for spec in cc.FIXTURES.values():
        assert spec["family"] in cc.FAMILIES


def test_the_glm_fixtures_are_named_through_box_artifacts():
    """No literal box path: the shared measurement root owns the address."""
    import box_artifacts

    for key in ("GLM_A8", "GLM_A16"):
        spec = cc.FIXTURES[key]
        root_key = spec["artifact"][0]
        assert root_key in box_artifacts.ROOTS
        resolved = box_artifacts.path(*spec["artifact"])
        assert resolved is not None and str(resolved).startswith(
            str(box_artifacts.root(root_key)))


def test_no_harness_option_is_a_torchrun_abbreviation():
    """The batch flag cannot be ``--m``: torchrun parses in front of this CLI.

    Measured on the pinned image: ``torch.distributed.run ... <script> --m 0``
    is refused with ``error: ambiguous option: --m`` (it abbreviates
    ``--max-restarts``, ``--monitor-interval``, ``--module``, ``--master-addr``
    and ``--master-port``) before the harness's own parser sees the argument,
    while ``--mset`` passes through.  A harness whose documented device command
    cannot run is a harness nobody can run, so the spelling is pinned here.

    The spellings are read out of the SOURCE rather than off ``_parser()``, so
    the test names the defect on the revision that had it instead of failing on
    a missing attribute.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(cc.__file__).read_text())
    options = {node.args[0].value
               for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == "add_argument"
               and node.args
               and isinstance(node.args[0], ast.Constant)}
    assert "--mset" in options
    assert "--m" not in options
    assert "--fixture" in options and "--family" in options
