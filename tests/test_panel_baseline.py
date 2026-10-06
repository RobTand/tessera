"""The #688 acceptance consumer: #685 baseline bands vs new panel rows.

Rule 8 record (2026-10-04).  First PrismaBuild run of this file, BEFORE the
compare() fix, through pbrun on dl380g10 (x86, 2 cpus, mem_gb=4, native
threads 1)::

    4 failed, 5 passed in 9.78s
    /…/checkout/src/tessera/serving/panel_baseline.py:151: AttributeError:
    'tuple' object has no attribute 'get'

— compare() matched panel rows against (row, index, view) tuples instead of
the rows themselves.  The five rule-reproduction tests passed from the start,
which is their own evidence: the preserved tables' recorded medians/quartiles
follow the bench's nearest-sample rule exactly, and NOT the interpolated
``statistics.quantiles(n=4, inclusive)`` of ``timing_panel.timing_summary``
(T16 M1 after: recorded IQR 0.003040 vs inclusive 0.002992).  A consumer that
rebuilt the recorded band with the panel's own summary rule would judge the
acceptance against a band the bench never recorded; the issue forbids exactly
that: "consume the original samples and p25/p75/IQR ... do not invent a
tolerance from rounded prose".
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import box_artifacts

from tessera.serving import timing_panel as tp

#: The preserved #685 tables, addressed through the box-artifact roots that
#: own them (tests/box_artifacts.py): a box without the tree skips with the
#: root, the variable and the path named, instead of a literal only one
#: machine resolves.
BEFORE = ("shared_runs", "pact-tradeoff-20260927", "full", "bench_linears.json")
AFTER = ("measurements", "kernel-640-pact-bench",
         "bench-after-20260928T092811Z", "bench_linears.json")
#: The digests the #688 handoff comment records for these two tables; the
#: comparator pins nothing by itself, the caller pins and the receipt records.
BEFORE_SHA256 = "9071f5146da3d7a3891b14ab980e3637a718b5436126fa337ab0ec330c14de69"
AFTER_SHA256 = "4d51fbe9c5113e0d1211c0b1f603a941e518bcfc8e6a95922281d3307f61bc2f"

FOUR_GROUPS = {"experts.T8", "experts.T16", "rate.experts.E4M3_R896", "experts.T4"}


def _present(artifact) -> Path:
    path = box_artifacts.path(*artifact)
    if path is None or not path.is_file():
        pytest.skip(box_artifacts.reason(artifact[0], path))
    return path


def test_the_panel_summary_rule_is_not_the_recorded_band_rule():
    """The trap, stated directly: same samples, two different IQRs."""
    from tessera.serving import panel_baseline as pb
    raw = json.loads(_present(AFTER).read_bytes())
    cell = next(r for r in raw["results"] if r["group"] == "experts.T16")["timings"]["1"]
    rebuilt = pb.bench_summarize(cell["samples_ms"])
    assert rebuilt["median_ms"] == cell["median_ms"]
    assert rebuilt["p25_ms"] == cell["p25_ms"] and rebuilt["p75_ms"] == cell["p75_ms"]
    assert rebuilt["iqr_ms"] == cell["iqr_ms"]
    panel_rule = tp.timing_summary(cell["samples_ms"])
    assert panel_rule["median_ms"] != cell["median_ms"]
    assert panel_rule["iqr_ms"] != cell["iqr_ms"]


@pytest.mark.parametrize("table", ["before", "after"])
def test_reproduces_every_recorded_statistic_of_both_preserved_tables(table):
    """Every cell of the preserved table, not just the four compared groups."""
    from tessera.serving import panel_baseline as pb
    path = _present(BEFORE if table == "before" else AFTER)
    doc = json.loads(path.read_bytes())
    checked = 0
    for group in doc["results"]:
        for m, cell in group["timings"].items():
            rebuilt = pb.bench_summarize(cell["samples_ms"])
            for key in ("median_ms", "p25_ms", "p75_ms", "iqr_ms", "min_ms", "n"):
                assert rebuilt[key] == cell[key], (
                    f"{table} {group['group']} M={m} {key}: "
                    f"recorded {cell[key]!r}, bench-rule reconstruction {rebuilt[key]!r}")
            checked += 1
    assert checked >= 22 * 11  # 22 groups x 11 M values in each preserved table


def test_compares_the_four_documented_groups_at_the_requested_m_values():
    from tessera.serving import panel_baseline as pb
    doc = json.loads(_present(AFTER).read_bytes())
    rows = []
    # experts.T8, layer 10, E4M3 K1 1024/1024, w13 2048x4096 / w2 4096x1024.
    t8 = next(r for r in doc["results"] if r["group"] == "experts.T8")
    for m in pb.REQUESTED_MS:
        rows.append(pb.row_view(structure="routed_moe",
                                module=t8["module"], family=t8["info"]["scheme_family"],
                                grid=t8["info"]["grid"], q256=(1024, 1024),
                                rank_local_shape=((2048, 4096), (4096, 1024)), m=m,
                                median_ms=t8["timings"][str(m)]["median_ms"], samples_n=30))
    receipt = pb.compare(_present(AFTER).read_bytes(), rows, baseline_sha256=AFTER_SHA256)
    assert receipt["schema"] == pb.SCHEMA
    assert receipt["baseline"]["sha256"] == AFTER_SHA256
    assert receipt["bench_rule"]["proven_reproduction"] is True
    verdicts = receipt["groups"]["experts.T8"]
    assert set(verdicts) == set(str(m) for m in pb.REQUESTED_MS)
    assert all(v["verdict"] == "reproduced" for v in verdicts.values())
    # The other three documented groups have no row: named, not silent.
    for selector in FOUR_GROUPS - {"experts.T8"}:
        assert all(v["verdict"] == "no_panel_row" for v in receipt["groups"][selector].values())


def test_a_median_outside_the_recorded_band_is_a_named_gap():
    from tessera.serving import panel_baseline as pb
    doc = json.loads(_present(AFTER).read_bytes())
    t4 = next(r for r in doc["results"] if r["group"] == "experts.T4")
    band = t4["timings"]["512"]
    row = pb.row_view(structure="routed_moe", module=t4["module"],
                      family=t4["info"]["scheme_family"], grid=t4["info"]["grid"],
                      q256=(896, 896), rank_local_shape=((2048, 4096), (4096, 1024)),
                      m=512, median_ms=band["p75_ms"] * 1.001, samples_n=30)
    receipt = pb.compare(_present(AFTER).read_bytes(), [row], baseline_sha256=AFTER_SHA256)
    verdict = receipt["groups"]["experts.T4"]["512"]
    assert verdict["verdict"] == "gap"
    assert verdict["recorded"]["p75_ms"] == band["p75_ms"]
    assert verdict["new"]["median_ms"] == band["p75_ms"] * 1.001


def test_identity_mismatch_never_compares_numbers():
    from tessera.serving import panel_baseline as pb
    doc = json.loads(_present(AFTER).read_bytes())
    t8 = next(r for r in doc["results"] if r["group"] == "experts.T8")
    # Same (structure, module, family, grid, rate) key, wrong rank-local shape:
    # the join must refuse to compare numbers across a different geometry.
    row = pb.row_view(structure="routed_moe", module=t8["module"],
                      family=t8["info"]["scheme_family"], grid=t8["info"]["grid"],
                      q256=(1024, 1024), rank_local_shape=((2048, 4096), (4096, 512)),
                      m=512, median_ms=t8["timings"]["512"]["median_ms"], samples_n=30)
    receipt = pb.compare(_present(AFTER).read_bytes(), [row], baseline_sha256=AFTER_SHA256)
    verdict = receipt["groups"]["experts.T8"]["512"]
    assert verdict["verdict"] == "identity_mismatch"
    assert "4096" in verdict["reason"] and verdict.get("new", {}).get("median_ms") is None


def test_wrong_baseline_bytes_refuse_against_a_pinned_digest():
    from tessera.serving import panel_baseline as pb
    doc = json.loads(_present(AFTER).read_bytes())
    with pytest.raises(ValueError, match="pinned baseline digest"):
        pb.compare(_present(AFTER).read_bytes(), [], baseline_sha256="0" * 64)


def test_panel_rows_project_from_a_validated_dense_panel(tmp_path):
    """The dense CPU fixture panel of tests/test_native_timing_panel.py, reused."""
    pytest.importorskip("torch")
    import test_native_timing_panel as base
    import torch
    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused_frame import pack_fused
    generator = torch.Generator().manual_seed(688)
    weight = torch.randn(16, 128, generator=generator) * 0.02
    exported, _, _ = encode_linear_planes(weight, grid=E4M3_GRID, q256=1024,
                                          name="weight", verify=False)
    blob = pack_fused([("weight", 16, exported.blob)])
    declaration = {"family": "TESSERA_FP8", "structure": "dense", "grid": "E4M3",
                   "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
                   "rows": 16, "columns": 128, "roles": [["weight", 16]], "wire_bytes": len(blob)}
    panel = base.build_dense_fixture_panel(tmp_path, blob=blob, declaration=declaration)
    from tessera.serving import panel_baseline as pb
    expected = copy.deepcopy(panel["runtime"])
    tp.validate_panel(panel, expected_runtime=expected)
    rows = pb.panel_row_views(panel)
    assert len(rows) == 1
    row = rows[0]
    assert row["structure"] == "dense" and row["family"] == "TESSERA_FP8"
    assert row["m"] == 512 and row["median_ms"] == 2.5
    # ... and the dense row matches no routed #685 group, by name.
    doc = json.loads(_present(AFTER).read_bytes())
    receipt = pb.compare(_present(AFTER).read_bytes(), rows, baseline_sha256=AFTER_SHA256)
    for selector in FOUR_GROUPS:
        assert all(v["verdict"] == "no_panel_row" for v in receipt["groups"][selector].values())


def test_unprojectable_panel_rows_are_named_not_dropped():
    from tessera.serving import panel_baseline as pb
    doc = json.loads(_present(AFTER).read_bytes())
    panel = {"schema": tp.SCHEMA, "rows": [{"scope_id": "x", "timing": {"median_ms": 1.0},
                                             "scheme": {}, "prefix": "p",
                                             "cell_id": "c"}],
             "plan": {"rows": [{"id": "x", "scope": {"structure": "routed_moe"}}]},
             "runtime": {}}
    with pytest.raises(ValueError, match="routed_moe"):
        pb.panel_row_views(panel)


TOOL = Path(__file__).resolve().parents[1] / "tools" / "tessera_panel_baseline.py"


def _run_cli(args, cwd):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_the_reviewer_cli_names_missing_samples_without_a_traceback(tmp_path):
    baseline = tmp_path / "missing-samples.json"
    baseline.write_text(json.dumps({"results": [{"group": "broken",
                                                "timings": {"1": {"n": 3}}}]}))
    proc = _run_cli(["verify-table", "--baseline", str(baseline)], tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "REFUSED" in proc.stderr and "samples_ms" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_the_reviewer_cli_proves_the_recorded_rule_over_the_after_table(tmp_path):
    path = _present(AFTER)
    proc = _run_cli(["verify-table", "--baseline", str(path),
                     "--sha256", AFTER_SHA256], tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["mode"] == "verify_rule" and out["schema"] == "tessera.shape_time_baseline_comparison.v1"
    assert out["baseline"]["sha256"] == AFTER_SHA256
    assert out["proof"]["cells_checked"] >= 22 * 11
    assert out["bench_rule"]["proven_reproduction"] is True


def test_the_reviewer_cli_refuses_a_wrong_baseline_pin(tmp_path):
    path = _present(AFTER)
    proc = _run_cli(["verify-table", "--baseline", str(path),
                     "--sha256", "0" * 64], tmp_path)
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr and "pinned digest differs" in proc.stderr


def test_the_reviewer_cli_compares_a_validated_panel_bytes_pinned(tmp_path):
    """compare mode end to end: validate, project, pin the panel bytes, judge."""
    pytest.importorskip("torch")
    import test_native_timing_panel as base
    import torch
    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused_frame import pack_fused
    generator = torch.Generator().manual_seed(688)
    weight = torch.randn(16, 128, generator=generator) * 0.02
    exported, _, _ = encode_linear_planes(weight, grid=E4M3_GRID, q256=1024,
                                          name="weight", verify=False)
    blob = pack_fused([("weight", 16, exported.blob)])
    declaration = {"family": "TESSERA_FP8", "structure": "dense", "grid": "E4M3",
                   "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
                   "rows": 16, "columns": 128, "roles": [["weight", 16]], "wire_bytes": len(blob)}
    panel = base.build_dense_fixture_panel(tmp_path, blob=blob, declaration=declaration)
    panel_path = tmp_path / "panel.json"
    panel_path.write_bytes(tp.canonical(panel))
    runtime_path = tmp_path / "runtime-expected.json"
    runtime_path.write_bytes(tp.canonical(copy.deepcopy(panel["runtime"])))
    proc = _run_cli(["compare", "--baseline", str(_present(AFTER)),
                     "--sha256", AFTER_SHA256,
                     "--panel", str(panel_path),
                     "--panel-sha256", hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                     "--expected-runtime", str(runtime_path)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["baseline"]["sha256"] == AFTER_SHA256
    assert receipt["baseline_sha256_pin"]["matched"] is True
    assert receipt["bench_rule"]["proven_reproduction"] is True
    # The dense fixture row can satisfy no routed #685 group, by name.
    for selector in FOUR_GROUPS:
        assert all(v["verdict"] == "no_panel_row"
                   for v in receipt["groups"][selector].values())


# --- corrective regression set (2026-10-06 review at
# d894dae67cb93b8d5eddbae5d1da720a95611c0b): geometry agreement, one owned
# buffer, and input grammar at the comparator boundary.  These run without
# Torch and without the preserved tables: the baseline is a synthetic table
# whose recorded statistics follow the bench rule exactly.


def _synthetic_table(groups=FOUR_GROUPS, samples=(1.0, 2.0, 3.0, 4.0)):
    """A valid recorded-rule table: every cell carries the bench's own stats."""
    from tessera.serving import panel_baseline as pb
    cell = {"samples_ms": list(samples)}
    cell.update(pb.bench_summarize(samples))
    results = []
    for selector in groups:
        documented = next(g for g in pb.BASELINE_GROUPS if g["selector"] == selector)
        results.append({
            "group": selector,
            "module": documented["module"],
            "kind": documented["kind"],
            "info": {"scheme_family": documented["family"],
                     "grid": documented["grid"],
                     "q256": {"w13": documented["q256"][0],
                              "w2": documented["q256"][1]}},
            "timings": {str(m): dict(cell) for m in pb.REQUESTED_MS},
        })
    return json.dumps({"results": results}).encode()


def _t8_row(pb, **overrides):
    documented = next(g for g in pb.BASELINE_GROUPS if g["selector"] == "experts.T8")
    kwargs = dict(structure="routed_moe", module=documented["module"],
                  family="TESSERA_FP8", grid="E4M3", q256=(1024, 1024),
                  rank_local_shape=((2048, 4096), (4096, 1024)),
                  m=512, median_ms=2.5, samples_n=30)
    kwargs.update(overrides)
    return pb.row_view(**kwargs)


def test_geometry_unknown_is_nonpassing_and_never_borrows_the_reference():
    """A row without its own rank-local geometry is not agreement with it."""
    from tessera.serving import panel_baseline as pb
    row = _t8_row(pb, rank_local_shape=None)
    receipt = pb.compare(_synthetic_table(), [row])
    verdict = receipt["groups"]["experts.T8"]["512"]
    assert verdict["verdict"] == "geometry_missing"
    assert verdict["identity"]["row"]["rank_local_shape"] is None
    assert verdict.get("new") is None and "recorded" not in verdict


def test_geometry_malformed_refuses_at_the_row_boundary_with_a_name():
    from tessera.serving import panel_baseline as pb
    for bad in (((2048, 4096), 512),                    # not a pair
                ((2048, 4096), (4096, 512.5)),          # not integers
                ((2048, 4096), (4096, "1024")),         # not integers
                (),                                     # nothing to agree with
                ((0, 4096), (4096, 1024))):             # nonpositive dimension
            with pytest.raises(ValueError, match="rank-local shape"):
                _t8_row(pb, rank_local_shape=bad)


def test_compare_refuses_malformed_geometry_on_raw_claims_by_name():
    """compare owns the boundary too: dict claims cannot smuggle geometry in."""
    from tessera.serving import panel_baseline as pb
    good = _t8_row(pb)
    for bad in (((2048,),), "2048x4096", ((2048, 4096), (4096, None))):
        with pytest.raises(ValueError, match="rank-local shape"):
            pb.compare(_synthetic_table(), [dict(good, rank_local_shape=bad)])


def test_verify_table_binds_the_very_bytes_it_proved_over_a_pipe(tmp_path):
    """Interleaved-writer regression: the published binding describes the
    bytes the proof actually verified.  A FIFO's second read returns empty
    bytes, so a tool that binds by re-reading publishes a digest of
    nothing; one owned buffer publishes the table's own digest."""
    from tessera.serving import panel_baseline as pb
    table = _synthetic_table()
    fifo = tmp_path / "baseline.json"
    os.mkfifo(fifo)
    proc = subprocess.Popen([sys.executable, str(TOOL), "verify-table",
                             "--baseline", str(fifo)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    with open(fifo, "wb") as handle:
        handle.write(table)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, err
    receipt = json.loads(out)
    assert receipt["baseline"]["sha256"] == hashlib.sha256(table).hexdigest()
    assert receipt["baseline"]["bytes"] == len(table)
    assert receipt["bench_rule"]["cells_checked"] >= len(FOUR_GROUPS)


def test_verify_table_replacement_and_pin_stay_hard_across_runs(tmp_path):
    """A real replacement between runs: the pin refuses the old digest, and
    each receipt's binding and proof describe that run's actual bytes."""
    path = tmp_path / "baseline.json"
    first = _synthetic_table()
    path.write_bytes(first)
    proc = _run_cli(["verify-table", "--baseline", str(path)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["baseline"]["sha256"] == hashlib.sha256(first).hexdigest()
    second = _synthetic_table(samples=(5.0, 6.0, 7.0, 8.0))
    replacement = tmp_path / "next.json"
    replacement.write_bytes(second)
    os.replace(replacement, path)
    proc = _run_cli(["verify-table", "--baseline", str(path),
                     "--sha256", hashlib.sha256(first).hexdigest()], tmp_path)
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr and "pinned digest differs" in proc.stderr
    proc = _run_cli(["verify-table", "--baseline", str(path)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["baseline"]["sha256"] == hashlib.sha256(second).hexdigest()
    assert receipt["baseline"]["bytes"] == len(second)


def test_timing_cell_must_be_an_object_named_at_the_boundary():
    from tessera.serving import panel_baseline as pb
    with pytest.raises(ValueError, match="timing cell"):
        pb.verify_recorded_statistics(
            json.dumps({"results": [{"group": "broken",
                                     "timings": {"1": "median"}}]}).encode())
    with pytest.raises(ValueError, match="timings"):
        pb.verify_recorded_statistics(
            json.dumps({"results": [{"group": "broken",
                                     "timings": ["1"]}]}).encode())


def test_the_reviewer_cli_names_a_nonobject_cell_without_a_traceback(tmp_path):
    baseline = tmp_path / "bad-cell.json"
    baseline.write_text(json.dumps({"results": [{"group": "broken",
                                                 "timings": {"1": "median"}}]}))
    proc = _run_cli(["verify-table", "--baseline", str(baseline)], tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "REFUSED" in proc.stderr and "timing cell" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_samples_must_be_numbers_named_at_the_boundary():
    from tessera.serving import panel_baseline as pb
    for bad in ([1.0, None, 3.0, 4.0], ["1", "2", "3"], [True, False, True, False],
                [1.0, [2.0], 3.0, 4.0]):
        with pytest.raises(ValueError, match="samples_ms"):
            pb.bench_summarize(bad)


def test_the_reviewer_cli_names_null_samples_without_a_traceback(tmp_path):
    baseline = tmp_path / "null-samples.json"
    baseline.write_text(json.dumps({"results": [{"group": "broken", "timings": {"1": {
        "samples_ms": [1.0, None, 3.0, 4.0], "median_ms": 2.0, "p25_ms": 1.0,
        "p75_ms": 3.0, "iqr_ms": 2.0, "min_ms": 1.0, "n": 4}}}]}))
    proc = _run_cli(["verify-table", "--baseline", str(baseline)], tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "REFUSED" in proc.stderr and "samples_ms" in proc.stderr
    assert "Traceback" not in proc.stderr
