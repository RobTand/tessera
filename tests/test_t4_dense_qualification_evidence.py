"""Acceptance of retained current-WINDOW dense qualification, not a GPU campaign.

The deliverable is a small, portable evidence index at PACKET, consuming the
existing coverage/measurement owners. It embeds v3 reports (and their CLI argv),
not tensors, and digest-binds repository-relative supporting receipts. Large
primary captures may stay in the existing custody store, linked by the receipts.
A receipt's authenticity and relevance still require review; hashes and a green
CPU test are not proof that CUDA ran. No serving-default/menu change is required.
These are CPU evidence-consumption checks, not a substitute for the source-bound
GPU campaign. Review must also reconcile coverage to the authoritative owner
manifest, inspect receipt scope and verify any published scope against admission.

Packet fields used here: coverage (normalized owner cases), reports (argv/report
pairs), baselines (existing T8 comparison objects), and supporting_evidence
(name -> {path, sha256}). Coverage cases carry id, family, rows, cols, roles,
ms (timing), correctness_ms, tp_cases ([[world_size, rank], ...]), and
disposition (measured/refused); refusals also name reason and evidence.
rows/cols/roles are rank-local module dimensions, as in bench_dense_module.MODULES;
reports retain local_rows/local_cols separately from global export dimensions.
Measured dense cells retain shape_tag == case id and role_checks in output order:
name, rows, output_start, numeric, graph_equal. This checks single roles too.
The body timing domain reaches M8192; head timing uses sampled decode positions,
not full body prefill, following the existing geometry owner's domain.
These are acceptance receipts, not a second canonical coverage/publication API.

Pre-implementation check (x86, Python 3.14.4, CPU torch): both tests failed at
_packet with "Current WINDOW/L14 dense qualification evidence is missing".
2 failed in 6.22s; no skips; NO CUDA; zero device allocations. No GPU run was
needed to establish that the portable evidence deliverable is absent. This is
not a pre-fix numerical or kernel-performance receipt.
Command: PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest tests/test_t4_dense_qualification_evidence.py -q -n 2
--dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"af4ce18906eabc910016ec049b9e80d536519cee7cf2014a1ca59c054d630389","size":3657}},"operation":"16ae914c6c4a0a106e788263886071a383b0d0b1cd17128505f09194dc487aa3","receipt":"0c831af41e3b125bf55764854d93a19f23f6c2c3a61c4f1ed1c31bab9209f203"}]

Final acceptance rerun after separating measurement from performance admission:
2 failed in 6.10s at the same missing-packet assertion; NO CUDA, no skips,
zero device allocations.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4af17894b2f1a2fd35abf1f5f1473460d3bd9a02fb16b36e779df2d83246d700","size":3657}},"operation":"eeceb0f543f93284b64a6f6ca4beb753f5fbe63683f8ef0ad1a86c8c729adc5b","receipt":"c4ed21b0f6b5bcefdfbed43f613765f2f67f68820d37267b74af561b637ce284"}]

Repository guards: tests/test_issue_refs.py and tests/test_impacted_tests.py,
with the same command options, passed 212 tests in 77.59s; NO CUDA, no skips,
zero device allocations.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"effa1e6f11244684cef00261076c4c34a0323521172ca6d6c69cdcb40d5b63d8","size":1775}},"operation":"30b50ad2f6d931544368dfb56b1cad511d4feb52572deb75613e12ed170e0670","receipt":"d5679b115f2e776cfadc8cddf49d7a7b87e2cef042f2b93ba5da8111fc10638a"}]
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from experiments.t4_code import t4_fused_qualify as qualify

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/measurements/t4-dense-qualification.json"
WIDTHS = set(range(128, 1025, 128))  # The eight pure widths requested by this issue.
# Existing bench_dense_module body sweep, not a new performance threshold.
BODY_MS = {1, 2, 4, 8, 16, 64, 512, 2048, 4096, 8192}
FAMILIES = {"dense_mlp", "shared_expert", "attention", "head", "vision"}
KDA_ROLES = [["q_proj", 4096], ["k_proj", 4096], ["v_proj", 4096],
             ["b_proj", 32], ["f_a_proj", 128], ["g_a_proj", 128]]


def _packet():
    assert PACKET.is_file(), (
        "Current WINDOW/L14 dense qualification evidence is missing: "
        "retain the owner-derived coverage, real v3 reports and supporting "
        "receipts in docs/measurements/t4-dense-qualification.json; "
        "native availability and historical TCQ timings are not qualification"
    )
    return json.loads(PACKET.read_text())


def _receipt(packet, name):
    ref = packet["supporting_evidence"][name]
    path = Path(ref["path"])
    assert not path.is_absolute() and ".." not in path.parts, ref
    resolved = (ROOT / path).resolve()
    assert resolved.is_relative_to(ROOT) and resolved.is_file(), ref
    payload = resolved.read_bytes()
    assert payload and hashlib.sha256(payload).hexdigest() == ref["sha256"], ref


def _reports(packet, mode):
    reports = []
    for entry in packet["reports"]:
        report = entry["report"]
        if report["mode"] != mode:
            continue
        args = qualify.build_parser().parse_args(entry["argv"])
        qualify.validate_args(args)
        assert args.mode == mode and args.part == "dense"
        assert report["schema"] == qualify.SCHEMA
        expected = set(qualify.requested_keys(args))
        # validate_report otherwise trusts the report's own population list.
        assert expected == set(report["requested_population"])
        assert expected == {c["key"] for c in report["cells"]}
        assert len(report["cells"]) == len(expected), "Duplicate measured cells"
        # Coverage completion is not a claim that every measured rung wins.
        # The existing report validator also fails a measured performance loss.
        # Validate execution/resources with its correctness branch, then below
        # independently recompute timing comparability and gate admitted cells.
        validation_report = copy.deepcopy(report)
        validation_args = copy.copy(args)
        if mode == "timing":
            validation_args.mode = "correctness"
            validation_report["mode"] = "correctness"
            for cell in validation_report["cells"]:
                cell.pop("t8_comparison", None)
                cell.pop("comparison_error", None)
                cell.pop("competitive_status", None)
        ok, failures = qualify.validate_report(validation_report, validation_args)
        assert ok, failures
        reports.append(report)
    assert reports, f"No retained real {mode} reports"
    return reports


def test_current_wire_dense_export_load_execution_qualification():
    packet = _packet()
    for name in ("coverage_owner", "arithmetic_1213", "arithmetic_1214",
                 "arithmetic_1215", "arithmetic_1216", "direct_consumer_1241",
                 "current_wire_oracle", "activation_byte_capture",
                 "source_and_loaded_build", "actual_census"):
        _receipt(packet, name)
    cases = packet["coverage"]
    assert FAMILIES <= {c["family"] for c in cases}
    assert FAMILIES <= {c["family"] for c in cases if c["disposition"] == "measured"}, (
        "Blanket family refusals do not establish dense projection qualification")
    assert len({c["id"] for c in cases}) == len(cases)
    kda = next(c for c in cases if c["id"] == "kda_in")
    assert (kda["rows"], kda["cols"], kda["roles"]) == (12576, 4096, KDA_ROLES)
    assert kda["disposition"] == "measured", "Partial KDA shapes are not full-module proof"
    cells = [c for r in _reports(packet, "correctness") for c in r["cells"]]
    for case in cases:
        assert case["disposition"] in ("measured", "refused")
        if case["disposition"] == "refused":
            assert case["reason"].strip()
            _receipt(packet, case["evidence"])
            continue
        observed = [c for c in cells if c["shape_tag"] == case["id"]]
        assert case["tp_cases"] and len({tuple(tp) for tp in case["tp_cases"]}) == len(case["tp_cases"])
        assert all(size > 0 and 0 <= rank < size for size, rank in case["tp_cases"])
        expected = {(q, m, pattern, size, rank) for q in WIDTHS
                    for m in case["correctness_ms"] for pattern in ("onehot", "random")
                    for size, rank in case["tp_cases"]}
        assert expected <= {(c["q256"], c["m"], c["pattern"], c["tp_size"], c["tp_rank"])
                            for c in observed}, case
        assert 1 in case["correctness_ms"] and any(m > 1 for m in case["correctness_ms"]), case
        assert 1 in case["ms"], case
        if case["family"] != "head":
            assert BODY_MS <= set(case["ms"]), case
        else:
            assert case["timing_domain"] == "sampled_decode_positions", case
        assert sum(rows for _, rows in case["roles"]) == case["rows"], case
        for cell in observed:
            assert cell["q256"] in WIDTHS
            assert cell["wire_bytes"] > 0
            for digest in (cell["wire_sha256"], cell["input_sha256"]):
                assert re.fullmatch(r"[0-9a-f]{64}", digest), cell["key"]
            assert cell["local_rows"] == case["rows"] and cell["local_cols"] == case["cols"]
            assert cell["recipe"] == {"body": "WINDOW", "window_bits": 14,
                                      "span": 1, "plane": "LUT", "grid": "E2M1x2"}
            assert cell["activation_contract"] == "e2m1_group16_ue4m3_static"
            start = 0
            assert len(cell["role_checks"]) == len(case["roles"])
            for check, (name, rows) in zip(cell["role_checks"], case["roles"]):
                assert (check["name"], check["rows"], check["output_start"]) == (name, rows, start)
                assert check["numeric"]["ok"] is True and check["graph_equal"] is True
                start += rows
    kda_cells = [c for c in cells if c["shape_tag"] == "kda_in"]
    assert {(q, rank) for q in WIDTHS for rank in (0, 1)} <= {
        (c["q256"], c["tp_rank"]) for c in kda_cells if c["tp_size"] == 2}


def test_current_wire_dense_d41_timing_and_supporting_evidence():
    packet = _packet()
    for name in ("d41_protocol", "profiles", "power_device_0", "power_device_1",
                 "work_j", "admission_disposition"):
        _receipt(packet, name)
    reports = _reports(packet, "timing")
    cells = [c for report in reports for c in report["cells"]]
    for case in packet["coverage"]:
        if case["disposition"] == "refused":
            continue
        assert {(q, m, size, rank) for q in WIDTHS for m in case["ms"]
                for size, rank in case["tp_cases"]} <= {
            (c["q256"], c["m"], c["tp_size"], c["tp_rank"])
            for c in cells if c["shape_tag"] == case["id"]}, case
    for cell in cells:
        for execution in ("eager", "graph"):
            samples = cell["timing"][execution]["samples_ms"]
            assert samples and all(math.isfinite(x) and x > 0 for x in samples)
        # Recompute comparability; do not trust a recorded pass_kill or label.
        matches = []
        for baseline in packet["baselines"]:
            candidate = dict(cell)
            candidate.pop("comparison_error", None)
            qualify.attach_t8_comparison(candidate, baseline)
            if "comparison_error" not in candidate:
                matches.append(candidate)
        assert len(matches) == 1, cell["key"]
        compared = matches[0]
        assert compared["competitive_status"] in (
            "measured", "unattainable_scalar_floor", "no_exact_match_found")
        assert type(cell.get("admitted")) is bool, (
            "Every measured cell needs an explicit admission disposition", cell["key"])
        if cell["admitted"]:
            assert compared["competitive_status"] == "measured"
            assert all(v["pass_kill"] for v in compared["t8_comparison"].values())
