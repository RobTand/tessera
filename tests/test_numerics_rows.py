"""Draft-7 outcomes on independent CPU values, scale reads, and boundary events."""
from fractions import Fraction as F
import itertools
import math

import pytest
import torch

from experiments.numerics_rows.common import BoundaryError, BoundaryRecorder
from experiments.numerics_rows.reference import bf16, exact_input, output, rn32, selftest
from experiments.numerics_rows.row import (
    Invalid, check_preconditions, compare_output, coverage, dispatch, new_unit,
    opaque_apply, outcome, record_cuda_error, require_x,
)
from experiments.numerics_rows.scale_index import digits, evaluate, joint_coverage, passes


def unit():
    result = new_unit({"decoder": "native_fused_window_dense", "arm": {}})
    result["completed"] = True
    return result


def test_equal_values_and_signed_zeros():
    scope = unit()
    result = compare_output(scope, "zero", torch.tensor([0.0, -0.0]).bfloat16(), torch.tensor([-0.0, 0.0]).bfloat16())
    assert result["equal"] and result["signed_zero_diffs"] == 2
    assert outcome(scope) == "PASS"


@pytest.mark.parametrize("mutation", ("bit", "nan", "shape", "dtype"))
def test_changed_output_fails(mutation):
    reference = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    actual = reference.clone()
    if mutation == "bit":
        actual.view(torch.int16)[1] ^= 1
    elif mutation == "nan":
        actual[1] = float("nan")
    elif mutation == "shape":
        actual = actual.reshape(1, 2)
    else:
        actual = actual.float()
    scope = unit()
    compare_output(scope, mutation, actual, reference)
    assert outcome(scope) == "FAIL"


def test_missing_pair_fails_but_named_refusal_is_unattested():
    for reason, expected in ((None, "FAIL"), ("row shape is outside this pair", "UNATTESTED")):
        scope = unit()
        dispatch(scope, ("symbol", "decoder"), [], [{"name": "library", "sha256": "a"}], {"a"}, reason)
        assert outcome(scope) == expected


@pytest.mark.parametrize("changed", ("symbol", "binary"))
def test_wrong_dispatch_identity_fails(changed):
    scope = unit()
    dispatch(scope, ("symbol", "decoder"), [("wrong" if changed == "symbol" else "symbol", "decoder")],
             [{"name": "library", "sha256": "b" if changed == "binary" else "a"}], {"a"})
    assert outcome(scope) == "FAIL"


@pytest.mark.parametrize("products", ([F(1, 2**133)], [F(1, 2**126), -F(129, 128 * 2**126)], [F(2**24), F(1)]))
def test_x_false_is_invalid_before_native(products):
    assert not exact_input(products)["holds"]
    with pytest.raises(Invalid, match="X false"):
        require_x(torch.tensor([[exact_input(products)["holds"]]]), sweep=True)


def test_subnormal_operand_is_exact_and_flushing_fails():
    product = F(1, 2**133) * 2**64
    assert exact_input([product])["holds"]
    reference = torch.tensor([output([product], [1])], dtype=torch.bfloat16)
    assert float(reference[0]) == 2**-69
    scope = unit()
    compare_output(scope, "flushed operand", torch.zeros_like(reference), reference)
    assert outcome(scope) == "FAIL"


@pytest.mark.parametrize("claimed,expected", ((False, "UNATTESTED"), (True, "INVALID")))
@pytest.mark.parametrize("section", ("separation", "scale-index"))
def test_incomplete_coverage(section, claimed, expected):
    scope = unit()
    coverage(scope, section, False, claimed)
    assert outcome(scope) == expected


def test_wrong_scale_index_fails():
    coding = passes(4, 0, 1)
    check = joint_coverage([(2,)], coding["coding"], lambda read, wrong, values: values[wrong[0]], 4)
    assert check == {"reads": 1, "reads_covered": 1, "uncovered_reads": []}
    scope = unit()
    for entry in coding["coding"]:
        compare_output(scope, "wrong index", torch.tensor([entry["values"][3]]).bfloat16(), torch.tensor([entry["values"][2]]).bfloat16())
    assert outcome(scope) == "FAIL"


class Table:
    limit = 10.0
    def __call__(self, value):
        # CPU fixture values. P3's GPU table is a separate integration input.
        return rn32(value / (1 + math.exp(-value)))


def mode0_case(g=1.0, u=1.0):
    return {"read": [0, 1], "g": g, "u": u, "a_scale": 1.0, "g_scale": 1.0, "u_scale": 1.0, "scale": "row"}


def test_clamp_hides_one_pass_but_later_pass_finds_wrong_index():
    case, table = mode0_case(g=8), Table()
    coding = [{"values": [2.0, 4.0]}, {"values": [0.25, 0.5]}]
    assert evaluate(case, (0, 1), coding[0]["values"], table) == evaluate(case, (1, 1), coding[0]["values"], table)
    assert evaluate(case, (0, 1), coding[1]["values"], table) != evaluate(case, (1, 1), coding[1]["values"], table)
    scope = unit()
    compare_output(scope, "second digit", torch.tensor([evaluate(case, (1, 1), coding[1]["values"], table)]).bfloat16(),
                   torch.tensor([evaluate(case, (0, 1), coding[1]["values"], table)]).bfloat16())
    assert outcome(scope) == "FAIL"


def test_joint_cancelled_indices_are_incomplete():
    case, table = mode0_case(g=2**-12), Table()
    coding = [{"values": [1.0, 2.0, 0.5]}]
    own = evaluate(case, (0, 0), coding[0]["values"], table)
    wrong = evaluate(case, (1, 2), coding[0]["values"], table)
    assert own == wrong == 2**-13
    result = joint_coverage([(0, 0)], coding, lambda read, wrong, values: evaluate(case, wrong, values, table), 3)
    assert result["reads_covered"] == 0
    scope = unit()
    coverage(scope, "scale-index", False, False)
    assert outcome(scope) == "UNATTESTED"


def test_joint_check_matches_independent_exhaustive_population():
    table, case = Table(), mode0_case()
    plan = passes(6, -2, 1, gate=range(3), up=range(3, 6), gate_constant=16, up_constant=16)
    assert plan["families"] == ["gate", "up"]
    assert plan["passes_per_family"] == 2 and plan["passes_total"] == 4
    reads = list(itertools.product(range(6), repeat=2))
    expected = []
    for own in reads:
        signature = [evaluate(case, own, p["values"], table) for p in plan["coding"]]
        aliases = [wrong for wrong in reads if wrong != own and signature == [evaluate(case, wrong, p["values"], table) for p in plan["coding"]]]
        if not aliases:
            expected.append(own)
    actual = joint_coverage(reads, plan["coding"], lambda read, wrong, values: evaluate(case, wrong, values, table), 6)
    assert actual["reads_covered"] == len(expected)
    assert {tuple(r["read"]) for r in actual["uncovered_reads"]} == set(reads) - set(expected)


def test_digit_order_and_one_element_array():
    assert digits(5, 2) == [[0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]
    assert digits(1, 2) == []
    assert joint_coverage([(0,)], [], lambda *_: 1, 1)["reads_covered"] == 1


def test_p4_difference_has_fail_precedence():
    scope = unit()
    assert not check_preconditions(scope, scale_equal=False, routed=True, slice_matched=False)
    assert outcome(scope) == "FAIL"


@pytest.mark.parametrize("slice_matched,monotonic", ((False, True), (True, False)))
def test_swiglu_proof_failure_stops_only_routed_attestation(slice_matched, monotonic):
    scope = unit()
    assert not check_preconditions(scope, scale_equal=True, routed=True, slice_matched=slice_matched, monotonic=monotonic)
    assert outcome(scope) == "UNATTESTED"
    dense = unit()
    assert check_preconditions(dense, scale_equal=True, slice_matched=slice_matched, monotonic=monotonic)
    assert outcome(dense) == "PASS"


def test_stopped_job_and_fail_precedence():
    scope = unit()
    scope["completed"] = False
    assert outcome(scope) == "STOPPED"
    check_preconditions(scope, scale_equal=False)
    assert outcome(scope) == "FAIL"


@pytest.mark.parametrize("where,expected", (("before", "STOPPED"), ("after", "FAIL"), ("apply", "STOPPED")))
def test_cuda_error_attribution(where, expected):
    calls = 0
    def synchronize():
        nonlocal calls
        calls += 1
        if calls == (1 if where == "before" else 2):
            raise RuntimeError("injected CUDA error")
    scope = unit()
    recorder = BoundaryRecorder(synchronize, scope["boundaries"])
    with pytest.raises(BoundaryError) as caught:
        recorder.call("native", "apply" if where == "apply" else "tessera", lambda: None)
    record_cuda_error(scope, caught.value)
    assert outcome(scope) == expected
    assert scope["boundaries"][0]["sync_before"] is (where != "before")


def test_error_inside_apply_is_unattributed_until_replay():
    scope = unit()
    calls = iter((None, RuntimeError("asynchronous failure")))
    def synchronize():
        value = next(calls)
        if value:
            raise value
    recorder = BoundaryRecorder(synchronize, scope["boundaries"])
    with pytest.raises(BoundaryError) as caught:
        opaque_apply(lambda: recorder.call("inner", "tessera", lambda: None), recorder)
    assert caught.value.attributed_to is None
    assert recorder.attribute
    record_cuda_error(scope, caught.value)
    assert outcome(scope) == "STOPPED"


def test_rn_ties_and_signed_zero():
    assert rn32(F(1) + F(1, 2**24)) == 1
    assert rn32(F(1) + F(3, 2**24)) == 1 + 2**-22
    assert bf16(1 + 2**-8) == 1
    assert math.copysign(1, bf16(-0.0)) == -1
    result = selftest(samples=100)
    assert result["samples"] == 100 and result["ties"] == 50
    assert result["subnormal_ties"] == result["bf16_subnormal_ties"] == 25
    assert result["mismatches"] == 0 and not any(result["backends"].values())


def test_compact_passes_keep_the_same_joint_coverage():
    from experiments.numerics_rows.scale_index import scale_values
    arguments = {"gate": range(4), "up": range(4, 8), "gate_constant": 16, "up_constant": 32}
    full = passes(8, -2, 1, **arguments)
    compact = passes(8, -2, 1, compact=True, **arguments)
    assert all("values" not in entry for entry in compact["coding"])
    assert [list(scale_values(entry)) for entry in compact["coding"]] == [entry["values"] for entry in full["coding"]]


def test_generator_reports_a_weight_without_separation():
    import numpy as np
    from experiments.numerics_rows.cases import column_coverage
    class Quantizer:
        gs, normal_j = 1.0, 0
        def __call__(self, value):
            return value, 1.0
    weights = np.array([[1.0, 2.0], [1.5, 3.0]])
    scales = np.ones(2, dtype=np.float32)
    direct = column_coverage(weights, scales, "e4m3", Quantizer())
    assert direct["complete"] and direct["both"] == 4
    hidden = column_coverage(weights, scales, "e4m3", Quantizer(), peer=np.zeros_like(weights),
                             peer_scales=scales, table=Table(), stage="up")
    assert not hidden["complete"] and hidden["both"] == 0
    scope = unit()
    coverage(scope, "separation", hidden["complete"], False)
    assert outcome(scope) == "UNATTESTED"


def test_routing_has_the_requested_count_and_distinct_experts():
    from experiments.numerics_rows.cases import routing
    ids, weights = routing(65, 24, 37, list(range(8)))
    assert all(len(row) == len(set(row)) == 8 for row in ids)
    assert all(sum(expert in row for row in ids) == 37 for expert in range(8))
    assert all(abs(sum(row) - 1) < 1e-14 for row in weights)
    assert (ids, weights) == routing(65, 24, 37, list(range(8)))


@pytest.mark.parametrize("changed", (None, "read", "array", "g", "u", "a_scale", "g_scale", "u_scale"))
def test_scale_case_is_bound_to_actual_operands(changed):
    from experiments.numerics_rows.scale_launch import validate_case
    case = {"read": [2, 6], "g": 6.0, "u": -3.0, "a_scale": 0.5, "g_scale": 4.0, "u_scale": 8.0}
    plan = {"scale": "row", "array": {"N": 8}}
    if changed == "read":
        case["read"] = [1, 6]
    elif changed == "array":
        plan["array"]["N"] = 7
    elif changed:
        case[changed] *= 2
    arguments = dict(aq=3.0, a_scale=0.5, weight=2.0, scale=4.0, peer=-1.0, peer_scale=8.0, n=8, read=[2, 6])
    if changed:
        with pytest.raises(Invalid, match="C1 scale case"):
            validate_case(case, plan, **arguments)
    else:
        validate_case(case, plan, **arguments)


def test_setup_failure_writes_stopped_receipt(tmp_path, monkeypatch):
    import json
    from experiments.numerics_rows.row import main
    argv = ["row", "--image", "G", "--name", "test"]
    for name in ("source", "manifest", "population", "build", "q1", "h1", "rule", "freeze"):
        argv += ["--" + name, str(tmp_path / name)]
    argv += ["--out", str(tmp_path / "results")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 3
    receipt = json.loads((tmp_path / "results/numerics_row.json").read_text())
    assert receipt["row"]["exit_status"] == 3 and receipt["row"]["stop_cause"]
    assert receipt["row"]["utc_end"] and receipt["units"] == []


def test_scale_coding_checks_incidental_outputs_before_launch():
    from types import SimpleNamespace
    from experiments.numerics_rows.scale_launch import finite_references
    weight = torch.tensor([[1.0], [torch.finfo(torch.bfloat16).max]], dtype=torch.float64)
    adapter = SimpleNamespace(family="value", gs=1.0, roles=[("dense", weight, torch.ones(2), None)])
    arguments = (adapter, {"name": "dense"}, {"input_row": 0, "column": 0}, "row")
    finite_references(*arguments, [1.0, 1.0], 1.0, 1.0, None, None, 2)
    with pytest.raises(Invalid, match="nonfinite incidental reference"):
        finite_references(*arguments, [1.0, 2.0], 1.0, 1.0, None, None, 2)


def test_c1_rejects_normal_products_with_subnormal_quantum():
    import numpy as np
    from experiments.numerics_rows.cases import column_coverage
    class Quantizer:
        gs, normal_j = 1.0, 0
        def __call__(self, value):
            return value, 1.0
    weight = np.array([[(129 / 128) * 2**-126]])
    result = column_coverage(weight, np.ones(1, dtype=np.float32), "value", Quantizer())
    assert not result["complete"] and result["x_false_candidates"] > 0
    assert result["recipes"][0]["values"] == []
