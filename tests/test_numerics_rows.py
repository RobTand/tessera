"""Draft-7 outcomes on independent CPU values, scale reads, and boundary events."""
from fractions import Fraction as F
import itertools
import math

import pytest
import torch

from experiments.numerics_rows.common import BoundaryError, BoundaryRecorder
from experiments.numerics_rows.reference import FiniteBound, bf16, exact_input, output, rn32, selftest
from experiments.numerics_rows.row import (
    Invalid, check_preconditions, compare_output, coverage, dispatch, new_unit,
    finite_outputs, opaque_apply, outcome, record_cuda_error, require_x, x_products,
)
from experiments.numerics_rows.scale_index import digits, evaluate, joint_coverage, passes


def unit():
    result = new_unit({"decoder": "native_fused_window_dense", "arm": {}, "role": "unit", "structure": "dense"})
    result["completed"] = True
    result["exact"]["sweep"]["elements"] = 1
    result["exact"]["separation"]["records"] = [{"both": 1}]
    result["exact"]["scale_index"]["arrays"] = [{"launches": 1}]
    result["exact"]["chunk_pairs"]["x_elements"] = 1
    result["exact"]["composition"]["cases"] = 1
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
        dispatch(scope, ("symbol", "decoder"), [], [{"name": "library", "sha256": "a"}], reason)
        assert outcome(scope) == expected


def test_wrong_dispatch_identity_fails():
    scope = unit()
    dispatch(scope, ("symbol", "decoder"), [("wrong", "decoder")], [{"name": "library", "sha256": "a"}])
    assert outcome(scope) == "FAIL"


@pytest.mark.parametrize("products", ([F(1, 2**133)], [F(1, 2**126), -F(129, 128 * 2**126)], [F(2**24), F(1)]))
def test_x_false_is_invalid_before_native(products):
    assert not exact_input(products)["holds"]
    mask = x_products(torch.tensor([[float(value) for value in products]], dtype=torch.float64))
    assert not bool(mask[0])
    with pytest.raises(Invalid, match="X false"):
        require_x(mask.reshape(1, 1), sweep=True)


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


def test_completed_without_any_cases_is_stopped():
    scope = new_unit({"role": "unit", "structure": "dense"})
    scope["completed"] = True
    assert outcome(scope) == "STOPPED"


@pytest.mark.parametrize("section", ("sweep", "separation", "scale_index", "chunk_pairs", "composition", "swiglu", "down_input", "token_sum"))
def test_pass_requires_each_required_case_family(section):
    scope = unit()
    scope["structure"] = "routed_moe"
    for name in ("swiglu", "down_input", "token_sum"):
        scope["exact"][name]["elements"] = 1
    assert outcome(scope) == "PASS"
    if section in ("separation", "scale_index"):
        scope["exact"][section]["records" if section == "separation" else "arrays"].clear()
    else:
        scope["exact"][section] = {key: 0 for key in scope["exact"][section]}
    assert outcome(scope) == "STOPPED"


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


@pytest.mark.parametrize("family", ("e4m3", "e2m1"))
@pytest.mark.parametrize("stage", ("dense", "gate"))
def test_scale_cases_need_only_the_first_coordinate_per_group(monkeypatch, family, stage):
    import numpy as np
    from experiments.numerics_rows.cases import scale_index_manifest
    class Quantizer:
        gs, normal_j = 1.0, 0
        def __call__(self, value):
            return value, 1.0
        raw = staticmethod(float)
    weight = torch.zeros(2, 32, dtype=torch.float64)
    weight[0, 3], weight[1, 5], weight[0, 18] = 1, 2, 4
    peer = torch.zeros_like(weight)
    peer[1, 5], peer[0, 18] = 3, 5
    def full_coordinates(_):
        pytest.fail("C1 allocates all nonzero coordinates for each input row")
    monkeypatch.setattr(np, "argwhere", full_coordinates)
    gate = stage == "gate"
    plans = scale_index_manifest(weight, torch.ones(2), family, stage, Table() if gate else None,
        (peer, torch.ones(2)) if gate else None, torch.ones(2) if gate else None,
        Quantizer(), 1, 0, [{"M": 1}, {"M": 8}])
    target = "sfa" if family == "e2m1" else "a_s"
    expected = ([(1, 5), (0, 18)] if gate else [(0, 3), (0, 18)]) if target == "sfa" else [(0, 18) if gate else (0, 3)]
    for plan in (p for p in plans if p["scale"] == target):
        assert [(c["output_row"], c["column"]) for c in plan["read_cases"]] == expected * plan["array"]["M"]


@pytest.mark.parametrize("stage", ("down", "gate"))
def test_routed_row_plan_metadata_keeps_its_own_coding_domain(monkeypatch, stage):
    from experiments.numerics_rows.cases import scale_index_manifest
    from experiments.numerics_rows import scale_index
    row_plan = scale_index.make_plan
    monkeypatch.setattr(scale_index, "make_plan", lambda scale, *args, **kw:
        row_plan(scale, *args, **kw) if scale == "row" else {"scale": scale})
    class Quantizer:
        gs, normal_j = 1.0, 0
        def __call__(self, value):
            return value, 1.0
    weight, scale = torch.ones(2, 16), torch.ones(2)
    gate = stage == "gate"
    configs = [{"M": m} for m in (1, 2, 8, 32, 64, 65, 128, 512, 2048)]
    plans = scale_index_manifest(weight, scale, "e4m3", stage, Table() if gate else None,
        (weight, scale) if gate else None, scale if gate else None,
        Quantizer(), 16, 0, configs)
    for config in configs:
        selected = [p for p in plans if p["scale"] == "row" and p["array"].get("M", config["M"]) == config["M"]]
        assert len(selected) == 1
        plan = selected[0]
        assert plan["array"]["target"] == "row"
        assert plan["array"]["N"] == 16 * weight.shape[0] * (2 if gate else 1)
        assert plan["coding"] and all(c["n"] == plan["array"]["N"] for c in plan["coding"])


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


def test_down_separation_uses_the_actual_router_weight_before_bf16():
    import numpy as np
    from experiments.numerics_rows.cases import column_coverage
    class Quantizer:
        gs, normal_j = 1.0, 0
        def __call__(self, value):
            return math.copysign(1.0, value), 1.0
    weight, scale = np.ones((1, 1)), np.ones(1, dtype=np.float32)
    assert bf16(float(np.float32(0.501))) == bf16(float(np.float32(255 / 256) * np.float32(0.501))) == 0.5
    direct = column_coverage(weight, scale, "value", Quantizer(), stage="down")
    weighted = column_coverage(weight, scale, "value", Quantizer(), stage="down", router_weight=0.501)
    assert direct["complete"] and direct["lower_excluded"] == 1
    assert not weighted["complete"] and weighted["lower_excluded"] == 0


@pytest.mark.parametrize("m,expert", ((1, 0), (64, 7), (2048, 16)))
def test_separation_route_binds_an_actual_census_weight(m, expert):
    import numpy as np
    from experiments.numerics_rows.cases import routing, separation_route
    route = separation_route(m, 24, expert)
    ids, weights = routing(m, 24, 1, route["selected"])
    row, slot = divmod(route["flat_index"], 8)
    assert ids[row][slot] == expert
    assert route["weight"] == float(np.float32(weights[row][slot]))


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
    for name in ("source", "manifest", "population", "build", "build-provenance", "q1", "h1", "rule", "freeze"):
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
    adapter = SimpleNamespace(family="value", gs=1.0, splits=1, roles=[("dense", weight, torch.ones(2), None)])
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


@pytest.mark.parametrize("kind", ("exact", "diagnostic", "scale-index", "staged"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_bounded_nonfinite_outputs_fail_before_x_or_scale_selection(kind, value):
    scope = unit()
    got = torch.tensor([[1.0, value]], dtype=torch.bfloat16)
    reference = torch.ones_like(got)
    mask = torch.tensor([[True, False]])
    bound = FiniteBound(torch.tensor([[1.0, 2.0**125]], dtype=torch.float64))
    if kind == "exact":
        result = compare_output(scope, kind, got, reference, mask, bound=bound)
        assert result["equal"] and result["elements"] == 1
    else:
        finite_outputs(scope, kind, got, bound)
    assert outcome(scope) == "FAIL"
    assert scope["exact"]["finite_certificate"] == {"elements": 2, "held": 2, "nonfinite_where_held": 1}


def test_universal_finiteness_uses_the_strict_bound():
    scope = unit()
    got = torch.full((2,), float("inf"), dtype=torch.bfloat16)
    finite_outputs(scope, "outside", got, FiniteBound(torch.tensor([2.0**126, 2.0**127], dtype=torch.float64)))
    assert not scope["findings"]
    finite_outputs(scope, "inside", got, FiniteBound(torch.tensor([2.0**125, 2.0**126], dtype=torch.float64)))
    assert outcome(scope) == "FAIL" and "1 nonfinite outputs" in scope["findings"][0]


def test_sparse_dot_bound_keeps_cancellation_and_all_epilogue_scales():
    from experiments.numerics_rows.row import dot_reference
    a = torch.tensor([[2.0, -2.0]], dtype=torch.float64)
    w = torch.tensor([[3.0, 3.0]], dtype=torch.float64)
    ref, mask, bound = dot_reference(a, w, torch.tensor([0.5]), torch.tensor([4.0]), with_bounds=True)
    assert ref.item() == 0 and bool(mask.item())
    assert bound.held.item() and 24 < bound.value.item() < 25


def test_q1_checks_both_sign_facts_used_by_gate_separation():
    import numpy as np
    from experiments.numerics_rows.quantizers import d_facts
    gf = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    d = np.array([-0.25, -0.125, 0, 0.5, 1], dtype=np.float32)
    assert all(d_facts(gf, d).values())
    d[0] = 0.25
    assert not d_facts(gf, d)["d_nonpositive_negative_gf"]
    d[3] = -0.5
    assert not d_facts(gf, d)["nonzero_act_has_uf_sign_positive_gf"]


@pytest.mark.parametrize("wrong", (False, True))
def test_scale_launch_checks_the_loaded_read_on_cpu(monkeypatch, wrong):
    from types import SimpleNamespace
    from experiments.numerics_rows import scale_launch
    from experiments.numerics_rows.scale_index import make_plan
    w = torch.ones(2, 1, dtype=torch.float64)
    adapter = SimpleNamespace(device="cpu", family="value", gs=1.0, cols=1, splits=1,
        roles=[("dense", w, torch.ones(2, dtype=torch.float32), None)])
    adapter.quantized = lambda x, recorder: (x, None, x.double(), None)
    cases = [{"read": [index], "g": 1.0, "u": 0.0, "a_scale": 1.0, "g_scale": 1.0, "u_scale": 1.0,
              "scale": "row", "column": 0, "input": 1.0, "output_row": index, "input_row": 0} for index in range(2)]
    plan = make_plan("row", 2, cases, metadata={"N": 2, "M": 1, "mode": 2})
    assert plan["reads_covered"] == plan["reads"] == 2
    def launch(adapter, role, q, sf, target, values, recorder):
        # Controlled native boundary: compute the output with the selected scale indices.
        indices = [0, 0] if wrong else [0, 1]
        scales = torch.tensor([values[index] for index in indices], dtype=torch.float32)
        return ((q.double() @ w.T).float() * scales).bfloat16()
    monkeypatch.setattr(scale_launch, "dense_launch", launch)
    scope = unit()
    scope["m_config"] = {"M": 1}
    scale_launch.check(adapter, scope, {"structure": "dense", "roles": [{"name": "dense", "expert": 0, "scale_index": [plan]}]},
        BoundaryRecorder(lambda: None, scope["boundaries"]))
    result = scope["exact"]["scale_index"]["arrays"][-1]
    assert result["launches"] > 0
    assert outcome(scope) == ("FAIL" if wrong else "PASS")
    assert (result["mismatches"] > 0) is wrong


def test_p4_checks_the_loaded_dense_scale_before_the_verdict():
    from types import SimpleNamespace
    from experiments.numerics_rows.row import Dense, scale_receipt
    adapter = Dense.__new__(Dense)
    adapter.family = "value"
    reader = torch.tensor([1.0, 2.0], dtype=torch.float32)
    loaded = reader.clone()
    adapter.roles = [("dense", torch.ones(2, 1), reader, None)]
    adapter.layer = SimpleNamespace(tessera_native=SimpleNamespace(role_bundles=[SimpleNamespace(scale=loaded)]))
    positive = scale_receipt(adapter, False)
    assert positive["equal"]
    loaded.view(torch.int32)[1] ^= 1
    negative = scale_receipt(adapter, False)
    assert not negative["equal"] and negative["c_loaded_sha256"] != positive["c_loaded_sha256"]
    scope = unit()
    check_preconditions(scope, scale_equal=negative["equal"])
    assert outcome(scope) == "FAIL"


def test_ordered_scale_overflow_has_no_certificate_and_is_invalid_before_launch():
    from experiments.numerics_rows.row import dot_reference, finite_reference
    a, w = torch.tensor([[448.0]]), torch.tensor([[448.0]])
    ref, mask, bound = dot_reference(a, w, torch.tensor([2.0**118]), torch.tensor([2.0**-118]), with_bounds=True)
    assert mask.item() and not bound.held.item() and torch.isinf(ref).item()
    scope = unit()
    finite_outputs(scope, "no certificate", ref, bound)
    assert scope["exact"]["finite_certificate"] == {"elements": 1, "held": 0, "nonfinite_where_held": 0}
    assert not scope["findings"]
    with pytest.raises(Invalid, match="before the native call") as error:
        finite_reference(ref)
    scope["invalid"].append(str(error.value))
    assert outcome(scope) == "INVALID"


def test_certificate_exact_dot_sum_is_rounded_outward():
    from experiments.numerics_rows.reference import FiniteBound
    exact = F(2**60) + 1
    bound = FiniteBound.exact_sum(torch.tensor([[2.0**60, 1.0]], dtype=torch.float64), -1).item()
    assert F(math.nextafter(bound, -math.inf)) < exact <= F(bound)


def fraction_sum_oracle(values, axis):
    import numpy as np
    from experiments.numerics_rows.reference import outward
    array = values.detach().double().cpu().numpy()
    def total(row):
        return outward(sum((F(float(x)) for x in row), F(0))) if np.isfinite(row).all() else math.inf
    return torch.as_tensor(np.apply_along_axis(total, axis, array), device=values.device)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_certificate_exact_sum_matches_fraction_bytes(dtype):
    import numpy as np
    rng = np.random.default_rng(1717)
    bits = rng.integers(0, 2**64, size=(128, 32), dtype=np.uint64)
    random_rows = bits.view(np.float64)
    largest, smallest = np.finfo(np.float64).max, np.nextafter(0.0, 1.0)
    edges = [[-0.0], [smallest, -smallest], [smallest, smallest, -smallest],
             [largest, -largest, smallest], [largest, -largest, -smallest],
             [1.0, -math.nextafter(1.0, 0.0), smallest], [2.0**1023, -(2.0**1023), 1.0],
             [largest, largest], [-largest, -largest], [math.inf], [-math.inf], [math.nan],
             [math.inf, -math.inf], [smallest, np.finfo(np.float64).tiny]]
    tiny32 = float(np.nextafter(np.float32(0), np.float32(1)))
    edges += [[tiny32, -tiny32], [tiny32, float(np.finfo(np.float32).tiny)],
              [float(np.finfo(np.float32).max), -float(np.finfo(np.float32).max), tiny32]]
    rows = [np.pad(row, (0, 32-len(row))) for row in edges] + list(random_rows)
    for row in rows:
        values = torch.from_numpy(row.copy()).to(dtype).reshape(1, -1)
        for axis in (0, -1):
            try:
                expected = fraction_sum_oracle(values, axis)
            except OverflowError:
                with pytest.raises(OverflowError):
                    FiniteBound.exact_sum(values, axis)
                continue
            actual = FiniteBound.exact_sum(values, axis)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.contiguous().view(torch.uint8).numpy().tobytes() == expected.contiguous().view(torch.uint8).numpy().tobytes()
    values = torch.tensor([[[1.0, -0.0], [2.0**60, 1.0]], [[-1.0, 0.0], [-2.0**60, -1.0]]], dtype=dtype)
    for axis in (0, 1, -1):
        actual, expected = FiniteBound.exact_sum(values.transpose(0, 1), axis), fraction_sum_oracle(values.transpose(0, 1), axis)
        assert actual.contiguous().view(torch.uint8).numpy().tobytes() == expected.contiguous().view(torch.uint8).numpy().tobytes()
    for row in rows:
        values = torch.from_numpy(row.copy()).to(dtype)
        try:
            expected = fraction_sum_oracle(values, 0)
        except OverflowError:
            with pytest.raises(OverflowError):
                FiniteBound.exact_sum(values, 0)
            continue
        actual = FiniteBound.exact_sum(values, 0)
        assert actual.shape == expected.shape == torch.Size([])
        assert actual.numpy().tobytes() == expected.numpy().tobytes()


@pytest.mark.parametrize("k,splits,top_k,held", [(1, 1, 0, True), (2**20-2, 1, 1, False)])
def test_certificate_checks_manifest_operation_limit(k, splits, top_k, held):
    bound = FiniteBound.dot(torch.tensor([[1.0]], dtype=torch.float64), k, splits, top_k)
    assert bound.held.item() is held


def test_certificate_keeps_failed_ancestors_and_bf16_underflow_error():
    failed = FiniteBound(torch.tensor([2.0**126], dtype=torch.float64)).multiply(0)
    assert not failed.held.item()
    tiny = FiniteBound(torch.tensor([2.0**-134 * (1 + 2**-23)], dtype=torch.float64)).rounded(bf16=True)
    assert tiny.value.item() >= 2.0**-133
    routes = FiniteBound(torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    total = routes.token_sum(2)
    assert total.held.item() and total.value.item() > 3


@pytest.mark.parametrize("d,held", [([-1.0, 0.0, 1.0], True), ([-2.0, 0.0, 1.0], False), ([float("nan"), 0.0, 1.0], False)])
def test_measured_d_table_controls_the_mode_zero_certificate(d, held):
    import numpy as np
    from experiments.numerics_rows.quantizers import d_certificate
    record = d_certificate(np.array([-1, 0, 1], dtype=np.float32), np.array(d, dtype=np.float32))
    assert record["held"] is held
    gate = FiniteBound(torch.tensor([100.0], dtype=torch.float64))
    up = FiniteBound(torch.tensor([1000.0], dtype=torch.float64))
    act = gate.activation(up, 10.0, record["held"])
    assert act.held.item() is held and 1000 < act.value.item() < 1010


def test_routed_certificate_keeps_negative_gate_and_router_order():
    from types import SimpleNamespace
    from experiments.numerics_rows.routed import Routed
    class Table:
        finite_certificate = {"held": True}
        def __call__(self, value):
            return value
    adapter = Routed.__new__(Routed)
    adapter.rows = adapter.inter = adapter.splits = 1
    adapter.top_k, adapter.family, adapter.limit, adapter.table = 2, "value", 10.0, Table()
    def weight(expert, projection):
        return torch.tensor([[{"gate_proj": -100, "up_proj": 100, "down_proj": 2}[projection]]]), torch.tensor([3.0 if projection == "down_proj" else 1.0])
    adapter.weight = weight
    routes = SimpleNamespace(flat_sorted=torch.tensor([1, 0]), rw_sorted=torch.tensor([1.0, 2.0]))
    ids = torch.tensor([[0, 1]])
    act, _, gate, mode1 = adapter.gate_reference(torch.ones(1, 1), None, ids, routes)
    assert act.tolist() == [[-1000], [-1000]]
    assert bool(gate.held.all()) and bool((gate.value > 1000).all())
    assert mode1.value.shape == (1, 2, 2)
    down, _, bound = adapter.down_reference(torch.ones(2, 1), None, ids, None, routes)
    assert down.tolist() == [[12], [6]]
    assert bound.token_sum(2).held.item() and bound.token_sum(2).value.item() > 18


def test_routed_campaign_lists_every_expert_role_rank_once():
    from experiments.numerics_rows.c1_campaign import items

    population = {"modules": [{"name": f"module-{m}", "experts": 288, "structure": "routed_moe", "family": "TESSERA_FP8"} for m in range(42)]}
    listed = list(items(population))
    assert len(listed) == len({r["job_name"] for r in listed}) == 72576
    assert {r["module"] for r in listed[:1728]} == {"module-0"}
    assert listed[1728]["module"] == "module-1"
    for module in population["modules"]:
        actual = {(r["expert"], r["role"], r["tp"]["size"], r["tp"]["rank"]) for r in listed if r["module"] == module["name"]}
        assert actual == {(e, role, 2, rank) for e in range(288) for role in ("gate_proj", "up_proj", "down_proj") for rank in range(2)}


def test_routed_profile_cap_keeps_the_profile_and_partial_phases(tmp_path, monkeypatch):
    import json
    import signal
    import sys
    import experiments.numerics_rows.profile_c1 as profile_c1
    from experiments.numerics_rows import cases, scale_index

    for module, names in ((cases, ("checkpoint_routed", "decode_roles", "column_coverage", "routed_separation", "exact_chunks", "scale_index_manifest")),
                          (scale_index, ("scale_values", "evaluate", "make_plan"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    def cap_inside_phase(*args, **kwargs):
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    monkeypatch.setattr(cases, "column_coverage", cap_inside_phase)
    monkeypatch.setattr(cases, "main", lambda: cases.column_coverage())
    timers = []
    monkeypatch.setattr(signal, "setitimer", lambda *args: timers.append(args))
    monkeypatch.setattr(sys, "argv", ["profile_c1", "--out", str(tmp_path), "--seconds", "1", "--", "--sample-expert", "0"])
    profile_c1.main()
    record = json.loads((tmp_path / "profile.json").read_text())
    assert record["status"] == "CAPPED" and record["generation_finished"] is False
    assert record["phases"][0]["phase"] == "column_coverage" and record["phases"][0]["finished"] is False
    assert (tmp_path / "c1.pstats").stat().st_size and (tmp_path / "c1-profile.txt").stat().st_size
    assert timers == [(signal.ITIMER_REAL, 1), (signal.ITIMER_REAL, 0)]


@pytest.mark.parametrize("scale", ["row", "ratio", "a_s", "sfa", "router"])
@pytest.mark.parametrize("arity", [1, 2])
def test_cached_evaluator_matches_fraction_bits_and_reuses_scale_values(scale, arity):
    from experiments.numerics_rows.reference import f32_bits
    from experiments.numerics_rows.scale_index import CachedEvaluator
    table, codes = Table(), [0.0, -0.0, 0.5, 1.0, 2.0]
    cases = [{**mode0_case(g=2**-12), "read": [i] * arity, "scale": scale, "router_weight": -0.5} for i in range(2)]
    cache = CachedEvaluator(cases, table, maxsize=128)
    for case in cases:
        for wrong in itertools.product(range(len(codes)), repeat=arity):
            expected = evaluate(case, wrong, codes, table)
            assert f32_bits(cache(tuple(case["read"]), wrong, codes)) == f32_bits(expected)
    info = cache.cached.cache_info()
    assert info.misses == len(codes)**arity and info.hits == len(codes)**arity


def test_cached_evaluator_counts_exact_tuples_across_evictions_and_unused_labels():
    from experiments.numerics_rows.scale_index import CachedEvaluator
    cases = [{**mode0_case(), "read": [i, i], "g_scale": float(i + 1), "column": i} for i in range(2)]
    metrics = {}
    cache = CachedEvaluator(cases, Table(), coding=[{"values": [0.0, -0.0, 1.0]}], metrics=metrics, maxsize=1)
    for case in cases:
        for wrong in itertools.product(range(3), repeat=2):
            cache(case["read"], wrong, [0.0, -0.0, 1.0])
    report = cache.report()
    assert metrics["operand_signatures"] == 1 and metrics["completed_reads"] == 1
    assert report["distinct_evaluate_tuples"] == 9 and report["requests"] == report["misses"] == 18
    assert report["currsize"] == 1


@pytest.mark.parametrize("scale", ["row", "a_s", "router"])
def test_cached_evaluator_keeps_every_used_operand_and_the_table(scale):
    from experiments.numerics_rows.reference import f32_bits
    from experiments.numerics_rows.scale_index import CachedEvaluator
    cases = [mode0_case()]
    for field in ("g", "u", "a_scale", "g_scale", "u_scale"):
        cases.append({**mode0_case(), field: 2.0})
    cases = [{**case, "scale": scale, "read": [i, i]} for i, case in enumerate(cases)]
    class OtherTable:
        limit = 10.0
        def __call__(self, value):
            return value
    for table in (Table(), OtherTable()):
        cache = CachedEvaluator(cases, table)
        for case in cases:
            assert f32_bits(cache(case["read"], (0, 1), [0.5, 1.0])) == f32_bits(evaluate(case, (0, 1), [0.5, 1.0], table))


@pytest.mark.parametrize("mode0", [False, True])
def test_memoized_plan_keeps_every_case_byte(mode0):
    import json
    from experiments.numerics_rows.scale_index import make_plan
    cases = [{**mode0_case(), "read": [i, i + 4] if mode0 else [i]} for i in range(4)]
    arguments = {"table": Table(), "gate": range(4), "up": range(4, 8)} if mode0 else {}
    old = make_plan("row", 8, cases, **arguments)
    metrics = {}
    new = make_plan("row", 8, cases, **arguments, memoize=True, metrics=metrics)
    assert json.dumps(old, sort_keys=True, allow_nan=False).encode() == json.dumps(new, sort_keys=True, allow_nan=False).encode()
    assert metrics["completed_reads"] == new["reads"] == 4
    assert metrics["hits"] > 0 and metrics["distinct_evaluate_tuples"] <= metrics["requests"]


def test_profile_phase_records_the_final_completed_read_count(tmp_path, monkeypatch):
    import json
    import signal
    import sys
    import experiments.numerics_rows.profile_c1 as profile_c1
    from experiments.numerics_rows import cases, scale_index

    for module, names in ((cases, ("checkpoint_routed", "decode_roles", "column_coverage", "routed_separation", "exact_chunks", "scale_index_manifest")),
                          (scale_index, ("scale_values", "evaluate", "make_plan"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    read_cases = [{**mode0_case(), "read": [i]} for i in range(2)]
    def interrupted_plan(*args, **kwargs):
        for case in read_cases:
            scale_index.evaluate(case, (0,), [1.0])
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    monkeypatch.setattr(scale_index, "make_plan", interrupted_plan)
    monkeypatch.setattr(cases, "main", lambda: scale_index.make_plan("row", 2, read_cases))
    monkeypatch.setattr(signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["profile_c1", "--out", str(tmp_path), "--", "--sample-expert", "0"])
    profile_c1.main()
    record = json.loads((tmp_path / "profile.json").read_text())
    assert record["status"] == "CAPPED"
    assert record["plans"][0]["completed_reads"] == 1
    assert record["phases"][0]["completed_reads"] == 1
