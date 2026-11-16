"""Run index-coded copies of the loaded operands and check the joint read plan."""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from fractions import Fraction as F

from experiments.numerics_rows.reference import bf16, exact_input, multiply
from experiments.numerics_rows.scale_index import evaluate, joint_coverage, scale_values


def coded_native(native, target, values, mode):
    """Copy scale storage. Keep every weight, table, start state, and other scale."""
    import torch

    result = copy.copy(native)
    fields = ("gate", "up") if mode == 0 else ("down",)
    first = 0
    changed = {}
    for field in fields:
        if target == "ratio":
            tensor = getattr(native, "ratio_" + field)
            count = tensor.numel()
            changed["ratio_" + field] = torch.tensor(values[first:first + count], device=tensor.device, dtype=tensor.dtype).reshape_as(tensor)
        else:
            bundle = getattr(native, field)
            tensor = bundle.scale_all
            count = tensor.numel()
            scale = torch.tensor(values[first:first + count], device=tensor.device, dtype=tensor.dtype).reshape_as(tensor)
            changed[field] = replace(bundle, scale_all=scale)
        first += count
    if first != len(values):
        raise ValueError("coded array does not match the loaded scale array")
    if target == "row":
        changed["uniform"] = replace(native.uniform, **{field: replace(getattr(native.uniform, field), bundle=changed[field]) for field in fields})
    return replace(native, **changed)


def dense_launch(adapter, role, q, sf, target, values, recorder):
    import torch
    from tessera import routed_fused as rf, routed_fused_e2m1 as fe

    index = [r[0] for r in adapter.roles].index(role["name"])
    out = torch.empty(q.shape[0], adapter.roles[index][1].shape[0], device=q.device, dtype=torch.bfloat16)
    if adapter.family == "e2m1":
        native = adapter.layer.tessera_a4_roles[index]
        if target == "ratio":
            native = replace(native, ratio=torch.tensor(values, device=q.device, dtype=torch.float32).reshape_as(native.ratio))
        fe.dense_forward_quantized(native, q, sf, out=out)
    else:
        bundle = adapter.layer.tessera_native.role_bundles[index]
        if target == "row":
            bundle = replace(bundle, scale=torch.tensor(values, device=q.device, dtype=torch.float32).reshape_as(bundle.scale))
        if adapter.library == "triton":
            recorder.call("scale-index:triton", "tessera", lambda: bundle(q, sf if adapter.family == "e4m3" else None, out=out))
        else:
            rf.dense_forward(rf.prepare_dense_role(bundle), q, sf if adapter.family == "e4m3" else None, out, torch.zeros(1, device=q.device, dtype=torch.int32))
    return out


def check(adapter, unit, manifest, recorder):
    import torch
    from experiments.numerics_rows.cases import routing
    from experiments.numerics_rows.row import Invalid, compare_output, coverage

    routed = manifest["structure"] == "routed_moe"
    m = unit["m_config"]["M"]
    for role in manifest["roles"]:
        for plan in role["scale_index"]:
            meta = plan["array"]
            if meta.get("M", m) != m:
                continue
            cases = {tuple(c["read"]): c for c in plan["read_cases"]}
            table = adapter.table if routed else None
            # Recompute coverage from the outputs, not C1's coverage count.
            verified = joint_coverage(list(cases), plan["coding"], lambda read, wrong, values: evaluate(cases[read], wrong, values, table), meta["N"])
            complete = verified["reads"] == verified["reads_covered"]
            coverage(unit, "scale-index", complete, plan["reads"] == plan["reads_covered"])
            record = {k: plan[k] for k in ("scale", "base", "families", "passes_per_family", "passes_total")}
            record.update(verified, mismatches=0)
            unit["exact"]["scale_index"].append(record)
            if not complete:
                continue
            mode = meta["mode"]
            down = routed and mode == 2
            if routed:
                raw_ids, raw_weights = routing(m, adapter.experts, m, meta.get("route_experts", role["expert"]))
                ids, weights = torch.tensor(raw_ids, device="cuda"), torch.tensor(raw_weights, device="cuda", dtype=torch.float32)
                routes = adapter.route(ids, weights)
            for case in cases.values():
                rows = routes.routes if down else m
                cols = adapter.inter if down else adapter.cols
                x = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
                x[case["input_row"], case["column"]] = case["input"]
                if routed:
                    q, sf, a, a_scale = adapter.quantized(x, adapter.gs2 if down else adapter.gs13, recorder)
                    w, _ = adapter.weight(role["expert"], role["name"])
                else:
                    q, sf, a, a_scale = adapter.quantized(x, recorder)
                    w = next(w for name, w, _, _ in adapter.roles if name == role["name"])
                aq = float(a[case["input_row"], case["column"]])
                if plan["scale"] == "sfa":
                    byte = int(q.view(torch.uint8)[case["input_row"], case["column"] // 2])
                    code = (byte >> (4 * (case["column"] % 2))) & 15
                    from experiments.routed_pair_oracle import e2m1_table
                    aq = float(e2m1_table("cpu")[code])
                product = F(aq) * F(float(w[case["output_row"], case["column"]]))
                if multiply(aq, float(w[case["output_row"], case["column"]])) != case["g"]:
                    raise Invalid("C1 scale case does not match the actual quantizer and reader")
                if routed and mode == 0:
                    peer, _ = adapter.weight(role["expert"], "up_proj")
                    if multiply(aq, float(peer[case["output_row"], case["column"]])) != case["u"]:
                        raise Invalid("C1 scale case has the wrong up dot")
                for coding in plan["coding"]:
                    if not exact_input([product * (F(scale_values(coding)[case["read"][0]]) if plan["scale"] == "sfa" else 1)])["holds"]:
                        raise Invalid("X false in a claimed scale-index case")
                    qq, ss = q, None if sf is None else sf.clone()
                    vals = scale_values(coding)
                    expected = evaluate(case, case["read"], vals, table)
                    if not math.isfinite(expected):
                        raise Invalid("scale-index reference is not finite")
                    copied = adapter
                    rr = routes if routed else None
                    if plan["scale"] == "a_s":
                        ss = torch.tensor(list(vals), device=q.device, dtype=torch.float32)
                    elif plan["scale"] == "sfa":
                        logical = torch.tensor(list(vals), device=q.device, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, cols // 16)
                        ss.view(torch.uint8).reshape(rows, -1)[:, :cols // 16] = logical
                    elif plan["scale"] == "router":
                        rr = replace(routes, rw_sorted=torch.tensor(list(vals), device=q.device, dtype=torch.float32))
                    elif routed:
                        copied = copy.copy(adapter)
                        copied.native = coded_native(adapter.native, plan["scale"], vals, mode)
                    if routed:
                        out = torch.empty(rr.routes, adapter.rows if down else adapter.inter, device=q.device, dtype=torch.bfloat16)
                        copied.launch(mode, qq, ss, rr, out, row_mode=int(down), weighted=down)
                        if down:
                            output_row = int(routes.flat_sorted[case["input_row"]])
                        else:
                            eligible = ((ids.flatten()[routes.flat_sorted.long()] == role["expert"]) &
                                        (routes.flat_sorted.long() // ids.shape[1] == case["input_row"]))
                            positions = eligible.nonzero().flatten()
                            if not positions.numel():
                                raise Invalid("C1 scale read has no matching route")
                            output_row = int(positions[0])
                    else:
                        out = dense_launch(adapter, role, qq, ss, plan["scale"], vals, recorder)
                        output_row = case["input_row"]
                    ref = torch.tensor([expected], device=q.device, dtype=torch.bfloat16)
                    got = out[output_row, case["output_row"]].reshape(1)
                    result = compare_output(unit, f"scale-index:{role['name']}:{coding['family']}:{coding['digit']}:{case['read']}", got, ref, operands={"input": case["input"], "column": case["column"], "read": case["read"],
                        "scales": [vals[i] for i in case["read"]], "g": case["g"], "u": case["u"]})
                    record["mismatches"] += result["mismatches"]
