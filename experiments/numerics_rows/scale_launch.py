"""Run index-coded copies of the loaded operands and check the joint read plan."""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from fractions import Fraction as F

from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.reference import exact_input, f32_bits, multiply
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


def validate_case(case, plan, *, aq, a_scale, weight, scale, peer=None, peer_scale=None, router_weight=1.0, n, read):
    """Bind a C1 case to the actual read tuple and represented launch operands."""
    if plan["array"]["N"] != n or case["read"] != read:
        raise Invalid("C1 scale case has the wrong read tuple or array size")
    checks = {"g": multiply(aq, weight), "g_scale": scale}
    if peer is not None:
        checks.update(u=multiply(aq, peer), u_scale=peer_scale)
    if plan["scale"] not in ("a_s", "sfa"):
        checks["a_scale"] = a_scale
    if len(read) == 1:
        checks["router_weight"] = router_weight
    if any(f32_bits(case.get(name, 1.0)) != f32_bits(value) for name, value in checks.items()):
        raise Invalid("C1 scale case differs from the actual quantizer, reader, or router scale")


def finite_references(adapter, role, case, target, values, aq, a_scale, ids, routes, mode, *, rows=None):
    """Check the other outputs that share this case's nonzero input before the native call."""
    import torch

    routed, down = ids is not None, ids is not None and mode == 2
    if routed:
        flat = routes.flat_sorted.long()
        positions = [case["input_row"]] if down else (flat // ids.shape[1] == case["input_row"]).nonzero().flatten().tolist()
        experts = [(position, int(ids.flatten()[flat[position]])) for position in positions]
    else:
        experts = [(case["input_row"], 0)]
    bound = None
    for position, expert in experts:
        names = ("gate_proj", "up_proj") if routed and mode == 0 else (role["name"],)
        side_bounds = []
        for side, name in enumerate(names):
            if routed:
                weight, scale = adapter.weight(expert, name)
            else:
                _, weight, scale, _ = next(r for r in adapter.roles if r[0] == name)
            vector = weight[:, case["column"]].double().cpu()
            scale = scale.float().cpu()
            if adapter.family == "e2m1":
                gs = (adapter.gs2 if down else adapter.gs13) if routed else adapter.gs
                scale = scale / torch.tensor(gs, dtype=torch.float32)
            cols = weight.shape[1]
            if target == "row":
                count = (adapter.experts if routed else 1) * weight.shape[0]
                start = side * count + expert * weight.shape[0]
                scale = torch.tensor(list(values[start:start + weight.shape[0]]), dtype=torch.float32)
            elif target == "ratio":
                scale = torch.tensor(values[side * adapter.experts + expert] if routed else values[0], dtype=torch.float32)
            product = vector * aq
            if target == "sfa":
                product *= values[case["input_row"] * (cols // 16) + case["column"] // 16]
            epilogue = product.float() * (values[case["input_row"]] if target == "a_s" else a_scale)
            total = product.abs() * abs(values[case["input_row"]] if target == "a_s" else a_scale)
            epilogue = epilogue * scale
            total *= scale.double().abs()
            if down:
                rw = values[position] if target == "router" else float(routes.rw_sorted[position])
                epilogue *= rw
                total *= abs(rw)
            if not bool(torch.isfinite(epilogue.bfloat16()).all()):
                raise Invalid("C1 scale coding gives a nonfinite incidental reference")
            side_bounds.append(total)
        total = side_bounds[0]
        if routed and mode == 0:
            total = torch.where((total < 2.0**127) & (side_bounds[1] < 2.0**127), adapter.table.limit**2, float("inf"))
        if bound is None:
            count = routes.routes if routed else (rows if rows is not None else case["input_row"] + 1)
            bound = torch.zeros(count, total.numel(), dtype=torch.float64)
        output_row = int(routes.flat_sorted[position]) if down else position
        bound[output_row] = total
    return bound


def check(adapter, unit, manifest, recorder):
    import torch
    from experiments.numerics_rows.cases import routing
    from experiments.numerics_rows.row import compare_output, coverage, finite_outputs

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
                    w, scale = adapter.weight(role["expert"], role["name"])
                else:
                    q, sf, a, a_scale = adapter.quantized(x, recorder)
                    _, w, scale, _ = next(r for r in adapter.roles if r[0] == role["name"])
                gs = (adapter.gs2 if down else adapter.gs13) if routed else adapter.gs
                if adapter.family == "e2m1":
                    scale = scale.float() / torch.tensor(gs, dtype=torch.float32, device=scale.device)
                scalar = lambda tensor: float(tensor.reshape(-1)[min(case["output_row"], tensor.numel() - 1)])
                aq = float(a[case["input_row"], case["column"]])
                if plan["scale"] == "sfa":
                    byte = int(q.view(torch.uint8)[case["input_row"], case["column"] // 2])
                    code = (byte >> (4 * (case["column"] % 2))) & 15
                    from experiments.routed_pair_oracle import e2m1_table
                    aq = float(e2m1_table("cpu")[code])
                product = F(aq) * F(float(w[case["output_row"], case["column"]]))
                peer, peer_scale = None, None
                if routed and mode == 0:
                    peer, peer_scale = adapter.weight(role["expert"], "up_proj")
                    if adapter.family == "e2m1":
                        peer_scale = peer_scale.float() / torch.tensor(gs, dtype=torch.float32)
                if plan["scale"] in ("row", "ratio"):
                    count = (adapter.experts if routed else 1) * (w.shape[0] if plan["scale"] == "row" else 1)
                    index = (role["expert"] * w.shape[0] + case["output_row"]) if plan["scale"] == "row" else role["expert"]
                    read, n = ([index, count + index], 2 * count) if routed and mode == 0 else ([index], count)
                else:
                    groups = cols // 16 if plan["scale"] == "sfa" else 1
                    index = case["input_row"] * groups + (case["column"] // 16 if groups > 1 else 0)
                    read, n = ([index, index] if routed and mode == 0 else [index]), rows * groups
                validate_case(case, plan, aq=aq, a_scale=1.0 if a_scale is None else float(a_scale[case["input_row"]]),
                    weight=float(w[case["output_row"], case["column"]]), scale=scalar(scale),
                    peer=None if peer is None else float(peer[case["output_row"], case["column"]]),
                    peer_scale=None if peer_scale is None else scalar(peer_scale),
                    router_weight=float(routes.rw_sorted[case["input_row"]]) if down else 1.0, n=n, read=read)
                for coding in plan["coding"]:
                    vals = scale_values(coding)
                    products = [product]
                    if peer is not None:
                        products.append(F(aq) * F(float(peer[case["output_row"], case["column"]])))
                    if not all(exact_input([p * (F(vals[read[i]]) if plan["scale"] == "sfa" else 1)])["holds"] for i, p in enumerate(products)):
                        raise Invalid("X false in a claimed scale-index case")
                    qq, ss = q, None if sf is None else sf.clone()
                    expected = evaluate(case, case["read"], vals, table)
                    if not math.isfinite(expected):
                        raise Invalid("scale-index reference is not finite")
                    bound = finite_references(adapter, role, case, plan["scale"], vals, aq,
                        1.0 if a_scale is None else float(a_scale[case["input_row"]]),
                        ids if routed else None, routes if routed else None, mode, rows=m)
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
                    finite_outputs(unit, "scale-index:all-outputs", out, bound)
                    ref = torch.tensor([expected], device=q.device, dtype=torch.bfloat16)
                    got = out[output_row, case["output_row"]].reshape(1)
                    result = compare_output(unit, f"scale-index:{role['name']}:{coding['family']}:{coding['digit']}:{case['read']}", got, ref, operands={"input": case["input"], "column": case["column"], "read": case["read"],
                        "scales": [vals[i] for i in case["read"]], "g": case["g"], "u": case["u"]})
                    record["mismatches"] += result["mismatches"]
