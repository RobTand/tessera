"""Execute routed exact cases on the served launch and its staged replay."""
from __future__ import annotations

from experiments.numerics_rows.row import Invalid, compare_output, coverage, finite_outputs, finite_reference, opaque_apply, require_x
from experiments.numerics_rows.common import BoundaryError


def token_reference(routed, top_k):
    import torch

    values = routed.reshape(-1, top_k, routed.shape[-1]).float()
    total = torch.zeros_like(values[:, 0])
    for index in range(top_k):
        total = total + values[:, index]
    return total.to(torch.bfloat16)


def graph_record(unit, name, function, out, recorder):
    from experiments.t4_code.fused_e2m1_check import graph_equal

    recorder.enabled = False
    try:
        equal = graph_equal(function, out, before_call=recorder.before_call)
        unit["graph_replay_equal"].append({"call": name, "equal": equal})
    finally:
        recorder.enabled = True


def run_case(adapter, unit, projection, column_values, ids, weights, recorder, *, graph=False):
    import torch
    from tessera import routed_fused as rf, routed_fused_e2m1 as fe

    m = unit["m_config"]["M"]
    down = projection == "down_proj"
    routes = adapter.route(ids, weights)
    rows, cols = (routes.routes, adapter.inter) if down else (m, adapter.cols)
    x = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
    for column, value in column_values:
        x[:, column] = value
    q, sf, a, a_scale = adapter.quantized(x, adapter.gs2 if down else adapter.gs13, recorder)
    mode = 2 if down else 0
    got = torch.empty(routes.routes, adapter.rows if down else adapter.inter, device="cuda", dtype=torch.bfloat16)
    launch = lambda: adapter.launch(mode, q, sf, routes, got, row_mode=1 if down else 0, weighted=down)
    if down:
        ref, mask, bound = adapter.down_reference(a, a_scale, ids, weights, routes)
    else:
        ref, mask, bound, mode1_bound = adapter.gate_reference(a, a_scale, ids, routes)
    finite_reference(ref)
    require_x(mask, sweep=len(column_values) == 1)
    launch()
    result = compare_output(unit, f"{projection}:{column_values}", got, ref, mask, bound=bound)
    section = "down_input" if down else "swiglu"
    unit["exact"][section]["elements"] += int(mask.sum())
    unit["exact"][section]["mismatches"] += result["mismatches"]
    if graph:
        graph_record(unit, f"mode-{mode}", launch, got, recorder)
    if down:
        library = fe._ext() if adapter.family == "e2m1" else rf._ext(adapter.native.library)
        summed = torch.empty(m, adapter.rows, device="cuda", dtype=torch.bfloat16)
        sum_call = lambda: library.token_sum(got, summed, ids.shape[1])
        sum_ref = token_reference(ref, ids.shape[1])
        finite_reference(sum_ref)
        sum_call()
        sum_mask = mask.reshape(m, ids.shape[1], adapter.rows).all(1)
        sum_bound = bound.token_sum(ids.shape[1])
        reduced = compare_output(unit, "token_sum", summed, sum_ref, sum_mask, bound=sum_bound)
        unit["exact"]["token_sum"]["elements"] += int(sum_mask.sum())
        unit["exact"]["token_sum"]["mismatches"] += reduced["mismatches"]
        if graph:
            graph_record(unit, "token_sum", sum_call, summed, recorder)
    else:
        # Mode 1 has no exact-equality claim. The universal finiteness rule still applies.
        observation = torch.empty(m, ids.shape[1], 2 * adapter.inter, device="cuda", dtype=torch.bfloat16)
        adapter.launch(1, q, sf, routes, observation)
        finite_outputs(unit, "mode-1", observation, mode1_bound)
        unit.setdefault("mode1", []).append({"shape": list(observation.shape), "nonfinite": int((~torch.isfinite(observation)).sum())})
    return result


def composition(adapter, unit, x, ids, weights, recorder):
    from experiments.numerics_rows.reference import compare

    bound = adapter.preflight(x, ids, weights, recorder)
    try:
        got = recorder.call("routed:apply", "apply", lambda: opaque_apply(adapter.apply, recorder, x, ids, weights))
    except BoundaryError as error:
        if error.attributed_to is None:
            adapter.staged(x, ids, weights, recorder)
        raise
    staged, _ = adapter.staged(x, ids, weights, recorder, unit=unit)
    finite_outputs(unit, "composition:apply", got, bound)
    result = compare(got, staged)
    unit["exact"]["composition"]["cases"] += 1
    unit["exact"]["composition"]["equal"] &= result["equal"]
    if not result["equal"]:
        unit["findings"].append("apply differs from the same staged routed launches")
    return got


def run(adapter, unit, manifest, recorder):
    import torch
    from experiments.numerics_rows.cases import chunk_pairs, routing

    m, width = unit["m_config"]["M"], unit["m_config"]["superblock_width"]
    every_expert = m in (1, 64, 2048)
    unit["separation_census_complete"] = every_expert
    unit["needs_separation_transfer"] = not every_expert
    selected = [list(range(start, min(start + 8, adapter.experts))) for start in range(0, adapter.experts, 8)] if every_expert else [[0]]
    for count in unit["m_config"]["route_counts"]:
        for group in selected:
            raw_ids, raw_weights = routing(m, adapter.experts, count, group)
            tested = set()
            ids = torch.tensor(raw_ids, device="cuda", dtype=torch.int64)
            weights = torch.tensor(raw_weights, device="cuda", dtype=torch.float32)
            for role in manifest["roles"]:
                if role["expert"] not in group:
                    continue
                separated = role["separation"]
                if "by_m" in separated:
                    separated = next((r for r in separated["by_m"] if r["route"]["M"] == m), separated["by_m"][0])
                coverage(unit, "separation", separated["complete"], separated["complete"] or manifest["complete"])
                if count == 1:
                    unit["exact"]["separation"]["records"].append({k: v for k, v in separated.items() if k not in ("recipes", "uncovered", "by_m")})
                for recipe in separated["recipes"]:
                    for activation in recipe["values"]:
                        stage = role["name"]
                        key = stage, recipe["column"], activation
                        if key in tested:
                            continue
                        tested.add(key)
                        result = run_case(adapter, unit, stage, [(recipe["column"], activation)], ids, weights, recorder)
                        unit["exact"]["sweep"]["columns"] += 1
                        unit["exact"]["sweep"]["elements"] += result.get("elements", 0)
                        unit["exact"]["sweep"]["mismatches"] += result["mismatches"]
                unit["exact"]["sweep"]["experts"] += 1
            for projection, cols in (("gate_proj", adapter.cols), ("down_proj", adapter.inter)):
                for pair in chunk_pairs(cols, unit["m_config"]["chunk_size"]):
                    result = run_case(adapter, unit, projection, [(column, 1) for column in pair], ids, weights, recorder)
                    unit["exact"]["chunk_pairs"]["rows"] += m
                    unit["exact"]["chunk_pairs"]["x_elements"] += result.get("elements", 0)
                    unit["exact"]["chunk_pairs"]["mismatches"] += result["mismatches"]
            x = torch.zeros(m, adapter.cols, device="cuda", dtype=torch.bfloat16)
            x[:, 0] = 1
            composition(adapter, unit, x, ids, weights, recorder)


def diagnostic_cases(adapter, unit, recorder, manifest):
    import torch
    from experiments.numerics_rows.cases import routing

    m = unit["m_config"]["M"]
    raw_ids, raw_weights = routing(m, adapter.experts, m, 0)
    ids = torch.tensor(raw_ids, device="cuda", dtype=torch.int64)
    weights = torch.tensor(raw_weights, device="cuda", dtype=torch.float32)
    for seed in (0, 1, 2):
        for distribution in ("gauss", "outlier", "zero"):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn(m, adapter.cols, generator=generator, device="cuda").bfloat16()
            if distribution == "zero":
                x.zero_()
            elif distribution == "outlier":
                x.flatten()[::256] *= 128
                scales = torch.exp2(torch.linspace(-8, 8, m, device="cuda"))
                x = (x.float() * scales[:, None]).bfloat16()
                if adapter.family == "e2m1":
                    x = (x.float() * min(1.0, 6 * 448 / (float(x.abs().max()) * adapter.gs13))).bfloat16()
            got = composition(adapter, unit, x, ids, weights, recorder)
            # Record the FP64 dot error on independent sorted-route down input.
            routes = adapter.route(ids, weights)
            down = torch.randn(routes.routes, adapter.inter, generator=generator, device="cuda").bfloat16()
            q, sf, a, a_scale = adapter.quantized(down, adapter.gs2, recorder)
            native = torch.empty(routes.routes, adapter.rows, device="cuda", dtype=torch.bfloat16)
            reference, _, bound = adapter.down_reference(a, a_scale, ids, weights, routes)
            finite_reference(reference)
            adapter.launch(2, q, sf, routes, native, row_mode=1, weighted=True)
            finite_outputs(unit, f"diagnostic:mode-2:{distribution}:{seed}", native, bound)
            record = diagnostic_down(adapter, unit, native, a, a_scale, ids, routes, distribution, seed)
            unit["diagnostic"].append(record)

    graphed = set()
    for role in manifest["roles"]:
        if role["name"] in graphed or role["expert"] != 0:
            continue
        separation = role["separation"]
        if "by_m" in separation:
            separation = next((r for r in separation["by_m"] if r["route"]["M"] == m), separation["by_m"][0])
        if separation["recipes"]:
            recipe = separation["recipes"][0]
            run_case(adapter, unit, role["name"], [(recipe["column"], recipe["values"][0])], ids, weights, recorder, graph=True)
            graphed.add(role["name"])


def diagnostic_down(adapter, unit, got, a, a_scale, ids, routes, distribution, seed):
    import torch
    from experiments.numerics_rows.row import diagnostic

    flat = routes.flat_sorted.long()
    experts = ids.flatten()[flat]
    reference, total = torch.empty_like(got, dtype=torch.float64), torch.empty_like(got, dtype=torch.float64)
    for expert in experts.unique().tolist():
        select = (experts == expert).nonzero().flatten()
        weight, scale = adapter.weight(expert, "down_proj")
        weight = weight.to(a.device)
        c = (scale.float() / torch.tensor(adapter.gs2, dtype=torch.float32)) if adapter.family == "e2m1" else scale
        c = c.to(a.device).double()
        aa = a[select].double()
        if a_scale is not None:
            aa = aa * a_scale[select, None].double()
        reference[flat[select]] = (aa @ weight.T) * c * routes.rw_sorted[select, None].double()
        total[flat[select]] = (aa.abs() @ weight.abs().T) * c.abs() * routes.rw_sorted[select, None].double().abs()
    return diagnostic(got, reference, total, a.shape[1], 1, distribution, seed)
