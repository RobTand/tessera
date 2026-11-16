"""Execute routed exact cases on the served launch and its staged replay."""
from __future__ import annotations

from experiments.numerics_rows.row import Invalid, compare_output, coverage, opaque_apply, require_x
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
        equal = graph_equal(function, out)
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
        ref, mask = adapter.down_reference(a, a_scale, ids, weights, routes)
    else:
        ref, mask = adapter.gate_reference(a, a_scale, ids, routes)
    require_x(mask, sweep=len(column_values) == 1)
    launch()
    result = compare_output(unit, f"{projection}:{column_values}", got, ref, mask)
    section = "down_input" if down else "swiglu"
    unit["exact"][section]["elements"] += int(mask.sum())
    unit["exact"][section]["mismatches"] += result["mismatches"]
    if graph:
        graph_record(unit, f"mode-{mode}", launch, got, recorder)
    if down:
        library = fe._ext() if adapter.family == "e2m1" else rf._ext(adapter.native.library)
        summed = torch.empty(m, adapter.rows, device="cuda", dtype=torch.bfloat16)
        sum_call = lambda: library.token_sum(got, summed, ids.shape[1])
        sum_call()
        sum_ref = token_reference(ref, ids.shape[1])
        sum_mask = mask.reshape(m, ids.shape[1], adapter.rows).all(1)
        reduced = compare_output(unit, "token_sum", summed, sum_ref, sum_mask)
        unit["exact"]["token_sum"]["elements"] += int(sum_mask.sum())
        unit["exact"]["token_sum"]["mismatches"] += reduced["mismatches"]
        if graph:
            graph_record(unit, "token_sum", sum_call, summed, recorder)
    else:
        # Mode 1 is an observation. It cannot decide the unit's outcome.
        observation = torch.empty(m, ids.shape[1], 2 * adapter.inter, device="cuda", dtype=torch.bfloat16)
        adapter.launch(1, q, sf, routes, observation)
        unit.setdefault("mode1", []).append({"shape": list(observation.shape), "nonfinite": int((~torch.isfinite(observation)).sum())})
    return result


def composition(adapter, unit, x, ids, weights, recorder):
    from experiments.numerics_rows.reference import compare

    try:
        got = recorder.call("routed:apply", "apply", lambda: opaque_apply(adapter.apply, recorder, x, ids, weights))
    except BoundaryError as error:
        if error.attributed_to is None:
            adapter.staged(x, ids, weights, recorder)
        raise
    staged = adapter.staged(x, ids, weights, recorder)
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
    if not every_expert:
        unit["unattested"].append("element separation at columns-only M awaits the lead clarification in questions.md")
    selected = [list(range(start, min(start + 8, adapter.experts))) for start in range(0, adapter.experts, 8)] if every_expert else [[0]]
    graphed = set()
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
                coverage(unit, "separation", separated["complete"], manifest["complete"])
                unit["exact"]["separation"].append({k: v for k, v in separated.items() if k not in ("recipes", "uncovered")})
                for recipe in separated["recipes"]:
                    for activation in recipe["values"]:
                        stage = role["name"]
                        key = stage, recipe["column"], activation
                        if key in tested:
                            continue
                        tested.add(key)
                        first = stage not in graphed
                        result = run_case(adapter, unit, stage, [(recipe["column"], activation)], ids, weights, recorder, graph=first)
                        graphed.add(stage)
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
    diagnostic_cases(adapter, unit, recorder)


def diagnostic_cases(adapter, unit, recorder):
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
            adapter.launch(2, q, sf, routes, native, row_mode=1, weighted=True)
            record = diagnostic_down(adapter, native, a, a_scale, ids, routes, distribution, seed)
            unit["diagnostic"].append(record)


def diagnostic_down(adapter, got, a, a_scale, ids, routes, distribution, seed):
    import torch
    from experiments.numerics_rows.row import diagnostic

    flat = routes.flat_sorted.long()
    experts = ids.flatten()[flat]
    reference, total = torch.empty_like(got, dtype=torch.float64), torch.empty_like(got, dtype=torch.float64)
    for expert in experts.unique().tolist():
        select = (experts == expert).nonzero().flatten()
        weight, scale = adapter.weight(expert, "down_proj")
        weight = weight.to(a.device)
        c = scale.to(a.device).double() / (adapter.gs2 if adapter.family == "e2m1" else 1)
        aa = a[select].double()
        if a_scale is not None:
            aa = aa * a_scale[select, None].double()
        reference[flat[select]] = (aa @ weight.T) * c * routes.rw_sorted[select, None].double()
        total[flat[select]] = a[select].double().abs() @ weight.abs().T
    return diagnostic(got, reference, total, a.shape[1], 1, distribution, seed)
