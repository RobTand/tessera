"""Produce actual T8 byte plans and timing rows through unchanged serving owners.

A byte plan precedes timing. Only an exact serialized-byte match can produce a
competitive timing row. A lower bound can prove the scalar T8 floor exceeds a
T4 budget. Other failed finite searches remain explicit unresolved matches.
Neither outcome removes a valid T4 correctness or D41 row.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.t4_code import t4_fused_qualify as q

KINDS = {"gate_up": ("mode0", "mode1"), "down": ("mode2",),
         "chain": ("chain", "router_input"), "dense": ("dense",)}


def t8_plane_bytes(rate, rows, cols):
    from tessera.alphabet import E4M3_GRID
    from tessera.calculator import terminal_rate
    value = terminal_rate(rate, rows, cols, with_scale_base=False,
        with_scale_refine=False, with_row_scale=True, with_diagonals=False,
        completion=0, cap=E4M3_GRID.payload_bits, arity=1, span=1,
        window_bits=14, code_bytes=E4M3_GRID.code_bytes) * rows * cols / 8
    if value.denominator != 1:
        raise ValueError("T8 plane accounting must produce integral bytes")
    return int(value)


def t8_bounds():
    from tessera.serving.contract import reader_rate_grid
    found = reader_rate_grid("TESSERA_FP8", "E4M3")
    if found is None:
        raise ValueError("The unchanged T8 route has no published reader range")
    _family, low, high, step = found
    return low, high, step


def encode_t8(weight, rate, structure):
    from tessera.alphabet import E4M3_GRID
    from tessera.export import served_recipe
    recipe = served_recipe(E4M3_GRID, rate, structure)
    return q.encode_artifact_bytes(weight, grid=E4M3_GRID, q256=rate,
                                   **q.recipe_kwargs(recipe))


def shapes(args):
    return {"gate": (args.inter, args.hidden), "up": (args.inter, args.hidden),
            "down": (args.hidden, args.inter)}


def make_frames(weights, rate, structure):
    from tessera.fused_frame import pack_fused
    result = {}
    for name, weight in weights.items():
        role = q.ROLE_NAMES[name] if name in q.ROLE_NAMES else "weight"
        blob = encode_t8(weight, rate, structure)
        result[name] = pack_fused([(role, weight.shape[0], blob)])
    return result


def frame_sizes(frames, parts, copies):
    return sum(len(frames[p]) for p in parts) * copies


def select_actual_rate(target_bytes, lower_plane_bytes, legal_rates, cost, predictions):
    """Use measured bytes for selection. Predictions only choose the probe order.

    Probe an arithmetic bracket, then a binary bracket in the actual byte
    costs. Inspect both sides and adjacent rates. Never replace real bytes
    with a forecast. A failed search does not prove that all rates failed.
    """
    if lower_plane_bytes > target_bytes:
        return dict(status="unattainable_scalar_floor", exact_match=False,
                    target_bytes=target_bytes, t8_plane_lower_bound=lower_plane_bytes,
                    proof="Every legal scalar T8 body has at least one bit per weight. "
                          "This bound also includes its table and row scales.", measured=[])
    measured = {}
    def probe(rate):
        if rate not in measured:
            measured[rate] = int(cost(rate))
        return measured[rate]
    promising = sorted(legal_rates, key=lambda r: (abs(predictions[r] - target_bytes), r))[:3]
    winner = None
    for rate in promising:
        if probe(rate) == target_bytes:
            winner = rate
            break
    if winner is None:
        low, high = 0, len(legal_rates) - 1
        while low <= high:
            middle = (low + high) // 2
            rate = legal_rates[middle]
            value = probe(rate)
            if value == target_bytes:
                winner = rate
                break
            if value < target_bytes:
                low = middle + 1
            else:
                high = middle - 1
        for index in range(max(0, high - 2), min(len(legal_rates), low + 3)):
            rate = legal_rates[index]
            if probe(rate) == target_bytes:
                winner = rate
                break
    observations = [dict(q256=rate, actual_serialized_bytes=value,
                         signed_slack_bytes=target_bytes - value)
                    for rate, value in sorted(measured.items())]
    result = dict(status="exact_match" if winner is not None else "no_exact_match_found",
                  exact_match=winner is not None, selected_q256=winner,
                  target_bytes=target_bytes, t8_plane_lower_bound=lower_plane_bytes,
                  measured=observations, legal_rate_count=len(legal_rates),
                  search_scope="Arithmetic probes, actual binary bracket, and adjacent rates. "
                               "A no-match result is not an exhaustive impossibility proof.")
    if observations:
        closest = min(observations, key=lambda row: abs(row["signed_slack_bytes"]))
        result["closest"] = closest
    return result


def plan_cases(args, t4_q256):
    import torch
    from tessera.fused_frame import pack_fused

    low, high, step = t8_bounds()
    legal = list(range(low, high + 1, step))
    targets = []
    if args.part in ("all", "routed"):
        weights = {p: q.synthetic(*shape, args.seed + q.ROLE_SEEDS[p], "cuda")
                   for p, shape in shapes(args).items()}
        t4 = {p: pack_fused([(q.ROLE_NAMES[p], w.shape[0],
            q.encode_bytes(w, t4_q256, structure="routed_moe")[0])]) for p, w in weights.items()}
        targets.extend((dict(case_id="routed:" + kind, group=kind,
            parts=parts, copies=args.experts, structure="routed_moe", weights=weights,
            t4_frames=t4) for kind, parts in (("gate_up", ("gate", "up")),
                                             ("down", ("down",)),
                                             ("chain", ("gate", "up", "down")))))
    if args.part in ("all", "dense"):
        for tag, rows, cols in args.dense_shapes:
            weight = q.synthetic(rows, cols, args.seed + rows, "cuda")
            t4 = {"weight": q.dense_frame(q.encode_bytes(weight, t4_q256)[0], rows)}
            targets.append(dict(case_id=f"dense:{tag}:{rows}x{cols}", group="dense",
                parts=("weight",), copies=1, structure="dense", weights={"weight": weight},
                t4_frames=t4, shape=(tag, rows, cols)))
    for target in targets:
        weights, parts, copies = target["weights"], target["parts"], target["copies"]
        cache = {}
        def frames(rate):
            if rate not in cache:
                cache[rate] = make_frames({p: weights[p] for p in parts}, rate, target["structure"])
            return cache[rate]
        budget = frame_sizes(target["t4_frames"], parts, copies)
        minimum = sum(t8_plane_bytes(low, *weights[p].shape) for p in parts) * copies
        if minimum > budget:
            choice = select_actual_rate(budget, minimum, legal, None, None)
        else:
            # The highest legal rate gives the smallest traceback workspace.
            # These actual bytes supply only the fixed-overhead prediction.
            base = frames(high)
            overhead = {p: len(base[p]) - t8_plane_bytes(high, *weights[p].shape) for p in parts}
            predictions = {rate: sum(t8_plane_bytes(rate, *weights[p].shape) + overhead[p]
                                     for p in parts) * copies for rate in legal}
            choice = select_actual_rate(budget, minimum, legal,
                lambda rate: frame_sizes(frames(rate), parts, copies), predictions)
        record = dict(choice, case_id=target["case_id"], group=target["group"],
            t4_q256=t4_q256, format="T8", structure=target["structure"],
            actual_t4_serialized_bytes=budget, copies=copies, parts=list(parts),
            t4_template_bytes={p: len(b) for p, b in target["t4_frames"].items()},
            source_weight_seeds={p: args.seed + q.ROLE_SEEDS[p] if p in q.ROLE_SEEDS
                                 else args.seed + weights[p].shape[0] for p in weights},
            source_weight_sha256={p: q.tensor_digest(w) for p, w in weights.items()},
            shape=list(target["shape"]) if "shape" in target else [args.hidden, args.inter])
        selected = None
        if choice.get("exact_match"):
            selected = dict(frames(choice["selected_q256"]))
            missing = {p: w for p, w in weights.items() if p not in selected}
            selected.update(make_frames(missing, choice["selected_q256"], target["structure"]))
        # Only a real exact-match frame set reaches a production owner.
        yield record, target, selected
        del cache
        torch.cuda.empty_cache()


def base_layer(args, size, rank):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    layer = torch.nn.Module()
    layer.moe_config = q.make_moe_config(args.hidden, args.inter, args.experts, args.top_k, size, rank)
    layer.activation = MoEActivation.SILU
    layer.expert_map = None
    layer.global_num_experts = args.experts
    layer.apply_router_weight_on_input = False
    layer.swiglu_limit = layer.swiglu_alpha = layer.swiglu_beta = None
    return layer


def load_t8_routed(args, templates, rate, size, rank):
    import torch
    from tessera.serving.moe_route import build_tessera_moe_method
    from tessera.routed_fused import FusedRoutedWindowMoE
    frames = {p: [template] * args.experts for p, template in templates.items()}
    scheme = q.routed_scheme(frames, args.hidden, args.inter, args.experts, rate)
    scheme.update(family="TESSERA_FP8", grid="E4M3", plane="CHANNEL")
    layer = base_layer(args, size, rank)
    method = build_tessera_moe_method(scheme, "qualification.t8.experts", "resident", layer)
    method.create_weights(layer, args.experts, args.hidden, args.inter // size,
                          torch.bfloat16, global_num_experts=args.experts)
    layer.to("cuda")
    for p, shard in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        parameter = layer.w2_wire if p == "down" else layer.w13_wire
        for expert, frame in enumerate(frames[p]):
            parameter.weight_loader(parameter, torch.frombuffer(bytearray(frame), dtype=torch.uint8),
                                    "weight", shard, expert)
    method.process_weights_after_loading(layer)
    owner = method._native
    if not isinstance(owner, FusedRoutedWindowMoE):
        raise ValueError("The real T8 route did not select its existing fused native owner")
    return layer, method, owner


def load_t8_dense(frame, shape, rate, size, rank, axis):
    import torch
    from tessera.serving.fp8_route import build_tessera_fp8_method
    _tag, rows, cols = shape
    scheme = q.dense_scheme(frame, rows, cols, rate)
    scheme.update(family="TESSERA_FP8", grid="E4M3", plane="CHANNEL")
    layer = torch.nn.Module()
    layer.tp_size, layer.tp_rank = size, rank
    method = build_tessera_fp8_method(scheme, "qualification.t8.dense", "resident")
    local_rows = rows // size if axis == "row" else rows
    local_cols = cols // size if axis == "column" else cols
    method.create_weights(layer, local_cols, [local_rows], cols, rows, torch.bfloat16)
    layer.wire_bytes.data.copy_(torch.frombuffer(bytearray(frame), dtype=torch.uint8))
    layer.to("cuda")
    method.process_weights_after_loading(layer)
    return layer, method


def timing_row(fn, metadata, args):
    import torch
    eager, graph, _output, equal = q.capture_callable(fn)
    finite = bool(torch.isfinite(eager).all())
    return dict(metadata, format="T8", serving_owner=True, eager_ok=finite, graph_equal=equal,
                numeric_scope="Finiteness and graph replay of unchanged T8 runtime; no new arithmetic claim.",
                timing={"eager": q.time_callable(fn, args.warmup, args.iters),
                        "graph": q.time_callable(graph.replay, args.warmup, args.iters)},
                output_sha256=q.tensor_digest(eager))


def routed_rows(args, record, target, templates, size, rank):
    import torch
    rate = record["selected_q256"]
    layer, method, owner = load_t8_routed(args, templates, rate, size, rank)
    base = dict(format="T8", q256=rate, target_t4_q256=record["t4_q256"],
        case_id=record["case_id"], hidden=args.hidden, inter=args.inter // size,
        experts=args.experts, top_k=args.top_k, tp_size=size, tp_rank=rank,
        wire_bytes=record["actual_t4_serialized_bytes"],
        total_serialized_bytes=sum(map(len, templates.values())) * args.experts,
        resident_bytes=q.storage_bytes(method.resident_tensors(layer)),
        source_weight_seeds=record["source_weight_seeds"],
        source_weight_sha256=record["source_weight_sha256"],
        runtime_launch_pair=list(owner.launch_pair), runtime_library=owner.library,
        actual_t8_template_bytes={p: len(b) for p, b in templates.items()},
        weight_bytes_scope=list(record["parts"]),
        serving_intake="moe_route create/wire loaders/finalize/apply")
    for m in args.ms:
        ids, rw = q.routing(m, args.experts, args.top_k, args.seed + m, "cuda")
        x = q.synthetic(m, args.hidden, args.seed + m, "cuda").to(torch.bfloat16)
        metadata = dict(base, m=m, seed=args.seed + m, input_sha256=q.tensor_digest(x),
                        routing_sha256=q.tensor_digest(ids), routing_weights_sha256=q.tensor_digest(rw))
        for kind in KINDS[record["group"]]:
            if kind == "mode0":
                routing = owner._routing(ids, rw)
                codes, scales = owner._quantized(x, None, routing.tokens)
                output = torch.empty(routing.routes, owner.down.cols, dtype=torch.bfloat16, device="cuda")
                def fn():
                    owner._launch(0, codes, scales, routing, a_row_mode=0, mul_weight=False,
                                  limit=float("inf"), out=output, counter=0)
                    return output
            elif kind == "mode1":
                fn = lambda: owner.gate_up(x, ids, rw)
            elif kind == "mode2":
                xd = q.synthetic(m * args.top_k, args.inter // size, args.seed + m + 1,
                                 "cuda").to(torch.bfloat16)
                metadata["input_sha256"] = q.tensor_digest(xd)
                fn = lambda: owner.down_routes(xd, ids, rw)
            elif kind == "chain":
                fn = lambda: method.apply(layer, x, rw, ids, None, None)
            else:
                top_ids, top_rw = ids[:, :1].contiguous(), rw[:, :1].contiguous()
                layer.apply_router_weight_on_input = True
                metadata.update(top_k=1, routing_sha256=q.tensor_digest(top_ids),
                                routing_weights_sha256=q.tensor_digest(top_rw))
                fn = lambda: method.apply(layer, x, top_rw, top_ids, None, None)
            yield timing_row(fn, dict(metadata, kind=kind), args)
        layer.apply_router_weight_on_input = False


def dense_rows(args, record, target, templates, size, rank):
    import torch
    shape, rate = target["shape"], record["selected_q256"]
    tag, rows, cols = shape
    for axis in q.dense_axes(args, shape, size):
        layer, method = load_t8_dense(templates["weight"], shape, rate, size, rank, axis)
        local_cols = cols // size if axis == "column" else cols
        for m in args.ms:
            x = q.synthetic(m, local_cols, args.seed + m, "cuda").to(torch.bfloat16)
            metadata = dict(format="T8", kind="dense", case_id=record["case_id"],
                q256=rate, target_t4_q256=record["t4_q256"], rows=rows, cols=cols,
                shape_tag=tag, cut_axis=axis, tp_size=size, tp_rank=rank, m=m, seed=args.seed + m,
                input_sha256=q.tensor_digest(x), wire_bytes=len(templates["weight"]),
                resident_bytes=q.storage_bytes(method.resident_tensors(layer)),
                source_weight_seeds=record["source_weight_seeds"],
                source_weight_sha256=record["source_weight_sha256"],
                weight_bytes_scope=["weight"],
                runtime_launch_pair=list(layer.tessera_native.launch_pair),
                serving_intake="fp8_route create/wire load/finalize/apply")
            yield timing_row(lambda: method.apply(layer, x), metadata, args)


def produce(args):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("Actual T8 baseline production requires the GB10 GPU")
    result = dict(schema="tessera.t4_t8_baseline.v1", mode="t8-baseline",
        **q.source_stamp(args), plan_only=args.plan_only, byte_plan=[], cells=[], skips=[])
    for target_rate in args.q256:
        for record, target, templates in plan_cases(args, target_rate):
            result["byte_plan"].append(record)
            q.save_report(args.out, result)
            if args.plan_only or not record["exact_match"]:
                continue
            for size, rank in q.rank_cases(args):
                try:
                    generator = (dense_rows(args, record, target, templates, size, rank)
                                 if record["group"] == "dense" else
                                 routed_rows(args, record, target, templates, size, rank))
                    for row in generator:
                        result["cells"].append(row)
                        q.save_report(args.out, result)
                except Exception as exc:
                    result["skips"].append(dict(case_id=record["case_id"], t4_q256=target_rate,
                        tp_size=size, tp_rank=rank, reason=f"{type(exc).__name__}: {exc}"))
        torch.cuda.empty_cache()
    result["unattainable"] = [r for r in result["byte_plan"] if r["status"] == "unattainable_scalar_floor"]
    result["unresolved_exact_matches"] = [r for r in result["byte_plan"] if r["status"] == "no_exact_match_found"]
    result["population"] = dict(planned=len(result["byte_plan"]), measured_cells=len(result["cells"]),
                                skips=len(result["skips"]))
    return result


def baseline_expected_keys(report, args):
    expected = set()
    for plan in report["byte_plan"]:
        if not plan.get("exact_match"):
            continue
        for size, rank in q.rank_cases(args):
            axes = (q.dense_axes(args, tuple(plan["shape"]), size)
                    if plan["group"] == "dense" else [None])
            for axis in axes:
                for m in args.ms:
                    for kind in KINDS[plan["group"]]:
                        expected.add((plan["t4_q256"], plan["case_id"], kind, m, size, rank, axis))
    return expected


def validate_baseline(report, args):
    failures = []
    case_ids = (["routed:gate_up", "routed:down", "routed:chain"]
                if args.part in ("all", "routed") else [])
    if args.part in ("all", "dense"):
        case_ids += [f"dense:{tag}:{rows}x{cols}" for tag, rows, cols in args.dense_shapes]
    expected_plans = {(rate, case) for rate in args.q256 for case in case_ids}
    plans = report.get("byte_plan", [])
    actual_plans = [(p.get("t4_q256"), p.get("case_id")) for p in plans]
    if len(actual_plans) != len(set(actual_plans)) or set(actual_plans) != expected_plans:
        failures.append("The byte-plan population is incomplete")
    for plan in plans:
        if plan.get("status") == "unattainable_scalar_floor":
            if plan.get("t8_plane_lower_bound", 0) <= plan.get("target_bytes", 0):
                failures.append("A scalar-floor exclusion has no valid byte lower bound")
        elif plan.get("status") == "exact_match":
            selected = [r for r in plan.get("measured", []) if r["q256"] == plan.get("selected_q256")
                        and r["actual_serialized_bytes"] == plan.get("target_bytes")]
            if len(selected) != 1:
                failures.append("An exact match has no actual encoded-byte observation")
        elif plan.get("status") != "no_exact_match_found" or not plan.get("measured"):
            failures.append("The finite byte search has no observations or classification")
    if report.get("skips"):
        failures.append("A matched baseline owner or timing cell failed")
    if args.plan_only:
        if report.get("cells"):
            failures.append("A byte-only plan must not claim timing cells")
        return not failures, failures
    expected = baseline_expected_keys(report, args)
    keys = [(r.get("target_t4_q256"), r.get("case_id"), r.get("kind"), r.get("m"),
             r.get("tp_size"), r.get("tp_rank"), r.get("cut_axis")) for r in report.get("cells", [])]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        failures.append("The matched T8 timing population is incomplete")
    for row in report.get("cells", []):
        if row.get("serving_owner") is not True or row.get("eager_ok") is not True:
            failures.append("An actual T8 serving result failed")
        if row.get("graph_equal") is not True:
            failures.append("A T8 graph result is missing or differs")
        for mode in ("eager", "graph"):
            if not row.get("timing", {}).get(mode, {}).get("samples_ms"):
                failures.append("A T8 raw timing stream is missing")
    return not failures, failures
