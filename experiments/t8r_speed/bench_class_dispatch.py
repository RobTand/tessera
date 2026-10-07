"""Production class-dispatch measurements at the GLM TP2 rank-local shapes.

Run through bench_t8r.sh with BENCH_PY=bench_class_dispatch.py. The same entry
accepts --cpu-preflight for imports, arguments, metadata and small packed reads.
The parent owns GPU admission, submission, power coverage and table publication.

Gate/up uses the native mode-zero SwiGLU epilogue. Down uses original flat-route
inputs and emits weighted BF16 route rows. Both include inverse gather, device
routing and activation quantization. The full companion calls the opaque
Tessera operator and also includes intermediate quantization and token reduction.
Input copies are outside timing. Ids and weights alternate on every replay.
Heterogeneous latency references are declared interpolations of measured pure
whole-stack controls, not a nonexistent heterogeneous single launch.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_geometry import geometry, kernel_usage, resource_usage
from bench_rates import Clock, EXPERTS, HIDDEN, INTER, TOP_K, sha
from bench_t8r import PowerSampler, kernel_profile, summarize
from class_dispatch_inputs import (SCHEDULES, Invocation, activations, packed_constants,
                                   routing_generations, stitched_references)
from tessera import routed_fused as rf
from tessera.expert_classes import inverse_expert_ids

KINDS = ("gate_up", "down", "full")


def source_evidence():
    package = Path(rf.__file__).resolve().parent
    sources = [package / name for name in ("routed_fused.py", "routed_class_dispatch.py", "native_window_moe.py",
        "expert_classes.py", "serving/native_window.py", "serving/csrc/routed_fused_window.cu")]
    sources += [Path(__file__).resolve(), Path(__file__).with_name("class_dispatch_inputs.py")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}


class Report:
    def __init__(self, args):
        self.path = Path(args.out) / "bench_class_dispatch.json"
        self.data = {"schema": "tessera.d41.production_classes.v1", "meta": {
            "start_unix": time.time(), "arguments": vars(args).copy(), "source_sha256": source_evidence(),
            "tessera_head": os.environ.get("TESSERA_HEAD"), "image": os.environ.get("ORACLE_IMAGE"),
            "host": os.environ.get("HOST_NAME", os.uname().nodename),
            "pb_action": os.environ.get("PB_ACTION_KEY", os.environ.get("PRISMABUILD_ACTION_KEY")),
            "admission": args.admission_label,
            "torch": torch.__version__, "shape": {"hidden": HIDDEN, "inter": INTER,
                "experts": EXPERTS, "top_k": TOP_K, "tensor_parallel_size": 2},
            "claim_scope": "synthetic feature proof and measurements only; no artifact or quality qualification",
            "timing_scope": "inverse gather, device routing and quantization plus production dispatch; input copies excluded",
            "full_companion": "tessera::routed_window_classes; inverse gather, both quantizers, SwiGLU, down and fixed-order token reduction",
            "projection_omissions": {"gate_up": "down, intermediate quantization and token reduction",
                                      "down": "gate/up, SwiGLU and token reduction"},
            "fixed_policy": "native SM-count grids; native BM selection; two production streams; no tuning",
            "orders": {"F": "pure controls then production classes", "R": "production classes then reverse pure controls"},
            "statistic": "mean of F/R medians; spread is absolute F/R difference divided by that mean",
            "comparator": "route-count-weighted pure whole-stack interpolation; not a measured mixed single launch",
            "failures": [], "status": "started"}, "groups": {}}
        self.save()

    def save(self):
        self.data["meta"]["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(json.dumps(self.data, indent=1) + "\n")
        temporary.replace(self.path)

    def fail(self, error):
        self.data["meta"]["status"] = "failed"
        self.data["meta"]["failures"].append({"unix": time.time(), "error": repr(error),
                                                "traceback": traceback.format_exc()})
        self.save()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--ms", default="1,16,2048,4096")
    parser.add_argument("--schedules", default="q3_q4,three", help="fixed named schedules, not rate tuning")
    parser.add_argument("--projections", default=",".join(KINDS))
    parser.add_argument("--timers", default="graph,eager")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--prof-reps", type=int, default=1)
    parser.add_argument("--power-s", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=6141)
    parser.add_argument("--config", default="", help="optional GLM config; read on CPU and GPU")
    parser.add_argument("--ncu", action="store_true", help="outside-timer profiler leg through BENCH_NCU=1")
    parser.add_argument("--admission-label", default="not declared", help="actual admission scope recorded on every timing row")
    args = parser.parse_args()
    ms = [int(m) for m in args.ms.split(",")]
    if not ms or len(set(ms)) != len(ms) or any(m not in (1, 16, 2048, 4096) for m in ms):
        parser.error("M must be distinct members of 1,16,2048,4096")
    for field, choices in (("schedules", SCHEDULES), ("projections", KINDS), ("timers", ("graph", "eager"))):
        values = getattr(args, field).split(",")
        if not values or len(set(values)) != len(values) or any(v not in choices for v in values):
            parser.error(f"invalid {field} population")
    if args.iters < 2 or args.warmup < 0 or args.prof_reps < 1 or args.power_s <= 0:
        parser.error("both routing generations need at least two samples; profiler and power legs must be positive")
    return args


def cpu_preflight(args, report):
    # Exercise imports used only by the GPU legs, without touching CUDA.
    from torch.profiler import ProfilerActivity, profile  # noqa: F401
    from tessera.serving.native_window import _routed_window_classes  # noqa: F401
    from class_dispatch_inputs import PureControlWorkspace
    workspace = PureControlWorkspace(torch.device("cpu"))
    counters = workspace.counters
    counter_pointer = counters.data_ptr()
    if counters.dtype != torch.int32 or counters.shape != (2,) or bool(counters.count_nonzero()):
        raise AssertionError("pure control counters must initialize once as zero int32[2]")
    workspace_checks = []
    for mode in (0, 2, 0, 2, 0, 2):
        counters.fill_(37)
        projection = 1 if mode == 2 else 0
        slot, empty = workspace.launch_seat(mode)
        if workspace.counters is not counters or counters.data_ptr() != counter_pointer:
            raise AssertionError("pure control replaced its persistent counter workspace")
        if slot.data_ptr() != counter_pointer + projection * counters.element_size():
            raise AssertionError("pure control counter seat must alias its persistent projection slot")
        if slot.shape != (1,) or slot.dtype != torch.int32 or int(slot[0]) != 0:
            raise AssertionError("pure control must zero the selected counter on every launch")
        if int(counters[1 - projection]) != 37:
            raise AssertionError("pure control reset another projection counter")
        if empty.shape != (0,) or empty.dtype != torch.float32 or empty.device != counters.device:
            raise AssertionError("old pure launch empty-scale fallback differs")
        slot.fill_(101)  # Dirty state from a previous launch must reset next time.
        workspace_checks.append({"mode": mode, "projection": projection, "persistent_storage": True,
                                 "selected_counter_reset": True, "other_counter_unchanged": True})
    report.data["meta"]["pure_control_workspace_preflight"] = {"checks": workspace_checks,
        "scope": "CPU counter allocation, alias lifetime and per-launch initialization only; no GPU result",
        "empty_scale_fallback": "old adapter new_zeros(0, dtype=float32) on every launch"}
    library = rf.library_for("e4m3")
    if library != "e4m3mma" or rf.PAIRED_K32_BUILD or rf.MMA8_A_RING:
        raise ValueError("this entry uses the current unpaired E4M3 decoder without tuning overrides")
    for q in (512, 768, 1024):
        for mode in (0, 2):
            geometry(rf, q // 256, None, mode, True)
            rf.launch_smem_bytes(mode, rf.slot_words_for_rate(q // 256), mma8=True, bm=rf.BM)
    if args.config:
        config = json.loads(Path(args.config).read_text())
        text = config.get("text_config", config)
        observed = (text["hidden_size"], text["moe_intermediate_size"] // 2,
                    text["n_routed_experts"], text["num_experts_per_tok"])
        if observed != (HIDDEN, INTER, EXPERTS, TOP_K):
            raise ValueError(f"config does not describe the requested GLM TP2 shapes: {observed}")
        report.data["meta"]["config_read"] = {"path": args.config,
            "sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
    tiny = []
    pure_rates = sorted({q for name in args.schedules.split(",") for q in SCHEDULES[name]})
    for rates in [SCHEDULES[name] for name in args.schedules.split(",")] + [(q,) for q in pure_rates]:
        packed, meta, inverse = packed_constants(rates, experts=24, hidden=256, inter=128,
                                                 device=torch.device("cpu"), seed=args.seed)
        if inverse.dtype != torch.int32 or inverse.numel() != 24:
            raise ValueError("inverse gather metadata is not int32[E]")
        ids = torch.tensor(meta["expert_ids"])
        if not torch.equal(ids.index_select(0, inverse.long()), torch.arange(24)):
            raise ValueError("map and inverse disagree")
        reads = []
        for role in (packed.gate, packed.up, packed.down):
            if role.words_all.numel() != int(role.total_words.sum()):
                raise ValueError("flat packed extent disagrees with metadata")
            for desc in meta["expert_classes"]:
                view = rf.grouped_class_view(role, desc["start"], desc["end"])
                if view.words_all.untyped_storage().data_ptr() != role.words_all.untyped_storage().data_ptr():
                    raise ValueError("class view copied packed words")
            reads.append({name: {"shape": list(getattr(role, name).shape),
                "dtype": str(getattr(role, name).dtype),
                "sample": getattr(role, name).flatten()[:4].tolist()}
                for name in ("words_all", "codes_all", "native_all", "runs_all", "word_off",
                             "run_off", "tile_words", "total_words", "init_all", "has_init", "scale_all", "perm_all")})
        generations, counts = routing_generations(meta, 2, device=torch.device("cpu"))
        if torch.equal(generations[0][0], generations[1][0]) or torch.equal(generations[0][1], generations[1][1]):
            raise ValueError("replay ids and weights must change")
        for route, weights in generations:
            storage = inverse.index_select(0, route.flatten().long()).view_as(route)
            widths = tuple(dict.fromkeys(rf.superblock_rows(rf.library_for("e4m3"), mode, route.shape[0])
                                         for mode in (0, 1, 2)))
            rf._routing_tables(storage, weights, packed.experts, torch.device("cpu"), widths)
        x, down_x = activations(2, 256, 128, torch.device("cpu"), args.seed)
        if x.shape != (2, 256) or down_x.shape != (16, 128):
            raise ValueError("small activation shape differs")
        tiny.append({"rates": list(rates), "metadata": meta, "reads": reads, "generations": counts})
    report.data["meta"].update(status="cpu_preflight_passed", cpu_preflight=True,
        cuda_timing=False, gpu_results=0, tiny_packed_reads=tiny, end_unix=time.time())
    report.save()
    print(json.dumps({"cpu_preflight": "passed", "gpu_results": 0, "report": str(report.path),
                      "peak_host_rss_bytes": report.data["meta"]["peak_host_rss_bytes"]}), flush=True)


def assert_bits(actual, expected, label):
    if actual.dtype != torch.bfloat16 or actual.shape != expected.shape:
        raise AssertionError(f"{label}: BF16 shape or dtype differs")
    if not torch.isfinite(actual).all() or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        raise AssertionError(f"{label}: BF16 bits differ from saved pure-template output")


def capture(invocation):
    for _ in range(2):
        invocation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invocation()
    return graph


def check_case(args, report, name, rates, tokens, *, route_meta=None):
    packed, meta, inverse = packed_constants(rates, experts=EXPERTS, hidden=HIDDEN, inter=INTER,
                                             device=torch.device("cuda"), seed=args.seed)
    adapter = packed.adapter()
    if route_meta is not None:
        inverse = torch.tensor(inverse_expert_ids(route_meta["expert_ids"]), dtype=torch.int32, device=adapter.device)
    generations, populations = routing_generations(route_meta or meta, tokens, device=adapter.device)
    x, down_x = activations(tokens, HIDDEN, INTER, adapter.device, args.seed)
    references = []
    for ids, weights in generations:
        storage = inverse.index_select(0, ids.flatten().long()).view_as(ids)
        outputs = stitched_references(adapter, x, down_x, storage, weights)
        if any(not bool(torch.isfinite(out).all()) or not bool(out.count_nonzero()) for out in outputs.values()):
            raise AssertionError("pure reference must contain finite nonzero synthetic results")
        references.append({kind: out.cpu() for kind, out in outputs.items()})
    reference_path = Path(args.out) / f"reference-{name}-M{tokens}.pt"
    torch.save({"reference_kind": "stitched class-local old pure-template BF16 outputs",
        "metadata": meta, "populations": populations, "seed": args.seed,
        "routing": [(ids.cpu(), weights.cpu()) for ids, weights in generations],
        "outputs": references}, reference_path)
    group = report.data["groups"].setdefault(f"{name}:M{tokens}", {"name": name, "M": tokens,
        "rates": list(rates), "metadata": meta, "routing_generations": populations,
        "reference_path": str(reference_path), "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "correctness": [], "measurements": {}})
    report.save()
    # Read the saved bits, not a fresh expected output from the candidate path.
    saved = torch.load(reference_path, map_location="cpu", weights_only=True)["outputs"]
    for order in (tuple(range(len(adapter.classes))), tuple(reversed(range(len(adapter.classes))))):
        selected = dataclasses.replace(adapter, class_issue_order=order)
        for kind in args.projections.split(","):
            invocation = Invocation(selected, inverse, x, down_x, generations, kind)
            for generation in (0, 1):
                invocation.select(generation)
                assert_bits(invocation(), saved[generation][kind].to(adapter.device), f"{name}/{tokens}/{kind}/eager/{generation}")
            invocation.select(0)
            graph = capture(invocation)
            for generation in (1, 0, 1, 0):
                invocation.select(generation)
                graph.replay()
                torch.cuda.synchronize()
                assert_bits(invocation.out, saved[generation][kind].to(adapter.device),
                            f"{name}/{tokens}/{kind}/graph/{generation}")
            del graph
            group["correctness"].append({"kind": kind, "class_issue_order": list(order),
                "eager_generations": [0, 1], "captured_replays": [1, 0, 1, 0], "bitwise": True})
            if len(rates) == 1:
                if meta["expert_ids"] != list(range(EXPERTS)):
                    raise AssertionError("uniform control must have an identity expert map")
                pure = Invocation(adapter, inverse, x, down_x, generations, kind, pure=True)
                for generation in (0, 1):
                    pure.select(generation)
                    assert_bits(pure(), saved[generation][kind].to(adapter.device), "uniform old pure eager")
                pure.select(0)
                pure_graph = capture(pure)
                for generation in (1, 0):
                    pure.select(generation)
                    pure_graph.replay()
                    torch.cuda.synchronize()
                    assert_bits(pure.out, saved[generation][kind].to(adapter.device), "uniform old pure captured")
                del pure_graph
                group["uniform_identity_old_pure_bitwise" if route_meta is None else "paired_population_old_pure_bitwise"] = True
            report.save()
    del invocation, adapter, packed
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def class_geometry(adapter, tokens, usage):
    props = torch.cuda.get_device_properties(adapter.device)
    result = []
    for index, cls in enumerate(adapter.classes):
        projections = {}
        for kind, mode, suffix, tile, slot in (
            ("gate_up", 0, "gate", cls.tile_words_gate_up, cls.slot_words_gate_up),
            ("down", 2, "down", cls.tile_words_down, cls.slot_words_down)):
            q = adapter.expert_classes[index]["q256"]["w2" if mode == 2 else "w13"][0]
            bm = rf.superblock_rows(adapter.library, mode, tokens)
            geometry_row = geometry(rf, q // 256, None, mode, rf.library_mma8(adapter.library))
            dynamic = rf.launch_smem_bytes(mode, slot, mma8=rf.library_mma8(adapter.library), bm=bm)
            compiled = kernel_usage(usage, mode, False, q // 256, False, bm, fp8=True)
            if compiled is None:
                raise RuntimeError(f"current library resource usage lacks mode={mode}, q={q}, BM={bm}")
            table = getattr(cls, "table_" + suffix)
            bundle = getattr(cls, suffix)
            geometry_row.update(q256=q, bits_per_256_weight_tile=q, packed_512_row_tile_words=tile,
                packed_512_row_tile_bits=32 * tile, rows=bundle.rows, cols=bundle.cols,
                BM=bm, slot_words=slot, smem_bytes=dynamic,
                static_shared_bytes=compiled.get("SHARED"),
                shared_fit=dynamic + compiled.get("SHARED", 0) <= props.shared_memory_per_block_optin,
                window_bits=rf.WINDOW_BITS, table_entries=table.shape[1],
                table_entry_bits=8 * table.element_size(), table_bytes_per_expert=table.shape[1] * table.element_size(),
                word_pointer_alignment_mod_16=getattr(cls, "words_" + suffix).data_ptr() % 16,
                word_stride_alignment_mod_16=getattr(cls, "words_" + suffix).stride(0) * 4 % 16,
                usage=compiled, grid_ctas=props.multi_processor_count,
                threads=rf._ext(adapter.library).THREADS, decoder="routed_fused_kernel; pure whole-bit; legacy word layout; unpaired")
            projections[kind] = geometry_row
        result.append({"class": index, "start": cls.start, "end": cls.end,
                       "stream_assignment": index % 2, "projections": projections})
    return result


def timed_samples(invocation, timer, args):
    invocation.select(0)
    graph = capture(invocation) if timer == "graph" else None
    call = graph.replay if graph is not None else invocation
    # Consecutive replay seats alternate generations, including warmups.
    generation_index = 0
    for _ in range(args.warmup):
        generation_index ^= 1
        invocation.select(generation_index)
        call()
    torch.cuda.synchronize()
    samples, generations = [], []
    start_unix = time.time()
    for _ in range(args.iters):
        generation_index ^= 1
        invocation.select(generation_index)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        call()
        b.record()
        b.synchronize()
        samples.append(float(a.elapsed_time(b)))
        generations.append(generation_index)
    result = {"timer": timer, "wall": summarize(samples), "generation_per_sample": generations,
        "by_generation": {str(g): summarize([s for s, actual in zip(samples, generations) if actual == g]) for g in (0, 1)},
        "window_unix": [start_unix, time.time()], "input_copy_scope": "outside start/end CUDA events"}
    del graph
    return result


def measure(args, report, name, rates, tokens, timer, order, arm, usage, power, clock):
    packed, own_meta, own_inverse = packed_constants(rates, experts=EXPERTS, hidden=HIDDEN, inter=INTER,
                                                     device=torch.device("cuda"), seed=args.seed)
    adapter = packed.adapter()
    from class_dispatch_inputs import metadata
    route_meta = metadata(SCHEDULES[name], EXPERTS)
    inverse = torch.tensor(inverse_expert_ids(route_meta["expert_ids"]), dtype=torch.int32, device=adapter.device)
    generations, _ = routing_generations(route_meta, tokens, device=adapter.device)
    x, down_x = activations(tokens, HIDDEN, INTER, adapter.device, args.seed)
    group = report.data["groups"][f"{name}:M{tokens}"]
    if arm == "production":
        group["geometry"] = class_geometry(adapter, tokens, usage)
        group["resident_bytes"] = packed.resident_bytes() + inverse.numel() * inverse.element_size()
    for kind in args.projections.split(","):
        invocation = Invocation(adapter, inverse, x, down_x, generations, kind, pure=arm != "production")
        # Class issue order remains production's fixed order. F/R is arm order.
        row = {"arm": arm, "kind": kind, "order": order, "clock_start": clock.read(), "start_unix": time.time(),
            "library": adapter.library, "launch_pair": adapter.launch_pair,
            "admission": args.admission_label,
            "route_metadata": route_meta, "packed_metadata": own_meta,
            "map_gather": "same original-to-storage gather for every paired arm",
            "entry": "tessera::routed_window_classes" if kind == "full" and arm == "production" else
                     "FusedRoutedWindowMoE._launch -> dispatch_class_projection" if arm == "production" else
                     "old pure whole-stack template control",
            "status": "started"}
        key = f"{kind}:{timer}:{order}:{arm}"
        group["measurements"][key] = row
        report.save()
        row.update(timed_samples(invocation, timer, args))
        row.update(clock_end=clock.read(), end_unix=time.time(), status="timed")
        report.save()
        # Profile and power are separate legs. They never enter event samples.
        invocation.select(0)
        graph = capture(invocation) if timer == "graph" else None
        work = graph.replay if graph is not None else invocation
        sequence = 0
        def observed_call():
            nonlocal sequence
            sequence ^= 1
            invocation.select(sequence)
            work()
        trace = Path(args.out) / f"trace-{name}-M{tokens}-{kind}-{timer}-{order}-{arm}.json"
        row["profile"] = kernel_profile(observed_call, reps=args.prof_reps, full_names=True, trace_path=trace)
        row["profile"].update(trace_path=str(trace), outside_timing=True,
            scope="input copies included in diagnostic trace; excluded from timed CUDA events")
        row["power"] = power.sample_during(observed_call, args.power_s, capture_series=True)
        row["power"].update(outside_timing=True, both_spark_netdata="parent harvest required; no energy claim")
        row["status"] = "complete"
        report.save()
        if args.ncu:
            torch.cuda.profiler.start()
            observed_call()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        del graph, invocation
    del adapter, packed, x, down_x, own_inverse
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def combine(report):
    for group in report.data["groups"].values():
        if group["name"] not in SCHEDULES:
            continue
        rows = group["measurements"]
        summaries = {}
        for kind in KINDS:
            for timer in ("graph", "eager"):
                key = f"{kind}:{timer}"
                required = [f"{key}:{order}:production" for order in ("F", "R")]
                if not all(k in rows for k in required):
                    continue
                passes = {order: rows[f"{key}:{order}:production"]["wall"]["median_ms"] for order in ("F", "R")}
                mean = sum(passes.values()) / 2
                summary = {"median_ms": mean, "F_median_ms": passes["F"], "R_median_ms": passes["R"],
                           "spread": abs(passes["F"] - passes["R"]) / mean,
                           "comparator_kind": "interpolation of pure whole-stack controls, not a mixed single launch",
                           "by_generation": {}}
                summary["admission"] = report.data["meta"]["admission"]
                for generation in (0, 1):
                    counts = group["routing_generations"][generation]["class_routes"]
                    order_values = {}
                    for order in ("F", "R"):
                        production = rows[f"{key}:{order}:production"]["by_generation"][str(generation)]["median_ms"]
                        pure = [rows[f"{key}:{order}:pure_q{q}"]["by_generation"][str(generation)]["median_ms"] for q in group["rates"]]
                        comparator = sum(n * ms for n, ms in zip(counts, pure)) / sum(counts)
                        order_values[order] = {"production_ms": production, "pure_control_ms": pure,
                                              "interpolated_ms": comparator, "ratio_to_interpolation": production / comparator}
                    summary["by_generation"][str(generation)] = order_values
                summaries[key] = summary
        group["summary"] = summaries
    report.save()


def run_gpu(args, report):
    if args.config:
        # The same real input read and shape check applies to GPU execution.
        text = json.loads(Path(args.config).read_text())
        text = text.get("text_config", text)
        if (text["hidden_size"], text["moe_intermediate_size"] // 2,
            text["n_routed_experts"], text["num_experts_per_tok"]) != (HIDDEN, INTER, EXPERTS, TOP_K):
            raise ValueError("config does not describe the requested GLM TP2 shapes")
    library = rf.library_for("e4m3")
    if library != "e4m3mma" or rf.PAIRED_K32_BUILD or rf.MMA8_A_RING:
        raise ValueError("this fixed entry measures the current unpaired E4M3 decoder, without tuning overrides")
    lib = rf._ext(library)
    usage = resource_usage(lib)
    if not usage.get("kernels"):
        raise RuntimeError(f"current library resource usage is unavailable: {usage}")
    props = torch.cuda.get_device_properties(0)
    report.data["meta"].update(library=library, library_path=lib.__file__,
        library_sha256=hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest(), resource_usage=usage,
        device=torch.cuda.get_device_name(), architecture=f"sm_{props.major}{props.minor}",
        sms=props.multi_processor_count, shared_memory_available=props.shared_memory_per_block_optin,
        device_total_memory=props.total_memory, cuda_runtime=torch.version.cuda,
        status="correctness_before_all_timing")
    report.save()
    ms = [int(m) for m in args.ms.split(",")]
    schedules = args.schedules.split(",")
    pure_rates = sorted({q for name in schedules for q in SCHEDULES[name]})
    for name, rates in [(f"uniform_q{q}", (q,)) for q in pure_rates] + [(name, SCHEDULES[name]) for name in schedules]:
        for tokens in ms:
            check_case(args, report, name, rates, tokens)
    from class_dispatch_inputs import metadata
    for name in schedules:
        route_meta = metadata(SCHEDULES[name], EXPERTS)
        for q in SCHEDULES[name]:
            for tokens in ms:
                check_case(args, report, f"{name}-pure_q{q}", (q,), tokens, route_meta=route_meta)
    report.data["meta"].update(status="all_correctness_passed_before_timing", correctness_end_unix=time.time())
    report.save()
    power, clock = PowerSampler(), Clock()
    for name in schedules:
        for tokens in ms:
            for timer in args.timers.split(","):
                controls = [(f"pure_q{q}", (q,)) for q in SCHEDULES[name]]
                for order in ("F", "R"):
                    arms = controls + [("production", SCHEDULES[name])] if order == "F" else [("production", SCHEDULES[name])] + list(reversed(controls))
                    for arm, rates in arms:
                        measure(args, report, name, rates, tokens, timer, order, arm, usage, power, clock)
                    combine(report)
    report.data["meta"].update(status="complete", end_unix=time.time(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        energy_status="held for parent review of both-Spark Netdata coverage")
    report.save()


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    report = Report(args)
    try:
        if args.cpu_preflight:
            cpu_preflight(args, report)
        else:
            run_gpu(args, report)
    except Exception as error:
        report.fail(error)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
