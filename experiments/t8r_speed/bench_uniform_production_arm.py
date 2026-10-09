"""One source-isolated production arm for the matched uniform D41 panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))


def digest_tensor(tensor):
    import torch
    value = tensor.detach().contiguous().view(torch.uint8).reshape(-1).cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(memoryview(value.numpy()))
    return digest.hexdigest()


def build_owner(arm, q256, device, *, experts, hidden, inter, seed):
    import torch
    from bench_rates import build_projection
    from tessera import routed_fused as rf
    from tessera.native_window_moe import PackedWindowMoeBundles
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    bundles, hashes = [], {}
    for role, rows, cols in ((0, inter, hidden), (1, inter, hidden), (2, hidden, inter)):
        p = build_projection(rf, experts, rows, cols, q256 // 256, 0,
                             seed + 100 * role, device, True)
        widths = torch.full((experts,), p["words"].shape[1], dtype=torch.int32, device=device)
        offsets = torch.arange(experts, dtype=torch.int32, device=device) * widths[0]
        native = torch.arange(256, dtype=torch.int32, device=device)
        native[(native & 127) == 127] -= 1
        native = native.to(torch.uint8).expand(experts, 256).contiguous()
        bundle = prepare_grouped_window_gemm_from_soa(
            words_all=p["words"].flatten(), table_all=torch.empty(0, dtype=torch.uint8, device=device),
            codes_all=p["table"], native_all=native, scale_all=p["scale"],
            runs_all=p["runs"][:, :4].contiguous(), init_all=p["init"], has_init=p["has_init"],
            word_off=offsets, tile_words=torch.full_like(widths, p["tile_words"]), total_words=widths,
            run_off=torch.arange(experts + 1, dtype=torch.int32, device=device), perm_all=p["perm"],
            rows=rows, cols=cols, experts=experts, window_bits=p["window_bits"],
            family="e4m3")
        for field in ("words_all", "codes_all", "native_all", "scale_all", "runs_all", "init_all",
                      "has_init", "word_off", "tile_words", "total_words", "run_off", "perm_all"):
            hashes[f"role{role}.{field}"] = digest_tensor(getattr(bundle, field))
        bundles.append(bundle)
    if arm == "master":
        packed = PackedWindowMoeBundles(*bundles, family="e4m3")
    else:
        classes = [{"start": 0, "end": experts, "q256": {"w13": [q256, q256], "w2": [q256]}}]
        packed = PackedWindowMoeBundles(*bundles, family="e4m3", expert_classes=classes)
    return packed, hashes


def inputs(tokens, experts, hidden, device, seed):
    import torch
    generator = torch.Generator(device=device).manual_seed(seed + tokens)
    x = (torch.randn(tokens, hidden, generator=generator, device=device) * .05).to(torch.bfloat16)
    base = torch.arange(tokens * 8, device=device).reshape(tokens, 8)
    routes = []
    for generation in (0, 1):
        ids = ((base + generation * 17) % experts).to(torch.int32)
        if generation:
            ids = ids.flip(1)
        weights = (base.remainder(11) + 1 + generation * 3).float()
        weights /= weights.sum(1, keepdim=True)
        routes.append((ids, weights))
    return x, routes


def capture(call):
    import torch
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def run(args):
    import torch
    from bench_t8r import summarize, time_events
    from tessera import routed_fused as rf
    from tessera.serving.glm53_shared_fold import native_call

    device = torch.device("cpu" if args.cpu_preflight else "cuda")
    experts, hidden, inter = (24, 256, 128) if args.cpu_preflight else (288, 4096, 1024)
    packed, planes = build_owner(args.arm, args.q256, device, experts=experts,
                                 hidden=hidden, inter=inter, seed=args.seed)
    tokens = 2 if args.cpu_preflight else args.M
    x, routes = inputs(tokens, experts, hidden, device, args.seed)
    input_hashes = {"x": digest_tensor(x)}
    for generation, (ids, weights) in enumerate(routes):
        input_hashes[f"ids{generation}"] = digest_tensor(ids)
        input_hashes[f"weights{generation}"] = digest_tensor(weights)
    source = Path(rf.__file__).resolve()
    result = {"arm": args.arm, "source_head": args.source_head, "q256": args.q256, "M": args.M,
              "timer": args.timer, "order": args.order, "admission": args.admission_label,
              "source_files": {str(source.parent / relative): hashlib.sha256((source.parent / relative).read_bytes()).hexdigest()
                               for relative in ("routed_fused.py", "native_window_moe.py", "window_gemm_grouped.py",
                                                "serving/native_window.py", "serving/glm53_shared_fold.py",
                                                "serving/csrc/routed_fused_window.cu")},
              "plane_hashes": planes, "input_hashes": input_hashes,
              "entry": "PackedWindowMoeBundles.adapter -> glm53_shared_fold.native_call -> FusedRoutedWindowMoE.__call__",
              "scope": "production adapter after the router and TP cut; external collective excluded",
              "input_copy_scope": "outside timed CUDA events", "rows": [], "start_unix": time.time()}
    if args.cpu_preflight:
        if not callable(rf.FusedRoutedWindowMoE.__call__) or packed.experts != experts:
            raise RuntimeError("the source has no complete production adapter entry")
        result.update(status="cpu_preflight_passed", gpu_results=0, tiny_geometry=[experts, hidden, inter])
        return result
    adapter = packed.adapter()
    if type(adapter).__name__ != "FusedRoutedWindowMoE" or adapter.library != "e4m3mma":
        raise RuntimeError("the declared uniform bank did not select the production E4M3 MMA adapter")
    packed = packed.native_owner()
    library_file = Path(rf._ext(adapter.library).__file__).resolve()
    result.update(library=adapter.library, launch_pair=list(adapter.launch_pair), device=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  native_library=str(library_file), native_library_sha256=hashlib.sha256(library_file.read_bytes()).hexdigest())
    ids, weights = (value.clone() for value in routes[0])
    inverse = torch.arange(experts, dtype=torch.int32, device=device) if args.arm == "pr" else None

    def call():
        # The PR serving boundary maps global IDs once. Master already serves identity IDs.
        stored = ids if inverse is None else inverse.index_select(0, ids.reshape(-1)).reshape_as(ids)
        return native_call(adapter, x, stored, weights, swiglu_limit=10.0,
                           apply_router_weight_on_input=False)

    expected = []
    for route in routes:
        ids.copy_(route[0]); weights.copy_(route[1])
        value = call()
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (args.M, hidden) or not torch.isfinite(value).all():
            raise AssertionError("the production entry returned invalid BF16 output")
        expected.append(value.detach().clone())
    ids.copy_(routes[0][0]); weights.copy_(routes[0][1])
    graph, graph_output = capture(call)
    for generation in (1, 0, 1, 0):
        ids.copy_(routes[generation][0]); weights.copy_(routes[generation][1]); graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(graph_output.view(torch.int16), expected[generation].view(torch.int16)):
            raise AssertionError("the production graph changed its eager BF16 bits after routing replay")
    for generation in (0, 1):
        ids.copy_(routes[generation][0]); weights.copy_(routes[generation][1])
        samples = time_events(graph.replay if args.timer == "graph" else call, args.warmup, args.iters)
        result["rows"].append({"generation": generation, "wall": summarize(samples),
                               "output_sha256": digest_tensor(expected[generation]),
                               "eager_graph_bitwise": True, "admission": args.admission_label})
    result.update(status="complete", end_unix=time.time(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=("master", "pr"), required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--q256", type=int, choices=(512, 768, 1024), required=True)
    parser.add_argument("--M", type=int, choices=(1, 2048), required=True)
    parser.add_argument("--timer", choices=("eager", "graph"), required=True)
    parser.add_argument("--order", choices=("F", "R"), required=True)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--seed", type=int, default=6141)
    parser.add_argument("--admission-label", default="GPU-exclusive, not quiet-host certified")
    args = parser.parse_args()
    if args.iters < 2 or args.warmup < 0:
        parser.error("the timer needs at least two samples and a nonnegative warmup")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = run(args)
    (out / "uniform_production_arm.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"arm": args.arm, "q256": args.q256, "M": args.M, "timer": args.timer, "status": result["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
