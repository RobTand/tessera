#!/usr/bin/env python3
"""Finite actual-vLLM TP2 SP tile screen; root owns the paired GPU window.

Run one explicitly configured rank on each host. This file neither launches
the peer nor places work. It loads only layer-1 mHC tensors, uses the stock
vLLM distributed context and communicator, and writes a rank-owned receipt.
No model construction or serving-default change occurs here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    ap.add_argument("--init-method", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 2048, 2049])
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--iterations", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--timeout-seconds", type=int, default=90)
    args = ap.parse_args(argv)
    parsed = urlsplit(args.init_method)
    if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port or parsed.path:
        ap.error("init-method must be an explicit tcp://host:port")
    if len(args.tokens) != len(set(args.tokens)) or not set(args.tokens) <= {512, 2048, 2049}:
        ap.error("tokens must be distinct members of the finite 512/2048/2049 protocol")
    if args.tile != 1024 or not 1 <= args.iterations <= 64 or not 1 <= args.pairs <= 2:
        ap.error("protocol requires tile1024, iterations1..64 and pairs1..2")
    if not 1 <= args.timeout_seconds <= 180:
        ap.error("timeout-seconds must be 1..180")
    return args


def _sha(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(-1).view(__import__("torch").uint8)
                          .cpu().numpy().tobytes()).hexdigest()


def _publish(path, record):
    # Each case is durable before advancing to the next one; an interrupted
    # run retains completed cases and never advertises a full screen pass.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    import torch
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import get_tp_group
    from experiments.bench_native_moe_operator import native_runtime_context
    import mhc_probe as probe
    import routed_pair_oracle as rpo
    from tessera.serving import glm53_prefill as gp
    from tessera.serving.sp_tile_overlap import (SiteOutputs, SpTilePlan,
        inspected_communication, make_pynccl_runner, pynccl_sp_decline, stock_post_pre_into)

    args.out.mkdir(parents=True, exist_ok=False)
    result_path = args.out / "result.json"
    modules, why = gp._import_all()
    if modules is None or gp._match_interface(modules)[0] is None:
        raise RuntimeError("stock SP interface unqualified: " + why)
    comm_modules, why = inspected_communication()
    if comm_modules is None:
        raise RuntimeError("stock communication interface unqualified: " + why)
    kernels, tilelang, deep_gemm = modules[5], modules[6], modules[8]
    forcer = gp.install_split_forcer(kernels)
    rendezvous = urlsplit(args.init_method)
    config = EngineArgs(model=str(args.model), load_format="dummy", enforce_eager=True,
        max_model_len=8448, trust_remote_code=True, tensor_parallel_size=2,
        disable_custom_all_reduce=True, distributed_executor_backend="mp", nnodes=2,
        language_model_only=True, node_rank=args.rank, master_addr=rendezvous.hostname,
        master_port=rendezvous.port).create_engine_config()
    config.parallel_config.distributed_timeout_seconds = args.timeout_seconds
    config.parallel_config.cpu_distributed_timeout_seconds = args.timeout_seconds
    source_paths = [Path(__file__), ROOT.parent / "src/tessera/serving/sp_tile_overlap.py",
        ROOT.parent / "src/tessera/serving/glm53_prefill.py",
        ROOT.parent / "src/tessera/serving/stock_interface.py",
        ROOT / "mhc/mhc_probe.py", ROOT / "routed_pair_oracle.py",
        ROOT / "bench_native_moe_operator.py", ROOT / "bench_native_operator.py"]
    source = [_sha(path) for path in source_paths]
    common = {"tokens": args.tokens, "tile": args.tile, "iterations": args.iterations,
              "pairs": args.pairs, "init_method": args.init_method,
              "source": [{k: v for k, v in row.items() if k != "path"} for row in source],
            "model_config": _sha(args.model / "config.json")["sha256"],
            "model_index": _sha(args.model / "model.safetensors.index.json")["sha256"]}
    common_sha = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    record = {"schema": "tessera.sp_tile_probe.v1", "status": "incomplete",
              "rank": args.rank, "host": socket.gethostname(), "common_request": common,
              "common_request_sha256": common_sha, "source": source,
              "torch": torch.__version__, "vllm": vllm.__version__,
              "image_declaration": os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE"),
              "image_resolution_declaration": os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE_DECLARATION"),
              "environment": {name: value for name, value in os.environ.items()
                              if name.startswith(("NCCL_", "GLOO_")) or name == "VLLM_HOST_IP"},
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "energy_status": "HOLD: raw power and Netdata retained; no work/J qualification",
              "cases": []}
    _publish(result_path, record)
    distributed = {"world_size": 2, "rank": args.rank, "init_method": args.init_method,
                   "timeout_seconds": args.timeout_seconds}
    sampler = rpo.PowerSampler(hz=10)
    sampler.start()
    try:
        with native_runtime_context(config, distributed=distributed):
            group = get_tp_group()
            identity = torch.tensor(list(bytes.fromhex(common_sha)), device="cuda", dtype=torch.uint8)
            both = group.all_gather(identity, dim=0)
            if not torch.equal(both[:32], both[32:]):
                raise RuntimeError("TP ranks have different source/protocol requests")
            dc = group.device_communicator
            why = pynccl_sp_decline(dc,
                symmetric_ag_rs=comm_modules[1].should_nccl_symm_mem_ag_rs(),
                rocm=torch.version.hip is not None)
            if why:
                raise RuntimeError("stock SP route declined: " + why)
            nccl = dc.pynccl_comm
            record["communication"] = {"owner": type(dc).__qualname__,
                "pynccl_owner": type(nccl).__qualname__, "world_size": nccl.world_size,
                "rank": nccl.rank, "device": str(nccl.device), "nccl_version": nccl.nccl_version,
                "route": "inspected stock dim-0 SP PyNccl; one side stream"}
            runner = make_pynccl_runner(torch, nccl, forcer.full_batch)
            k = SimpleNamespace(torch=torch, post=kernels._MHC_POST_TILELANG_KERNEL,
                                gemm=tilelang._hc_prenorm_gemm_outputs,
                                pre=kernels._MHC_PRE_BIG_FUSE_TILELANG_KERNEL)
            def barrier():
                torch.distributed.barrier(group=group.cpu_group)

            for which in ("attn", "ffn"):
                prm = probe.mhc_params(args.model, which)
                call = probe.mhc_call(prm)
                adapter = stock_post_pre_into(k, fn=prm["fn"], scale=prm["scale"], base=prm["base"],
                    rms_eps=probe.RMS_EPS, hc_pre_eps=probe.HC_EPS, hc_sinkhorn_eps=probe.HC_EPS,
                    hc_post_mult_value=probe.POST_MULT, sinkhorn_repeat=probe.SINKHORN,
                    norm_weight=prm["norm"], norm_eps=probe.RMS_EPS)
                for tokens in args.tokens:
                    if not gp.shard_split_exact(kernels, deep_gemm, 2, tokens, probe.HIDDEN, probe.HC):
                        raise RuntimeError(f"stock split-k path unavailable at {tokens}")
                    plan = SpTilePlan(tokens, args.tile)
                    gen = torch.Generator(device="cuda").manual_seed(7000 + tokens)
                    _, residual, post, comb = probe.mhc_inputs(tokens, gen, call)
                    common_inputs = [_tensor_sha(t) for t in (residual, post, comb)]
                    input_id = hashlib.sha256(json.dumps(common_inputs).encode()).digest()
                    both_inputs = group.all_gather(torch.tensor(list(input_id), device="cuda",
                                                                dtype=torch.uint8), dim=0)
                    if not torch.equal(both_inputs[:32], both_inputs[32:]):
                        raise RuntimeError("TP ranks have different full mHC state")
                    gen.manual_seed(17000 + tokens + args.rank)
                    x = torch.randn(tokens, probe.HIDDEN, device="cuda", generator=gen).bfloat16()
                    # Stock SP has contiguous rank halves; candidate owns each tile's rank half.
                    stock_state = [modules[3].sp_shard(t).contiguous() for t in (residual, post, comb)]
                    tile_state = [plan.shard(t, args.rank, torch) for t in (residual, post, comb)]
                    def serialized():
                        reduced = modules[3].sp_reduce_scatter(x)
                        with forcer.full_batch(tokens):
                            outputs = call(reduced, *stock_state)
                        gathered = modules[3].sp_all_gather(outputs[3])[:tokens]
                        return SiteOutputs(*outputs[:3], gathered)
                    def overlap():
                        return runner.run(plan, x, *tile_state, adapter)

                    reference, candidate = serialized(), overlap()
                    torch.cuda.synchronize()
                    equal = {"layer_input": torch.equal(reference.layer_input, candidate.layer_input)}
                    for field in ("residual", "post", "comb"):
                        full_ref = modules[3].sp_all_gather(getattr(reference, field))[:tokens]
                        full_candidate = runner.gather(plan, getattr(candidate, field))
                        torch.cuda.synchronize()
                        equal[field] = torch.equal(full_ref, full_candidate)
                    identities = torch.arange(tokens, device="cuda", dtype=torch.float32).unsqueeze(1)
                    rebuilt = runner.gather(plan, plan.shard(identities, args.rank, torch))
                    torch.cuda.synchronize()
                    equal["token_identity"] = torch.equal(rebuilt, identities)
                    if not all(equal.values()):
                        raise RuntimeError(f"TP2 exactness failed at {which}/{tokens}: {equal}")
                    # Wrong-layout control retains valid collective ordering, but must
                    # disagree once there is more than one global tile.
                    mutant = None
                    if tokens > args.tile:
                        wrong = runner.run(plan, x, *stock_state, adapter)
                        torch.cuda.synchronize()
                        mutant = not torch.equal(wrong.layer_input, reference.layer_input)
                        if not mutant:
                            raise RuntimeError("contiguous-shard mutant had no output witness")
                    del reference, candidate
                    for _ in range(3):
                        serialized(); overlap()
                    torch.cuda.synchronize()
                    timings = []
                    for pair in range(args.pairs):
                        for arm in ("serialized", "overlap", "overlap", "serialized"):
                            barrier()
                            fn = serialized if arm == "serialized" else overlap
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            t0 = time.time()
                            start.record()
                            for _ in range(args.iterations): fn()
                            end.record()
                            end.synchronize()
                            t1 = time.time()
                            timings.append({"pair": pair, "arm": arm, "utc_window": [t0, t1],
                                "iterations": args.iterations,
                                "ms_per_site": start.elapsed_time(end) / args.iterations,
                                "host_ms_per_site": (t1 - t0) * 1000 / args.iterations,
                                "power": sampler.window(t0, t1)})
                    profiles = []
                    for arm, fn in (("serialized", serialized), ("overlap", overlap)):
                        barrier()
                        path = args.out / f"{which}-{tokens}-{arm}.trace.json"
                        from torch.profiler import ProfilerActivity, profile
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                     record_shapes=True) as prof:
                            fn()
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(str(path))
                        profiles.append({"arm": arm, "trace": _sha(path),
                                         "kernels": rpo.kernel_table(prof, limit=200)})
                    record["cases"].append({"which": which, "tokens": tokens, "tile": args.tile,
                        "plan": [asdict(t) for t in plan.tiles], "equal": equal,
                        "contiguous_shard_mutant_differs": mutant, "state_sha256": common_inputs,
                        "parameter_sha256": {key: _tensor_sha(value) for key, value in prm.items()},
                        "timings": timings, "profiles": profiles,
                        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_peak_bytes": torch.cuda.max_memory_reserved()})
                    _publish(result_path, record)
            record["status"] = "passed_operator_screen"
            record["scope"] = "actual TP2 communicator+mHC; no model/serving/quality/energy promotion"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.stop_flag = True
        sampler.join(timeout=6)
        record["power_samples"] = sampler.samples
        record["power_source"] = sampler.source
        if record["cases"]:
            t0 = record["cases"][0]["timings"][0]["utc_window"][0]
            t1 = record["cases"][-1]["timings"][-1]["utc_window"][1]
            try:
                record["netdata"] = rpo.netdata_window(t0, t1)
            except Exception as exc:
                record["netdata_unavailable"] = f"{type(exc).__name__}: {exc}"
        _publish(result_path, record)
    print(json.dumps({"result": _sha(result_path), "status": record["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
