"""T8R speed attribution: per-module kernel time, power and bytes at TP2 rank-0 shapes.

One process, one GB10.  Every Tessera group loads the REAL wire of one module of
the GLM-5.3-Flash Tessera-8 release artifact (all 288 experts for a routed
stack) and calls the serving route the vLLM plugin dispatches:

* routed: ``moe_route._RankLocalPackedIntake`` -> ``finish`` -> ``adapter()``
  (the fused routed window lane wherever the stack's rates admit it);
* dense / shared: ``lane.build_tessera_method`` -> ``create_weights`` (TP2 rank-0
  partition sizes) -> ``process_weights_after_loading`` -> ``apply``;
* bf16: the attention / router / lm_head Linears the artifact leaves at source
  precision, as vLLM's unquantized method runs them (``F.linear``), at their TP2
  rank-0 partition shapes, on random weights (time does not depend on values).

Routing is synthetic and BALANCED (token t picks experts (8t+j) mod 288), so
M = 1 touches 8 experts and M >= 36 touches all 288, as a serve does.

Per (group, M) it records:
* ``wall``: CUDA events around one eager call, median / IQR of ``--iters``;
* ``graph``: the same call replayed from a captured CUDA graph (M <= 8 only;
  the FULL_DECODE_ONLY proxy), median of ``--iters``;
* ``kernel_us``: torch.profiler self device time per call, by kernel name;
* ``power``: NVML board power sampled at 10 Hz over a ``--power-s`` loop of
  back-to-back calls, mean / max, against the 140 W envelope, and calls/J;
* ``bytes``: weight bytes the call must read (wire of touched experts, or bf16).

Usage: bench_t8r.py --out DIR [--groups a,b] [--ms 1,2,4,8,512,2048]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import types
import zlib

import torch


def _install_vllm_stubs():
    """Inert stand-ins for the two vLLM symbols the route classes subclass, used
    only where vLLM is absent (the pinned image has it; the real runtime wins)."""
    try:
        import vllm  # noqa: F401
        return False
    except ImportError:
        pass
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.linear", "vllm.model_executor.parameter"]
    mods = {n: types.ModuleType(n) for n in names}

    class LinearMethodBase:
        pass

    class BasevLLMParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data=data, requires_grad=False)

        def __init__(self, data, weight_loader=None, **kwargs):
            self._weight_loader = weight_loader

    mods["vllm.model_executor.layers.linear"].LinearMethodBase = LinearMethodBase
    mods["vllm.model_executor.parameter"].BasevLLMParameter = BasevLLMParameter
    for n, m in mods.items():
        sys.modules[n] = m
    return True


VLLM_STUBBED = _install_vllm_stubs()
os.environ.setdefault("TESSERA_SERVE_MODE", "resident")


def _init_vllm_world1(rdv_dir):
    """vLLM's LinearMethodBase reads the TP group at construction.  One process,
    one rank (world-1 gloo); the TP2 cut is the layer's tp_rank/tp_size, which is
    what the Tessera shard planner reads.  The rendezvous file lives in the
    output directory, never /tmp."""
    import contextlib
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    stack = contextlib.ExitStack()
    stack.enter_context(set_current_vllm_config(VllmConfig()))
    init_distributed_environment(world_size=1, rank=0,
                                 distributed_init_method="file://" + os.path.join(rdv_dir, "rdv"),
                                 local_rank=0, backend="gloo")
    ensure_model_parallel_initialized(1, 1)
    return stack

from safetensors import safe_open  # noqa: E402

ARTIFACT = "/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported"
P = "model.language_model.layers."
TOP_K, EXPERTS, SWIGLU_LIMIT = 8, 288, 10.0
TP_SIZE, TP_RANK = 2, 0
ENVELOPE_W = 140.0
HIDDEN = 4096

# (group id, kind, module).  kind: routed | dense_col (column-parallel, output
# cut) | dense_row (row-parallel, input cut).  Rates are the T8R manifest's.
TESSERA_GROUPS = [
    ("experts.R1024.L10", "routed", P + "10.mlp.experts"),
    ("experts.R1088.L11", "routed", P + "11.mlp.experts"),
    ("experts.R832.L42", "routed", P + "42.mlp.experts"),
    ("shared_gate_up.R1024.L11", "dense_col", P + "11.mlp.shared_experts.gate_up_proj"),
    ("shared_gate_up.R1088.L10", "dense_col", P + "10.mlp.shared_experts.gate_up_proj"),
    ("shared_gate_up.R960.L13", "dense_col", P + "13.mlp.shared_experts.gate_up_proj"),
    ("shared_gate_up.R832.L25", "dense_col", P + "25.mlp.shared_experts.gate_up_proj"),
    ("shared_down.R1024.L12", "dense_row", P + "12.mlp.shared_experts.down_proj"),
    ("shared_down.R1088.L10", "dense_row", P + "10.mlp.shared_experts.down_proj"),
    ("shared_down.R960.L11", "dense_row", P + "11.mlp.shared_experts.down_proj"),
    ("shared_down.R832.L17", "dense_row", P + "17.mlp.shared_experts.down_proj"),
    ("dense_gate_up.R960.L0", "dense_col", P + "0.mlp.gate_up_proj"),
    ("dense_down.R1088.L0", "dense_row", P + "0.mlp.down_proj"),
    ("dense_gate_up.R1024.L2", "dense_col", P + "2.mlp.gate_up_proj"),
    ("dense_down.R832.L1", "dense_row", P + "1.mlp.down_proj"),
]
# (group id, out_features, in_features) per TP2 rank; merged where vLLM merges.
# KDA (34 layers): q/k/v column-parallel 8192 -> 4096 each (merged here as one
# 12288-row GEMM and also timed as three), o_proj row-parallel 8192 -> 4096 in,
# f_a / g_a replicated 128 x 4096, f_b / g_b column-parallel 4096 x 128,
# b_proj column-parallel 32 x 4096.  MLA (11 layers): q_a + kv_a replicated
# (1536 + 512) x 4096, q_b column-parallel 8192 x 1536, kv_b 16384 x 512,
# o_proj row-parallel 4096 x 8192; indexer (replicated) wq_b 4096 x 1536, wk
# 128 x 4096, weights_proj 32 x 4096.  Router 288 x 4096 (replicated).
# lm_head vocab-parallel 77440 x 4096.
BF16_GROUPS = [
    ("bf16.kda_qkv", 12288, 4096),
    ("bf16.kda_q", 4096, 4096),
    ("bf16.kda_o", 4096, 4096),
    ("bf16.kda_fa_ga", 256, 4096),
    ("bf16.kda_fb", 4096, 128),
    ("bf16.kda_b", 32, 4096),
    ("bf16.mla_qa_kva", 2048, 4096),
    ("bf16.mla_qb", 8192, 1536),
    ("bf16.mla_kvb", 16384, 512),
    ("bf16.mla_o", 4096, 8192),
    ("bf16.idx_wqb", 4096, 1536),
    ("bf16.idx_wk", 128, 4096),
    ("bf16.idx_weights", 32, 4096),
    ("bf16.router", 288, 4096),
    ("bf16.lm_head", 77440, 4096),
]


class Store:
    def __init__(self, root):
        self.root = root
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.config = json.load(open(os.path.join(root, "config.json")))
        self._open = {}
        self.schemes = {}
        for g in self.config["quantization_config"]["config_groups"].values():
            for t in g["targets"]:
                self.schemes[t] = g["scheme"]

    def get(self, name):
        shard = self.index[name]
        if shard not in self._open:
            self._open[shard] = safe_open(os.path.join(self.root, shard), "pt", device="cpu")
        return self._open[shard].get_tensor(name)


# ------------------------------------------------------------------ power
class PowerSampler:
    """10 Hz board power.  NVML if importable, else ``nvidia-smi`` polling."""

    def __init__(self):
        self.source = None
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self._nv.nvmlDeviceGetPowerUsage(self._h)
            self.source = "pynvml"
        except Exception:  # noqa: BLE001
            import shutil
            if shutil.which("nvidia-smi"):
                self.source = "nvidia-smi"

    def read_w(self):
        if self.source == "pynvml":
            return self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        if self.source == "nvidia-smi":
            import subprocess
            out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                                  "-i", str(torch.cuda.current_device())],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            return float(out.split()[0])
        return None

    def sample_during(self, work, seconds):
        """Run ``work()`` back to back for ``seconds`` while sampling; return stats."""
        if self.source is None:
            n = 0
            t0 = time.time()
            while time.time() - t0 < seconds:
                work(); n += 1
            torch.cuda.synchronize()
            return {"source": None, "calls": n, "seconds": time.time() - t0}
        samples, stop = [], threading.Event()

        def loop():
            while not stop.is_set():
                try:
                    samples.append((time.time(), self.read_w()))
                except Exception:  # noqa: BLE001
                    pass
                stop.wait(0.1)
        th = threading.Thread(target=loop, daemon=True)
        # warm the pipeline so the sampled window is steady state
        for _ in range(3):
            work()
        torch.cuda.synchronize()
        th.start()
        t0 = time.time(); n = 0
        while time.time() - t0 < seconds:
            work(); n += 1
            if n % 16 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        t1 = time.time()
        stop.set(); th.join()
        w = [v for ts, v in samples if v is not None and t0 <= ts <= t1]
        mean = sum(w) / len(w) if w else None
        return {"source": self.source, "calls": n, "seconds": t1 - t0, "window_unix": [t0, t1],
                "samples": len(w), "mean_w": mean, "max_w": max(w) if w else None,
                "envelope_frac": (mean / ENVELOPE_W) if mean else None,
                "calls_per_j": (n / ((t1 - t0) * mean)) if mean else None}


# ------------------------------------------------------------------ builders
def build_dense(store, module, kind):
    from tessera.serving.lane import build_tessera_method
    scheme = store.schemes[module]
    method = build_tessera_method(scheme, module, "resident")
    layer = torch.nn.Module()
    layer.tp_rank, layer.tp_size = TP_RANK, TP_SIZE
    layer.prefix = module
    rows, cols = int(scheme["rows"]), int(scheme["columns"])
    role_rows = [int(r) for _n, r in scheme["roles"]]
    if kind == "dense_col":
        ops, inpp = [r // TP_SIZE for r in role_rows], cols
    else:
        ops, inpp = role_rows, cols // TP_SIZE
    method.create_weights(layer, input_size_per_partition=inpp, output_partition_sizes=ops,
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16,
                          weight_loader=None)
    layer.wire_bytes.data = store.get(module + ".wire_bytes").clone()
    layer.to("cuda")
    with torch.no_grad():
        method.process_weights_after_loading(layer)
    local = (int(layer.tessera_rows), int(layer.tessera_columns))
    q = int(scheme["q256"])
    info = {"family": scheme["family"], "q256": q, "local_shape": list(local),
            "decoder": getattr(layer, "tessera_decoder", None),
            "symbol": getattr(layer, "tessera_symbol", None)}
    wire_rank = int(scheme["wire_bytes"]) / TP_SIZE

    def fn(x):
        return method.apply(layer, x)
    return fn, local[1], info, layer, (lambda m: wire_rank)


def build_routed(store, module):
    from tessera.serving.scheme import validate_tessera_moe_scheme
    from tessera.serving.moe_route import _RankLocalPackedIntake
    scheme = store.schemes[module]
    declared = validate_tessera_moe_scheme(scheme, module)
    dev = torch.device("cuda")
    intake = _RankLocalPackedIntake(declared, module, dev, TP_RANK, TP_SIZE)
    if not intake.compact:
        raise RuntimeError("compact routed lane not published in this build")
    w13_len = torch.zeros(EXPERTS, 2, dtype=torch.long)
    w2_len = torch.zeros(EXPERTS, dtype=torch.long)
    wire_total = 0
    for e in range(EXPERTS):
        for group, index, proj in (("w13", 0, "gate_proj"), ("w13", 1, "up_proj"), ("w2", 0, "down_proj")):
            wire = store.get(f"{module}.{e}.{proj}.wire")
            wire_total += wire.numel()
            intake.load(group, index, e, wire, device=dev)
            if group == "w13":
                w13_len[e, index] = wire.numel()
            else:
                w2_len[e] = wire.numel()
    packed = intake.finish(w13_len, w2_len)
    native = packed.adapter()
    per_expert_rank = wire_total / EXPERTS / TP_SIZE
    info = {"family": scheme["family"], "q256": {g: v["q256"] for g, v in scheme["groups"].items()},
            "adapter": type(native).__name__,
            "adapter_attrs": sorted(a for a in vars(native) if not a.startswith("__"))[:40]
            if hasattr(native, "__dict__") else None,
            "resident_bytes": int(packed.resident_bytes()) if hasattr(packed, "resident_bytes") else None,
            "wire_bytes_rank": wire_total / TP_SIZE}

    def touched(m):
        return min(EXPERTS, m * TOP_K) * per_expert_rank

    def fn(x, ids, w):
        return native(x, ids, w, swiglu_limit=SWIGLU_LIMIT, apply_router_weight_on_input=False)
    return fn, info, packed, touched


def build_bf16(out_f, in_f):
    w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02

    def fn(x):
        return torch.nn.functional.linear(x, w)
    return fn, in_f, {"shape": [out_f, in_f]}, w, (lambda m: out_f * in_f * 2)


def balanced_routing(m, device):
    t = torch.arange(m, device=device).unsqueeze(1) * TOP_K + torch.arange(TOP_K, device=device)
    ids = (t % EXPERTS).to(torch.int32)
    w = torch.full((m, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    return ids, w


# ------------------------------------------------------------------ timing
def summarize(samples):
    s = sorted(samples)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    return {"median_ms": q(0.5), "p25_ms": q(0.25), "p75_ms": q(0.75), "min_ms": s[0], "n": len(s)}


def time_events(call, warmup, iters):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); call(); b.record(); b.synchronize()
        out.append(float(a.elapsed_time(b)))
    return out


def kernel_profile(call, reps=5):
    from torch.profiler import profile, ProfilerActivity
    call(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            call()
        torch.cuda.synchronize()
    per = {}
    total = 0.0
    launches = 0
    for evt in prof.key_averages():
        dev_us = getattr(evt, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(evt, "self_cuda_time_total", 0.0)
        if dev_us and evt.device_type is not None and str(evt.device_type).endswith("CUDA"):
            per[evt.key[:120]] = {"us_per_call": dev_us / reps, "count_per_call": evt.count / reps}
            total += dev_us / reps
        if evt.key in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC"):
            launches += evt.count / reps
    top = dict(sorted(per.items(), key=lambda kv: -kv[1]["us_per_call"])[:12])
    return {"kernel_us_per_call": total, "launches_per_call": launches, "top": top}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="all")
    ap.add_argument("--ms", default="1,2,4,8,512,2048")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--power-s", type=float, default=3.0)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--ncu", action="store_true",
                    help="one call per (group, M) between cudaProfilerStart/Stop; no timing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ms = [int(v) for v in args.ms.split(",")]
    wanted = None if args.groups == "all" else set(args.groups.split(","))
    torch.manual_seed(0)
    dev = torch.device("cuda")
    store = Store(ARTIFACT)
    power = PowerSampler()
    import tessera
    meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "tp": [TP_RANK, TP_SIZE],
            "top_k": TOP_K, "experts": EXPERTS, "ms": ms, "warmup": args.warmup, "iters": args.iters,
            "power_source": power.source, "tessera_file": tessera.__file__, "artifact": ARTIFACT,
            "tessera_head": os.environ.get("TESSERA_HEAD"), "tessera_state": os.environ.get("TESSERA_STATE"),
            "image": os.environ.get("ORACLE_IMAGE"), "pb_action": os.environ.get("PB_ACTION_KEY"),
            "host": os.environ.get("HOST_NAME"), "kernel_sha": os.environ.get("KERNEL_SHA"),
            "start_unix": time.time()}
    meta["vllm_stubbed"] = VLLM_STUBBED
    if not VLLM_STUBBED:
        import vllm
        meta["vllm"] = getattr(vllm, "__version__", None)
    try:
        import importlib.metadata as md
        meta["tessera_dist"] = md.version("tessera_quant")
    except Exception as exc:  # noqa: BLE001
        meta["tessera_meta_error"] = repr(exc)
    ctx = None if VLLM_STUBBED else _init_vllm_world1(args.out)  # noqa: F841 -- held open
    results = []
    plan = [(g, k, mod) for g, k, mod in TESSERA_GROUPS] + [(g, "bf16", (o, i)) for g, o, i in BF16_GROUPS]
    for gid, kind, module in plan:
        if wanted is not None and gid not in wanted and gid.split(".")[0] not in wanted:
            continue
        t0 = time.time()
        rec = {"group": gid, "kind": kind, "module": module if kind != "bf16" else None}
        fn = holder = None
        try:
            with torch.inference_mode():
                if kind == "routed":
                    fn, info, holder, bytes_for = build_routed(store, module)
                    width = int(store.schemes[module]["groups"]["w13"]["columns"])
                elif kind == "bf16":
                    fn, width, info, holder, bytes_for = build_bf16(*module)
                else:
                    fn, width, info, holder, bytes_for = build_dense(store, module, kind)
                rec["info"] = info
                rec["load_s"] = time.time() - t0
                rec["cells"] = {}
                for m in ms:
                    # seeded per (group, M), so two arms that run different
                    # group sets still see the same x and can compare outputs
                    torch.manual_seed(zlib.crc32(f"{gid}:{m}".encode()))
                    x = torch.randn(m, width, device=dev, dtype=torch.bfloat16)
                    xa = (x,) if kind != "routed" else (x, *balanced_routing(m, dev))
                    call = lambda: fn(*xa)  # noqa: E731
                    if args.ncu:
                        # ncu --profile-from-start off: exactly one profiled call per (group, M).
                        for _ in range(3):
                            call()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        rec["cells"][str(m)] = {"ncu": True, "bytes": bytes_for(m)}
                        print(json.dumps({"group": gid, "M": m, "ncu": True}), flush=True)
                        del x, xa
                        continue
                    cell = {"bytes": bytes_for(m)}
                    # the output's bytes, for a bitwise A/B across kernel arms
                    y = call()
                    torch.cuda.synchronize()
                    cell["out_sha256"] = hashlib.sha256(
                        y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                    cell["out_shape"] = list(y.shape)
                    del y
                    ts = time.time()
                    cell["wall"] = summarize(time_events(call, args.warmup, args.iters))
                    cell["wall_window_unix"] = [ts, time.time()]
                    cell["profile"] = kernel_profile(call)
                    if not args.no_graph and m <= 8:
                        try:
                            s = torch.cuda.Stream()
                            s.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s):
                                for _ in range(3):
                                    call()
                            torch.cuda.current_stream().wait_stream(s)
                            torch.cuda.synchronize()
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                call()
                            cell["graph"] = summarize(time_events(g.replay, args.warmup, args.iters))
                            replay = g.replay
                        except Exception as exc:  # noqa: BLE001
                            cell["graph_error"] = repr(exc)[:400]
                            replay = call
                    else:
                        replay = call
                    cell["power"] = power.sample_during(replay, args.power_s)
                    best = cell.get("graph", cell["wall"])["median_ms"]
                    cell["eff_gbps"] = cell["bytes"] / (best * 1e-3) / 1e9
                    k_us = cell["profile"]["kernel_us_per_call"]
                    cell["eff_gbps_kernel"] = cell["bytes"] / (k_us * 1e-6) / 1e9 if k_us else None
                    rec["cells"][str(m)] = cell
                    print(json.dumps({"group": gid, "M": m, "wall_ms": round(cell["wall"]["median_ms"], 4),
                                      "graph_ms": round(cell["graph"]["median_ms"], 4) if "graph" in cell else None,
                                      "kernel_ms": round(k_us / 1000, 4),
                                      "GBps_kernel": round(cell["eff_gbps_kernel"] or 0, 1),
                                      "W": round(cell["power"].get("mean_w") or 0, 1)}), flush=True)
                    del x, xa
            rec["ok"] = True
        except Exception:  # noqa: BLE001
            import traceback
            rec["ok"] = False
            rec["error"] = traceback.format_exc()
            print(json.dumps({"group": gid, "ok": False, "err": rec["error"][-800:]}), flush=True)
        finally:
            fn = holder = None
            torch.cuda.empty_cache()
        rec["group_wall_s"] = time.time() - t0
        results.append(rec)
        json.dump({"meta": meta, "results": results}, open(os.path.join(args.out, "bench_t8r.json"), "w"),
                  indent=1, default=repr)
    meta["end_unix"] = time.time()
    json.dump({"meta": meta, "results": results}, open(os.path.join(args.out, "bench_t8r.json"), "w"),
              indent=1, default=repr)
    bad = [r["group"] for r in results if not r["ok"]]
    print("done; failed groups:", bad, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
