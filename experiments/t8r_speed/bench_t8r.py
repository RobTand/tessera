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
M = 1 touches 8 experts and M >= 36 touches all 288, as a serve does.  With
``--routing DIR``, every routed group also replays the top-k ids a serve
recorded (``DIR/m<M>/*.pt``), one ``<M>@<file>`` cell per file, so a real
routing's per-expert skew can be timed against the balanced cell.

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
import io
import json
import os
import statistics
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
    def __init__(self, root, inputs=None):
        self.root = root
        self.inputs = inputs
        self.index = self.metadata("model.safetensors.index.json")["weight_map"]
        self.config = self.metadata("config.json")
        self._open = {}
        self.schemes = {}
        for g in self.config["quantization_config"]["config_groups"].values():
            for t in g["targets"]:
                self.schemes[t] = g["scheme"]

    def metadata(self, name):
        path = os.path.join(self.root, name)
        return self.inputs.json(path) if self.inputs else json.load(open(path))

    def get(self, name):
        if self.inputs:
            return self.inputs.tensor(self.root, name, index=self.index)
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

    def sample_during(self, work, seconds, *, capture_series=False):
        """Run ``work()`` back to back for ``seconds`` while sampling; return stats."""
        if self.source is None:
            n = 0
            t0 = time.time()
            while time.time() - t0 < seconds:
                work(); n += 1
            torch.cuda.synchronize()
            return {"source": None, "calls": n, "seconds": time.time() - t0}
        samples, clocks, stop = [], [], threading.Event()

        def loop():
            while not stop.is_set():
                try:
                    ts = time.time()
                    samples.append((ts, self.read_w()))
                    if capture_series and self.source == "pynvml":
                        clocks.append((ts, self._nv.nvmlDeviceGetClockInfo(self._h, self._nv.NVML_CLOCK_SM),
                                       self._nv.nvmlDeviceGetTemperature(self._h, self._nv.NVML_TEMPERATURE_GPU)))
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
        extra = {}
        if capture_series:
            extra = {"power_series_unix_w": samples, "clock_temperature_series": clocks, "energy_status": "HOLD_pending_Netdata_coverage",
                     "gpu_clock_mhz": self._nv.nvmlDeviceGetClockInfo(self._h, self._nv.NVML_CLOCK_SM)
                         if self.source == "pynvml" else None,
                     "gpu_temperature_c": self._nv.nvmlDeviceGetTemperature(self._h, self._nv.NVML_TEMPERATURE_GPU)
                         if self.source == "pynvml" else None}
        return {**extra, "source": self.source, "calls": n, "seconds": t1 - t0, "window_unix": [t0, t1],
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
    library = getattr(native, "library", None)
    library_path = library_sha = None
    if library is not None:
        from tessera import routed_fused as rf
        library_path = os.path.realpath(rf._ext(library).__file__)
        with open(library_path, "rb") as handle:
            library_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    expected_library = os.environ.get("BENCH_EXPECT_LIBRARY_SHA256")
    if expected_library and library_sha != expected_library:
        raise ValueError("routed adapter's loaded native library differs from the expected binary")
    per_expert_rank = wire_total / EXPERTS / TP_SIZE
    info = {"family": scheme["family"], "q256": {g: v["q256"] for g, v in scheme["groups"].items()},
            "adapter": type(native).__name__,
            "resident_word_layouts": {role: getattr(packed, role).word_layout
                                      for role in ("gate", "up", "down")},
            "native_piece_major": getattr(native, "piece_major", None),
            "native_library": library,
            "native_library_path": library_path,
            "native_library_sha256": library_sha,
            "requested_piece_major": os.environ.get("TESSERA_ROUTED_PIECE_MAJOR", "0"),
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


def routing_files(root, ms):
    """``{M: [path, ...]}`` of recorded top-k ids under ``root/m<M>/*.pt|*.json``.

    A ``.json`` file is a list of ``.pt`` paths (relative to ``root``) whose ids
    are concatenated in order: consecutive prefill chunks of one layer make that
    layer's routing for one larger step.
    """
    out = {}
    for m in ms:
        d = os.path.join(root, f"m{m}")
        if os.path.isdir(d):
            out[m] = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith((".pt", ".json")))
    return out


def recorded_routing(path, m, device):
    """Top-k ids a serve recorded (``{"ids": int32 [M, top_k]}``), uniform weights.

    Time depends on how many routes each expert receives, not on the weights,
    so the weights stay 1/top_k as in ``balanced_routing``.
    """
    if path.endswith(".json"):
        root = os.path.dirname(os.path.dirname(path))
        parts = [torch.load(os.path.join(root, q), map_location="cpu", weights_only=False)["ids"]
                 for q in json.load(open(path))]
        ids = torch.cat(parts, 0)
    else:
        ids = torch.load(path, map_location="cpu", weights_only=False)["ids"]
    if tuple(ids.shape) != (m, TOP_K):
        raise ValueError(f"{path}: ids shape {tuple(ids.shape)} != ({m}, {TOP_K})")
    ids = ids.to(device=device, dtype=torch.int32)
    w = torch.full((m, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    return ids, w


def routing_stats(ids):
    """Per-expert route counts and the fused kernel's 64-route superblocks."""
    n = torch.bincount(ids.flatten().long().cpu(), minlength=EXPERTS)
    return {"experts_touched": int((n > 0).sum()), "max_routes": int(n.max()),
            "superblocks": int(((n + 63) // 64).sum()),
            "superblocks_128": int(((n + 127) // 128).sum())}


# ------------------------------------------------------------------ timing
def summarize(samples):
    raw = list(samples)
    s = sorted(raw)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    return {"median_ms": statistics.median(s), "p25_ms": q(0.25), "p75_ms": q(0.75),
            "min_ms": s[0], "n": len(s), "raw_samples_ms": raw}


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


def kernel_profile(call, reps=5, *, full_names=False):
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
            per[evt.key if full_names else evt.key[:120]] = {"us_per_call": dev_us / reps, "count_per_call": evt.count / reps}
            total += dev_us / reps
        if evt.key in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC"):
            launches += evt.count / reps
    top = dict(sorted(per.items(), key=lambda kv: -kv[1]["us_per_call"])[:12])
    return {"kernel_us_per_call": total, "launches_per_call": launches, "top": top}


def require_single_replay_options(args, *, stubbed=False):
    if getattr(args,"profile_native_file",None):
        expected = "/mnt/shared/astra-routed-gate-20261002/retained-native-0f953b69/tessera_routed_fused_mma_e4m3.so"
        if not args.single_routing_file or not args.ncu or args.profile_native_file!=expected:
            raise ValueError("retained native artifact requires the exact counter-only replay")
    if args.single_routing_file:
        if (args.groups != "experts.R1024.L10" or args.ms != "2048"
                or not args.input_manifest or args.routing or not args.no_graph
                or args.power_s < 30 or args.warmup != 10 or args.iters != 30):
            raise ValueError("single replay requires L10/M2048, staged inputs, no graph/menu and >=30s power")
        if args.artifact != "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported":
            raise ValueError("single replay requires the actual A8SE artifact")
        if stubbed:
            raise ValueError("single replay refuses stubbed vLLM")
    elif args.input_manifest or args.artifact != ARTIFACT:
        raise ValueError("artifact/staged-input overrides require the exact single replay")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="all")
    ap.add_argument("--ms", default="1,2,4,8,512,2048")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--power-s", type=float, default=3.0)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--outputs-only", action="store_true",
                    help="record output bits and native/layout identity without timing, power or graph work")
    ap.add_argument("--routing", default=None,
                    help="directory with m<M>/*.pt recorded top-k ids; each file adds a "
                         "'<M>@<file>' cell to every routed group (balanced cells stay)")
    ap.add_argument("--ncu", action="store_true",
                    help="one call per (group, M) between cudaProfilerStart/Stop; no timing")
    ap.add_argument("--artifact", default=ARTIFACT)
    ap.add_argument("--single-routing-file", default=None,
                    help="one sealed historical real-ID replay; no balanced/menu cases")
    ap.add_argument("--input-manifest", default=None,
                    help="exact PB-staged readset for the single replay")
    ap.add_argument("--profile-native-file", default=None,
                    help="one declared retained native-code artifact for counter-only recovery")
    args = ap.parse_args()
    require_single_replay_options(args, stubbed=VLLM_STUBBED)
    os.makedirs(args.out, exist_ok=True)
    ms = [int(v) for v in args.ms.split(",")]
    wanted = None if args.groups == "all" else set(args.groups.split(","))
    torch.manual_seed(0)
    dev = torch.device("cuda")
    inputs = None
    native_owner = None
    if args.single_routing_file:
        from pb_staged_store import StagedInputs
        inputs = StagedInputs(args.input_manifest)
    try:
        store = Store(args.artifact, inputs)
        if inputs:
            # Bind publisher declarations to the exact staged bytes intake reads.
            published = store.metadata("tessera_serving_manifest.json")
            inputs.bind_roles(store.root, published["modules"][P + "10.mlp.experts"]["roles"])
            if args.profile_native_file:
                from pb_staged_store import NativeCallback
                from tessera import routed_fused as rf
                native_owner = NativeCallback(inputs,args.profile_native_file,rf,
                    os.path.join(args.out,"native-artifact"),
                    expected_sha256=os.environ["BENCH_EXPECT_LIBRARY_SHA256"],
                    source_sha256=os.environ["KERNEL_SHA"])

        power = PowerSampler()
        import tessera
        meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "tp": [TP_RANK, TP_SIZE],
                "top_k": TOP_K, "experts": EXPERTS, "ms": ms, "warmup": args.warmup, "iters": args.iters,
                "power_source": power.source, "tessera_file": tessera.__file__, "artifact": args.artifact,
                "tessera_head": os.environ.get("TESSERA_HEAD"), "tessera_state": os.environ.get("TESSERA_STATE"),
                "image": os.environ.get("ORACLE_IMAGE"), "pb_action": os.environ.get("PB_ACTION_KEY"),
                "host": os.environ.get("HOST_NAME"), "kernel_sha": os.environ.get("KERNEL_SHA"),
                "e4m3_mma": os.environ.get("TESSERA_FUSED_E4M3_MMA"),
                "start_unix": time.time()}
        recorded = routing_files(args.routing, ms) if args.routing else {}
        if args.routing and not any(recorded.values()):
            raise SystemExit(f"--routing {args.routing}: no m<M>/*.pt for M in {ms} "
                             "(is the directory mounted into the container?)")
        meta["routing"] = {"root": args.routing, "files": {str(m): len(v) for m, v in recorded.items()}}
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
        if inputs:
            meta["single_replay"] = {"scope": "historical IDs, seeded random x and uniform weights; not VB capture",
                                      "reference_baseline_source": "608bbdf0d6909548ff7c6919e5cdb834c1fcef7c",
                                      "manifest_sha256": inputs.manifest_sha256,
                                      "sdk_version": inputs.sdk.SDK_VERSION}
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
                    cases = []
                    for m in ms:
                        if args.single_routing_file:
                            cases.append((f"{m}@{os.path.splitext(os.path.basename(args.single_routing_file))[0]}",
                                          m, args.single_routing_file))
                            continue
                        cases.append((str(m), m, None))
                        if kind == "routed":
                            cases += [(f"{m}@{os.path.splitext(os.path.basename(f))[0]}", m, f)
                                      for f in recorded.get(m, [])]
                    for key, m, rfile in cases:
                        # seeded per (group, M), so two arms that run different
                        # group sets still see the same x and can compare outputs
                        torch.manual_seed(zlib.crc32(f"{gid}:{m}".encode()))
                        x = torch.randn(m, width, device=dev, dtype=torch.bfloat16)
                        if kind != "routed":
                            xa = (x,)
                        elif rfile is None:
                            xa = (x, *balanced_routing(m, dev))
                        elif inputs:
                            loaded = torch.load(io.BytesIO(inputs.read(rfile)), map_location="cpu", weights_only=True)
                            ids = loaded["ids"]
                            if tuple(ids.shape) != (2048, TOP_K) or ids.dtype != torch.int32:
                                raise ValueError("single replay routing shape/dtype differs")
                            if int(ids.min()) < 0 or int(ids.max()) >= EXPERTS:
                                raise ValueError("single replay expert ids out of range")
                            weights = torch.full((m, TOP_K), 1.0/TOP_K, dtype=torch.float32, device=dev)
                            xa = (x, ids.to(device=dev), weights)
                            from tessera import routed_fused as rf
                            expected = "3a32d040668cc1fe5678c2088deb5afa8cd6f227a7920dfbd899c90863c03254"
                            source = os.path.join(os.path.dirname(rf.__file__), "serving/csrc/routed_fused_window.cu")
                            if hashlib.sha256(open(source, "rb").read()).hexdigest() != expected:
                                raise ValueError("single replay kernel source differs from baseline608")
                            if rf.library_for("e4m3") != "e4m3mma" or rf.superblock_rows("e4m3mma", 0, m) != 128:
                                raise ValueError("single replay library/width differs")
                            meta["single_replay"]["input_hashes"] = {
                                "x": hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                                "weights": hashlib.sha256(weights.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                                "ids": hashlib.sha256(ids.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                            meta["single_replay"]["seed"] = zlib.crc32(f"{gid}:{m}".encode())
                            from tessera.serving.backend import platform_token
                            token = platform_token(torch=torch)
                            meta["single_replay"]["compile_flags"] = rf._cflags(token, True, True)
                            lib = rf._ext("e4m3mma")
                            if native_owner:
                                native_owner.bind(lib)
                            meta["single_replay"]["library_path"] = lib.__file__
                            meta["single_replay"]["library_sha256"] = hashlib.sha256(open(lib.__file__, "rb").read()).hexdigest()
                            expected_library = os.environ.get("BENCH_EXPECT_LIBRARY_SHA256")
                            if expected_library and meta["single_replay"]["library_sha256"] != expected_library:
                                raise ValueError("profile recovery native binary differs from measured library")
                            meta["single_replay"]["build_platform"] = token
                        else:
                            xa = (x, *recorded_routing(rfile, m, dev))
                        call = lambda: fn(*xa)  # noqa: E731
                        if args.ncu:
                            # ncu --profile-from-start off: exactly one profiled call per (group, M).
                            for _ in range(args.warmup if inputs else 3):
                                call()
                            torch.cuda.synchronize()
                            torch.cuda.cudart().cudaProfilerStart()
                            call()
                            torch.cuda.synchronize()
                            torch.cuda.cudart().cudaProfilerStop()
                            rec["cells"][key] = {"ncu": True, "bytes": bytes_for(m)}
                            print(json.dumps({"group": gid, "M": key, "ncu": True}), flush=True)
                            del x, xa
                            continue
                        cell = {"bytes": bytes_for(m)}
                        if kind == "routed":
                            cell["routing"] = dict(routing_stats(xa[1]), source=rfile or "balanced")
                        # the output's bytes, for a bitwise A/B across kernel arms
                        y = call()
                        torch.cuda.synchronize()
                        cell["out_sha256"] = hashlib.sha256(
                            y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                        cell["out_shape"] = list(y.shape)
                        del y
                        if args.outputs_only:
                            cell["outputs_only"] = True
                            rec["cells"][key] = cell
                            print(json.dumps({"group": gid, "M": key, "out_sha256": cell["out_sha256"],
                                              "outputs_only": True}), flush=True)
                            del x, xa
                            continue
                        ts = time.time()
                        cell["wall"] = summarize(time_events(call, args.warmup, args.iters))
                        cell["wall_window_unix"] = [ts, time.time()]
                        cell["profile"] = kernel_profile(call, full_names=bool(inputs))
                        if inputs:
                            wanted_kernel = "routed_fused_kernel<true, 0, false, false, 4, false, 128>"
                            mode0 = [v for k,v in cell["profile"]["top"].items() if wanted_kernel in k]
                            if len(mode0) != 1 or mode0[0]["count_per_call"] != 1:
                                raise ValueError("actual profiled single replay is not one mode0/RL4/BMT128 kernel")
                            cell["mode0_profile"] = mode0[0]
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
                        cell["power"] = power.sample_during(replay, args.power_s, capture_series=bool(inputs))
                        if inputs:
                            cell["power"]["scope"] = "whole single-owner forward; not gate-only or full-model energy"
                            cell["power"]["calls_per_j"] = None
                            if cell["power"].get("source") is None:
                                raise ValueError("single replay requires fast power instrument")
                        best = cell.get("graph", cell["wall"])["median_ms"]
                        cell["eff_gbps"] = cell["bytes"] / (best * 1e-3) / 1e9
                        k_us = cell["profile"]["kernel_us_per_call"]
                        cell["eff_gbps_kernel"] = cell["bytes"] / (k_us * 1e-6) / 1e9 if k_us else None
                        rec["cells"][key] = cell
                        print(json.dumps({"group": gid, "M": key, "wall_ms": round(cell["wall"]["median_ms"], 4),
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
        if native_owner:
            native_owner.finish(torch.cuda.synchronize)
            meta["native_code_artifact"] = native_owner.record
        if inputs:
            meta["staged_reads"] = inputs.reads
            inputs.close()
        meta["end_unix"] = time.time()
        json.dump({"meta": meta, "results": results}, open(os.path.join(args.out, "bench_t8r.json"), "w"),
                  indent=1, default=repr)
        bad = [r["group"] for r in results if not r["ok"]]
        print("done; failed groups:", bad, flush=True)
        return 1 if bad else 0
    finally:
        try:
            if native_owner:
                native_owner.finish(torch.cuda.synchronize)
        finally:
            if inputs:
                inputs.close()


if __name__ == "__main__":
    sys.exit(main())
