"""Time the window-body GEMV (``kernel_window_gemv``) against what it replaces.

Arms (``--arm``), each re-runnable on its own on a contended box:

``gemv``    per Linear shape of Qwen3-0.6B / Qwen3-4B / an assumed 27B list, at
            M in 1..8: the fused GEMV (kernel alone, and the op = zero + kernel
            + bf16 cast) against bf16 cuBLAS, the resident-FP8 path the lane
            serves today (per-token quant + ``torch._scaled_mm``, eager and
            compiled), ``_scaled_mm`` alone, and a plain streaming read of the
            same wire bytes (the per-shape bandwidth denominator).  Cold by
            rotating through >= 96 MB of replicas.  Per-model per-token totals
            follow from the per-shape rows.
``plans``   the launch-shape sweep on the 4B shapes at M=1 (rows per lane,
            warps, resident blocks, columns per item, table dtype).
``ablate``  where the time goes: the kernel with the table gather, the wire
            read, the FMA, or both reads deleted (``ablation`` 1..4).

Every row carries the box's power and the count of other CUDA processes
sampled while it ran (CLAUDE.md principle 15: utilisation is non-diagnostic
on GB10; power against the envelope is the load).  ``bench_group`` interleaves
arms and reports min-of-rounds, for the reasons its docstring gives.

Usage (on the GPU box, inside ``experiments/run_gemv_bench.sh`` for the lock)::

    PYTHONPATH=src TMPDIR=/home/rob/tmp TORCH_EXTENSIONS_DIR=/home/rob/tmp/torch-ext-gemv \\
    python experiments/bench_kernel_window_gemv.py --arm all --out /mnt/shared/tessera-runs/gemv
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/rob/tmp/torch-ext-gemv")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import kernel_window_gemv as kg   # noqa: E402

COLD_BYTES = 96e6
SPEC_GB_S = 273.0
L = 14

#: `(name, rows, cols)` per model, one entry per *unit* -- the granularity the
#: lane issues a GEMV at (q/k/v are three members of one fused module).
MODELS = {
    "Qwen3-0.6B": (28, [
        ("q_proj", 2048, 1024), ("k_proj", 1024, 1024), ("v_proj", 1024, 1024),
        ("o_proj", 1024, 2048), ("gate_proj", 3072, 1024), ("up_proj", 3072, 1024),
        ("down_proj", 1024, 3072),
    ]),
    "Qwen3-4B": (36, [
        ("q_proj", 4096, 2560), ("k_proj", 1024, 2560), ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096), ("gate_proj", 9728, 2560), ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ]),
    # ASSUMED dense 27B list (hidden 5120, intermediate 17408, 64 layers, 40 q
    # heads / 8 kv heads at head_dim 128); no 27B config is on this box.
    "27B-assumed": (64, [
        ("q_proj", 5120, 5120), ("k_proj", 1024, 5120), ("v_proj", 1024, 5120),
        ("o_proj", 5120, 5120), ("gate_proj", 17408, 5120), ("up_proj", 17408, 5120),
        ("down_proj", 5120, 17408),
    ]),
}


# --- power ----------------------------------------------------------------------


class Power:
    """Samples ``nvidia-smi`` power (and who else is on the GPU) in a thread."""

    def __init__(self, period: float = 0.12):
        self._samples: list[tuple[float, float, int, int]] = []
        self._stop = threading.Event()
        self._period = period
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                p, clk = out.splitlines()[0].split(",")
                apps = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                n = len([ln for ln in apps.splitlines() if ln.strip()])
                self._samples.append((time.time(), float(p), n, int(clk)))
            except Exception:
                pass
            self._stop.wait(self._period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> dict:
        rows = [(p, k, c) for t, p, k, c in self._samples if t0 <= t <= t1]
        if not rows:
            return {"mean_w": None, "max_w": None, "n": 0}
        w = [p for p, _k, _c in rows]
        procs = [k for _p, k, _c in rows]
        clk = [c for _p, _k, c in rows]
        return {"mean_w": round(sum(w) / len(w), 1), "max_w": round(max(w), 1),
                "min_w": round(min(w), 1), "n": len(w), "sm_mhz_mean": round(sum(clk) / len(clk)),
                "cuda_procs_max": max(procs), "cuda_procs_min": min(procs)}


POWER: Power | None = None


def bench_group(fns: dict, n: int = 64, warm: int = 8, seconds: float = 1.5, rounds: int = 0) -> dict:
    """Time several arms **interleaved**; report min-of-rounds (the least
    interfered sample on a box that is never quiet), with median/mean/max
    beside it so the contention is visible, and the power over the group."""
    calls, iters = {}, {}
    for name, value in fns.items():
        if isinstance(value, tuple):
            calls[name], iters[name] = value
        else:
            calls[name], iters[name] = value, n
    fns = calls
    names = list(fns)
    for name in names:
        for _ in range(min(warm, iters[name])):
            fns[name]()
    torch.cuda.synchronize()
    per = {name: [] for name in names}
    events = {name: (torch.cuda.Event(True), torch.cuda.Event(True)) for name in names}
    t0 = time.time()
    r = 0
    while (rounds and r < rounds) or (not rounds and time.time() - t0 < seconds) or r < 3:
        for name in names:
            start, end = events[name]
            start.record()
            for _ in range(iters[name]):
                fns[name]()
            end.record()
            torch.cuda.synchronize()
            per[name].append(start.elapsed_time(end) / iters[name] * 1000)
        r += 1
        if rounds and r >= rounds:
            break
    t1 = time.time()
    power = POWER.window(t0, t1) if POWER is not None else {}
    out = {}
    for name in names:
        v = sorted(per[name])
        out[name] = {
            "us": round(v[0], 3), "median_us": round(v[len(v) // 2], 3),
            "mean_us": round(sum(v) / len(v), 3), "max_us": round(v[-1], 3),
            "rounds": len(v), "iters_per_round": iters[name], **power,
        }
    return out


# --- units ------------------------------------------------------------------------


class Rotor:
    def __init__(self, items):
        self.items = list(items)
        self.i = 0

    def next(self):
        item = self.items[self.i]
        self.i = (self.i + 1) % len(self.items)
        return item


def synth(rows: int, cols: int, rate: int = 4, seed: int = 0, M: int = 1, plan=None):
    """A window unit of the given shape from random bits: random R-bit codes,
    a table of random finite E4M3 values (bf16-exact), a random row scale."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.int32, device="cuda", generator=g).to(torch.uint8)
    bytes_ = torch.randint(0, 256, (1 << L,), dtype=torch.int32, device="cuda", generator=g).to(torch.uint8)
    bytes_ = torch.where((bytes_ & 0x7F) == 0x7F, bytes_ - 1, bytes_)        # no NaN patterns
    values = bytes_.view(torch.float8_e4m3fn).float().bfloat16()
    scale = torch.rand(rows, device="cuda", generator=g) * 0.01 + 0.001
    return kg.prepare_value_unit(body, (rate,) * cols, L, values, scale=scale, M=M, plan=plan)


def cold_units(rows, cols, target=COLD_BYTES, **kw_):
    first = synth(rows, cols, **kw_)
    n = max(2, int(target // max(1, first.rep.nbytes)) + 1)
    units = [first]
    for _ in range(1, n):
        # replicas share everything but the wire bytes (and so the item tables)
        units.append(dataclasses.replace(first, rep=dataclasses.replace(first.rep, words=first.rep.words.clone())))
    return Rotor(units)


def cold_tensors(make, nbytes, target=COLD_BYTES):
    n = max(2, int(target // max(1, nbytes)) + 1)
    return Rotor([make() for _ in range(n)])


def _fp8_pair(rows, cols):
    tile = torch.randint(0, 200, (rows, cols), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    scale = torch.rand(rows, 1, device="cuda") * 0.01 + 0.001
    return tile, scale


def _fp8_per_token(x: torch.Tensor):
    """Per-token dynamic E4M3 quantisation: the A side the FP8 route serves."""
    amax = x.abs().amax(dim=1, keepdim=True).to(torch.float32).clamp_min(1e-12)
    scale = amax / 448.0
    return (x.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


_QUANT_C = None


def _fused_quant():
    """The A side as the LANE runs it: one fused quantisation kernel (vLLM's
    ``native_fp8_quant`` is not importable here; ``torch.compile`` of the same
    expression is the stand-in, and the eager 5-6 launch version is reported
    beside it so the launch overhead is visible)."""
    global _QUANT_C
    if _QUANT_C is None:
        cfg = torch._dynamo.config
        for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                     "accumulated_cache_size_limit"):
            if hasattr(cfg, name):
                setattr(cfg, name, 512)
        _QUANT_C = torch.compile(_fp8_per_token, dynamic=False)
    return _QUANT_C


_READ = None


def _reader():
    """A plain streaming reader of a byte tensor (the same kernel as
    ``bw_probe_cuda.py``): the per-shape denominator."""
    global _READ
    if _READ is None:
        from torch.utils.cpp_extension import load_inline
        kg._ensure_toolchain_on_path()
        src = r"""
        #include <torch/extension.h>
        __global__ void read_kernel(const uint4* __restrict__ src, long n_vec, float* __restrict__ out) {
            long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
            long stride = (long)gridDim.x * blockDim.x;
            unsigned acc = 0;
            for (; i < n_vec; i += stride) { uint4 v = __ldg(src + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
            if (acc == 0xDEADBEEF) out[blockIdx.x] = 1.0f;
        }
        void read_bytes(torch::Tensor src, torch::Tensor out, int blocks, int threads) {
            long n_vec = (src.numel() * src.element_size()) / 16;
            read_kernel<<<blocks, threads>>>((const uint4*)src.data_ptr(), n_vec, out.data_ptr<float>());
        }
        """
        major, minor = torch.cuda.get_device_capability()
        _READ = load_inline(
            "gemv_bench_reader",
            cpp_sources="#include <torch/extension.h>\nvoid read_bytes(torch::Tensor, torch::Tensor, int, int);\n",
            cuda_sources=src, functions=["read_bytes"],
            extra_cuda_cflags=["-O3", "-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"],
            build_directory=_mk(os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], "gemv_bench_reader")),
        )
    return _READ


def _mk(p):
    os.makedirs(p, exist_ok=True)
    return p


def parse_plan(spec: str | None):
    """``rpl,warps,blocks,cpi,dtype[,item_cost|fixed]`` -> Plan, or None for the default."""
    if not spec:
        return None
    parts = spec.split(",")
    rpl, warps, blocks, cpi, dt = parts[:5]
    balanced, item_cost = True, 24
    if len(parts) > 5:
        if parts[5] == "fixed":
            balanced = False
        else:
            item_cost = int(parts[5])
    return kg.Plan(rpl=int(rpl), warps=int(warps), blocks=int(blocks), cols_per_item=int(cpi),
                   table_dtype=torch.float32 if dt in ("f32", "fp32", "float32") else torch.bfloat16,
                   balanced=balanced, item_cost=item_cost)


def plan_str(p: kg.Plan) -> str:
    tail = f"/i{p.item_cost}" if p.balanced else "/fixed"
    return f"rpl{p.rpl}/w{p.warps}/b{p.blocks}/c{p.cols_per_item}/{'f32' if p.table_dtype == torch.float32 else 'bf16'}{tail}"


# --- arms ---------------------------------------------------------------------------


def arm_gemv(out: dict, models, batches, plan_spec=None, seconds=2.5, quick=False):
    shapes = {}
    for model in models:
        for _name, r, c in MODELS[model][1]:
            shapes.setdefault((r, c), []).append(model)
    reader = _reader()
    sink = torch.zeros(8192, device="cuda")
    table = {}
    for (r, c), used_by in sorted(shapes.items()):
        rot = cold_units(r, c)
        wire = rot.items[0].rep.nbytes
        bf16 = cold_tensors(lambda: torch.randn(r, c, dtype=torch.bfloat16, device="cuda"), r * c * 2)
        fp8 = cold_tensors(lambda: _fp8_pair(r, c), r * c)
        entry = {"rows": r, "cols": c, "used_by": used_by, "wire_bytes": wire,
                 "bf16_bytes": r * c * 2, "fp8_bytes": r * c, "wire_replicas": len(rot.items), "M": {}}
        for m in (batches if not quick else (1,)):
            plan = parse_plan(plan_spec) or kg.default_plan(r, c, m)
            units = [rot.items[0].with_plan(plan)]
            units += [u.with_plan(plan, share_from=units[0]) for u in rot.items[1:]]
            urot = Rotor(units)
            x = torch.randn(m, c, dtype=torch.bfloat16, device="cuda")
            xq0, xs0 = _fp8_per_token(x)
            quant = _fused_quant()
            quant(x)
            scratch = torch.zeros(m, r, dtype=torch.float32, device="cuda")

            def kernel_only():
                return kg.window_gemv(urot.next(), x, out=scratch)     # accumulates; timing only

            def fused_op():
                return kg.window_gemv(urot.next(), x).to(torch.bfloat16)

            def bf16_call():
                return torch.nn.functional.linear(x, bf16.next())

            def fp8_eager_call():
                w, ws = fp8.next()
                xq, xs = _fp8_per_token(x)
                return torch._scaled_mm(xq, w.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)

            def fp8_lane_call():
                w, ws = fp8.next()
                xq, xs = quant(x)
                return torch._scaled_mm(xq, w.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)

            def fp8_mm_only():
                w, ws = fp8.next()
                return torch._scaled_mm(xq0, w.t(), scale_a=xs0, scale_b=ws.t(), out_dtype=torch.bfloat16)

            def wire_read():
                reader.read_bytes(urot.next().rep.words, sink, 192, 256)

            arms = {"fused_kernel": kernel_only, "fused_op": fused_op, "bf16_linear": bf16_call,
                    "fp8_quant_plus_mm": fp8_eager_call, "fp8_lane_quant_plus_mm": fp8_lane_call,
                    "fp8_mm_only": fp8_mm_only, "wire_read": wire_read}
            n = 64 if r * c <= 4 << 20 else 16
            try:
                g = bench_group(arms, n=n, seconds=seconds)
            except Exception as exc:
                print(f"  {r}x{c} M={m}: FAILED {exc}")
                continue
            for k, nb in (("fused_kernel", wire), ("fused_op", wire), ("wire_read", wire),
                          ("bf16_linear", r * c * 2), ("fp8_mm_only", r * c), ("fp8_lane_quant_plus_mm", r * c)):
                g[k]["GB_per_s"] = round(nb / (g[k]["us"] * 1e-6) / 1e9, 1)
                g[k]["frac_of_273"] = round(g[k]["GB_per_s"] / SPEC_GB_S, 3)
            g["fused_kernel"]["frac_of_wire_read"] = round(g["wire_read"]["us"] / g["fused_kernel"]["us"], 3)
            g["speedup_kernel_vs_fp8_lane"] = round(g["fp8_lane_quant_plus_mm"]["us"] / g["fused_kernel"]["us"], 3)
            g["speedup_op_vs_fp8_lane"] = round(g["fp8_lane_quant_plus_mm"]["us"] / g["fused_op"]["us"], 3)
            g["speedup_op_vs_fp8_eager"] = round(g["fp8_quant_plus_mm"]["us"] / g["fused_op"]["us"], 3)
            g["speedup_kernel_vs_mm_only"] = round(g["fp8_mm_only"]["us"] / g["fused_kernel"]["us"], 3)
            g["speedup_op_vs_bf16"] = round(g["bf16_linear"]["us"] / g["fused_op"]["us"], 3)
            g["lane_quant_is_fused"] = bool(g["fp8_lane_quant_plus_mm"]["us"] < g["fp8_quant_plus_mm"]["us"])
            g["plan"] = plan_str(plan)
            g["items"] = int(units[0].items_for(m)[0].shape[0])
            entry["M"][str(m)] = g
            print(f"  {r:6d}x{c:<6d} M={m} [{plan_str(plan)} items={g['items']}] "
                  f"kernel {g['fused_kernel']['us']:8.1f} us ({g['fused_kernel']['GB_per_s']:5.1f} GB/s, "
                  f"{g['fused_kernel']['frac_of_wire_read']:.2f} of read) op {g['fused_op']['us']:8.1f} | "
                  f"bf16 {g['bf16_linear']['us']:8.1f} | fp8 lane {g['fp8_lane_quant_plus_mm']['us']:8.1f} "
                  f"eager {g['fp8_quant_plus_mm']['us']:8.1f} mm {g['fp8_mm_only']['us']:8.1f} "
                  f"({g['fp8_mm_only']['GB_per_s']:5.1f} GB/s) | read {g['wire_read']['us']:7.1f} "
                  f"({g['wire_read']['GB_per_s']:5.1f} GB/s) | x{g['speedup_op_vs_fp8_lane']:.3f} op/lane "
                  f"x{g['speedup_kernel_vs_mm_only']:.3f} k/mm | {g['fused_kernel'].get('mean_w')} W "
                  f"procs {g['fused_kernel'].get('cuda_procs_min')}-{g['fused_kernel'].get('cuda_procs_max')}",
                  flush=True)
        table[f"{r}x{c}"] = entry
        del rot, bf16, fp8, units, urot
        torch.cuda.empty_cache()
    out["gemv"] = table
    out["totals"] = totals(table, models, batches if not quick else (1,))
    for model, t in out["totals"].items():
        for m, row in t.items():
            print(f"  {model} M={m}: per token {row}")


def totals(table, models, batches):
    res = {}
    for model in models:
        layers, units = MODELS[model]
        res[model] = {}
        for m in batches:
            acc = {}
            ok = True
            for _name, r, c in units:
                g = table.get(f"{r}x{c}", {}).get("M", {}).get(str(m))
                if g is None:
                    ok = False
                    break
                for arm in ("fused_kernel", "fused_op", "bf16_linear", "fp8_quant_plus_mm",
                            "fp8_lane_quant_plus_mm", "fp8_mm_only", "wire_read"):
                    acc[arm] = acc.get(arm, 0.0) + g[arm]["us"]
            if not ok:
                continue
            row = {k: round(v * layers, 1) for k, v in acc.items()}
            row["layers"] = layers
            row["speedup_op_vs_fp8_lane"] = round(row["fp8_lane_quant_plus_mm"] / row["fused_op"], 3)
            row["speedup_kernel_vs_fp8_lane"] = round(row["fp8_lane_quant_plus_mm"] / row["fused_kernel"], 3)
            row["speedup_op_vs_bf16"] = round(row["bf16_linear"] / row["fused_op"], 3)
            row["speedup_kernel_vs_mm_only"] = round(row["fp8_mm_only"] / row["fused_kernel"], 3)
            res[model][str(m)] = row
    return res


def arm_plans(out: dict, shapes, seconds=1.5, M=1):
    """The launch-shape sweep at M=1, kernel alone, cold."""
    res = {}
    for r, c in shapes:
        rot = cold_units(r, c)
        x = torch.randn(M, c, dtype=torch.bfloat16, device="cuda")
        scratch = torch.zeros(M, r, dtype=torch.float32, device="cuda")
        rows = {}
        grid = []
        for dt in (torch.bfloat16, torch.float32):
            for rpl in (16, 8):
                for warps in (8, 16):
                    for blocks in (48, 96, 144):
                        for cpi, ic, bal in ((1024, 8, True), (1024, 24, True), (1024, 64, True), (1024, 160, True),
                                             (256, 24, True), (128, 24, False), (256, 24, False), (1024, 24, False)):
                            if dt == torch.float32 and blocks > 96:
                                continue
                            if rpl == 8 and (dt == torch.float32 or warps == 8):
                                continue
                            grid.append(kg.Plan(rpl=rpl, warps=warps, blocks=blocks, cols_per_item=cpi,
                                                table_dtype=dt, item_cost=ic, balanced=bal))
        best = None
        for plan in grid:
            units = [rot.items[0].with_plan(plan)]
            units += [u.with_plan(plan, share_from=units[0]) for u in rot.items[1:]]
            urot = Rotor(units)

            def call():
                return kg.window_gemv(urot.next(), x, out=scratch)
            try:
                g = bench_group({"k": call}, n=16 if r * c > 4 << 20 else 48, seconds=seconds)["k"]
            except Exception as exc:
                rows[plan_str(plan)] = {"error": str(exc)[:200]}
                continue
            g["GB_per_s"] = round(rot.items[0].rep.nbytes / (g["us"] * 1e-6) / 1e9, 1)
            g["items"] = int(units[0].items_for(M)[0].shape[0])
            rows[plan_str(plan)] = g
            if best is None or g["us"] < best[1]:
                best = (plan_str(plan), g["us"])
            print(f"  {r}x{c} {plan_str(plan):28s} items={g['items']:5d} {g['us']:8.1f} us {g['GB_per_s']:6.1f} GB/s "
                  f"{g.get('mean_w')} W", flush=True)
        res[f"{r}x{c}"] = {"rows": rows, "best": best, "default": plan_str(kg.default_plan(r, c, M))}
        print(f"  {r}x{c} best {best}  default {plan_str(kg.default_plan(r, c, M))}", flush=True)
        del rot
        torch.cuda.empty_cache()
    out["plans"] = res


ABLATIONS = {0: "kernel", 1: "no_gather", 2: "no_wire_read", 3: "no_fma", 4: "no_gather_no_read"}


def arm_ablate(out: dict, shapes, plan_spec=None, seconds=2.0, M=1):
    res = {}
    for r, c in shapes:
        plan = parse_plan(plan_spec) or kg.default_plan(r, c, M)
        rot = cold_units(r, c, plan=plan)
        x = torch.randn(M, c, dtype=torch.bfloat16, device="cuda")
        scratch = torch.zeros(M, r, dtype=torch.float32, device="cuda")
        arms = {}
        for a, name in ABLATIONS.items():
            def call(a=a):
                return kg.window_gemv(rot.next(), x, out=scratch, ablation=a)
            arms[name] = call
        g = bench_group(arms, n=16 if r * c > 4 << 20 else 48, seconds=seconds)
        base = g["kernel"]["us"]
        for name in g:
            g[name]["delta_vs_kernel"] = round(1 - g[name]["us"] / base, 3)
        g["plan"] = plan_str(plan)
        res[f"{r}x{c}"] = g
        print(f"  {r}x{c} [{plan_str(plan)}] " + "  ".join(
            f"{name} {g[name]['us']:.1f}us ({g[name]['delta_vs_kernel']:+.0%})" for name in ABLATIONS.values()),
            flush=True)
        del rot
        torch.cuda.empty_cache()
    out["ablate"] = res


def main(argv=None):
    global POWER
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "gemv", "plans", "ablate"])
    ap.add_argument("--out", default="/mnt/shared/tessera-runs/gemv")
    ap.add_argument("--models", default="Qwen3-0.6B,Qwen3-4B,27B-assumed")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--plan", default=None, help="rpl,warps,blocks,cpi,dtype (default: default_plan)")
    ap.add_argument("--shapes", default="4096x2560,1024x2560,2560x4096,9728x2560,2560x9728",
                    help="for plans/ablate")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")
    batches = tuple(int(b) for b in args.batches.split(","))
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    out = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with Power() as pw:
        POWER = pw
        if args.arm in ("all", "gemv"):
            print("== gemv ==", flush=True)
            arm_gemv(out, models, batches, args.plan, seconds=args.seconds, quick=args.quick)
        if args.arm in ("all", "plans"):
            print("== plans ==", flush=True)
            arm_plans(out, shapes, seconds=min(args.seconds, 1.5))
        if args.arm in ("all", "ablate"):
            print("== ablate ==", flush=True)
            arm_ablate(out, shapes, args.plan, seconds=args.seconds)
    tag = f"_{args.tag}" if args.tag else ""
    path = outdir / f"bench_{args.arm}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
