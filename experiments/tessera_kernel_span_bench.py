"""Kernel-lane baseline and delta for the span-2 decode (principle 15).

One GLM routed expert (2048 x 4096, the shipping shape) encoded at the E2M1x2
cap (R = 7).  Arms: the tuple GEMV on a span-1 body (the kernel as it stood
before the span-2 decode), the tuple GEMV on a span-2 body (after it), and a
bf16 torch GEMV as the bandwidth anchor for this box.  Each arm runs a timed
loop of at least ``--seconds`` wall-clock so the Netdata power series has a
window to read: the in-process number is CUDA-event ms per call and plane
bytes per call; the box-level number is power against the envelope, read from
``nvidia_smi.gpu_power_draw`` over the printed epoch window.  A torch.profiler
pass records the kernel's self CUDA time so the ms/call is attributed to the
kernel and not to launch or Python.  Results append to
``experiments/results/tessera_kernel_span_bench.json``.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402
from tessera.manifest import ScalePlaneKind  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402
from tessera.wire import nvfp4_scale_bytes, nvfp4_scale_bytes_lut  # noqa: E402

CODE = ConvCode(memory=6)


def timed(fn, seconds, sync_every=32):
    fn(); torch.cuda.synchronize()
    t0 = time.time(); ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record(); n = 0
    while time.time() - t0 < seconds:
        for _ in range(sync_every):
            fn()
        n += sync_every
        torch.cuda.synchronize()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / n, n, t0, time.time()


def profiled(fn, calls=50):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    rows = sorted(prof.key_averages(), key=lambda r: -r.self_device_time_total)
    return [{"name": r.key[:80], "self_cuda_ms_per_call": r.self_device_time_total / 1e3 / calls,
             "count": r.count} for r in rows[:4]]


def unit_scales(unit, rows, cols):
    if unit.scale_plane is ScalePlaneKind.LUT:
        e4m3, g = nvfp4_scale_bytes_lut(unit.scale_refine, unit.scale_lut, unit.scale_global)
    else:
        e4m3, g = nvfp4_scale_bytes(unit.scale_base, unit.scale_refine, unit.group, unit.half)
    return e4m3.reshape(rows, cols // 16).t().contiguous(), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--arms", nargs="+", default=["bf16", "span1", "span2"])
    ap.add_argument("--label", default="")
    ap.add_argument("--lanes", type=int, default=64)
    ap.add_argument("--split-k", type=int, default=32)
    ap.add_argument("--out", default="experiments/results/tessera_kernel_span_bench.json")
    a = ap.parse_args()
    from tessera.kernel import (
        build_anchor_values, build_tuple_index_lut, pack_kernel_planes, tessera_gemv_tuple,
    )
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    grid = tuple_grid(E2M1_GRID, 2)
    rate = grid.rate_cap
    forests = {rate: build_forest(rate, grid=grid)}
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    name = f"model.language_model.layers.{a.layer}.mlp.experts.0.{a.proj}.weight"
    with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
        w = f.get_tensor(name).contiguous().cuda().float()
    rows, cols = w.shape
    torch.manual_seed(0)
    x = torch.randn(cols, device="cuda")
    records = []
    log = lambda s: print(s, flush=True)
    log(f"{name} {tuple(w.shape)}  commit {commit}  label {a.label!r}  lanes {a.lanes} split_k {a.split_k}")

    if "bf16" in a.arms:
        wb, xb = w.to(torch.bfloat16), x.to(torch.bfloat16)
        ms, n, t0, t1 = timed(lambda: wb @ xb, a.seconds)
        rec = {"arm": "bf16 torch gemv", "ms_per_call": ms, "calls": n, "bytes_per_call": wb.numel() * 2,
               "epoch": [t0, t1], "profile": profiled(lambda: wb @ xb)}
        rec["GBps"] = rec["bytes_per_call"] / ms / 1e6
        records.append(rec)
        log(f"  bf16 torch gemv          {ms:8.4f} ms/call  {rec['GBps']:7.1f} GB/s  window {t0:.0f}-{t1:.0f}")

    for arm, span, plane in (("span1", 1, ScalePlaneKind.S6B), ("span2", 2, ScalePlaneKind.LUT)):
        if arm not in a.arms:
            continue
        unit = encode_unit(w, forests, (rate,) * cols, CODE, span=span, scale_plane=plane,
                           scale_refit=0, completion=0)
        reference = reconstruct_unit(unit, forests, CODE, completion=0).float()
        scales, g = unit_scales(unit, rows, cols)
        try:
            planes = pack_kernel_planes(unit.body_bits, rate=rate, span=span)
        except GrammarError as e:
            log(f"  {arm}: pack refused -- {e}")
            records.append({"arm": f"tessera tuple gemv {arm}", "refused": str(e)})
            continue
        index_lut = build_tuple_index_lut(forests[rate], CODE)
        values = build_anchor_values(forests[rate])
        if span == 1:
            select, point = planes
            fn = lambda: tessera_gemv_tuple(x, select, point, index_lut, values, scales, g, rows, cols,
                                            rate=rate, arity=2, lanes=a.lanes, split_k=a.split_k)
            plane_bytes = select.numel() + point.numel()
        else:
            from tessera.kernel import gemv_from_packed, pack_unit_for_kernel
            packed = pack_unit_for_kernel(unit, forests[rate], CODE)
            fn = lambda: gemv_from_packed(x, packed, lanes=a.lanes, split_k=a.split_k)
            plane_bytes = packed["select"].numel() + packed["label"].numel() + packed["point"].numel()
            scales = packed["nibbles"]      # the scale plane at its wire size
        got = fn(); want = reference @ x
        rel = float((got - want).norm() / want.norm())
        assert rel < 1e-5, f"{arm}: kernel disagrees with the reference decode, rel {rel}"
        ms, n, t0, t1 = timed(fn, a.seconds)
        rec = {"arm": f"tessera tuple gemv {arm}", "ms_per_call": ms, "calls": n,
               "bytes_per_call": plane_bytes + scales.numel(), "plane_bytes": plane_bytes,
               "scale_bytes": scales.numel(), "rel_err_vs_reference": rel,
               "epoch": [t0, t1], "profile": profiled(fn)}
        rec["GBps"] = rec["bytes_per_call"] / ms / 1e6
        records.append(rec)
        log(f"  tessera tuple gemv {arm}  {ms:8.4f} ms/call  {rec['GBps']:7.1f} GB/s of body+plane  "
            f"({rec['bytes_per_call']/1e6:.2f} MB/call)  rel {rel:.2e}  window {t0:.0f}-{t1:.0f}")
        for p in rec["profile"][:2]:
            log(f"      {p['name'][:60]:<60} {p['self_cuda_ms_per_call']:8.4f} ms/call self CUDA")
        del unit, reference
    out = Path(a.out)
    prior = json.load(open(out)) if out.exists() else []
    prior.append({"commit": commit, "label": a.label, "tensor": name, "shape": [rows, cols],
                  "seconds": a.seconds, "lanes": a.lanes, "split_k": a.split_k,
                  "records": records, "time": time.time()})
    json.dump(prior, open(out, "w"), indent=1)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
