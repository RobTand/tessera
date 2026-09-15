#!/usr/bin/env python3
"""tessera#486 stage 2 probe: the fused LUT swap passes against the reference.

Checks on one CUDA device; it writes nothing but stdout:

  * identity: ``encode._fit_lut`` with ``TESSERA_LUT_FUSED=0`` and ``=1``
    returns the same bytes and the same table bit for bit, and the fused passes
    answered the fit (``lut_fused.STATS``) with no tripwire;
  * trial costs: ``lut_fused.position_costs`` equals ``encode._lut_cost`` of
    every trial table bit for bit, at sizes on both sides of every change in
    torch's reduction layout, and counts the costs a sequential sum would get
    wrong (the check's power);
  * walls, with ``--walls`` only: synchronised ``_fit_lut`` walls, fused and
    not.  A wall on a shared box is not a measurement.

    PYTHONPATH=src python experiments/tessera486_lut_fused_probe.py [--quick] [--walls]
"""
import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import tessera.encode as enc  # noqa: E402
import tessera.lut_fused as lf  # noqa: E402

SIZES = [128, 129, 131, 1000, 2051, 7681, 130560, 130561, 131072, 150001, 229375, 229376,
         229377, 300007, 524288, 1048576, 2097153, 4194304]
QUICK = [128, 131, 2051, 130561, 229376, 524288]
DISTS = ["halves", "wide", "lattice", "dead"]


def bits(x: torch.Tensor) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def sample(n, dist, gen, dev):
    if dist == "halves":        # per-half LUT targets and energies over a few octaves
        t = torch.exp(torch.randn(n, device=dev, generator=gen) * 0.7)
        w = torch.exp(torch.randn(n, device=dev, generator=gen) * 1.2)
    elif dist == "wide":        # a bracket as wide as the grid allows
        t = torch.exp(torch.randn(n, device=dev, generator=gen) * 2.0)
        w = torch.rand(n, device=dev, generator=gen) * 50
    elif dist == "lattice":     # targets on grid-commensurate points, few weight levels: exact ties
        t = torch.randint(1, 200, (n,), device=dev, generator=gen).float() / 32
        w = torch.randint(1, 4, (n,), device=dev, generator=gen).float()
    else:                       # "dead": a quarter of the halves carry no energy
        t = torch.exp(torch.randn(n, device=dev, generator=gen) * 0.7)
        w = torch.exp(torch.randn(n, device=dev, generator=gen))
        w = torch.where(torch.rand(n, device=dev, generator=gen) < 0.25, torch.zeros_like(w), w)
    return t.contiguous(), w.contiguous()


def global_scale(t):
    return float(2.0 ** (math.floor(math.log2(float(t.max()))) - 6.0))


def fit(t, w, gs, entries, swaps, fused):
    os.environ["TESSERA_LUT_FUSED"] = "1" if fused else "0"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = enc._fit_lut(t, w, gs, entries, swaps=swaps)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def identity(sizes, dev, report):
    totals = {"cases": 0, "identical": 0, "fused": 0, "tripped": 0, "nonfinite": 0}
    gen = torch.Generator(device=dev)
    for n in sizes:
        for dist in DISTS:
            for entries, swaps in ((16, 32), (16, 1), (8, 32)):
                if entries != 16 and n not in (131, 229376):
                    continue
                gen.manual_seed(n * 7 + len(dist) * 131 + entries + swaps)
                t, w = sample(n, dist, gen, dev)
                gs = global_scale(t)
                before = dict(lf.STATS)
                (rb, rt), t_ref = fit(t, w, gs, entries, swaps, fused=False)
                (gb, gt), t_fus = fit(t, w, gs, entries, swaps, fused=True)
                delta = {k: lf.STATS[k] - before[k] for k in lf.STATS}
                same = (rb.dtype == gb.dtype and rt.dtype == gt.dtype
                        and torch.equal(rb, gb)
                        and torch.equal(rt.view(torch.int32), gt.view(torch.int32)))
                live = int((w > 0).sum())
                totals["cases"] += 1
                totals["identical"] += int(same)
                for k in ("fused", "tripped", "nonfinite"):
                    totals[k] += delta[k]
                row = {"n": n, "live": live, "dist": dist, "entries": entries, "swaps": swaps,
                       "identical": same, **delta}
                if not same or delta["tripped"]:
                    row["ref_bytes"] = rb.tolist()
                    row["fused_bytes"] = gb.tolist()
                report["identity"].append(row)
    report["identity_totals"] = totals
    return totals


def trial_costs(sizes, dev, report):
    totals = {"costs": 0, "mismatch": 0, "order_sensitive": 0}
    gen = torch.Generator(device=dev)
    cpu = torch.Generator().manual_seed(486)
    for n in sizes:
        for dist in DISTS:
            gen.manual_seed(n * 11 + len(dist))
            t, w = sample(n, dist, gen, dev)
            live = w > 0
            s, ww = t[live].contiguous(), w[live].contiguous()
            if s.numel() < 128:
                continue
            grid = enc.e4m3_positive_values(dev) * global_scale(t)
            pick = torch.randperm(grid.numel(), generator=cpu)
            table, values = grid[pick[:16].to(dev)], grid[pick[16:56].to(dev)].sort().values
            for position in (0, 7, 15):
                got = lf.position_costs(s, ww, table, values, position)
                for u in range(values.numel()):
                    trial = table.clone()
                    trial[position] = values[u]
                    want = enc._lut_cost(s, ww, trial)
                    gap = (s[:, None] - trial[None, :]).abs().amin(dim=1)
                    seq = torch.cumsum(ww * gap * gap, 0)[-1]
                    totals["costs"] += 1
                    if bits(want) != bits(got[u]):
                        totals["mismatch"] += 1
                        if totals["mismatch"] <= 20:
                            report["cost_mismatch"].append(
                                {"n": s.numel(), "dist": dist, "position": position, "u": u,
                                 "torch": float(want), "fused": float(got[u]),
                                 "plan": str(lf._plan(s))})
                    totals["order_sensitive"] += int(bits(want) != bits(seq))
    report["cost_totals"] = totals
    return totals


def walls(dev, report):
    gen = torch.Generator(device=dev)
    for n in (229376, 524288):
        gen.manual_seed(n)
        t, w = sample(n, "halves", gen, dev)
        gs = global_scale(t)
        fit(t, w, gs, 16, 32, fused=True)          # compile and warm
        rows = []
        for _ in range(3):
            (_, _), t_ref = fit(t, w, gs, 16, 32, fused=False)
            (_, _), t_fus = fit(t, w, gs, 16, 32, fused=True)
            rows.append({"reference_s": t_ref, "fused_s": t_fus})
        report["walls"].append({"n": n, "rows": rows})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--walls", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    sizes = QUICK if args.quick else SIZES
    report = {"torch": torch.__version__, "device": props.name,
              "multi_processor_count": int(props.multi_processor_count),
              "max_threads_per_multi_processor": int(props.max_threads_per_multi_processor),
              "warp_size": int(props.warp_size), "tile": lf._resolve_tile(),
              "allocator": torch.cuda.get_allocator_backend(),
              "plans": {str(n): str(lf._plan(torch.empty(n, device=dev))) for n in sizes},
              "identity": [], "cost_mismatch": [], "walls": []}
    t0 = time.perf_counter()
    costs = trial_costs(sizes, dev, report)
    ident = identity(sizes, dev, report)
    if args.walls:
        walls(dev, report)
    report["wall_s"] = time.perf_counter() - t0
    print(json.dumps(report, indent=1))
    ok = (costs["mismatch"] == 0 and ident["identical"] == ident["cases"]
          and ident["tripped"] == 0 and ident["fused"] > 0)
    print(f"SUMMARY costs={costs['costs']} cost_mismatch={costs['mismatch']} "
          f"order_sensitive={costs['order_sensitive']} identity={ident['identical']}/"
          f"{ident['cases']} fused={ident['fused']} tripped={ident['tripped']} "
          f"nonfinite={ident['nonfinite']} ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
