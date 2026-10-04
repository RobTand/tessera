#!/usr/bin/env python3
"""The fused mHC post/pre kernel (#783) against the stock split-k sequence.

Run inside the serving image through ``experiments/mhc/mhc_probe.sh`` with
``MHC_PROBE_SCRIPT=mhc_fused_probe.py``.  Real layer-1 ``hc_attn``/``hc_ffn``
projections, scales, bases and norm weights of the served checkpoint.

``bitwise``  Every output (residual, post mix, comb mix, layer input) of
             ``tessera.serving.mhc_fusion.fused_post_pre`` against stock
             ``mhc_fused_post_pre_tilelang`` at the same split, bit for bit:
             each case runs both with ``compute_num_split`` answering for the
             case's full batch (``tokens == full_batch`` is a plain call; a
             smaller ``tokens`` is an exact-SP shard).  Realistic inputs (one
             stock call's own mixes) and adversarial ones (zero and negative-zero
             rows, large magnitudes), ragged token counts, a CUDA-graph replay of
             the fused call, and a second fused run (determinism).  Any
             mismatch fails the probe (exit 1).

``timing``   Per-site ms under CUDA-graph replay over enough input copies to
             exceed L2, the stock and fused arms interleaved round by round,
             at the served shapes; per-kernel device time (CUPTI); the fused
             grid swept; power samples per window.  Matched before/after
             microbenchmark only -- served end to end is not measured here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mhc_probe as mp  # noqa: E402
import routed_pair_oracle as rpo  # noqa: E402

from tessera.serving import mhc_fusion as mf  # noqa: E402

#: (tokens, full_batch): the call's own tokens and the batch whose split it runs at.
BITWISE_CASES = ((33, 33), (64, 64), (100, 100), (256, 256), (256, 512), (511, 511), (512, 512),
                 (1000, 1000), (1024, 1024), (1024, 2048), (2048, 2048), (2049, 2049),
                 (4096, 4096), (4096, 8192), (8192, 8192))
TIMING_CASES = ((1024, 2048), (2048, 2048), (512, 512), (256, 512), (4096, 8192), (8192, 8192))
OUTPUTS = ("residual_cur", "post_mix", "comb_mix", "layer_input")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Forced:
    """``compute_num_split`` answering for ``full_batch`` tokens (SplitForcer's rule), recording calls."""

    def __init__(self, tk, full_batch):
        self.tk, self.stock, self.full_batch, self.seen = tk, tk.compute_num_split, full_batch, []

    def __call__(self, block_k, k, grid_size):
        s = self.stock(block_k, k, -(-self.full_batch // 64))
        self.seen.append(s)
        return s

    def __enter__(self):
        self.tk.compute_num_split = self
        return self

    def __exit__(self, *exc):
        self.tk.compute_num_split = self.stock


def adversarial(x, res, post, comb, gen):
    x, res = x.clone(), res.clone()
    t = x.shape[0]
    rows = torch.randperm(t, generator=torch.Generator().manual_seed(t))[: max(1, t // 8)].tolist()
    for n, r in enumerate(rows):
        kind = n % 4
        if kind == 0:
            res[r].zero_(); x[r].zero_()
        elif kind == 1:
            res[r].fill_(-0.0); x[r].fill_(-0.0)
        elif kind == 2:
            res[r].mul_(1.0e4); x[r].mul_(-3.0e3)
        else:
            res[r, :, ::7] = 0.0; x[r, ::5] = -0.0
    return x, res, post, comb


def fused_call(lib, tk, prm):
    def call(x, residual, post, comb, grid=None, clocks=None, tile=None):
        return mf.fused_post_pre(lib, tk, x, residual, post, comb, prm["fn"], prm["scale"], prm["base"],
                                 mp.RMS_EPS, mp.HC_EPS, mp.HC_EPS, mp.POST_MULT, mp.SINKHORN,
                                 prm["norm"], mp.RMS_EPS, grid=grid, clocks=clocks, tile=tile)
    return call


PHASES = ("post", "gemm", "mixes", "layer_input")


TILES = (16, 32, 48, 64)


def phase_clocks(fused, ins, reps=3):
    """Per-tile phase durations (us, median over tiles and reps) from the kernel's own stamps."""
    tokens = ins[0].shape[0]
    tm = mf.tile_tokens(tokens, mf.default_grid(torch, ins[0].device))
    tiles = -(-tokens // tm)
    per_phase = {k: [] for k in PHASES}
    spans = []
    for _ in range(reps):
        clocks = torch.zeros(tiles * 5, dtype=torch.int64, device="cuda")
        fused(*ins, clocks=clocks, tile=tm)
        torch.cuda.synchronize()
        c = clocks.view(tiles, 5).double().cpu()
        for i, k in enumerate(PHASES):
            per_phase[k] += ((c[:, i + 1] - c[:, i]) / 1e3).tolist()
        spans.append(float((c[:, 4].max() - c[:, 0].min()) / 1e3))
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    return {"tile_tokens": tm, "tiles": tiles, "median_tile_us": {k: med(v) for k, v in per_phase.items()},
            "kernel_span_us": med(spans), "note": "%globaltimer stamps by thread 0; diagnostic, not a timing arm"}


def part_bitwise(args, model_dir, lib, tk):
    out = {"cases": [], "failures": []}
    for which in ("attn", "ffn"):
        prm = mp.mhc_params(model_dir, which)
        stock = mp.mhc_call(prm)
        fused = fused_call(lib, tk, prm)
        for tokens, full in BITWISE_CASES:
            gen = torch.Generator(device="cuda").manual_seed(tokens * 7 + full)
            base = mp.mhc_inputs(tokens, gen, stock)
            for variant in ("realistic", "adversarial"):
                ins = base if variant == "realistic" else adversarial(*base, gen)
                with Forced(tk, full) as f_stock:
                    ref = stock(*ins)
                with Forced(tk, full) as f_fused:
                    got = fused(*ins)
                    again = fused(*ins)
                    by_tile = {tm: fused(*ins, tile=tm) for tm in TILES}
                rec = {"which": which, "tokens": tokens, "full_batch": full, "variant": variant,
                       "split_stock": f_stock.seen, "split_fused": f_fused.seen,
                       "deterministic": all(torch.equal(a, b) for a, b in zip(got, again)),
                       "tiles_equal": {tm: all(torch.equal(a, b) for a, b in zip(o, ref)) for tm, o in by_tile.items()}}
                for name, a, b in zip(OUTPUTS, got, ref):
                    rec[name] = mp.compare(a, b)
                    rec[name]["shape_equal"] = tuple(a.shape) == tuple(b.shape)
                if tokens in (1024, 2048) and variant == "realistic":
                    with Forced(tk, full):
                        static = [v.clone() for v in ins]
                        fused(*static)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = fused(*static)
                    graph.replay()
                    torch.cuda.synchronize()
                    rec["graph_replay_equal"] = all(torch.equal(a, b) for a, b in zip(captured, ref))
                rec["bitwise"] = (all(rec[n]["equal"] and rec[n]["shape_equal"] for n in OUTPUTS)
                                  and rec["deterministic"] and all(rec["tiles_equal"].values())
                                  and rec.get("graph_replay_equal", True)
                                  and f_stock.seen[-1:] == f_fused.seen[-1:])
                out["cases"].append(rec)
                if not rec["bitwise"]:
                    out["failures"].append({k: rec[k] for k in ("which", "tokens", "full_batch", "variant")})
                log("bitwise", which, tokens, full, variant, "split", f_stock.seen, f_fused.seen, rec["bitwise"],
                    " ".join(f"{n}:{rec[n]['elements_differing']}" for n in OUTPUTS))
    out["passed"] = not out["failures"] and bool(out["cases"])
    return out


def part_timing(args, model_dir, lib, tk, sampler):
    out = {"cells": [], "grids": args.grids, "rounds": args.rounds}
    for which in ("attn", "ffn"):
        prm = mp.mhc_params(model_dir, which)
        stock = mp.mhc_call(prm)
        fused = fused_call(lib, tk, prm)
        for tokens, full in TIMING_CASES:
            gen = torch.Generator(device="cuda").manual_seed(tokens)
            x, res, post, comb = mp.mhc_inputs(tokens, gen, stock)
            floor = tokens * (2 * mp.HC * mp.HIDDEN * 2 + 2 * mp.HIDDEN * 2 + 2 * (mp.HC + mp.HC * mp.HC) * 4)
            k = mp.copies_for(floor)
            ins = [(x.clone(), res.clone(), post.clone(), comb.clone()) for _ in range(k)]
            arms = {"stock": lambda a: stock(*a)}
            for g in args.grids:
                arms[f"fused_g{g}"] = (lambda a, g=g: fused(*a, grid=g))
            for tm in TILES:
                arms[f"tile{tm}"] = (lambda a, tm=tm: fused(*a, tile=tm))
            arms["fused_default"] = lambda a: fused(*a)
            samples = {name: [] for name in arms}
            power = {name: [] for name in arms}
            with Forced(tk, full) as forced:
                for _ in range(args.rounds):  # interleaved arms, one graph per arm per round
                    for name, fn in arms.items():
                        calls = [(lambda a=a, fn=fn: fn(a)) for a in ins]
                        t0 = time.time()
                        samples[name].append(mp.graph_ms(calls, args.reps))
                        power[name].append(sampler.window(t0, time.time()))
                phases = phase_clocks(fused, ins[0])
                kern = {}
                for name in ("stock", "fused_default"):
                    rows = mp.kernel_device_us(lambda fn=arms[name]: [fn(a) for a in ins[:4]], 2)
                    kern[name] = {n[:120]: v for n, v in rows.items()
                                  if any(s in n for s in ("mhc", "prenorm", "tessera"))}
            cell = {"which": which, "tokens": tokens, "full_batch": full, "split": forced.seen[-1],
                    "copies": k, "floor_bytes": floor, "floor_ms_at_peak": floor / mp.PEAK_DRAM_GBS / 1e6,
                    "default_grid": mf.default_grid(torch, x.device),
                    "ms_per_site": {n: sorted(v) for n, v in samples.items()},
                    "median_ms_per_site": {n: sorted(v)[len(v) // 2] for n, v in samples.items()},
                    "kernels": kern, "power": power, "fused_phases": phases}
            med = cell["median_ms_per_site"]
            cell["fraction_of_floor"] = {n: cell["floor_ms_at_peak"] / v for n, v in med.items()}
            out["cells"].append(cell)
            log("timing", which, tokens, full, "split", cell["split"],
                " ".join(f"{n}={v:.4f}" for n, v in med.items()), f"floor={cell['floor_ms_at_peak']:.4f}",
                "phases", json.dumps(phases["median_tile_us"]), f"tile={phases['tile_tokens']}",
                f"span={phases['kernel_span_us']:.1f}us")
            del ins
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported")
    ap.add_argument("--parts", default="bitwise,timing")
    ap.add_argument("--grids", type=int, nargs="+", default=[24, 32, 48])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import vllm
    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk

    meta = {"host": os.environ.get("HOST_NAME", socket.gethostname()), "image": os.environ.get("ORACLE_IMAGE"),
            "tessera_head": os.environ.get("TESSERA_HEAD"), "tessera_state": os.environ.get("TESSERA_STATE"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "torch": torch.__version__, "vllm": vllm.__version__,
            "device": torch.cuda.get_device_name(0), "model": args.model, "argv": sys.argv,
            "peak_dram_gbs": mp.PEAK_DRAM_GBS, "utc_start": time.time()}
    t0 = time.time()
    lib = mf.library()
    meta.update(build_s=time.time() - t0, build_id=lib.build_id, library=lib.path,
                source_sha256=lib.source_sha256, flags=list(mf.FLAGS), nvcc_version=lib.nvcc_version)
    log("meta", json.dumps(meta))
    res = {"meta": meta}
    sampler = rpo.PowerSampler()
    sampler.start()
    status = 0
    for part in args.parts.split(","):
        if part == "bitwise":
            res["bitwise"] = part_bitwise(args, Path(args.model), lib, tk)
            if not res["bitwise"]["passed"]:
                status = 1
        elif part == "timing":
            res["timing"] = part_timing(args, Path(args.model), lib, tk, sampler)
        else:
            raise SystemExit(f"unknown part {part}")
        (out_dir / "mhc_fused_probe.json").write_text(json.dumps(res, indent=1) + "\n")
    sampler.stop_flag = True
    meta["utc_end"] = time.time()
    meta["power_sampler"] = sampler.source
    try:
        res["netdata"] = rpo.netdata_window(meta["utc_start"], meta["utc_end"])
    except Exception as exc:  # noqa: BLE001
        res["netdata"] = {"error": f"{type(exc).__name__}: {exc}"}
    (out_dir / "mhc_fused_probe.json").write_text(json.dumps(res, indent=1) + "\n")
    log("done", "bitwise passed" if status == 0 else "BITWISE FAILED", out_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
