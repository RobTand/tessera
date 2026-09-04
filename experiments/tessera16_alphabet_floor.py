"""Is there an alphabet floor?  The window body over a BF16 grid, R = 4..8.

Rob, 2026-09-02: PrismaQuant should be able to allocate ``W(n<=A) A{4,8,16}``
-- trellis-coded weights at any rate up to the activation width, decoded to
the serving tile (NVFP4 tile at A4, per-channel FP8 at A8, a plain BF16 tile
at A16).  Before any library work the question is the **floor**: what error
does the shipping window body reach when the reconstruction alphabet stops
being the binding constraint?  This measures that, and nothing else.  It is a
table of numbers, not a design.

**The arms.**  Everything is the shipping E4M3 wire recipe -- window body,
CHANNEL scale plane, L=14, ``scale_refit=4``, scale-weighted trellis
(``export.E4M3_RECIPE``) -- with one thing changed: the payload grid the
window table snaps to.

  * ``E4M3``   the shipping grid, 256 codes, run through the REAL wire
               (``encode_linear`` -> bytes -> ``read_unit_artifact``), priced
               at the bytes actually written.
  * ``BF16``   the finite normal bf16 values (sign x 8-bit exponent x 7-bit
               mantissa) over a 32-binade window, 8192 codes.  Not
               serialisable -- ``SERIALISABLE_GRIDS`` is closed and the
               ALPHABET plane is one byte per table entry -- so this arm runs
               the identical encoder in memory (``encode_unit`` ->
               ``reconstruct_unit``) and is priced as the E4M3 wire plus the
               second byte its 2^L table entries would cost.  ``--parity``
               proves the in-memory path is byte-for-byte the wire path on
               E4M3 before any BF16 number is believed.

**Why the grid is a window and not all of bf16.**  The table is 2^L
equal-mass Gaussian quantiles at ``sigma`` grid units; at sigma=1 and L=14
they span [7.6e-5, 4.05].  Exponents -29..2 (32 binades x 128 mantissas x 2
signs = 8192 codes, a legal ``PayloadGrid`` size) contain that with ~15
binades of margin at each end, and every value in the window is exactly a
bf16 value, so the table snap **is** bf16 rounding (nearest-value; it differs
from round-to-nearest-even only on exact midpoints, counted by ``--checks``).

**Why sigma is passed explicitly on BF16.**  ``scale_channel._default_sigma``
searches a quarter-binade ladder for the spread whose nearest-value error is
smallest.  On a floating-point grid with 8 exponent bits that error is
scale-free over ~30 binades, so the search is degenerate -- it is choosing
between equals.  sigma=1.0 is stated instead: it puts the table inside the
grid with margin and gives reach 4.00 sigma, next to E4M3's 4.08.
``--checks`` re-runs R=4 at sigma=4.0 and reports the (nil) difference.

**References, all on the same rows.**  EXL3 K=4/5/6/8 from
``/home/rob/dq-runs/exl3-ref`` (the reconstructions
``exl3_reference_quantise.py`` wrote, scored on the last 1024 capture rows --
the split those files were built against), and the per-channel FP8 RTN
LS-refit floor from ``tessera8_bounds`` -- the error no E4M3-tile format at
any rate goes below.  Arithmetic means over the six tensors reproduce
``experiments/results/tessera8_targets.json`` (K6 out 0.017662, K8 0.004785).

**The fold (part B).**  An A16 tile has no scale tensor, so the served weight
is ``bf16(code * row_scale)`` rather than the fp32 product the encoder scored.
Each Tessera arm carries ``out_bf16`` / ``out_fp16`` -- the same reconstruction
rounded to the serving tile's dtype -- and the fold's own contribution
``sqrt(out_fold^2 - out^2)``, which is independent of the trellis.  The
no-fold alternative is the unrounded column: a streamed decode that keeps the
fp16 row word (2 bytes per row, already priced) and multiplies in fp32.

Stages::

    # A + B + D, six GLM-5.3-Flash routed-expert projections
    PYTHONPATH=src:experiments:/home/rob/prismaquant TMPDIR=/home/rob/tmp \
      TRITON_CACHE_DIR=/home/rob/.triton-cache \
      /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
      experiments/tessera16_alphabet_floor.py --stage glm \
      --out experiments/results/tessera16_alphabet_floor_glm.json
    # (as written that is six tensors x two grids at R = 8, ~11 h; the run of
    #  record was stopped after L5.gate_proj.  Add ``--layers 5 --projs
    #  gate_proj`` to reproduce just those rows.)

    # A + B, the six-tensor geomean at R = 4..7 (R = 8 is one tensor only;
    # see the note on the rate-8 cost in ``stage_dense``)
    ... --stage glm --rungs 1024 1280 1536 1792 --no-parity \
        --out experiments/results/tessera16_alphabet_floor_glm47.json

    # C, all 196 Qwen3-0.6B Linears at R = 4, one layer also at R = 8
    ... --stage dense --rungs 1024 2048 --rungs-all 1024 --subset-layer 2 \
        --out experiments/results/tessera16_alphabet_floor_dense.json

    # D's cheap checks (table saturation, snap exactness)
    ... --stage checks --window-bits 12 14 16 \
        --out experiments/results/tessera16_alphabet_floor_checks.json

    # D, is the BF16 alphabet scale-free?  The second line re-runs the large
    # sigma on a shifted binade window, because a sigma the *experiment's*
    # grid window cannot hold is not a fact about the alphabet.
    ... --stage sigma --sigmas 0.25 1.0 4.0 --rungs 1024 \
        --out experiments/results/tessera16_alphabet_floor_sigma.json
    ... --stage sigma --sigmas 4.0 --bf16-exp -27 4 --rungs 1024 \
        --out experiments/results/tessera16_alphabet_floor_sigma_wide.json
"""
from __future__ import annotations

import argparse, glob, json, math, sys, time

import numpy as np
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import GAUSSIAN_SOURCE, E4M3_GRID, PayloadGrid   # noqa: E402
from tessera.decode import reconstruct_unit                            # noqa: E402
from tessera.encode import encode_unit, window_table                   # noqa: E402
from tessera.export import (                                           # noqa: E402
    DEFAULT_CODE, DEFAULT_SCALE_REFIT, DEFAULT_TRELLIS_WEIGHTING,
    E4M3_WINDOW_BITS, _plan_for, encode_linear_planes)
from tessera.manifest import BodyKind, ScalePlaneKind                  # noqa: E402
from tessera.scale_channel import default_channel_sigma                # noqa: E402
from tessera.stock import materialize_stock                            # noqa: E402
from tessera.unit_artifact import read_unit_artifact                   # noqa: E402
from tessera8_bounds import e4m3_rtn, ls_refit                         # noqa: E402
from tessera8_targets import ACT, E4M3_MAX, EXL3, SRC                  # noqa: E402

#: The BF16 grid's exponent window.  32 binades is the smallest power-of-two
#: count of binades that brackets the L=14 table's [7.6e-5, 4.05] with margin
#: at both ends; a PayloadGrid size must be a power of two.
BF16_EXP_LO, BF16_EXP_HI = -29, 2
#: bf16 is scale-free over its exponent field, so this is a placement, not a fit.
BF16_SIGMA = 1.0
DENSE_H = "/home/rob/tessera-runs/stock/h_diag.pt"
DENSE_SRC = "/home/rob/models/Qwen3-0.6B"
DENSE_FP8RTN = "/home/rob/tessera-runs/stock/qwen3-0.6b-fp8-rtn"
DENSE_NVFP4 = "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported"


def bf16_grid(lo: int = BF16_EXP_LO, hi: int = BF16_EXP_HI) -> PayloadGrid:
    """Signed normal bf16 values with unbiased exponent in ``[lo, hi]``.

    ``(1 + m/128) * 2^e`` for m in 0..127 is exactly the bf16 significand
    ladder, so every value here is a bf16 value and the nearest-value snap the
    window table performs is bf16 rounding over the window.  Zero and the
    subnormals are outside it and are never nearest to any table quantile.
    """
    mags = [(1.0 + m / 128.0) * 2.0 ** e for e in range(lo, hi + 1) for m in range(128)]
    values = tuple([+v for v in mags] + [-v for v in mags])
    return PayloadGrid(f"BF16e{lo}:{hi}", values)


def geomean(xs) -> float:
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


# ---------------------------------------------------------------- encoders

def window_kwargs(grid: PayloadGrid, q256: int, cols: int, L: int, sigma, refit: int):
    """The exporter's own window/CHANNEL settings, resolved for ``encode_unit``.

    Mirrors ``export.encode_linear_planes`` under ``E4M3_RECIPE`` exactly --
    including ``trellis_weighting='scale'``, which ``encode_unit`` does NOT
    default to -- so the in-memory arm and the wire arm are one encoder.
    """
    rates, forest = _plan_for(grid, q256, cols, BodyKind.WINDOW, sigma)
    return rates, dict(
        forest=forest, rates=rates, code=DEFAULT_CODE, completion=0,
        scale_refit=refit, span=1, scale_plane=ScalePlaneKind.CHANNEL,
        trellis_weighting=DEFAULT_TRELLIS_WEIGHTING, body=BodyKind.WINDOW,
        window_bits=L, window_seed=0, window_sigma=None, channel_sigma=sigma,
    )


def encode_memory(w, grid, q256, L, sigma, refit=DEFAULT_SCALE_REFIT):
    """Encode in memory and reconstruct.  Returns ``(hat, rates, unit, secs)``."""
    t0 = time.time()
    rates, kw = window_kwargs(grid, q256, w.shape[1], L, sigma, refit)
    forest = kw.pop("forest"); kw.pop("rates"); code = kw.pop("code")
    unit = encode_unit(w, forest, rates, code, **kw)
    hat = reconstruct_unit(unit, forest, code)
    return hat, rates, unit, time.time() - t0


def encode_wire(w, grid, q256, L, sigma, refit=DEFAULT_SCALE_REFIT, name="unit"):
    """Encode to real artifact bytes and read them back.  ``(hat, bytes, secs)``."""
    t0 = time.time()
    exported, unit, forests = encode_linear_planes(
        w, grid=grid, q256=q256, name=name, scale_refit=refit,
        body=BodyKind.WINDOW, window_bits=L, scale_plane=ScalePlaneKind.CHANNEL,
        span=1, channel_sigma=sigma, verify=True,
    )
    return read_unit_artifact(exported.blob, device=w.device), len(exported.blob), time.time() - t0


def payload_bits_per_weight(rates, arity: int) -> float:
    return sum(rates) / (len(rates) * arity)


# ---------------------------------------------------------------- stage: glm

GLM_RUNGS = [1024, 1280, 1536, 1792, 2048]          # R = 4, 5, 6, 7, 8
REPRO_RUNGS = [960, 1216]                            # the reach-json rungs


def stage_glm(a):
    from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

    gb = bf16_grid()
    sig8 = default_channel_sigma(E4M3_GRID)
    out = {"args": vars(a), "grid": {"bf16": {"name": gb.name, "codes": gb.size,
                                              "sigma": BF16_SIGMA},
                                     "e4m3": {"codes": E4M3_GRID.size, "sigma": sig8}},
           "experts": {}}
    out_path = Path(a.out)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)
        Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    dev = "cuda"
    parity_done = not a.parity
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
        g = select_mse_grid_input_global_scale([x_fit])
        xq4 = nvfp4_activation_qdq_served(x_ev, g).float()
        xq8 = fp8_dynamic_activation_qdq_vllm(x_ev).dequant.float()
        del x_fit, xa, blob
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            tname = f"L{layer}.{proj}"
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            res = {}
            log(f"\n== {tname} {tuple(w.shape)}")
            log(f"    {'arm':<44} {'bpp':>7} {'wt':>8} {'out':>8} {'o_bf16':>8} "
                f"{'o_fp16':>8} {'a4':>8} {'a8':>8} {'s':>5}")

            def rec(arm, hat, bpp, secs=0.0, extra=None):
                fold_b = hat.to(torch.bfloat16).float()
                fold_h = hat.to(torch.float16).float()
                r = {"bpp": bpp,
                     "wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "out_bf16": float((x_ev @ fold_b.T - y).norm() / ny),
                     "out_fp16": float((x_ev @ fold_h.T - y).norm() / ny),
                     "wt_bf16": float((fold_b - w).norm() / nw),
                     "a4": float((xq4 @ hat.T - y).norm() / ny),
                     "a8": float((xq8 @ hat.T - y).norm() / ny),
                     "secs": secs}
                if extra:
                    r.update(extra)
                res[arm] = r
                log(f"    {arm:<44} {bpp:7.4f} {r['wt']:8.5f} {r['out']:8.5f} "
                    f"{r['out_bf16']:8.5f} {r['out_fp16']:8.5f} {r['a4']:8.5f} "
                    f"{r['a8']:8.5f} {secs:5.0f}")

            for K in a.exl3:
                p = Path(EXL3) / f"L{layer}_{proj}_K{K}.pt"
                if p.exists():
                    rec(f"EXL3 K={K}", torch.load(p, map_location=dev).float(), K + 0.0117)
            s = w.abs().amax(dim=1, keepdim=True) / E4M3_MAX
            s = ls_refit(w, s, 6, dim=1)
            rec("FP8 RTN per-channel LS-refit (E4M3 floor)",
                e4m3_rtn(w / s) * s, 8.0 + 32 / w.shape[1])

            e4m3_bpp = {}
            for L in a.window_bits:
                for q in (REPRO_RUNGS if L == E4M3_WINDOW_BITS else []) + list(a.rungs):
                    hat, nbytes, secs = encode_wire(w, E4M3_GRID, q, L, sig8, name=tname)
                    bpp = 8 * nbytes / w.numel()
                    e4m3_bpp[(L, q)] = bpp
                    rec(f"E4M3 window q{q} L={L}", hat, bpp, secs,
                        extra={"bytes": nbytes, "table_bytes": 1 << L})
                    del hat
                    torch.cuda.empty_cache()
            if not parity_done:
                # The in-memory path must BE the wire path before any BF16
                # number built on it is believed.
                q = a.rungs[0]; L = a.window_bits[0]
                hat_m, rates, unit, _ = encode_memory(w, E4M3_GRID, q, L, sig8)
                hat_w, _, _ = encode_wire(w, E4M3_GRID, q, L, sig8, name=tname)
                out["parity"] = {
                    "tensor": tname, "q256": q, "L": L,
                    "dtypes": [str(hat_m.dtype), str(hat_w.dtype)],
                    "equal": bool(torch.equal(hat_m.float(), hat_w.float())),
                    "max_abs_diff": float((hat_m.float() - hat_w.float()).abs().max())}
                log(f"    [parity] encode_unit == wire on {tname} q{q} L={L}: "
                    f"{out['parity']['equal']}")
                del hat_m, hat_w, unit
                torch.cuda.empty_cache()
                parity_done = True
            for L in a.window_bits:
                for q in a.rungs:
                    hat, rates, unit, secs = encode_memory(w, gb, q, L, BF16_SIGMA)
                    # Priced as the E4M3 wire at the same rung plus the second
                    # byte each of the 2^L table entries would carry.
                    base = e4m3_bpp.get((L, q))
                    extra_bpp = (1 << L) * 8 / w.numel()
                    bpp = None if base is None else base + extra_bpp
                    analytic = (payload_bits_per_weight(rates, gb.arity)
                                + (w.shape[0] * 2 + 4) * 8 / w.numel()
                                + (1 << L) * 2 * 8 / w.numel())
                    rec(f"BF16 window q{q} L={L}", hat,
                        analytic if bpp is None else bpp, secs,
                        extra={"bpp_analytic": analytic, "table_bytes": (1 << L) * 2,
                               "distinct_rows": int(unit.scale_rows.unique().numel())})
                    del hat, unit
                    torch.cuda.empty_cache()
            out["experts"][tname] = res
            out_path.write_text(json.dumps(out, indent=1))
            del w, y
            torch.cuda.empty_cache()
        del x_ev, xq4, xq8
        torch.cuda.empty_cache()

    summarise(out, log)
    out_path.write_text(json.dumps(out, indent=1))
    log(f"\nwrote {a.out}")


def summarise(out, log):
    arms, keys = {}, ("bpp", "wt", "out", "out_bf16", "out_fp16", "a4", "a8")
    for res in out["experts"].values():
        for arm, r in res.items():
            arms.setdefault(arm, []).append(r)
    n = len(out["experts"])
    out["summary"] = {}
    log(f"\n== arithmetic mean over {n} tensors "
        f"(the aggregation tessera8_targets.json uses)")
    log(f"    {'arm':<44} {'bpp':>7} {'wt':>8} {'out':>8} {'o_bf16':>8} {'o_fp16':>8} "
        f"{'a4':>8} {'a8':>8} {'gm(out)':>8}")
    for arm, rs in arms.items():
        if len(rs) != n:
            continue
        m = {k: sum(r[k] for r in rs) / len(rs) for k in keys}
        m["out_geomean"] = geomean([r["out"] for r in rs])
        m["wt_geomean"] = geomean([r["wt"] for r in rs])
        m["fold_bf16_only"] = math.sqrt(max(m["out_bf16"] ** 2 - m["out"] ** 2, 0.0))
        m["fold_fp16_only"] = math.sqrt(max(m["out_fp16"] ** 2 - m["out"] ** 2, 0.0))
        out["summary"][arm] = m
        log(f"    {arm:<44} {m['bpp']:7.4f} {m['wt']:8.5f} {m['out']:8.5f} "
            f"{m['out_bf16']:8.5f} {m['out_fp16']:8.5f} {m['a4']:8.5f} {m['a8']:8.5f} "
            f"{m['out_geomean']:8.5f}")
    # Ratios to each EXL3 rung, geomean over tensors (the memory-note convention).
    exl3 = {arm: rs for arm, rs in arms.items() if arm.startswith("EXL3")}
    out["ratios"] = {}
    for arm, rs in arms.items():
        if arm.startswith("EXL3") or len(rs) != n:
            continue
        out["ratios"][arm] = {
            ref: {"out": geomean([r["out"] / q["out"] for r, q in zip(rs, qs)]),
                  "wt": geomean([r["wt"] / q["wt"] for r, q in zip(rs, qs)]),
                  "dbpp": out["summary"][arm]["bpp"] - sum(q["bpp"] for q in qs) / n}
            for ref, qs in exl3.items()}


# -------------------------------------------------------------- stage: dense

def open_all(d):
    handles = [safe_open(f, framework="pt") for f in sorted(glob.glob(d + "/*.safetensors"))]
    idx = {}
    for h in handles:
        for k in h.keys():
            idx[k] = h
    return idx


def stage_dense(a):
    """C: all 196 Qwen3-0.6B Linears, H-weighted, against the census anchors."""
    dev = "cuda"
    H = {k: v.to(dev, torch.float32) for k, v in torch.load(DENSE_H).items()}
    src = open_all(DENSE_SRC)
    f8 = open_all(DENSE_FP8RTN)
    nv = open_all(DENSE_NVFP4)
    gb = bf16_grid()
    sig8 = default_channel_sigma(E4M3_GRID)
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)

    def nvfp4(name, rows, cols):
        pk = nv[name + ".weight_packed"].get_tensor(name + ".weight_packed").to(dev)
        s = nv[name + ".weight_scale"].get_tensor(name + ".weight_scale").to(dev).float()
        g = nv[name + ".weight_global_scale"].get_tensor(name + ".weight_global_scale").to(dev).float()
        lo, hi = (pk & 0xF).long(), (pk >> 4).long()
        dq = lambda t: e2m1[t & 7] * torch.where(t >= 8, -1.0, 1.0)
        return torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols) * (s / g).repeat_interleave(16, dim=1)

    def fp8pair(idx, name):
        return (idx[name + ".weight"].get_tensor(name + ".weight").to(dev).float()
                * idx[name + ".weight_scale"].get_tensor(name + ".weight_scale").to(dev).float())

    def err(W, Wd, h):
        E = Wd - W
        return (math.sqrt((E * E).sum().item() / (W * W).sum().item()),
                math.sqrt(((E * E).sum(0) * h).sum().item() / ((W * W).sum(0) * h).sum().item()))

    out = {"args": vars(a), "tensors": {}}
    out_path = Path(a.out)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)
        out_path.with_suffix(".log").write_text("\n".join(lines) + "\n")

    t0 = time.time()
    names = sorted(H)[: a.limit] if a.limit else sorted(H)
    # The rate-8 arms run on a subset, and the reason is mechanical, not a
    # judgement about which tensors matter: the fused window Viterbi's class
    # scan is a ``tl.static_range(1, 2^R)`` unroll, so at R=8 the tile is 4
    # classes x 2 columns and one 2048x4096 encode takes ~12 minutes against
    # 33 s at R=7 and 5 s at R=4.  Over 440 M Qwen parameters that is ~10 h
    # per arm.  The subset is every role of one mid-network layer, so the
    # comparison is like-for-like: every arm is re-summarised over it.  The
    # default layer 2 was chosen a priori from the dense-outlier memo's
    # pointer at rows that overrun the window table's reach; measured, its
    # R=4 arms land slightly *easier* than the 196-tensor census average
    # (E4M3 0.0738 vs 0.0765), so the subset is representative, not a pick.
    sub = [n for n in names
           if any(f".layers.{L}." in n for L in a.subset_layer)]
    out["subset"] = sub
    log(f"rate-8 arms on {len(sub)} tensors of layers {a.subset_layer}: "
        + ", ".join(".".join(x.split(".")[-4:-1]) for x in sub))
    for i, name in enumerate(names):
        W = src[name + ".weight"].get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
        rows, cols = W.shape
        h = H[name]
        rec = {"rows": rows, "cols": cols}
        rec["fp8rtn"] = err(W, fp8pair(f8, name), h)
        rec["nvfp4"] = err(W, nvfp4(name, rows, cols), h)
        for q in (a.rungs if name in sub else a.rungs_all):
            # E4M3 through the exporter's own path (the census's arm B).
            _, unit, forests = encode_linear_planes(
                W, grid=E4M3_GRID, q256=q, name=name, verify=False)
            st = materialize_stock(unit, forests, DEFAULT_CODE)
            rec[f"e4m3_q{q}"] = err(
                W, st["weight"].to(dev).float() * st["weight_scale"].to(dev).float(), h)
            if i == 0:
                ref = reconstruct_unit(unit, forests, DEFAULT_CODE)
                mat = st["weight"].to(dev).float() * st["weight_scale"].to(dev).float()
                out["materialise_parity"] = {
                    "tensor": name, "q256": q,
                    "max_abs_diff": float((ref - mat).abs().max()),
                    "rel": float((ref - mat).norm() / ref.norm())}
            del unit, forests, st
            hat, rates, unit, _ = encode_memory(W, gb, q, E4M3_WINDOW_BITS, BF16_SIGMA)
            rec[f"bf16_q{q}"] = err(W, hat, h)
            rec[f"bf16fold_q{q}"] = err(W, hat.to(torch.bfloat16).float(), h)
            del hat, unit
            torch.cuda.empty_cache()
        out["tensors"][name] = rec
        if i % 10 == 0:
            log(f"[{i}/{len(names)}] {name} {time.time()-t0:.0f}s "
                + " ".join(f"{k} {v[1]:.4f}" for k, v in rec.items() if isinstance(v, (list, tuple))))
            out_path.write_text(json.dumps(out, indent=1))
        del W
        torch.cuda.empty_cache()

    rs = list(out["tensors"].values())
    subs = [out["tensors"][n] for n in sub if n in out["tensors"]]
    def table(rows_, label, store):
        keys = sorted({k for r in rows_ for k, v in r.items()
                       if isinstance(v, (list, tuple)) and len(v) == 2}
                      , key=lambda k: (not k.startswith("fp8"), k))
        keys = [k for k in keys if all(k in r for r in rows_)]
        out[store] = {k: {"plain": geomean([r[k][0] for r in rows_]),
                          "weighted": geomean([r[k][1] for r in rows_])} for k in keys}
        log(f"\n== geomean over {len(rows_)} {label}")
        log(f"    {'arm':<16} {'plain':>9} {'H-weighted':>11}")
        for k, v in out[store].items():
            log(f"    {k:<16} {v['plain']:9.4f} {v['weighted']:11.4f}")
    table(rs, "Qwen3-0.6B Linears (all)", "summary")
    if subs:
        table(subs, f"Linears of layer {a.subset_layer} (the rate-8 subset)",
              "summary_subset")
    out_path.write_text(json.dumps(out, indent=1))
    log(f"\nwrote {a.out}")


# ------------------------------------------------------------- stage: checks

def stage_checks(a):
    """D: table saturation, snap exactness, sigma invariance of the BF16 grid."""
    out = {"args": vars(a), "tables": {}, "snap": {}}
    log_lines = []

    def log(s):
        print(s, flush=True); log_lines.append(s)

    gb = bf16_grid()
    sig8 = default_channel_sigma(E4M3_GRID)
    gv_b = torch.tensor(gb.values)
    gv_8 = torch.tensor(E4M3_GRID.values)
    log(f"BF16 grid {gb.name}: {gb.size} codes, peak {float(gv_b.abs().max())}, "
        f"min |v| {float(gv_b.abs().min()):.3e}")
    for L in a.window_bits:
        for gname, grid, gv, sigma in (("E4M3", E4M3_GRID, gv_8, sig8),
                                       ("BF16", gb, gv_b, BF16_SIGMA)):
            tab = window_table(grid, L, sigma=sigma, seed=0, half=16)
            vals = gv[tab.long()]
            out["tables"][f"{gname}_L{L}"] = {
                "states": 1 << L,
                "distinct_values": int(vals.unique().numel()),
                "grid_codes": grid.size,
                "reach_grid_units": float(vals.abs().max()),
                "reach_sigma": float(vals.abs().max()) / sigma,
            }
            log(f"  {gname} L={L}: {int(vals.unique().numel())} distinct of {1<<L} states "
                f"({grid.size} codes available), reach {float(vals.abs().max())/sigma:.3f} sigma")
            # The snap against exact bf16 round-to-nearest-even.
            if gname == "BF16":
                q = torch.tensor(GAUSSIAN_SOURCE(1 << L, sigma))
                rne = q.to(torch.bfloat16).float()
                snapped = torch.sort(vals).values
                diff = (snapped - torch.sort(rne).values).abs()
                out["snap"][f"L{L}"] = {
                    "entries": 1 << L,
                    "differ_from_rne": int((diff > 0).sum()),
                    "max_abs_diff": float(diff.max()),
                    "max_rel_diff": float((diff / torch.sort(rne).values.abs().clamp_min(1e-30)).max()),
                }
                log(f"    snap vs bf16 RNE: {int((diff>0).sum())} of {1<<L} entries differ, "
                    f"max |d| {float(diff.max()):.2e}")
    out["log"] = log_lines
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


def stage_price(a):
    """C's pricing, and the row-scale census.

    ``stage_dense`` records errors only, because its arms are read against
    each other on identical bytes-per-tensor.  The census anchors, though,
    are quoted in bpp, so the arms have to be priced the same way: total
    bytes over total quantizable parameters, param-weighted, table included.
    The table is why this cannot be a constant -- 2^14 entries is 0.008 bpp
    on an 8 M-parameter GLM expert and 0.125 bpp (0.25 for BF16) on a
    1024x1024 Qwen Linear, and the acceptance is that every arm carries it.

    The same pass answers the last part of D on a second model: whether the
    per-row scale is doing work is the spread of the row RMS, which is a
    property of the weights and needs no encode.
    """
    d = json.load(open(a.dense_json))
    src = open_all(DENSE_SRC)
    rows = d["tensors"]
    out = {"args": vars(a), "dense_json": a.dense_json, "price": {}, "row_rms": {}}

    def bytes_for(arm, r):
        n = r["rows"] * r["cols"]
        if arm == "fp8rtn":                    # one byte a weight + fp32 row scale
            return n + r["rows"] * 4
        if arm == "nvfp4":                     # 4 bits + one e4m3 scale per 16
            return n // 2 + n // 16 + 4
        q = int(arm.rsplit("_q", 1)[1])
        # The window wire: payload at q256/256 bits a weight, the CHANNEL
        # plane (one fp16 word a row plus the fp32 global), and the ALPHABET
        # plane -- one byte an entry for E4M3, two for BF16.
        body = n * q / 256.0 / 8.0
        plane = r["rows"] * 2 + 4
        table = (1 << E4M3_WINDOW_BITS) * (2 if arm.startswith("bf16") else 1)
        return body + plane + table

    arms = sorted({k for r in rows.values() for k, v in r.items()
                   if isinstance(v, (list, tuple)) and len(v) == 2})
    for arm in arms:
        rs = [r for r in rows.values() if arm in r]
        nb = sum(bytes_for(arm, r) for r in rs)
        npar = sum(r["rows"] * r["cols"] for r in rs)
        out["price"][arm] = {
            "tensors": len(rs), "params": npar, "bpp": 8.0 * nb / npar,
            "plain": geomean([r[arm][0] for r in rs]),
            "weighted": geomean([r[arm][1] for r in rs])}
    print(f"  {'arm':<16} {'n':>4} {'bpp':>7} {'plain':>9} {'H-weighted':>11}")
    for k, v in out["price"].items():
        print(f"  {k:<16} {v['tensors']:4d} {v['bpp']:7.4f} "
              f"{v['plain']:9.4f} {v['weighted']:11.4f}", flush=True)

    sp = []
    for name in sorted(rows):
        W = src[name + ".weight"].get_tensor(name + ".weight").float()
        rm = W.pow(2).mean(1).sqrt()
        sp.append(float(rm.max() / rm.min()))
    out["row_rms"] = {"tensors": len(sp), "median_spread": float(np.median(sp)),
                      "max_spread": max(sp), "min_spread": min(sp)}
    print(f"  row RMS spread over {len(sp)} Linears: median {np.median(sp):.2f}x, "
          f"max {max(sp):.2f}x, min {min(sp):.2f}x", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


def stage_sigma(a):
    """D: is the BF16 grid scale-free?  Same tensor, two sigmas, one rung."""
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    name = f"model.language_model.layers.{a.layers[0]}.mlp.experts.0.{a.projs[0]}.weight"
    with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
        w = f.get_tensor(name).contiguous().cuda().float()
    gb = bf16_grid(*a.bf16_exp)
    print(f"  grid {gb.name}: {gb.size} codes, peak {max(gb.values):.5g}",
          flush=True)
    rms = w.pow(2).mean(1).sqrt()
    out = {"args": vars(a), "tensor": name, "sigma": {},
           # Whether a per-row scale is *needed* at all is a property of the
           # weights, not of the refit: it is the spread of the row RMS.
           "row_rms": {"max": float(rms.max()), "min": float(rms.min()),
                       "median": float(rms.median()),
                       "spread": float(rms.max() / rms.min()),
                       "p99_over_p1": float(rms.quantile(0.99) / rms.quantile(0.01))}}
    print(f"  row RMS spread max/min {out['row_rms']['spread']:.2f}x, "
          f"p99/p1 {out['row_rms']['p99_over_p1']:.2f}x", flush=True)
    for sigma in a.sigmas:
        for refit in a.refits:
            hat, rates, unit, secs = encode_memory(
                w, gb, a.rungs[0], a.window_bits[0], sigma, refit)
            key = f"sigma{sigma}_refit{refit}"
            out["sigma"][key] = {"wt": float((hat - w).norm() / w.norm()),
                                 "row_scales_distinct": int(unit.scale_rows.unique().numel()),
                                 "rows": int(unit.scale_rows.numel()), "secs": secs}
            print(f"  {key}: wt {out['sigma'][key]['wt']:.6f} "
                  f"({out['sigma'][key]['row_scales_distinct']} distinct row words) "
                  f"{secs:.0f}s", flush=True)
            del hat, unit
            torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["glm", "dense", "checks", "sigma",
                                        "price"], default="glm")
    ap.add_argument("--dense-json",
                    default="experiments/results/tessera16_alphabet_floor_dense.json",
                    help="price: the dense stage's output to price")
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rungs", type=int, nargs="+", default=GLM_RUNGS)
    ap.add_argument("--window-bits", type=int, nargs="+", default=[E4M3_WINDOW_BITS])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--bf16-exp", type=int, nargs=2,
                    default=[BF16_EXP_LO, BF16_EXP_HI],
                    help="sigma stage: the binade window of the BF16 grid.  A "
                         "sigma the window cannot hold is the window's fault, "
                         "not the alphabet's, so the scale-freeness check "
                         "re-runs the large sigma on a shifted window.")
    ap.add_argument("--refits", type=int, nargs="+", default=[DEFAULT_SCALE_REFIT, 0])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rungs-all", type=int, nargs="+", default=[1024],
                    help="dense: rungs run on every tensor")
    ap.add_argument("--subset-layer", type=int, nargs="+", default=[2],
                    help="dense: the layers whose Linears also get --rungs")
    ap.add_argument("--parity", action="store_true", default=True)
    ap.add_argument("--no-parity", dest="parity", action="store_false")
    ap.add_argument("--out", default="experiments/results/tessera16_alphabet_floor.json")
    a = ap.parse_args()
    {"glm": stage_glm, "dense": stage_dense, "checks": stage_checks,
     "sigma": stage_sigma, "price": stage_price}[a.stage](a)


if __name__ == "__main__":
    main()
