"""``(L, sigma)`` for the 16-bit route: the two numbers ``BF16_RECIPE`` inherited.

Issue #18.  ``export.BF16_RECIPE`` is ``E4M3_RECIPE`` with the alphabet
swapped, so its window width ``L = 14`` and its modelled source spread
``channel_sigma = 1.0`` were **stated, not searched** -- and they are exactly
the pair that decides *reach*, the residual the route's own receipt names.
This searches them, through the real wire, at the bytes each arm actually
writes.

**The claim this registers before it measures anything.**  On the BF16 recipe
``window_sigma`` is ``None``, so the table is built at ``sigma =
channel_sigma`` (``encode_unit``'s CHANNEL branch) and the rows are scaled to
``channel_sigma`` grid units of RMS.  Both ends move together, so
``channel_sigma`` is a **gauge**, not a parameter -- and on a grid closed under
multiplication by two it is an *exact* one:

* ``window_table``'s entries are equal-mass Gaussian quantiles times sigma,
  snapped to the nearest grid value.  bf16 values are closed under x2 away
  from the exponent extremes and nearest-value snapping commutes with x2, so
  ``table(2s) = 2 table(s)`` exactly.
* ``initial_channel_scale`` divides each row's RMS by sigma (or, past the
  body's reach, by ``reach * rms / amax`` -- and the reach doubles too), so
  every row scale halves exactly.
* ``channel_global`` is a power of two, so it halves exactly and the **stored
  fp16 word does not change at all**.
* The trellis therefore sees targets twice as large against a table twice as
  large, picks the same codes, and reconstructs the same numbers.

So the ``gauge`` stage predicts **a bit-identical decoded tensor** at dyadic
sigmas on BF16, and predicts the same shift is *not* a gauge on E4M3, whose
256 values are not closed under x2 at either end of a table that already spans
[0.007, 381] grid units at the default spread of 94.2.  A prediction that
fails is the finding; a prediction that holds says the sigma half of #18 has
nothing to search.

**The invariant is the tensor, not the file.**  A gauge shift is *written
down*: the ALPHABET plane holds the doubled table and the fp32 global halves,
so the artifact's bytes change while the tensor they decode to does not.  Both
hashes are reported for that reason, and the gauge claim is made on ``tsha``
(the decoded tensor) -- reading it off the file hash would report a gauge as a
difference.

**What is left after the gauge is the real axis**, and the ``reach`` stage
sweeps it: the *ratio* between the table's spread and the row's, held apart by
passing ``window_sigma`` explicitly.  ``reach / channel_sigma`` is how many
row-RMS the body can emit; lower spends resolution on the tail, higher spends
it on the bulk and hands more rows to the per-row reach start.

**And ``L`` is a byte question, not a quality question**, which is why it is
swept at measured bpp and read off a frontier.  A BF16 table costs ``2^L x 2``
bytes on the ALPHABET plane -- 32 KB at L=14, which is 0.031 bpp on a 2048x4096
GLM expert and **0.25 bpp on a 1024x1024 Qwen Linear**.  At R=8 that is 3% of
the artifact.  Whether the shaping a wider window buys is worth its own table
is therefore a per-shape answer, and both shapes are measured.

**The controls.**  Every arm of a stage runs in **one process**, in a fixed
order, and the **default arm is run first and repeated last**.  The encoder is
deterministic, so the repeat is not a noise estimate: it pins that no arm
leaked state into a later one (a mis-keyed table cache is exactly the failure
it catches) by asserting the two are byte-identical, and it reports both wall
clocks so box drift shows up as disagreement between two baselines rather than
as a factor in a ratio.  Every arm is priced at the bytes it wrote, so no arm
borrows another's plane.

Stages::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=src:experiments"

    # is channel_sigma a gauge?  (minutes)
    env $E $P experiments/bf16_l_sigma_sweep.py --stage gauge --out OUT/gauge.json

    # the table's reach in row-RMS units (tens of minutes)
    env $E $P experiments/bf16_l_sigma_sweep.py --stage reach --out OUT/reach.json

    # L at matched bytes, small units and large ones
    env $E $P experiments/bf16_l_sigma_sweep.py --stage dense-l --out OUT/dense_l.json
    env $E $P experiments/bf16_l_sigma_sweep.py --stage glm-l   --out OUT/glm_l.json

    # the PAIR: L x ratio jointly, across the rate range, against a
    # byte-matched shipped reference built at the rung that spends the same
    # bytes.  Pre-registered reading in the comment above ``PAIR_BITS``.
    env $E $P experiments/bf16_l_sigma_sweep.py --stage pair-glm \\
        --layers 5 20 42 --projs gate_proj up_proj --out OUT/pair_glm.json
    env $E $P experiments/bf16_l_sigma_sweep.py --stage pair-dense --out OUT/pair_dense.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import BF16_GRID, E4M3_GRID  # noqa: E402
from tessera.encode import grid_vector_table, window_table  # noqa: E402
from tessera.export import (  # noqa: E402
    BF16_CHANNEL_SIGMA,
    BF16_WINDOW_BITS,
    encode_linear_planes,
)
from tessera.scale_channel import default_channel_sigma  # noqa: E402
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

# The same sources, the same held-out rows and the same metric definitions the
# route's own weight-space receipt used, so these numbers sit beside those.
from bf16_route_weight_space import (  # noqa: E402
    DENSE_H,
    DENSE_SRC,
    EXL3,
    GLM_ACT,
    GLM_SRC,
    fp8_floor,
    geomean,
    open_all,
)

#: Four dense Qwen Linears spanning the shapes where the table's bytes bite,
#: including the two roles the reach fix moved most (``down_proj``, ``k_proj``).
DENSE_UNITS = [
    "model.layers.2.mlp.down_proj",      # 1024 x 3072
    "model.layers.2.self_attn.q_proj",   # 2048 x 1024
    "model.layers.2.self_attn.k_proj",   # 1024 x 1024, the smallest
    "model.layers.14.mlp.gate_proj",     # 3072 x 1024
]


#: The grids whose reach can be swept.  ``channel_sigma`` is a gauge on BF16
#: and provably not one on E4M3 (#36), so the stage has to name which it is on.
GRIDS = {"bf16": BF16_GRID, "e4m3": E4M3_GRID}


def base_channel_sigma(grid) -> float:
    """The spread the shipped recipe would pick for this grid.

    BF16 carries its own constant because its alphabet is closed under x2 and
    the recipe pins 1.0 rather than deriving it; every other grid asks
    ``default_channel_sigma``, which is the dyadic ladder #36 is about.
    """
    return BF16_CHANNEL_SIGMA if grid is BF16_GRID else default_channel_sigma(grid)


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def tensor_sha(t: torch.Tensor) -> str:
    """The decoded tensor's hash -- the quantity a gauge leaves alone."""
    return sha(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())


def frontier(points: "list[tuple[float, float, str]]") -> "list[str]":
    """The labels on the lower-left hull of ``(bpp, error)`` -- the RD frontier.

    An arm is on it when no other arm is at least as cheap *and* at least as
    accurate.  That is the matched-bytes reading of an ``L`` sweep: a wider
    table is worth its own bytes only if nothing narrower reaches its error
    for less.
    """
    out = []
    for bpp, err, label in points:
        if not any(b <= bpp and e <= err and (b, e) != (bpp, err)
                   for b, e, _ in points):
            out.append(label)
    return out


def reach_stats(w: torch.Tensor, grid, window_bits: int, table_sigma: float,
                channel_sigma: float) -> dict:
    """How far the body can reach, in row-RMS, and how many rows exceed it.

    The same quantities ``initial_channel_scale`` computes, reported rather
    than only acted on: a sweep over the spread is a sweep over this.
    """
    codes = window_table(grid, window_bits, sigma=table_sigma, seed=0,
                         half=16, device=w.device)
    reach = float(grid_vector_table(grid, w.device)[codes.long()].abs().max())
    rms = w.float().pow(2).mean(dim=1).sqrt()
    amax = w.float().abs().amax(dim=1)
    over = (amax * channel_sigma > reach * rms)
    return {
        "reach_grid_units": reach,
        "reach_row_rms": reach / channel_sigma,
        "rows_over_reach": float(over.float().mean()),
        "max_z": float((amax / rms.clamp_min(1e-30)).max()),
    }


class Bench:
    """One stage's arms, run in order in one process, each priced at its bytes."""

    def __init__(self, out_path: str):
        self.out = Path(out_path)
        self.lines: "list[str]" = []
        self.doc: dict = {}

    def log(self, s: str) -> None:
        print(s, flush=True)
        self.lines.append(s)
        self.out.with_suffix(".log").write_text("\n".join(self.lines) + "\n")

    def save(self) -> None:
        self.out.write_text(json.dumps(self.doc, indent=1))

    def header(self, cols) -> None:
        self.log("    " + f"{'arm':<30}" + "".join(f"{c:>10}" for c in cols)
                 + f"{'sha':>18}{'s':>7}")

    def row(self, arm: str, r: dict, cols) -> None:
        self.log("    " + f"{arm:<30}"
                 + "".join(f"{r.get(c, float('nan')):10.5f}" for c in cols)
                 + f"{r.get('sha', ''):>18}{r.get('secs', 0.0):7.0f}")


def encode_arm(w, grid, q256, name, *, window_bits, window_sigma, channel_sigma):
    started = time.time()
    exported, _unit, _forests = encode_linear_planes(
        w, grid=grid, q256=q256, name=name, window_bits=window_bits,
        window_sigma=window_sigma, channel_sigma=channel_sigma, verify=True,
    )
    hat = read_unit_artifact(exported.blob, device=w.device)
    return hat, float(exported.bpp), sha(exported.blob), time.time() - started


def try_arm(b: "Bench", label: str, fn):
    """Run one arm; record a failure rather than losing the rest of the stage.

    A wide table is the arm most likely to run out of memory (L=16 is 65536
    Viterbi states), and an unattended chain that dies on it loses every arm
    behind it.  The failure is written into the record so it is visible as a
    gap and not as an absence.
    """
    try:
        return fn()
    except Exception as exc:                                  # noqa: BLE001
        torch.cuda.empty_cache()
        b.log(f"    {label:<30} !! FAILED: {type(exc).__name__}: {exc}")
        return None


def check_repeat_tensor(b: Bench, first: dict, last: dict, label: str) -> dict:
    """``check_repeat`` on the decoded tensor as well as the file."""
    v = check_repeat(b, first, last, label)
    v["tsha_first"] = first.get("tsha")
    v["tsha_repeat"] = last.get("tsha")
    v["tensor_identical"] = first.get("tsha") == last.get("tsha")
    return v


def score(w, hat, h=None, x=None, y=None, ny=None) -> dict:
    e = hat - w
    r = {"wt": float(e.norm() / w.norm())}
    if h is not None:
        r["h"] = float(math.sqrt(float(
            ((e * e).sum(0) * h).sum() / ((w * w).sum(0) * h).sum())))
    if x is not None:
        r["out"] = float((x @ hat.T - y).norm() / ny)
    return r


def check_repeat(b: Bench, first: dict, last: dict, label: str) -> dict:
    """The repeated-baseline control: identical bytes, and the two wall clocks."""
    same = first.get("sha") == last.get("sha")
    verdict = {
        "arm": label, "sha_first": first.get("sha"), "sha_repeat": last.get("sha"),
        "bytes_identical": bool(same),
        "secs_first": first.get("secs"), "secs_repeat": last.get("secs"),
        "wall_drift": (None if not first.get("secs") else
                       last.get("secs", 0.0) / first["secs"]),
    }
    b.log(f"    control: {label} repeated last -> bytes "
          f"{'IDENTICAL' if same else '!! DIFFER'}, wall "
          f"{first.get('secs', 0):.0f}s then {last.get('secs', 0):.0f}s"
          + ("" if not first.get("secs") else
             f" ({verdict['wall_drift']:.2f}x box drift)"))
    return verdict


# --------------------------------------------------------------- stage gauge


def stage_gauge(a) -> None:
    """Is ``channel_sigma`` a gauge?  Dyadic and non-dyadic spreads, both grids."""
    b = Bench(a.out)
    src = open_all(DENSE_SRC)
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    b.doc = {"args": vars(a), "units": {}}
    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = H[name]
        res: dict = {"rows": w.shape[0], "cols": w.shape[1]}
        b.log(f"\n== {name} {tuple(w.shape)}  (window_sigma tracks channel_sigma)")
        for label, grid, base in (
            ("bf16", BF16_GRID, BF16_CHANNEL_SIGMA),
            ("e4m3", E4M3_GRID, default_channel_sigma(E4M3_GRID)),
        ):
            b.log(f"  -- {label}, base sigma {base:.6g}; tsha is the DECODED tensor")
            b.header(("bpp", "wt", "h"))
            arms = ([("dyadic", m) for m in a.dyadic]
                    + [("odd", m) for m in a.non_dyadic] + [("dyadic", 1.0)])
            for i, (kind, mult) in enumerate(arms):
                cs = base * mult
                arm = (f"{label} x{mult:g} ({kind})"
                       + (" [repeat]" if i == len(arms) - 1 else ""))
                got = try_arm(b, arm, lambda cs=cs: encode_arm(
                    w, grid, a.rung, name, window_bits=a.window_bits,
                    window_sigma=None, channel_sigma=cs))
                if got is None:
                    continue
                hat, bpp, s, secs = got
                r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                     "sigma": cs, "mult": mult, "kind": kind, **score(w, hat, h=h)}
                res[arm] = r
                b.row(arm, r, ("bpp", "wt", "h"))
                b.log(f"        file {s}  tensor {r['tsha']}")
                del hat
                torch.cuda.empty_cache()
            if f"{label} x1 (dyadic)" not in res:
                continue
            first = res[f"{label} x1 (dyadic)"]
            last = res.get(f"{label} x1 (dyadic) [repeat]", {})
            res[f"{label}_control"] = check_repeat_tensor(
                b, first, last, f"{label} sigma={base:.6g}")
            group = sorted(k for k, v in res.items()
                           if isinstance(v, dict) and v.get("tsha") == first["tsha"])
            res[f"{label}_gauge"] = {
                "base_sigma": base,
                "reference_tsha": first["tsha"],
                "tensor_identical_arms": group,
                "dyadic_all_identical": all(
                    v["tsha"] == first["tsha"] for v in res.values()
                    if isinstance(v, dict) and v.get("kind") == "dyadic"),
                "file_identical_arms": sorted(
                    k for k, v in res.items()
                    if isinstance(v, dict) and v.get("sha") == first["sha"]),
                "ratio_wt_to_default": {
                    f"x{v['mult']:g}({v['kind']})": v["wt"] / first["wt"]
                    for v in res.values() if isinstance(v, dict) and "mult" in v},
            }
            b.log(f"    {label}: {len(group)} of {len(arms)} arms decode to the SAME "
                  f"tensor as sigma={base:.6g}; every dyadic arm identical = "
                  f"{res[f'{label}_gauge']['dyadic_all_identical']}; "
                  f"{len(res[f'{label}_gauge']['file_identical_arms'])} share its FILE")
        b.doc["units"][name] = res
        b.save()
        del w
        torch.cuda.empty_cache()
    b.log(f"\nwrote {a.out}")


# --------------------------------------------------------------- stage reach


def stage_reach(a) -> None:
    """The real axis: the table's reach in row-RMS units.

    Two knobs, and they are not the same axis.  ``--channel-sigmas`` moves the
    spread the rows are scaled to; ``--table-ratios`` moves the table's sigma
    *relative* to that spread.  Ratio 1.0 passes ``window_sigma=None``, which
    is what the shipped recipe stores and what makes the table track the
    channel scale (``encode.py``'s ``table_sigma = channel_sigma`` under a
    CHANNEL plane), so the ``x1 r1`` arm is the default encode itself and not
    a re-spelling of it.  Pinning the table at an absolute sigma while the
    channel scale moves -- what this stage did while it was BF16-only, where
    the base spread is 1.0 and the two coincide -- silently sweeps the ratio
    on any grid whose base is not 1.0.
    """
    b = Bench(a.out)
    grid = GRIDS[a.grid]
    base = base_channel_sigma(grid)
    src = open_all(DENSE_SRC)
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    b.doc = {"args": vars(a), "grid": a.grid, "base_channel_sigma": base,
             "units": {}}
    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = H[name]
        res: dict = {"rows": w.shape[0], "cols": w.shape[1]}
        for q in a.rungs:
            b.log(f"\n== {name} {tuple(w.shape)}  R={q / 256:g}  grid={a.grid}  "
                  f"base_csigma={base:g}  ratios={a.table_ratios} "
                  f"(r1 = window_sigma None, the table tracks the channel)")
            b.header(("bpp", "wt", "h", "reach_rms", "over"))
            # The default arm runs first and again last -- the issue's own gate.
            pairs = [(m, r) for r in a.table_ratios
                     for m in a.channel_sigmas if (m, r) != (1.0, 1.0)]
            pairs = [(1.0, 1.0)] + pairs + [(1.0, 1.0)]
            for i, (m, ratio) in enumerate(pairs):
                cs = m * base
                # None, not cs: the value the recipe stores, so the control
                # arm exercises the default path rather than mirroring it.
                ws = None if ratio == 1.0 else ratio * cs
                arm = f"R{q} x{m:g} r{ratio:g}" + (
                    " [repeat]" if i == len(pairs) - 1 else "")
                st = reach_stats(w, grid, a.window_bits,
                                 cs if ws is None else ws, cs)
                got = try_arm(b, arm, lambda cs=cs, ws=ws: encode_arm(
                    w, grid, q, name, window_bits=a.window_bits,
                    window_sigma=ws, channel_sigma=cs))
                if got is None:
                    continue
                hat, bpp, s, secs = got
                r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                     "channel_sigma": cs, "channel_sigma_mult": m,
                     "table_ratio": ratio, "window_sigma": ws,
                     "reach_rms": st["reach_row_rms"], "over": st["rows_over_reach"],
                     **st, **score(w, hat, h=h)}
                res[arm] = r
                b.row(arm, r, ("bpp", "wt", "h", "reach_rms", "over"))
                del hat
                torch.cuda.empty_cache()
            if f"R{q} x1 r1 [repeat]" in res:
                res[f"R{q}_control"] = check_repeat_tensor(
                    b, res[f"R{q} x1 r1"], res[f"R{q} x1 r1 [repeat]"],
                    f"R{q} x1 r1")
        b.doc["units"][name] = res
        b.save()
        del w
        torch.cuda.empty_cache()
    summarise_sweep(b, "channel_sigma_mult")
    b.save()
    b.log(f"\nwrote {a.out}")


# ------------------------------------------------------------------- stage L


def stage_dense_l(a) -> None:
    b = Bench(a.out)
    src = open_all(DENSE_SRC)
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    b.doc = {"args": vars(a), "units": {}}
    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = H[name]
        res: dict = {"rows": w.shape[0], "cols": w.shape[1]}
        res["FP8 RTN LS-refit"] = {
            "bpp": 8.0 + 32 / w.shape[1], **score(w, fp8_floor(w), h=h)}
        for q in a.rungs:
            b.log(f"\n== {name} {tuple(w.shape)}  R={q / 256:g}   table at "
                  f"L={a.window_bits} = "
                  f"{2 ** a.window_bits * 2 * 8 / w.numel():.4f} bpp")
            b.header(("bpp", "wt", "h"))
            arms = [L for L in a.window_bits_list if L * 256 >= q] + [a.window_bits]
            for i, L in enumerate(arms):
                arm = f"R{q} L={L}" + (" [repeat]" if i == len(arms) - 1 else "")
                got = try_arm(b, arm, lambda L=L: encode_arm(
                    w, BF16_GRID, q, name, window_bits=L,
                    window_sigma=None, channel_sigma=None))
                if got is None:
                    continue
                hat, bpp, s, secs = got
                r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                     "L": L, "table_bpp": 2 ** L * 2 * 8 / w.numel(),
                     **score(w, hat, h=h)}
                res[arm] = r
                b.row(arm, r, ("bpp", "wt", "h"))
                del hat
                torch.cuda.empty_cache()
            if f"R{q} L={a.window_bits} [repeat]" in res:
                res[f"R{q}_control"] = check_repeat_tensor(
                    b, res[f"R{q} L={a.window_bits}"],
                    res[f"R{q} L={a.window_bits} [repeat]"], f"R{q} L={a.window_bits}")
        res["frontier"] = {
            axis: frontier([(v["bpp"], v[axis], k) for k, v in res.items()
                            if isinstance(v, dict) and "L" in v and "[repeat]" not in k])
            for axis in ("wt", "h")}
        b.log(f"    frontier(wt) {res['frontier']['wt']}")
        b.log(f"    frontier(h)  {res['frontier']['h']}")
        b.doc["units"][name] = res
        b.save()
        del w
        torch.cuda.empty_cache()
    summarise_sweep(b, "L")
    b.save()
    b.log(f"\nwrote {a.out}")


def stage_glm_l(a) -> None:
    index = json.load(open(f"{GLM_SRC}/model.safetensors.index.json"))["weight_map"]
    b = Bench(a.out)
    b.doc = {"args": vars(a), "units": {}}
    for layer in a.layers:
        blob = torch.load(
            f"{GLM_ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        x = xa[xa.shape[0] - a.eval_rows:].contiguous().cuda()
        del xa, blob
        h = (x * x).sum(dim=0)
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{GLM_SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            tname = f"L{layer}.{proj}"
            # The captured blob is the experts' INPUT, so it feeds the
            # projections that read the hidden dim and no others.  Naming
            # ``down_proj`` used to run a whole layer's worth of arms and then
            # die inside the matmul; refuse it here, where the message can say
            # which activations exist rather than which shapes disagreed.
            if w.shape[1] != x.shape[1]:
                raise SystemExit(
                    f"{tname}: this stage feeds the experts' input activations "
                    f"({x.shape[1]} wide), and {proj} reads {w.shape[1]}. Only "
                    "projections on the hidden dim (gate_proj, up_proj) are "
                    "expressible from this capture; down_proj would need the "
                    "intermediate activations, which are not captured."
                )
            y = x @ w.T
            ny = y.norm()
            res: dict = {"rows": w.shape[0], "cols": w.shape[1]}
            for k in a.exl3:
                p = Path(EXL3) / f"L{layer}_{proj}_K{k}.pt"
                if p.exists():
                    res[f"EXL3 K={k}"] = {
                        "bpp": k + 0.0117,
                        **score(w, torch.load(p, map_location="cuda").float(),
                                h=h, x=x, y=y, ny=ny)}
            for q in a.rungs:
                b.log(f"\n== {tname} {tuple(w.shape)}  R={q / 256:g}   table at "
                      f"L={a.window_bits} = "
                      f"{2 ** a.window_bits * 2 * 8 / w.numel():.4f} bpp")
                b.header(("bpp", "wt", "h", "out"))
                for k in a.exl3:
                    if f"EXL3 K={k}" in res:
                        b.row(f"EXL3 K={k}", res[f"EXL3 K={k}"],
                              ("bpp", "wt", "h", "out"))
                arms = [L for L in a.window_bits_list if L * 256 >= q] + [a.window_bits]
                for i, L in enumerate(arms):
                    arm = f"R{q} L={L}" + (" [repeat]" if i == len(arms) - 1 else "")
                    got = try_arm(b, arm, lambda L=L: encode_arm(
                        w, BF16_GRID, q, name, window_bits=L,
                        window_sigma=None, channel_sigma=None))
                    if got is None:
                        continue
                    hat, bpp, s, secs = got
                    r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                         "L": L, "table_bpp": 2 ** L * 2 * 8 / w.numel(),
                         **score(w, hat, h=h, x=x, y=y, ny=ny)}
                    res[arm] = r
                    b.row(arm, r, ("bpp", "wt", "h", "out"))
                    del hat
                    torch.cuda.empty_cache()
                if f"R{q} L={a.window_bits} [repeat]" in res:
                    res[f"R{q}_control"] = check_repeat_tensor(
                        b, res[f"R{q} L={a.window_bits}"],
                        res[f"R{q} L={a.window_bits} [repeat]"],
                        f"R{q} L={a.window_bits}")
            res["frontier"] = {
                axis: frontier([(v["bpp"], v[axis], k) for k, v in res.items()
                                if isinstance(v, dict) and "L" in v
                                and "[repeat]" not in k])
                for axis in ("wt", "h", "out")}
            for axis in ("wt", "h", "out"):
                b.log(f"    frontier({axis}) {res['frontier'][axis]}")
            b.doc["units"][tname] = res
            b.save()
            del w, y
            torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()
    summarise_sweep(b, "L")
    b.save()
    b.log(f"\nwrote {a.out}")


def summarise_sweep(b: Bench, axis_name: str) -> None:
    arms: "dict[str, list]" = {}
    for res in b.doc["units"].values():
        for arm, r in res.items():
            if isinstance(r, dict) and "bpp" in r and "[repeat]" not in arm:
                arms.setdefault(arm, []).append(r)
    n = len(b.doc["units"])
    keys = ("wt", "h", "out")
    b.doc["summary"] = {}
    b.log(f"\n== geomean over {n} unit(s), swept on {axis_name}")
    b.log("    " + f"{'arm':<30}" + f"{'bpp':>10}" + "".join(f"{k:>10}" for k in keys))
    for arm, rs in arms.items():
        if len(rs) != n:
            continue
        row = {"bpp": sum(r["bpp"] for r in rs) / n}
        for k in keys:
            if all(k in r for r in rs):
                row[k] = geomean([r[k] for r in rs])
        b.doc["summary"][arm] = row
        b.log("    " + f"{arm:<30}" + f"{row['bpp']:10.5f}"
              + "".join(f"{row.get(k, float('nan')):10.5f}" for k in keys))
    pts = {axis: [(v["bpp"], v[axis], k) for k, v in b.doc["summary"].items()
                  if axis in v and k.startswith("R")]
           for axis in keys}
    b.doc["summary_frontier"] = {ax: frontier(p) for ax, p in pts.items() if p}
    for ax, f in b.doc["summary_frontier"].items():
        b.log(f"    frontier({ax}) {f}")


# ---------------------------------------------------------------- stage pair
#
# THE PRE-REGISTERED READING.  Written before the stage was run; the commit
# that adds it precedes every artifact it produced, which is the only proof
# of pre-registration that survives a rewrite of this comment.
#
# **What is already searched, and what is not.**  ``channel_sigma`` is a
# dyadic gauge on this grid (``--stage gauge``, four dense units, decoded
# tensor bit-identical over 16x).  The two axes that are left -- the table's
# width ``L`` and the table/row spread *ratio* ``window_sigma /
# channel_sigma`` -- have each been swept with **the other held at its
# shipped value**: ``--stage dense-l``/``glm-l`` moved ``L`` at ratio 1.0,
# and #48 moved the ratio at ``L=14``.  A pair is not searched by two
# one-dimensional sweeps through the same point unless the axes are
# separable, and nothing has measured whether they are.  That is this stage.
#
# **Hypothesis, stated with its mechanism.**  ``L`` buys code resolution and
# the ratio buys reach, and they spend the same resource: a table of ``2^L``
# entries laid over ``ratio * channel_sigma`` grid units of range.  Widening
# the reach at fixed ``L`` therefore thins the table, so *if the axes
# interact at all*, deeper ``L`` should be worth more at wider ratio and the
# optimal ratio should widen with ``L``.  H0 (separable): the argmin over
# ratio is the same at every ``L`` and the argmin over ``L`` is the same at
# every ratio, at every rung.  H1 (interacting): they move together in the
# direction above.  H2 (interacting the other way) is possible and would
# falsify the mechanism rather than the interaction.
#
# **The gate metric, and why not ``wt``.**  ``wt`` is disqualified in
# advance: it decreases monotonically in ``L`` on 4 of 4 dense units at both
# rungs (#18 thread), so a sweep gated on it answers "deeper" before it
# starts.  The gate is ``h`` on the dense set (the captured diagonal) and
# ``out`` on the GLM experts (the held-out capture rows, the same rows the
# route's own receipt scored).  ``wt`` is still reported, as a control on
# the gate rather than as evidence.
#
# **Matched bytes, exactly, not by slope.**  ``L`` costs bytes and the ratio
# does not: an arm at ratio r != 1 is *bit-for-bit the same size* as the
# shipped arm, so its ratio is read directly.  An arm at ``L != 14`` is not,
# and the rate axis is continuous in ``q256``, so the reference is built
# rather than interpolated: for each ``L`` this encodes the **shipped pair
# (L=14, ratio 1.0) at the rung whose bpp equals that arm's**, and asserts
# the two bpp are equal to within 1e-9.  A table 0.09 bpp wide on a GLM
# expert is 0.75 bpp on a 1024x1024 Qwen Linear, which is exactly where the
# thread's two-point 1.903x-per-bpp slope would be doing the deciding.
#
# **The decision rule, per rung and per population.**  A *unit win* is the
# candidate beating its own byte-matched shipped reference on the gate
# metric.  The encoder is deterministic and every arm here is a real
# difference, not a draw from a distribution, so "win" is strict inequality;
# a 1% margin count is reported beside it because a 0.2% win is a true
# statement about a number nobody should spend a wire change on.
#
#   * ADOPT-WORTHY (the strongest verdict this stage can reach):  a strict
#     majority of per-unit wins AND a geomean below 1.00 on the gate metric
#     AND the six-expert GLM cross-check no worse than 1.00x.
#   * CONFIRMED:  the shipped pair (14, 1.0) is the argmin, or nothing else
#     clears the bar above.  This is a full answer and changes nothing.
#   * INCONCLUSIVE:  wins and geomean disagree, or the optimum sits on the
#     edge of the swept grid and is unbracketed.
#
# **No default flips out of this stage regardless of the outcome**, and that
# is settled before the numbers exist.  Moving the ratio needs an explicit
# ``window_sigma`` in ``BF16_RECIPE`` and moving ``L`` moves
# ``BF16_WINDOW_BITS``; both change the bytes and the ``encoder_profile_id``
# -- a wire change -- and the BF16 route has no serving lane, so nothing
# here is promotable under principle 3 in any case.  An adopt-worthy result
# is recorded as the measured optimum for the reach term #48 describes and
# #84 bounds, not spent.

#: The pair.  ``L`` from the width the route ships (14) one step either way;
#: ratios bracketing #48's BF16 turnover (best at 1.25-1.5 on ``wt`` and
#: 1.5-1.75 on ``h`` at R=8, rising past 2.0), including 1.4142 = the
#: ``channel_sigma = 0.707`` this issue's own reach sweep found.
PAIR_BITS = [12, 14, 16]
PAIR_RATIOS = [1.0, 1.25, 1.4142135623730951, 1.75]


def table_bpp(window_bits: int, numel: int) -> float:
    """The ALPHABET plane's cost on a BF16 table: one 16-bit word per entry."""
    return (1 << window_bits) * 16 / numel


def bytematched_rung(q256: int, window_bits: int, default_bits: int,
                     numel: int) -> "int | None":
    """The rung at which the shipped ``L`` spends exactly this ``L``'s bytes.

    ``q256`` is the rate in 1/256 bits per weight and the axis is continuous
    in it, so the byte match is *built* -- ``None`` only if the table delta
    is not an integral number of q256 steps on this shape, in which case the
    caller must say it interpolated instead of pretending it did not.
    """
    delta = (table_bpp(window_bits, numel) - table_bpp(default_bits, numel)) * 256
    step = round(delta)
    if abs(delta - step) > 1e-6 or q256 + step <= 0:
        return None
    return q256 + int(step)


def pair_arm_key(q256: int, window_bits: int, ratio: float) -> str:
    return f"R{q256} L={window_bits} r={ratio:g}"


def run_pair_unit(b: "Bench", a, tname: str, w: torch.Tensor, name: str,
                  *, h=None, x=None, y=None, ny=None) -> dict:
    """One unit's joint ``(L, ratio)`` grid at every rung, plus its references."""
    numel = w.numel()
    default_L = a.window_bits
    axes = ("wt", "h") if x is None else ("wt", "h", "out")
    res: dict = {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "numel": numel,
                 "gate": a.gate, "default_L": default_L}
    for q in a.rungs:
        Ls = [L for L in a.pair_bits if L * 256 >= q]
        refs = {L: (q if L == default_L else bytematched_rung(q, L, default_L, numel))
                for L in Ls}
        b.log(f"\n== {tname} {tuple(w.shape)}  R={q / 256:g}  grid=bf16  "
              f"gate={a.gate}  L={Ls}  ratios={[round(r, 4) for r in a.pair_ratios]}")
        b.log(f"    byte-matched shipped rungs (L=14 spending each L's bytes): "
              + ", ".join(f"L{L}->R{refs[L]}" if refs[L] else f"L{L}->NONE"
                          for L in Ls))
        b.header(("bpp", *axes, "reach_rms", "over"))
        arms = [(q, default_L, 1.0, "")]
        arms += [(q, L, r, "") for L in Ls for r in a.pair_ratios
                 if not (L == default_L and r == 1.0)]
        arms += [(refs[L], default_L, 1.0, f" [bytematch L={L}]")
                 for L in Ls if L != default_L and refs[L] is not None]
        arms += [(q, default_L, 1.0, " [repeat]")]
        for qq, L, ratio, tag in arms:
            arm = pair_arm_key(qq, L, ratio) + tag
            if arm in res:
                continue
            # ratio 1.0 is ``window_sigma=None``, the value the recipe stores,
            # so the reference arms exercise the shipped path and not a
            # re-spelling of it.
            ws = None if ratio == 1.0 else ratio * BF16_CHANNEL_SIGMA
            st = reach_stats(w, BF16_GRID, L,
                             BF16_CHANNEL_SIGMA if ws is None else ws,
                             BF16_CHANNEL_SIGMA)
            got = try_arm(b, arm, lambda qq=qq, L=L, ws=ws: encode_arm(
                w, BF16_GRID, qq, name, window_bits=L, window_sigma=ws,
                channel_sigma=BF16_CHANNEL_SIGMA))
            if got is None:
                continue
            hat, bpp, s, secs = got
            r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                 "q256": qq, "rung": q, "L": L, "ratio": ratio,
                 "table_bpp": table_bpp(L, numel),
                 "reach_rms": st["reach_row_rms"], "over": st["rows_over_reach"],
                 **st, **score(w, hat, h=h, x=x, y=y, ny=ny)}
            res[arm] = r
            b.row(arm, r, ("bpp", *axes, "reach_rms", "over"))
            del hat
            torch.cuda.empty_cache()
        first, last = pair_arm_key(q, default_L, 1.0), \
            pair_arm_key(q, default_L, 1.0) + " [repeat]"
        if last in res:
            res[f"R{q}_control"] = check_repeat_tensor(b, res[first], res[last],
                                                       f"R{q} shipped pair")
        # The byte match is an assertion, not a hope: each candidate is
        # compared only against a reference it is provably the same size as.
        cmp: dict = {}
        for L in Ls:
            # The reference is stored under its own tagged key, so build the
            # same string here: looking up the untagged one silently drops
            # every arm at a width the shipped recipe does not carry, which
            # is exactly the set this stage exists to price.
            ref_key = None if refs[L] is None else (
                pair_arm_key(refs[L], default_L, 1.0)
                + ("" if L == default_L else f" [bytematch L={L}]"))
            ref = res.get(ref_key) if ref_key else None
            for ratio in a.pair_ratios:
                key = pair_arm_key(q, L, ratio)
                if key not in res or ref is None:
                    continue
                gap = abs(res[key]["bpp"] - ref["bpp"])
                cmp[key] = {
                    "ref": ref_key, "bpp_gap": gap,
                    "bytes_matched": gap < 1e-9,
                    **{f"{ax}_ratio": res[key][ax] / ref[ax] for ax in axes
                       if ax in ref and ref[ax] > 0},
                }
        res[f"R{q}_vs_shipped"] = cmp
        bad = [k for k, v in cmp.items() if not v["bytes_matched"]]
        b.log(f"    byte match: {len(cmp) - len(bad)} of {len(cmp)} arms sit at "
              f"their reference's exact bpp" + (f"; UNMATCHED {bad}" if bad else ""))
        gate = a.gate
        best = min((v[f"{gate}_ratio"], k) for k, v in cmp.items()
                   if f"{gate}_ratio" in v)
        b.log(f"    best on {gate} at matched bytes: {best[1]} at {best[0]:.4f}x")
        for L in Ls:
            row = [(v[f"{gate}_ratio"], res[k]["ratio"]) for k, v in cmp.items()
                   if res[k]["L"] == L and f"{gate}_ratio" in v]
            if row:
                b.log(f"      L={L:<3} best ratio {min(row)[1]:g} "
                      f"({min(row)[0]:.4f}x)  " + "  ".join(
                          f"r{rr:g}={vv:.4f}" for vv, rr in row))
    return res


def stage_pair_dense(a) -> None:
    """The joint grid on dense Qwen Linears, gated on the captured diagonal."""
    b = Bench(a.out)
    a.gate = a.gate or "h"
    src = open_all(DENSE_SRC)
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    b.doc = {"args": vars(a), "grid": "bf16", "population": "dense-qwen",
             "gate": a.gate, "units": {}}
    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        b.doc["units"][name] = run_pair_unit(b, a, name, w, name, h=H[name])
        b.save()
        del w
        torch.cuda.empty_cache()
    summarise_pair(b)
    b.save()
    b.log(f"\nwrote {a.out}")


def stage_pair_glm(a) -> None:
    """The joint grid on GLM routed experts, gated on held-out ``out``.

    Scope, stated once and inherited by every number below: the capture
    holds the **MoE block's input** rows, so ``out`` is this expert's error
    over all of them and not over the subset the router would send it, and
    only projections reading the hidden dim are expressible (``down_proj``
    would need the intermediate activations, which are not captured).  The
    expert index is swept, which the earlier six-tensor sets did not do:
    "six experts" there means six (layer, proj) cells of expert 0.
    """
    index = json.load(open(f"{GLM_SRC}/model.safetensors.index.json"))["weight_map"]
    b = Bench(a.out)
    a.gate = a.gate or "out"
    b.doc = {"args": vars(a), "grid": "bf16", "population": "glm-experts",
             "gate": a.gate, "units": {}}
    for layer in a.layers:
        blob = torch.load(
            f"{GLM_ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        x = xa[xa.shape[0] - a.eval_rows:].contiguous().cuda()
        del xa, blob
        h = (x * x).sum(dim=0)
        for expert in a.experts:
            for proj in a.projs:
                name = (f"model.language_model.layers.{layer}.mlp.experts."
                        f"{expert}.{proj}.weight")
                with safe_open(f"{GLM_SRC}/{index[name]}", framework="pt") as f:
                    w = f.get_tensor(name).contiguous().cuda().float()
                tname = f"L{layer}.e{expert}.{proj}"
                if w.shape[1] != x.shape[1]:
                    raise SystemExit(
                        f"{tname}: this stage feeds the experts' input "
                        f"activations ({x.shape[1]} wide) and {proj} reads "
                        f"{w.shape[1]}; only projections on the hidden dim are "
                        "expressible from this capture.")
                y = x @ w.T
                b.doc["units"][tname] = run_pair_unit(
                    b, a, tname, w, name, h=h, x=x, y=y, ny=y.norm())
                b.save()
                del w, y
                torch.cuda.empty_cache()
        del x, h
        torch.cuda.empty_cache()
    summarise_pair(b)
    b.save()
    b.log(f"\nwrote {a.out}")


def summarise_pair(b: "Bench") -> None:
    """Per-unit wins first, then the geomean -- never the geomean alone.

    Every ratio here is against the **byte-matched shipped pair**, so a row
    reading below 1.0 is that arm beating ``(L=14, ratio 1.0)`` at bytes the
    two provably share, on units counted one at a time.
    """
    gate = b.doc["gate"]
    units = b.doc["units"]
    n = len(units)
    rungs = sorted({int(k[1:].split("_")[0]) for res in units.values()
                    for k in res if k.endswith("_vs_shipped")})
    b.doc["summary"] = {}
    axes = ("wt", "h", "out")
    for q in rungs:
        cmps = {u: res.get(f"R{q}_vs_shipped", {}) for u, res in units.items()}
        arms = sorted({k for c in cmps.values() for k in c},
                      key=lambda k: (int(k.split("L=")[1].split()[0]),
                                     float(k.split("r=")[1])))
        b.log(f"\n== R={q / 256:g}: {n} unit(s), each arm against the "
              f"byte-matched shipped pair (L=14, r=1); gate = {gate}")
        b.log("    " + f"{'arm':<20}{'wins':>8}{'/n':>4}" + f"{'win@1%':>8}"
              + "".join(f"{ax + ' geo':>10}" for ax in axes) + f"{'bpp':>10}")
        table = {}
        for arm in arms:
            vals = {ax: [c[arm][f"{ax}_ratio"] for c in cmps.values()
                         if arm in c and f"{ax}_ratio" in c[arm]] for ax in axes}
            g = [v for v in vals.get(gate, []) if v == v]
            if len(g) != n:
                continue
            bpps = [units[u][arm]["bpp"] for u in units if arm in units[u]]
            row = {
                "n": n, "wins": sum(1 for v in g if v < 1.0),
                "wins_1pct": sum(1 for v in g if v < 0.99),
                "per_unit": {u: cmps[u][arm].get(f"{gate}_ratio")
                             for u in units if arm in cmps[u]},
                "bpp": sum(bpps) / len(bpps),
                **{f"{ax}_geomean": geomean(vals[ax]) for ax in axes
                   if len(vals[ax]) == n},
            }
            table[arm] = row
            b.log("    " + f"{arm.split(' ', 1)[1]:<20}{row['wins']:>8}{n:>4}"
                  f"{row['wins_1pct']:>8}"
                  + "".join(f"{row.get(ax + '_geomean', float('nan')):10.4f}"
                            for ax in axes)
                  + f"{row['bpp']:10.5f}")
        b.doc["summary"][f"R{q}"] = table
        if not table:
            continue
        # Separability: does the best ratio depend on L, and the best L on ratio?
        best_r = {}
        for L in sorted({int(k.split("L=")[1].split()[0]) for k in table}):
            row = [(v[f"{gate}_geomean"], float(k.split("r=")[1]))
                   for k, v in table.items()
                   if int(k.split("L=")[1].split()[0]) == L]
            best_r[L] = min(row)[1]
            b.log(f"      L={L:<3} best ratio {min(row)[1]:g} ({min(row)[0]:.4f}x "
                  f"geomean, {table[[k for k in table if int(k.split('L=')[1].split()[0]) == L and float(k.split('r=')[1]) == min(row)[1]][0]]['wins']}/{n} units)")
        best_L = {}
        for ratio in sorted({float(k.split("r=")[1]) for k in table}):
            row = [(v[f"{gate}_geomean"], int(k.split("L=")[1].split()[0]))
                   for k, v in table.items() if float(k.split("r=")[1]) == ratio]
            best_L[ratio] = min(row)[1]
            b.log(f"      r={ratio:<6g} best L {min(row)[1]} ({min(row)[0]:.4f}x geomean)")
        b.doc["summary"][f"R{q}_separability"] = {
            "best_ratio_at_L": best_r, "best_L_at_ratio": best_L,
            "ratio_argmin_independent_of_L": len(set(best_r.values())) == 1,
            "L_argmin_independent_of_ratio": len(set(best_L.values())) == 1,
        }
        b.log(f"      separable? best-ratio same at every L: "
              f"{len(set(best_r.values())) == 1}; best-L same at every ratio: "
              f"{len(set(best_L.values())) == 1}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gauge", "reach", "dense-l", "glm-l",
                             "pair-dense", "pair-glm"])
    ap.add_argument("--units", nargs="+", default=DENSE_UNITS)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj"])
    # The expert index the GLM stages address.  Every "six expert" set on this
    # issue to date is six (layer, proj) cells of expert 0; sweeping this is
    # what widens the population rather than the shape.
    ap.add_argument("--experts", type=int, nargs="+", default=[0])
    ap.add_argument("--rung", type=int, default=1024)
    ap.add_argument("--rungs", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--window-bits", type=int, default=BF16_WINDOW_BITS)
    ap.add_argument("--window-bits-list", type=int, nargs="+",
                    default=[8, 10, 12, 14, 16])
    ap.add_argument("--dyadic", type=float, nargs="+", default=[1.0, 0.25, 0.5, 2.0, 4.0])
    ap.add_argument("--non-dyadic", type=float, nargs="+", default=[0.75, 1.5, 3.0])
    ap.add_argument("--table-sigma", type=float, default=1.0)
    ap.add_argument("--grid", choices=sorted(GRIDS), default="bf16",
                    help="which grid the reach stage sweeps (#36 is E4M3)")
    # Multipliers of the grid's own base spread, so one list means the same
    # thing on both grids.  On BF16 the base is 1.0 and these are absolute.
    ap.add_argument("--channel-sigmas", type=float, nargs="+",
                    default=[0.25, 0.5, 0.7071067811865476,
                             1.4142135623730951, 2.0])
    # window_sigma / channel_sigma.  1.0 is the shipped tracking recipe.
    ap.add_argument("--table-ratios", type=float, nargs="+", default=[1.0])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[4, 6, 8])
    # The joint stage: the pair, not two sweeps through one point.
    ap.add_argument("--pair-bits", type=int, nargs="+", default=PAIR_BITS)
    ap.add_argument("--pair-ratios", type=float, nargs="+", default=PAIR_RATIOS)
    ap.add_argument("--gate", choices=["wt", "h", "out"], default=None,
                    help="the metric the reading is gated on; wt is "
                         "disqualified in advance (monotone in L) and is "
                         "reported as a control, never as the gate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    {"gauge": stage_gauge, "reach": stage_reach,
     "dense-l": stage_dense_l, "glm-l": stage_glm_l,
     "pair-dense": stage_pair_dense, "pair-glm": stage_pair_glm}[a.stage](a)


if __name__ == "__main__":
    main()
