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
    """The real axis: the table's reach in row-RMS units, at a pinned table."""
    b = Bench(a.out)
    src = open_all(DENSE_SRC)
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    b.doc = {"args": vars(a), "units": {}}
    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = H[name]
        res: dict = {"rows": w.shape[0], "cols": w.shape[1]}
        for q in a.rungs:
            b.log(f"\n== {name} {tuple(w.shape)}  R={q / 256:g}  "
                  f"window_sigma={a.table_sigma} (table pinned)")
            b.header(("bpp", "wt", "h", "reach_rms", "over"))
            arms = list(a.channel_sigmas) + [BF16_CHANNEL_SIGMA]
            for i, cs in enumerate(arms):
                arm = f"R{q} csigma={cs:g}" + (" [repeat]" if i == len(arms) - 1 else "")
                st = reach_stats(w, BF16_GRID, a.window_bits, a.table_sigma, cs)
                got = try_arm(b, arm, lambda cs=cs: encode_arm(
                    w, BF16_GRID, q, name, window_bits=a.window_bits,
                    window_sigma=a.table_sigma, channel_sigma=cs))
                if got is None:
                    continue
                hat, bpp, s, secs = got
                r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                     "channel_sigma": cs,
                     "reach_rms": st["reach_row_rms"], "over": st["rows_over_reach"],
                     **st, **score(w, hat, h=h)}
                res[arm] = r
                b.row(arm, r, ("bpp", "wt", "h", "reach_rms", "over"))
                del hat
                torch.cuda.empty_cache()
            if f"R{q} csigma={BF16_CHANNEL_SIGMA:g} [repeat]" in res:
                res[f"R{q}_control"] = check_repeat_tensor(
                    b, res[f"R{q} csigma={BF16_CHANNEL_SIGMA:g}"],
                    res[f"R{q} csigma={BF16_CHANNEL_SIGMA:g} [repeat]"],
                    f"R{q} csigma={BF16_CHANNEL_SIGMA:g}")
        b.doc["units"][name] = res
        b.save()
        del w
        torch.cuda.empty_cache()
    summarise_sweep(b, "channel_sigma")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gauge", "reach", "dense-l", "glm-l"])
    ap.add_argument("--units", nargs="+", default=DENSE_UNITS)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj"])
    ap.add_argument("--rung", type=int, default=1024)
    ap.add_argument("--rungs", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--window-bits", type=int, default=BF16_WINDOW_BITS)
    ap.add_argument("--window-bits-list", type=int, nargs="+",
                    default=[8, 10, 12, 14, 16])
    ap.add_argument("--dyadic", type=float, nargs="+", default=[1.0, 0.25, 0.5, 2.0, 4.0])
    ap.add_argument("--non-dyadic", type=float, nargs="+", default=[0.75, 1.5, 3.0])
    ap.add_argument("--table-sigma", type=float, default=1.0)
    ap.add_argument("--channel-sigmas", type=float, nargs="+",
                    default=[0.25, 0.5, 0.7071067811865476, 1.0,
                             1.4142135623730951, 2.0])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    {"gauge": stage_gauge, "reach": stage_reach,
     "dense-l": stage_dense_l, "glm-l": stage_glm_l}[a.stage](a)


if __name__ == "__main__":
    main()
