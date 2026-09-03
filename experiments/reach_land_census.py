"""Issue #87: how often does ``initial_channel_scale``'s reach bound land short?

``initial_channel_scale`` raises a row whose loudest weight would land past the
body's ``reach`` so that the weight lands *exactly on* the reach, at ``scale =
amax / reach``.  That is a **lower bound**: any effective scale below it puts
``amax / effective`` past the reach and the trellis clips the weight the raise
exists for.  The scale is then landed on an fp16 word with
``land_channel_scale``, which rounds to **nearest**, so about half the raised
rows land one ulp low.

This census counts them, with no encode: it is arithmetic on the same tensors
``initial_channel_scale`` computes.  Per unit and arm it reports

* ``over``     -- the fraction of rows the reach raise touched,
* ``short``    -- of those, how many landed below their own bound,
* ``worst``    -- the largest relative shortfall ``(floor - eff) / floor``,
* ``clip_h``   -- the h-weighted energy of the clip that shortfall causes,
                  ``sum_r h[j*_r] (amax_r - reach * eff_r)^2 / sum_rj h_j w_rj^2``
                  over the short rows, ``j*_r`` the row's loudest column.  It is
                  the extra squared error at the one weight the bound is about,
                  in the units of the sweep's ``h`` metric squared.

Scope.  The CHANNEL plane is what calls ``initial_channel_scale``, and
``wire_recipe`` puts exactly two grids on it: **E4M3** (every rung, L=14) and
**BF16** (every rung).  E2M1 and E2M1x2 ride LUT16/S6B and never reach this
code, so "extend to E2M1" is structurally empty rather than unmeasured -- the
script refuses a non-CHANNEL grid by name rather than forcing a plane nothing
ships.

    env $E $P experiments/reach_land_census.py --grid e4m3 --mult 1.0 1.5
    env $E $P experiments/reach_land_census.py --grid e4m3 --all-dense
    env $E $P experiments/reach_land_census.py --grid bf16
    env $E $P experiments/reach_land_census.py --grid e4m3 --source glm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import BF16_GRID, E4M3_GRID  # noqa: E402
from tessera.encode import grid_vector_table, window_table  # noqa: E402
from tessera.export import BF16_RECIPE, E4M3_RECIPE  # noqa: E402
from tessera.scale_channel import (  # noqa: E402
    channel_global, default_channel_sigma, initial_channel_scale,
    land_channel_scale,
)

from bf16_route_weight_space import DENSE_H, DENSE_SRC, GLM_SRC, open_all  # noqa: E402

#: The eight dense Linears issue #80's reach sweep ran, in its order.
DENSE_UNITS = [
    "model.layers.2.mlp.down_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.14.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.27.self_attn.o_proj",
    "model.layers.14.self_attn.v_proj",
]

GRIDS = {"e4m3": (E4M3_GRID, E4M3_RECIPE), "bf16": (BF16_GRID, BF16_RECIPE)}


def base_sigma(key: str, grid) -> float:
    """The spread the shipped recipe states for this grid."""
    stated = GRIDS[key][1].channel_sigma
    return float(default_channel_sigma(grid)) if stated is None else float(stated)


def census(w: torch.Tensor, h: "torch.Tensor | None", grid, window_bits: int,
           sigma: float, seed: int, half: int) -> dict:
    """The row-landing census for one (weight, arm).  No encode."""
    w = w.float()
    codes = window_table(grid, window_bits, sigma=sigma, seed=seed, half=half,
                         device=w.device)
    reach = float(grid_vector_table(grid, w.device)[codes.long()].abs().max())
    rms = w.pow(2).mean(dim=1).sqrt()
    amax, amax_col = w.abs().max(dim=1)
    over = amax * sigma > reach * rms
    floor = amax / reach                       # the bound the raise computes
    stored, eff, g = initial_channel_scale(w, sigma, reach=reach)
    short = over & (eff < floor)
    rel = torch.where(short, (floor - eff) / floor.clamp_min(1e-30),
                      torch.zeros_like(eff))
    out = {
        "rows": int(w.shape[0]), "cols": int(w.shape[1]),
        "reach_grid_units": reach, "reach_row_rms": reach / sigma,
        "global": float(g),
        "rows_over": int(over.sum()), "over": float(over.float().mean()),
        "rows_short": int(short.sum()),
        "short_of_over": (float(short.sum()) / float(over.sum())
                          if int(over.sum()) else 0.0),
        "worst_rel_shortfall": float(rel.max()),
        "max_z": float((amax / rms.clamp_min(1e-30)).max()),
    }
    # The clip the shortfall causes, at the one weight the bound is about.
    excess = (amax - reach * eff).clamp_min(0.0)
    excess = torch.where(short, excess, torch.zeros_like(excess))
    if h is not None:
        hv = h.float().to(w.device)
        num = float((hv[amax_col] * excess.pow(2)).sum())
        den = float(((w * w) * hv[None, :]).sum())
        out["clip_h"] = num / den
    out["clip_wt"] = float(excess.pow(2).sum()) / float(w.pow(2).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="e4m3", choices=sorted(GRIDS))
    ap.add_argument("--source", default="dense", choices=["dense", "glm"])
    ap.add_argument("--mult", type=float, nargs="+", default=[1.0])
    ap.add_argument("--window-bits", type=int, default=None)
    ap.add_argument("--all-dense", action="store_true",
                    help="every Linear in the dense checkpoint, not the eight")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    grid, recipe = GRIDS[a.grid]
    bits = a.window_bits if a.window_bits is not None else recipe.window_bits
    sigma0 = base_sigma(a.grid, grid)
    dev = torch.device(a.device)

    if a.source == "dense":
        idx = open_all(DENSE_SRC)
        h = torch.load(DENSE_H, map_location="cpu")
        if a.all_dense:
            names = sorted(k[:-len(".weight")] for k in idx
                           if k.endswith(".weight") and ".layers." in k
                           and idx[k].get_slice(k).get_shape().__len__() == 2
                           and "norm" not in k and "embed" not in k)
        else:
            names = list(DENSE_UNITS)
    else:
        idx = open_all(GLM_SRC)
        h = None
        names = sorted(k[:-len(".weight")] for k in idx
                       if k.endswith(".weight") and ".experts." in k)
    if a.limit:
        names = names[:a.limit]

    doc = {"grid": a.grid, "source": a.source, "window_bits": bits,
           "base_channel_sigma": sigma0, "mults": a.mult, "units": {}}
    print(f"{a.grid} L={bits} sigma0={sigma0:.6g} reach on "
          f"{len(names)} {a.source} units", flush=True)
    print(f"{'unit':<44}{'m':>5}{'rows':>7}{'over':>8}{'short':>8}"
          f"{'/over':>8}{'worst':>11}{'clip_h':>12}", flush=True)
    for name in names:
        t = idx[name + ".weight"].get_tensor(name + ".weight")
        if t.ndim != 2:
            continue
        w = t.to(dev).float()
        hv = None
        if h is not None:
            hk = h.get(name) if isinstance(h, dict) else None
            if hk is not None and hk.numel() == w.shape[1]:
                hv = hk
        rec = {}
        for m in a.mult:
            r = census(w, hv, grid, bits, sigma0 * m, recipe.window_seed, 16)
            r["channel_sigma_mult"] = m
            r["channel_sigma"] = sigma0 * m
            rec[f"x{m:g}"] = r
            print(f"{name:<44}{m:>5g}{r['rows']:>7d}{r['over']:>8.4f}"
                  f"{r['rows_short']:>8d}{r['short_of_over']:>8.3f}"
                  f"{r['worst_rel_shortfall']:>11.3e}"
                  f"{r.get('clip_h', float('nan')):>12.3e}", flush=True)
        doc["units"][name] = rec
        del w
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
