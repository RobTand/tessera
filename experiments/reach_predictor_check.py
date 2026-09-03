"""Does a *derived* predictor pick the same per-unit reach arm the sweep did?

Issue #80 says ``channel_sigma``'s optimum is per-unit.  A per-unit value found
by sweeping each unit is a lookup table; the question this answers is whether
the winner is **predictable from the unit itself** -- from ``(w, h, codebook)``,
with no Viterbi -- because only then is there a rule to ship.

The predictor.  Under a CHANNEL plane the window table's entries are equal-mass
Gaussian quantiles at ``table_sigma`` snapped to the grid, and the body is the
table: at each position the trellis may land on the ``2^R`` entries its ``R``
new bits reach from the current state.  Those entries are a permutation of the
quantile set, so the first-order model of one position is *the nearest of ``K =
2^R`` draws from the table's own empirical distribution* -- Tseng et al.'s random
Gaussian codebook, which is what the table is built to be.  For a target ``t``:

    D(t) = E[ min_{K draws c} (t - c)^2 ] = int_0^inf 2x (1 - F(t+x) + F(t-x))^K dx

with ``F`` the table's empirical CDF.  It is a pure function of the codebook and
the rate, tabulated once per arm.  The unit's predicted h-weighted error is then

    pred(arm) = sqrt( sum_rj h_j s_r^2 D(w_rj / s_r) / sum_rj h_j w_rj^2 )

where ``s_r`` is the row scale ``initial_channel_scale`` actually assigns under
that arm -- the production function, called, not modelled, so the reach-aware
per-row start (and with it the whole squeeze-vs-clip trade this issue is about)
is in the prediction rather than beside it.

What it does NOT model: the trellis's path memory (this is the one-step bound),
the scale refit, and the LDLQ compensation.  If the ranking survives all three
the predictor is usable; if it does not, that is the finding.

Scored against ``/mnt/shared/tessera-runs/reach/reach_e4m3_*.json`` -- the arms
already on disk, so no encode is re-run and the realised numbers are the ones
the issue was filed on.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E4M3_GRID, BF16_GRID          # noqa: E402
from tessera.encode import grid_vector_table, window_table  # noqa: E402
from tessera.scale_channel import default_channel_sigma, initial_channel_scale  # noqa: E402

DENSE_SRC = "/home/rob/models/Qwen3-0.6B"
DENSE_H = "/mnt/shared/tessera-runs/bf16/refs/h_diag.pt"
RUNS = "/mnt/shared/tessera-runs/reach"
GRIDS = {"e4m3": E4M3_GRID, "bf16": BF16_GRID}
BF16_CHANNEL_SIGMA = 1.0


def open_all(path: str):
    handles = [safe_open(str(f), framework="pt") for f in sorted(Path(path).glob("*.safetensors"))]
    return {k: h for h in handles for k in h.keys()}


def codebook(grid, window_bits: int, table_sigma: float, device) -> torch.Tensor:
    """The table's reconstruction values, in grid units, with multiplicity."""
    codes = window_table(grid, window_bits, sigma=table_sigma, seed=0, half=16, device=device)
    return grid_vector_table(grid, device)[codes.long()].reshape(-1).float()


def distortion_table(values: torch.Tensor, K: int, n_t: int = 2049, n_x: int = 3072):
    """``D(t)`` on a grid of ``t``, plus the grid, for the nearest-of-K model.

    ``F`` is the empirical CDF of ``values``; the integral is taken on a
    log-spaced ``x`` grid that runs from far below the codebook's finest
    spacing to far above its span, so the tail of ``g^K`` is resolved at both
    ends.
    """
    v = values.sort().values
    span = float(v.abs().max())
    t = torch.linspace(-span, span, n_t, device=v.device, dtype=torch.float64)
    x = torch.logspace(math.log10(span * 1e-6), math.log10(span * 4.0), n_x,
                       device=v.device, dtype=torch.float64)
    n = v.numel()
    vd = v.double()
    # F(t+x) - F(t-x) for every (t, x): the mass within x of t.
    hi = torch.searchsorted(vd, (t[:, None] + x[None, :]).contiguous(), right=True)
    lo = torch.searchsorted(vd, (t[:, None] - x[None, :]).contiguous(), right=False)
    inside = (hi - lo).double() / n
    g = (1.0 - inside).clamp(0.0, 1.0)
    integrand = 2.0 * x[None, :] * g.pow(K)
    D = torch.trapz(integrand, x, dim=1)
    return t.float(), D.float()


def predict(w, h, grid, window_bits, channel_sigma, window_sigma, K):
    device = w.device
    table_sigma = channel_sigma if window_sigma is None else window_sigma
    vals = codebook(grid, window_bits, table_sigma, device)
    reach = float(vals.abs().max())
    _stored, eff, _g = initial_channel_scale(w, channel_sigma, reach=reach)
    t_grid, D = distortion_table(vals, K)
    step = float(t_grid[1] - t_grid[0])
    lo = float(t_grid[0])
    tgt = (w / eff[:, None]).clamp(lo, -lo)
    idx = ((tgt - lo) / step)
    i0 = idx.floor().long().clamp(0, D.numel() - 2)
    frac = (idx - i0.float()).clamp(0, 1)
    d = D[i0] * (1 - frac) + D[i0 + 1] * frac
    num = ((d * (eff[:, None] ** 2)) * h[None, :]).sum()
    den = ((w * w) * h[None, :]).sum()
    return float((num / den).sqrt()), reach, float((w.abs().amax(1) * channel_sigma
                                                   > reach * w.pow(2).mean(1).sqrt()).float().mean())


def load_measured(tag: str):
    """``{(unit, rung, m, ratio): h}`` from the sweeps already on disk."""
    out, base = {}, None
    for name in ("spread", "ratio", "wide"):
        p = Path(RUNS) / f"reach_{tag}_{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        base = d["base_channel_sigma"]
        for unit, res in d["units"].items():
            for arm, r in res.items():
                if not isinstance(r, dict) or "wt" not in r or "[repeat]" in arm:
                    continue
                rung = int(arm.split()[0][1:])
                out[(unit, rung, r["channel_sigma_mult"], r["table_ratio"])] = r["h"]
    return out, base


def spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main(tag="e4m3", window_bits=14):
    grid = GRIDS[tag]
    base = BF16_CHANNEL_SIGMA if grid is BF16_GRID else default_channel_sigma(grid)
    measured, file_base = load_measured(tag)
    assert file_base is None or abs(file_base - base) < 1e-9, (file_base, base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = open_all(DENSE_SRC)
    H = {k: v.to(device).float() for k, v in torch.load(DENSE_H).items()}
    units = sorted({k[0] for k in measured})
    rungs = sorted({k[1] for k in measured})
    arms = sorted({(k[2], k[3]) for k in measured})
    print(f"grid {tag}  L={window_bits}  base sigma {base:.6g}  "
          f"{len(units)} units x {len(rungs)} rungs x {len(arms)} arms")
    doc = {"grid": tag, "window_bits": window_bits, "base_channel_sigma": base, "cells": {}}
    hits = tot = 0
    for rung in rungs:
        K = 1 << (rung // 256)
        print(f"\n=== R{rung}  (K = 2^{rung // 256} = {K} reachable entries per position)")
        head = f"{'unit':<34}" + "".join(f"{f'{m:g}/{r:g}':>9}" for m, r in arms)
        print("  measured (h ratio vs default), then predicted (same normalisation)")
        print(head + f"{'argmin':>12}{'pred':>12}{'rho':>7}")
        for unit in units:
            w = src[unit + ".weight"].get_tensor(unit + ".weight").to(device).float().contiguous()
            h = H[unit]
            mrow, prow, keys = [], [], []
            for m, ratio in arms:
                key = (unit, rung, m, ratio)
                if key not in measured:
                    continue
                cs = m * base
                ws = None if ratio == 1.0 else ratio * cs
                p, reach, over = predict(w, h, grid, window_bits, cs, ws, K)
                keys.append((m, ratio)); mrow.append(measured[key]); prow.append(p)
            mb = mrow[keys.index((1.0, 1.0))]; pb = prow[keys.index((1.0, 1.0))]
            am = keys[int(torch.tensor(mrow).argmin())]
            ap = keys[int(torch.tensor(prow).argmin())]
            hits += int(am == ap); tot += 1
            doc["cells"][f"{unit}|R{rung}"] = {
                "arms": [list(k) for k in keys], "measured": mrow, "predicted": prow,
                "argmin_measured": list(am), "argmin_predicted": list(ap),
                "spearman": spearman(mrow, prow),
                "realised_if_predicted": measured[(unit, rung, *ap)] / mb,
                "oracle": min(mrow) / mb,
            }
            print(f"{unit.replace('model.', ''):<34}"
                  + "".join(f"{v / mb:9.4f}" for v in mrow)
                  + f"{f'{am[0]:g}/{am[1]:g}':>12}{f'{ap[0]:g}/{ap[1]:g}':>12}"
                  + f"{spearman(mrow, prow):7.3f}")
            print(f"{'  ^predicted':<34}" + "".join(f"{v / pb:9.4f}" for v in prow))
            del w
            torch.cuda.empty_cache() if device == "cuda" else None
        rc = [doc["cells"][f'{u}|R{rung}'] for u in units]
        g_rule = math.exp(sum(math.log(c["realised_if_predicted"]) for c in rc) / len(rc))
        g_orc = math.exp(sum(math.log(c["oracle"]) for c in rc) / len(rc))
        print(f"    R{rung} geomean: rule-as-predicted {g_rule:.4f}   oracle {g_orc:.4f}   "
              f"argmin agreement {sum(1 for c in rc if c['argmin_measured'] == c['argmin_predicted'])}/{len(rc)}")
        doc["cells"][f"R{rung}_summary"] = {"geomean_rule": g_rule, "geomean_oracle": g_orc}
    print(f"\nargmin agreement overall {hits}/{tot}")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"/home/rob/tmp/musefix/predictor_{tag}.json")
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e4m3")
