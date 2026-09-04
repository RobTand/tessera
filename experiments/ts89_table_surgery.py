"""#89's mechanism, isolated on one column at a time and on the CPU.

``viterbi_window`` is exact and carries no state across columns -- "encoding a
range is bit-identical to the same columns of a full pass over the same targets
and scale" -- so the whole 1.36x effect, if it is real, must be reproducible on
the handful of Hessian-dominant columns that carry the h-weighted error, with
no GPU and no export.  That makes the *mechanism* question answerable by
surgery rather than by another sweep: hold everything fixed and swap defined
subsets of one table's entries for the other's.

The two tables are the same seeded permutation of the same equal-mass Gaussian
quantiles -- ``torch.randperm`` is seeded on ``seed`` alone, so state ``s``
draws quantile ``q_s`` under both sigmas.  Only the *snap* differs.  So entry
``s`` of the two tables is the same quantile landed on two different grid
values, and replacing "the top N entries of the 0.75 table with the 1.0
table's, rescaled" is a well-defined operation on matched entries.  The N at
which the differential dies names which entries carry it:

    N ~ 2      the reach itself
    N ~ 50     the tail
    N in the hundreds or more    the interior

Stages::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp PYTHONPATH=src:experiments"
    env $E $P experiments/ts89_table_surgery.py --out OUT/surgery.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E4M3_GRID  # noqa: E402
from tessera.encode import (  # noqa: E402
    grid_vector_table,
    viterbi_window,
    window_table,
)
from tessera.scale_channel import default_channel_sigma, initial_channel_scale  # noqa: E402
from bf16_route_weight_space import DENSE_H, DENSE_SRC, open_all  # noqa: E402

UNIT = "model.layers.2.mlp.down_proj"
L, RATE = 14, 8


def table_values(sigma: float) -> torch.Tensor:
    codes = window_table(E4M3_GRID, L, sigma=sigma, seed=0, half=16)
    return grid_vector_table(E4M3_GRID)[codes.long()].squeeze(-1).float()


def scales(w: torch.Tensor, sigma: float, reach: float) -> torch.Tensor:
    _stored, effective, _g = initial_channel_scale(w, sigma, reach=reach)
    return effective


def run(w_cols: torch.Tensor, scale: torch.Tensor, values: torch.Tensor) -> float:
    """SSE of one set of columns against one table, in the weight's own units.

    One Viterbi pass, no scale refit -- a REDUCED model of the production
    encoder (which runs ``scale_refit=4``).  The branch metric carries
    production's ``trellis_weighting="scale"`` weight, ``(c / max c)^2`` per
    position, so the path this picks is the path production's first pass
    picks; the SSE returned is recomputed in w-space from the codes, so it is
    the true error of that path whatever the objective was.

    One deliberate infidelity: ``scale`` here is production's *effective* fp32
    row scale, where the artifact carries an fp16 word times a global.  The
    reduction still agreed with production's ``scale_refit=0`` arm to four
    decimals (1.0369 against 1.0367), which is itself a small finding -- the
    fp16 landing is inert before the refit runs.
    """
    scale_col = scale.view(-1, 1)
    targets = (w_cols / scale_col).contiguous()
    weights = ((scale_col / scale_col.amax()) ** 2).expand_as(targets).contiguous()
    state, _sse = viterbi_window(targets, values.view(-1, 1), L, RATE,
                                 weights=weights, impl="reference")
    hat = values[state] * scale_col
    return float(((hat - w_cols) ** 2).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--columns", type=int, default=8,
                   help="how many of the top h-energy columns to run")
    p.add_argument("--ns", type=int, nargs="+",
                   default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                            2048, 4096, 8192, 16384])
    a = p.parse_args()

    src = open_all(DENSE_SRC)
    w = src[UNIT + ".weight"].get_tensor(UNIT + ".weight").float().contiguous()
    h = torch.load(DENSE_H)[UNIT].float()
    base = default_channel_sigma(E4M3_GRID)

    doc: dict = {"unit": UNIT, "L": L, "rate": RATE, "base": base,
                 "shape": list(w.shape)}
    # The columns that carry the h-weighted budget.
    budget = h * (w * w).sum(0)
    order = budget.argsort(descending=True)
    pick = order[: a.columns]
    doc["columns"] = [[int(c), float(budget[c] / budget.sum())] for c in pick.tolist()]
    print("top h-energy columns (index, share of budget):", doc["columns"], flush=True)

    arms = {}
    tables, scale_by = {}, {}
    for name, sigma in (("s1.00", base), ("s0.75", 0.75 * base)):
        v = table_values(sigma)
        tables[name] = v
        scale_by[name] = scales(w, sigma, float(v.abs().max()))
        arms[name] = sigma

    sub = w[:, pick].contiguous()
    hw = h[pick]
    base_sse = {}
    for name in tables:
        per_col = []
        for j in range(pick.numel()):
            per_col.append(run(sub[:, j: j + 1], scale_by[name], tables[name]))
        base_sse[name] = per_col
        print(f"{name}: per-column sse {[f'{x:.6g}' for x in per_col]}", flush=True)
    doc["per_column_sse"] = base_sse
    hsum = {k: float((torch.tensor(v) * hw).sum()) for k, v in base_sse.items()}
    doc["h_weighted_sse"] = hsum
    doc["h_ratio_on_these_columns"] = (hsum["s0.75"] / hsum["s1.00"]) ** 0.5
    print(f"h-weighted sse {hsum}; h ratio on these columns "
          f"{doc['h_ratio_on_these_columns']:.4f}", flush=True)

    # ---- surgery: give the 0.75 table the 1.00 table's top-N entries -------
    # Matched by STATE: both tables draw the same quantile at the same state,
    # so entry s of one is entry s of the other landed on a different grid
    # value.  "Top N" is by the magnitude of the quantile, i.e. by the 1.00
    # table's own |value| order, and the donated value is rescaled by 0.75 so
    # the graft is a change of *snap* and not of spread.
    v1, v075 = tables["s1.00"], tables["s0.75"]
    rank = v1.abs().argsort(descending=True)
    doc["surgery"] = []
    for n in a.ns:
        grafted = v075.clone()
        if n:
            take = rank[:n]
            grafted[take] = 0.75 * v1[take]
        sc = scales(w, 0.75 * base, float(grafted.abs().max()))
        per_col = [run(sub[:, j: j + 1], sc, grafted) for j in range(pick.numel())]
        got = float((torch.tensor(per_col) * hw).sum())
        row = {"n": n, "h_weighted_sse": got,
               "h_ratio": (got / hsum["s1.00"]) ** 0.5,
               "reach": float(grafted.abs().max()),
               "n_distinct": int(torch.unique(grafted.abs()).numel())}
        doc["surgery"].append(row)
        print(f"  graft top {n:6d} entries -> h ratio {row['h_ratio']:.4f} "
              f"(reach {row['reach']:.1f})", flush=True)
        Path(a.out).write_text(json.dumps(doc, indent=1))
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
