"""Candidate 3: does the rung mispricing track the LS scale refit?

THE PREMISE, STATED HONESTLY.  Issue #4 puts it as "a cost measured on an
un-refit render prices a different object than the one that ships".  For *this*
allocation that premise is false: ``docs/measurements/tessera-allocated-served-2026-09-02.md``
proved the priced blobs and the served bytes were the same objects, and the
exporter's default ``scale_refit`` was on for both.  So there is no un-refit /
refit mismatch to blame here.

What is still worth measuring, and what this script measures:

  (a) whether the *shape* of the measured rung curve depends on the refit at
      all -- i.e. whether the plateau above R1006 is an artifact of refitting;
  (b) whether the L1 surrogate's per-rung residual (how far its prediction is
      from the measured KL, after the best per-unit scalar) is *correlated*
      with how much the refit moved that rung -- which is the testable form of
      "the mispricing tracks refit gain";
  (c) what a cost priced on the un-refit render would have picked, evaluated on
      the measured (refit) table -- the literal candidate.

The H-aware half of #4's candidate 3 -- LDLQ and the exact full-H refit -- is
NOT measured here: those change the encode itself, so the rung curve would have
to be re-swept under them.  Named as untested, not asserted either way.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
UNIFORM = 1006
PQ = Path("/mnt/shared/tessera-runs/pq-continuous/qwen06b")


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def knapsack(cost, bytes_, rungs, budget, gran=64, min_frac=0.999):
    groups = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
              ("self_attn.o_proj",), ("mlp.gate_proj", "mlp.up_proj"),
              ("mlp.down_proj",))
    units = []
    for g in groups:
        opts = []
        for q in rungs:
            opts.append((q, sum(bytes_[(r, q)] for r in g),
                         sum(cost[(r, q)] for r in g)))
        units.append((g, opts))
    nb = budget // gran + 1
    lo = int(np.ceil(budget * min_frac / gran))
    INF = float("inf")
    dp = np.full(nb, INF)
    dp[0] = 0.0
    choice = [np.full(nb, -1, dtype=np.int32) for _ in units]
    for ui, (_g, opts) in enumerate(units):
        nxt = np.full(nb, INF)
        ch = choice[ui]
        for oi, (_q, b, c) in enumerate(opts):
            step = b // gran
            if step >= nb:
                continue
            cand = np.full(nb, INF)
            cand[step:] = dp[:nb - step] + c
            m = cand < nxt
            nxt = np.where(m, cand, nxt)
            ch[m] = oi
        dp = nxt
    feasible = np.where(np.isfinite(dp[lo:]))[0]
    best = lo + int(feasible[np.argmin(dp[lo:][feasible])])
    pick = {}
    cur = best
    for ui in range(len(units) - 1, -1, -1):
        g, opts = units[ui]
        oi = int(choice[ui][cur])
        q, b, _c = opts[oi]
        for r in g:
            pick[r] = q
        cur -= b // gran
    return pick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="rd_table.json (refit on)")
    ap.add_argument("--table-norefit", required=True)
    ap.add_argument("--mse", required=True, help="output_mse_resolution.json (refit)")
    ap.add_argument("--mse-norefit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def load(p):
        t = json.loads(Path(p).read_text())
        arms = {a["arm"]: a for a in t["arms"]}
        base = arms["anchor:uniform_R1006"]["kl_full"]
        rungs = tuple(t["rungs"])
        d = {}
        for r in ROLES:
            for q in rungs:
                v = base if q == UNIFORM else arms[f"sweep:{r}@R{q}"]["kl_full"]
                d[(r, q)] = v - base
        wire = {(r, q): t["wire_bytes"][f"{r}|{q}"] for r in ROLES for q in rungs}
        return t, arms, base, rungs, d, wire

    t_on, arms_on, base_on, rungs_on, dkl_on, wire = load(args.table)
    t_off, arms_off, base_off, rungs_off, dkl_off, _ = load(args.table_norefit)
    shared = tuple(q for q in rungs_off if q in rungs_on)

    mse_on = json.loads(Path(args.mse).read_text())["settings"]["seed0_rows256"]["output_mse"]
    mse_off = json.loads(Path(args.mse_norefit).read_text())["settings"]["seed0_rows256"]["output_mse"]
    stats = pickle.load((PQ / "probe.pkl").open("rb"))["stats"]
    h = {r: float(stats[f"model.layers.0.{r}"]["h_trace"]) for r in ROLES}

    out = {"schema": "tessera.rung_refit_tracking/1",
           "shared_rungs": list(shared),
           "baseline_kl": {"refit_on": base_on, "refit_off": base_off},
           "note": ("The priced and served blobs of the audited allocation were "
                    "both the refit render; the un-refit arm here is a "
                    "counterfactual, not what shipped."),
           "untested": ("LDLQ and the exact full-H refit change the encode, so "
                        "the rung curve under them was not swept.")}

    # (a) does the refit change the SHAPE of the curve?
    shape = {}
    for r in ROLES:
        row = {}
        for q in shared:
            row[f"R{q}"] = {
                "dkl_refit_on": dkl_on[(r, q)],
                "dkl_refit_off": dkl_off[(r, q)],
                "mse_refit_on": mse_on[f"{r}|{q}"],
                "mse_refit_off": mse_off[f"{r}|{q}"],
                "refit_mse_gain": mse_off[f"{r}|{q}"] / mse_on[f"{r}|{q}"],
            }
        shape[r] = row
    out["per_unit"] = shape
    out["curve_shape_spearman_on_vs_off"] = {
        r: spearman([dkl_on[(r, q)] for q in shared],
                    [dkl_off[(r, q)] for q in shared]) for r in ROLES}

    # (b) does the L1 residual track the refit gain?
    #     residual_{r,q} = log(measured dkl / (a_r * L1 cost)), a_r the best
    #     per-unit scalar, over rungs where the measured dkl is positive.
    l1 = {(r, q): 0.5 * h[r] * mse_on[f"{r}|{q}"] for r in ROLES for q in shared}
    scale = {}
    for r in ROLES:
        xs = [(l1[(r, q)], dkl_on[(r, q)]) for q in shared if dkl_on[(r, q)] > 0]
        scale[r] = float(np.exp(np.mean([np.log(y) - np.log(x) for x, y in xs]))) if len(xs) >= 2 else float("nan")
    resid, gain, keys = [], [], []
    for r in ROLES:
        for q in shared:
            if dkl_on[(r, q)] <= 0 or not np.isfinite(scale[r]):
                continue
            resid.append(float(np.log(dkl_on[(r, q)] / (scale[r] * l1[(r, q)]))))
            gain.append(float(np.log(mse_off[f"{r}|{q}"] / mse_on[f"{r}|{q}"])))
            keys.append(f"{r}|R{q}")
    out["residual_vs_refit_gain"] = {
        "n": len(resid), "keys": keys,
        "log_residual": resid, "log_refit_gain": gain,
        "spearman": spearman(resid, gain),
        "pearson": float(np.corrcoef(resid, gain)[0, 1]) if len(resid) > 2 else float("nan"),
        "per_unit_scale": scale,
    }

    # (c) the literal candidate: price on the un-refit render, deploy the refit one
    budget = sum(wire[(r, UNIFORM)] for r in ROLES)
    uni = base_on + sum(dkl_on[(r, UNIFORM)] for r in ROLES)
    picks = {}
    for label, cost in (("L1_on_refit_render", l1),
                        ("L1_on_unrefit_render",
                         {(r, q): 0.5 * h[r] * mse_off[f"{r}|{q}"]
                          for r in ROLES for q in shared})):
        pk = knapsack(cost, wire, shared, budget)
        kl = base_on + sum(dkl_on[(u, pk[u])] for u in ROLES)
        picks[label] = {"pick": pk, "bytes": sum(wire[(u, pk[u])] for u in ROLES),
                        "budget": budget, "predicted_kl_on_measured_table": kl,
                        "vs_uniform": kl / uni}
    out["reallocate_shared_rungs"] = picks

    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)
    print("curve-shape spearman(on,off):",
          {r: round(v, 3) for r, v in out["curve_shape_spearman_on_vs_off"].items()})
    print("residual vs refit gain: spearman",
          round(out["residual_vs_refit_gain"]["spearman"], 3),
          " pearson", round(out["residual_vs_refit_gain"]["pearson"], 3),
          f"(n={out['residual_vs_refit_gain']['n']})")
    for k, v in picks.items():
        print(f"  {k:24s} vs_uniform {v['vs_uniform']:.3f}  {v['pick']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
