"""Score every candidate cost against the measured rung table, offline.

Issue #4 asks whether any candidate cost "orders the seven units' rungs
correctly".  Taken literally that question is vacuous and this script says so
with a number: within one unit every candidate is monotone in rate and so is
the measured curve over most of its range, so a per-unit rank correlation reads
~1 for all of them and discriminates nothing.

The ranking the allocator actually consumes is the **cross-unit exchange rate**:
a multi-choice knapsack compares *marginal cost per byte* across every (unit,
step) pair it could buy or sell.  So three scores are reported, in increasing
order of how much they resemble the decision:

  1. ``spearman_within_unit``  -- the literal reading, reported and dismissed.
  2. ``spearman_marginal_pooled`` -- candidate marginal-cost-per-byte against
     measured marginal-KL-per-byte, pooled over every (unit, consecutive-rung)
     step.  This is the quantity the DP ranks.
  3. ``reallocate`` -- run the knapsack on the candidate at the served byte
     budget, then evaluate its pick on the **measured** table.  Printing
     ``alloc/uniform`` is #4's own closure criterion, done without a serve.

Plus the proposal's section 3a split: fit one scalar per unit to the measured
curve given the candidate's shape, and ask whether a per-unit reweighting alone
would have recovered the served ordering.  If yes the curves are fine and the
exchange rate is wrong; if no, no reweighting saves the form.

The measured table is ``rd_table.json``: ``dkl(q, r) = KL(unit q at rung r,
the other six at R1006, the rest of the body BF16) - KL(all seven at R1006)``,
full-vocabulary KL against the local BF16 teacher on 4088 scored positions.
``rd_positions.npz`` carries the per-position KL of every arm, so each ``dkl``
gets a paired bootstrap interval and a difference at the top of the curve can
be called real or not.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

PQ = Path("/mnt/shared/tessera-runs/pq-continuous/qwen06b")
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
#: fused serving units: the DP solves over these, so a re-allocation must too
GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
          ("self_attn.o_proj",), ("mlp.gate_proj", "mlp.up_proj"),
          ("mlp.down_proj",))
ALLOC = {"self_attn.q_proj": 1083, "self_attn.k_proj": 1083, "self_attn.v_proj": 1083,
         "self_attn.o_proj": 934, "mlp.gate_proj": 1107, "mlp.up_proj": 1107,
         "mlp.down_proj": 749}
UNIFORM = 1006


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def knapsack(cost, bytes_, rungs, budget, groups=GROUPS, gran=64, min_frac=0.999):
    """Least-cost rung per fused group inside ``budget`` bytes (exact DP on a
    ``gran``-byte lattice; group bytes rounded UP so a pick never overspends).

    ``min_frac`` is the byte-match floor ``tessera.control.assert_byte_matched``
    enforces (0.1% by default): a pick that leaves more than that on the table
    is not a byte-matched comparison, it is a cheaper artifact, and the leftover
    bytes are quality the candidate declined to buy.  Solutions below the floor
    are refused here rather than compared later."""
    caps = int(budget // gran)
    INF = float("inf")
    best = np.full(caps + 1, INF)
    best[0] = 0.0
    choice = np.zeros((len(groups), caps + 1), dtype=np.int32)
    for gi, group in enumerate(groups):
        nxt = np.full(caps + 1, INF)
        pick = np.zeros(caps + 1, dtype=np.int32)
        for ri, r in enumerate(rungs):
            w = int(np.ceil(sum(bytes_[(u, r)] for u in group) / gran))
            c = sum(cost[(u, r)] for u in group)
            if w > caps:
                continue
            cand = np.full(caps + 1, INF)
            cand[w:] = best[:caps + 1 - w] + c
            upd = cand < nxt
            nxt = np.where(upd, cand, nxt)
            pick = np.where(upd, ri, pick)
        best, choice[gi] = nxt, pick
    lo = int(np.ceil(budget * float(min_frac) / gran))
    window = np.full_like(best, INF)
    window[lo:] = best[lo:]
    cap = int(np.argmin(window)) if np.isfinite(window).any() else int(np.argmin(best))
    out, c = {}, cap
    for gi in range(len(groups) - 1, -1, -1):
        ri = int(choice[gi][c])
        r = rungs[ri]
        for u in groups[gi]:
            out[u] = r
        c -= int(np.ceil(sum(bytes_[(u, r)] for u in groups[gi]) / gran))
    return out, float(best[cap])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--positions", default=None)
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--aura", required=True)
    ap.add_argument("--refit", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    table = json.loads(Path(args.table).read_text())
    rungs = tuple(table["rungs"])
    arms = {a["arm"]: a for a in table["arms"]}
    base = arms["anchor:uniform_R1006"]["kl_full"]
    wire = {(r, q): table["wire_bytes"][f"{r}|{q}"] for r in ROLES for q in rungs}

    dkl = {}
    for role in ROLES:
        for q in rungs:
            v = base if q == UNIFORM else arms[f"sweep:{role}@R{q}"]["kl_full"]
            dkl[(role, q)] = v - base

    # ---- paired bootstrap on the measured table -----------------------
    se = {}
    if args.positions:
        npz = np.load(args.positions)
        b = npz["anchor:uniform_R1006"]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, b.size, size=(args.bootstrap, b.size))
        for role in ROLES:
            for q in rungs:
                if q == UNIFORM:
                    se[(role, q)] = 0.0
                    continue
                d = npz[f"sweep:{role}@R{q}"] - b
                se[(role, q)] = float(d[idx].mean(axis=1).std(ddof=1))

    # ---- candidates ----------------------------------------------------
    stats = pickle.load((PQ / "probe.pkl").open("rb"))["stats"]
    h = {r: float(stats[f"model.layers.0.{r}"]["h_trace"]) for r in ROLES}
    shipped = pickle.load((PQ / "cost.pkl").open("rb"))["costs"]
    res = json.loads(Path(args.resolution).read_text())["settings"]
    aura = json.loads(Path(args.aura).read_text())

    cand = {}
    cand["L1_shipped_menu"] = {
        (r, q): 0.5 * h[r] * shipped[f"model.layers.0.{r}"][f"TESSERA_E4M3_K1_R{q}"]["output_mse"]
        for r in ROLES for q in rungs}
    for label, key in (("L1_remeasured_4x256", "seed0_rows256"),
                       ("L1_10x_rows2560", "seed0_rows2560"),
                       ("L1_40x_rows20480", "seed0_rows20480")):
        cand[label] = {(r, q): 0.5 * h[r] * res[key]["output_mse"][f"{r}|{q}"]
                       for r in ROLES for q in rungs}
    cand["AURA_kl_adjoint"] = {(r, q): aura["predicted_dloss"][f"{r}|{q}"]
                               for r in ROLES for q in rungs}
    cand["EMPIRICAL_unit_kl"] = {(r, q): dkl[(r, q)] for r in ROLES for q in rungs}

    # ---- candidate 2, honestly: a coarse empirical surface, interpolated --
    # Scored against itself the empirical table is exact by construction, which
    # measures nothing.  What a shipping empirical cost would actually be is a
    # few measured anchors plus an interpolant, and the campaign's own surface
    # interpolates linearly in (q256, log2 dloss) -- so do that, on the arm's
    # absolute KL (which is positive and interpolable where dkl is not).
    anchor_sets: dict[str, list[int]] = {}

    def interpolated(anchors):
        surf = {}
        for r in ROLES:
            xs = sorted(anchors)
            ys = [np.log(base + dkl[(r, a)]) for a in xs]
            for q in rungs:
                surf[(r, q)] = float(np.exp(np.interp(q, xs, ys))) - base
        return surf

    for n_anchors, anchors in (("5anchor", (320, 640, 900, 1083, 1340)),
                               ("7anchor", (320, 640, 826, 970, 1083, 1200, 1340)),
                               ("9anchor", (320, 512, 749, 900, 1006, 1083, 1150, 1262, 1340))):
        cand[f"EMPIRICAL_interp_{n_anchors}"] = interpolated(anchors)
        anchor_sets[n_anchors] = list(anchors)

    if args.refit:
        refit = json.loads(Path(args.refit).read_text())
    else:
        refit = None

    out = {"schema": "tessera.rung_cost_scoring/1", "rungs": list(rungs),
           "baseline_kl": base, "empirical_anchor_sets": anchor_sets,
           "candidates": {}}

    # measured marginal KL per byte on each consecutive step
    steps = [(r, rungs[i], rungs[i + 1]) for r in ROLES for i in range(len(rungs) - 1)]
    m_marg = np.array([(dkl[(r, b1)] - dkl[(r, b2)]) / max(1, wire[(r, b2)] - wire[(r, b1)])
                       for r, b1, b2 in steps])

    # three budgets: the receipt's 4.0-bpp point, and the uniform arms one
    # rung ladder step either side of it, because "nothing to allocate" at one
    # budget is not a statement about the axis.
    BUDGET_RUNGS = (749, UNIFORM, 1262)
    budget = sum(wire[(r, UNIFORM)] for r in ROLES)
    uniform_pred = base + sum(dkl[(r, UNIFORM)] for r in ROLES)
    alloc_pred = base + sum(dkl[(r, ALLOC[r])] for r in ROLES)

    for name, cost in cand.items():
        per_unit = {r: spearman([cost[(r, q)] for q in rungs],
                                [dkl[(r, q)] for q in rungs]) for r in ROLES}
        c_marg = np.array([(cost[(r, b1)] - cost[(r, b2)]) / max(1, wire[(r, b2)] - wire[(r, b1)])
                           for r, b1, b2 in steps])
        by_budget = {}
        for bq in BUDGET_RUNGS:
            bud = sum(wire[(r, bq)] for r in ROLES)
            uni = base + sum(dkl[(r, bq)] for r in ROLES)
            pk, _ = knapsack(cost, wire, rungs, bud)
            kl = base + sum(dkl[(u, pk[u])] for u in ROLES)
            by_budget[f"uniform_R{bq}"] = {
                "pick": pk, "bytes": sum(wire[(u, pk[u])] for u in ROLES),
                "budget": bud, "uniform_kl": uni,
                "predicted_kl_on_measured_table": kl, "vs_uniform": kl / uni}
        pick, _ = knapsack(cost, wire, rungs, budget)
        pick_kl = base + sum(dkl[(u, pick[u])] for u in ROLES)
        pick_bytes = sum(wire[(u, pick[u])] for u in ROLES)
        # inversions: steps the candidate prices as a gain that measured flat or worse
        inv = []
        for (r, b1, b2), cm, mm in zip(steps, c_marg, m_marg):
            if cm > 0 and mm <= 0:
                inv.append({"unit": r, "step": f"R{b1}->R{b2}",
                            "cand_gain_per_byte": float(cm),
                            "measured_gain_per_byte": float(mm),
                            "cand_dcost": float(cost[(r, b1)] - cost[(r, b2)]),
                            "measured_dkl": float(dkl[(r, b2)] - dkl[(r, b1)])})
        out["candidates"][name] = {
            "spearman_within_unit": per_unit,
            "spearman_within_unit_mean": float(np.mean(list(per_unit.values()))),
            "spearman_marginal_pooled": spearman(c_marg, m_marg),
            "reallocate": {"pick": pick, "bytes": pick_bytes, "budget": budget,
                           "predicted_kl_on_measured_table": pick_kl,
                           "vs_uniform": pick_kl / uniform_pred},
            "reallocate_by_budget": by_budget,
            "n_inversions": len(inv), "inversions": inv,
        }

    # ---- oracle and the served points ----------------------------------
    oracle, _ = knapsack({k: v for k, v in dkl.items()}, wire, rungs, budget)
    oracle_by_budget = {}
    for bq in BUDGET_RUNGS:
        bud = sum(wire[(r, bq)] for r in ROLES)
        uni = base + sum(dkl[(r, bq)] for r in ROLES)
        pk, _ = knapsack({k: v for k, v in dkl.items()}, wire, rungs, bud)
        kl = base + sum(dkl[(u, pk[u])] for u in ROLES)
        oracle_by_budget[f"uniform_R{bq}"] = {
            "pick": pk, "bytes": sum(wire[(u, pk[u])] for u in ROLES), "budget": bud,
            "uniform_kl": uni, "kl": kl, "vs_uniform": kl / uni}
    out["oracle_by_budget"] = oracle_by_budget
    out["reference"] = {
        "uniform_R1006": {"kl": uniform_pred, "bytes": budget},
        "served_allocation": {"pick": ALLOC, "kl_additive": alloc_pred,
                              "kl_measured": arms["anchor:allocated"]["kl_full"],
                              "bytes": sum(wire[(r, ALLOC[r])] for r in ROLES),
                              "vs_uniform_additive": alloc_pred / uniform_pred,
                              "vs_uniform_measured":
                                  arms["anchor:allocated"]["kl_full"] / base},
        "oracle_on_measured_table": {
            "pick": oracle, "kl": base + sum(dkl[(u, oracle[u])] for u in ROLES),
            "bytes": sum(wire[(u, oracle[u])] for u in ROLES),
            "vs_uniform": (base + sum(dkl[(u, oracle[u])] for u in ROLES)) / uniform_pred},
        "additivity": {
            "sum_of_single_unit_dkl": alloc_pred - base,
            "measured_joint_dkl": arms["anchor:allocated"]["kl_full"] - base,
            "share": (alloc_pred - base) / (arms["anchor:allocated"]["kl_full"] - base)},
    }

    # ---- proposal 3a: is it the weights or the curves? ------------------
    # fit one scalar per unit: min_a sum_r (log dkl - log(a*cost))^2, on the
    # rungs where the measured dkl is positive and resolved
    threea = {}
    for name, cost in cand.items():
        scale = {}
        for r in ROLES:
            xs = [(cost[(r, q)], dkl[(r, q)]) for q in rungs
                  if dkl[(r, q)] > 0 and cost[(r, q)] > 0]
            if len(xs) < 3:
                continue
            scale[r] = float(np.exp(np.mean([np.log(y) - np.log(x) for x, y in xs])))
        if len(scale) < len(ROLES):
            threea[name] = {"fitted": False}
            continue
        rw = {(r, q): scale[r] * cost[(r, q)] for r in ROLES for q in rungs}
        pick, _ = knapsack(rw, wire, rungs, budget)
        kl = base + sum(dkl[(u, pick[u])] for u in ROLES)
        threea[name] = {"fitted": True, "per_unit_scale": scale, "pick": pick,
                        "kl": kl, "vs_uniform": kl / uniform_pred,
                        "implied_h_ratio": {r: scale[r] * h[r] for r in ROLES}}
    out["reweighted_3a"] = threea
    out["measured_dkl"] = {f"{r}|{q}": dkl[(r, q)] for r in ROLES for q in rungs}
    out["dkl_bootstrap_se"] = {f"{r}|{q}": se.get((r, q)) for r in ROLES for q in rungs}
    out["wire_bytes"] = {f"{r}|{q}": wire[(r, q)] for r in ROLES for q in rungs}
    if refit is not None:
        out["refit"] = refit
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
