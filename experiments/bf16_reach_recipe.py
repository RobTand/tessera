"""The BF16 reach term, measured through the built path (issue #48).

``BF16_RECIPE`` used to leave ``window_sigma`` unset, which ties the window
table's spread to the row scale and pins the body's **reach** -- how many
row-RMS the largest table entry can express -- to one value at every rung.
Issue #18's ``--stage reach`` found that value optimal at R=4 on four dense
Qwen Linears and 19% off at R=8, and could only find it by pinning
``window_sigma`` outside the recipe.  ``export._window_sigma_for`` now carries
the term, and this is the measurement of the thing that was built rather than
of a proxy for it.

**The claim this registers before it measures anything.**

1.  The built path at R=4 writes **the same bytes as the pinned wire**.  The
    reference rung is where the recipe is calibrated, and ``window_sigma =
    channel_sigma = 1.0`` is what ``encode_unit`` resolves an unset spread to
    on a CHANNEL plane, so "explicit 1.0" and "unset" must be the same file,
    not merely the same error.  If they are not, the term is not a
    re-parameterisation of the pinned wire and everything below is measuring
    two changes at once.
2.  The built path at R=8 reproduces the sweep's ``0.813x`` on ``wt``, **to
    within the gauge tolerance and no closer**.  The sweep spent the ratio as
    ``(window_sigma 1.0, channel_sigma 0.707)``; the recipe spends it as
    ``(1.414, 1.0)``.  Those are the same reach and differ by a factor of
    ``sqrt(2)`` on both ends -- a *non-dyadic* gauge shift, so by #18's own
    gauge result they land in different orbits and agree to ~0.02%, not
    exactly.  The sweep's own arm is run here, in the same table, so the
    tolerance is measured rather than asserted.
3.  The law ``reach*(R) = reach*(4) sqrt(R/4)`` predicts rungs it was never
    fitted to.  R=2 and R=6 were not in the sweep.  At **R=2 the law predicts
    a reach of 2.86 row-RMS -- less than the pinned 4.05, the opposite
    direction from the R=8 finding** -- which is the falsifiable half: a law
    that only ever says "reach further" would be indistinguishable from a
    monotone drift.  A rung where the bracket beats the law is the finding.

**Note on the recorded R<=3 rows.**  The sub-reference arms in the recorded
JSON were run *before* the floor existed: ``arms_for`` reads its ``law`` value
from ``wire_recipe``, and the recipe is now floored at the reference rung, so a
re-run at R<=3 gives ``built (recipe) == pinned 1.0`` and the unfloored law is
no longer one of the default arms.  The floor was added **on** that result --
it is the finding, not a precondition of it.  To reproduce the unfloored
sub-reference arms, pass the values explicitly with ``--extra-sigmas`` (R=1
0.5, R=2 0.7071, R=3 0.8660).

**The controls.**  Every arm of a unit runs in one process in a fixed order;
the built arm is run first at each rung and repeated last, and the repeat is
asserted byte- *and* tensor-identical -- the encoder is deterministic, so this
is not a noise estimate but a check that no arm leaked state (a mis-keyed
window-table cache is exactly the failure it catches).  Every arm within a
(unit, rung) writes the same number of bytes, which is what makes the ratios
readable at all; the bpp column is printed so that is visible and not trusted.

Weight space, four dense units, no serve.  Principle 3: the next gate is a
served A/B at matched bytes in one vLLM session on the BF16 lane, and nothing
here is promotable without it.

Run::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=src:experiments"
    env $E $P experiments/bf16_reach_recipe.py --out OUT/reach_recipe.json
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import BF16_GRID                              # noqa: E402
from tessera.export import (                                        # noqa: E402
    BF16_CHANNEL_SIGMA,
    BF16_WINDOW_SIGMA,
    encode_linear_planes,
    wire_recipe,
)
from tessera.unit_artifact import read_unit_artifact                # noqa: E402

from bf16_l_sigma_sweep import (                                    # noqa: E402
    DENSE_UNITS,
    Bench,
    check_repeat_tensor,
    reach_stats,
    score,
    sha,
    tensor_sha,
)
from bf16_route_weight_space import DENSE_H, DENSE_SRC, geomean, open_all  # noqa: E402

ROOT2 = math.sqrt(2.0)


def arms_for(q256: int, extra=()) -> "list[tuple[str, float, float]]":
    """``(label, window_sigma, channel_sigma)`` for one rung, built arm first.

    ``None`` means "let the recipe decide" -- the built path, and the only arm
    that measures the change rather than a hand-set spread.  The bracket is
    the law halved and doubled in reach; where a bracket coincides with the
    pinned wire it is dropped rather than run twice under two names, and the
    log says so.
    """
    law = wire_recipe(BF16_GRID, q256).window_sigma
    arms = [("built (recipe)", None, BF16_CHANNEL_SIGMA)]
    # The old wire, spelled explicitly.  Kept even at the reference rung,
    # where it is the *point*: there the recipe resolves to this value and the
    # two arms must be the same file, not merely the same error.
    arms.append((f"pinned 1.0 (old wire) s={BF16_WINDOW_SIGMA:.4g}",
                 BF16_WINDOW_SIGMA, BF16_CHANNEL_SIGMA))
    seen = {round(law, 9), round(BF16_WINDOW_SIGMA, 9)}
    for label, sigma in (("law/sqrt2", law / ROOT2), ("law*sqrt2", law * ROOT2)):
        if round(sigma, 9) in seen:
            continue
        seen.add(round(sigma, 9))
        arms.append((f"{label} s={sigma:.4g}", sigma, BF16_CHANNEL_SIGMA))
    # The sweep's own parameterisation of the same reach: a non-dyadic gauge
    # shift of the built arm, so it bounds how exactly a re-parameterisation
    # can be expected to reproduce a measurement.  Degenerate at the reference
    # rung, where it is the pinned arm again.
    if round(law, 9) != round(BF16_WINDOW_SIGMA, 9):
        arms.append((f"gauge twin (1.0, {1.0 / law:.4g})", 1.0, 1.0 / law))
    # Spreads named on the command line, for asking whether an optimum the
    # bracket only bounds on one side is interior.  Below the reference rung
    # the law's bracket sits entirely under the pinned wire, so "the law loses
    # to pinned" there does not yet say pinned is the best of them.
    for sigma in extra:
        if round(float(sigma), 9) in seen:
            continue
        seen.add(round(float(sigma), 9))
        arms.append((f"extra s={float(sigma):.4g}", float(sigma), BF16_CHANNEL_SIGMA))
    return arms


def encode_arm(w, q256, name, sigma, csigma):
    started = time.time()
    kwargs = {} if sigma is None else {"window_sigma": float(sigma)}
    exported, _unit, _forests = encode_linear_planes(
        w, grid=BF16_GRID, q256=q256, name=name,
        channel_sigma=float(csigma), verify=True, **kwargs)
    hat = read_unit_artifact(exported.blob, device=w.device)
    return hat, float(exported.bpp), sha(exported.blob), time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/shared/tessera-runs/bf16/qsweep/reach_recipe.json")
    ap.add_argument("--units", nargs="*", default=DENSE_UNITS)
    ap.add_argument("--rungs", nargs="*", type=int, default=[512, 1024, 1536, 2048])
    ap.add_argument("--extra-sigmas", nargs="*", type=float, default=[])
    a = ap.parse_args()

    b = Bench(a.out)
    src = open_all(DENSE_SRC)
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    b.doc = {"args": vars(a), "units": {}}
    b.log(f"BF16 reach recipe: window_sigma per rung, channel_sigma pinned at "
          f"{BF16_CHANNEL_SIGMA} (a gauge).  The built arm names no spread.")

    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = H[name]
        res: dict = {"rows": w.shape[0], "cols": w.shape[1]}
        for q in a.rungs:
            recipe = wire_recipe(BF16_GRID, q)
            b.log(f"\n== {name} {tuple(w.shape)}  R={q / 256:g}  recipe "
                  f"L={recipe.window_bits} window_sigma={recipe.window_sigma:.6f}")
            b.header(("bpp", "wt", "h", "reach_rms", "over"))
            arms = arms_for(q, a.extra_sigmas)
            arms.append((arms[0][0] + " [repeat]", arms[0][1], arms[0][2]))
            for label, sigma, csigma in arms:
                key = f"R{q} {label}"
                table_sigma = recipe.window_sigma if sigma is None else sigma
                st = reach_stats(w, BF16_GRID, recipe.window_bits, table_sigma, csigma)
                try:
                    hat, bpp, s, secs = encode_arm(w, q, name, sigma, csigma)
                except Exception as exc:                          # noqa: BLE001
                    torch.cuda.empty_cache()
                    b.log(f"    {key:<34} !! FAILED: {type(exc).__name__}: {exc}")
                    continue
                r = {"bpp": bpp, "sha": s, "tsha": tensor_sha(hat), "secs": secs,
                     "window_sigma": table_sigma, "channel_sigma": csigma,
                     "reach_rms": st["reach_row_rms"], "over": st["rows_over_reach"],
                     **st, **score(w, hat, h=h)}
                res[key] = r
                b.row(key, r, ("bpp", "wt", "h", "reach_rms", "over"))
                del hat
                torch.cuda.empty_cache()
            first, last = res.get(f"R{q} {arms[0][0]}"), res.get(f"R{q} {arms[-1][0]}")
            if first and last:
                res[f"R{q}_control"] = check_repeat_tensor(
                    b, first, last, f"R{q} {arms[0][0]}")
            if first:
                bpps = {round(v["bpp"], 6) for k, v in res.items()
                        if isinstance(v, dict) and k.startswith(f"R{q} ")}
                res[f"R{q}_bytes"] = {"distinct_bpp": sorted(bpps),
                                      "identical": len(bpps) == 1}
                b.log(f"    bytes: {len(bpps)} distinct bpp across the rung's arms "
                      f"-> {'IDENTICAL' if len(bpps) == 1 else '!! DIFFER'}")
                res[f"R{q}_ratios"] = {
                    k: {axis: v[axis] / first[axis] for axis in ("wt", "h")}
                    for k, v in res.items()
                    if isinstance(v, dict) and k.startswith(f"R{q} ") and "wt" in v}
        b.doc["units"][name] = res
        b.save()
        del w
        torch.cuda.empty_cache()

    # Geomeans over the units, per rung, relative to the built arm.
    summary = {}
    for q in a.rungs:
        labels = {k for u in b.doc["units"].values() for k in u
                  if isinstance(u[k], dict) and k.startswith(f"R{q} ")}
        summary[f"R{q}"] = {}
        for label in sorted(labels):
            for axis in ("wt", "h"):
                vals = [u[f"R{q}_ratios"][label][axis]
                        for u in b.doc["units"].values()
                        if f"R{q}_ratios" in u and label in u[f"R{q}_ratios"]]
                if len(vals) == len(b.doc["units"]):
                    summary[f"R{q}"].setdefault(label, {})[axis] = geomean(vals)
                    summary[f"R{q}"][label][f"{axis}_worse_than_built"] = sum(
                        v > 1.0 for v in vals)
    b.doc["summary_vs_built"] = summary
    b.log("\n== geomean over the units, each arm against the built arm "
          "(>1 means the built recipe wins)")
    for rung, arms in summary.items():
        b.log(f"  {rung}")
        for label, v in arms.items():
            b.log(f"    {label:<40} wt {v['wt']:.4f}  h {v['h']:.4f}  "
                  f"(wt worse than built on {v['wt_worse_than_built']} of "
                  f"{len(b.doc['units'])})")
    b.save()
    b.log(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
