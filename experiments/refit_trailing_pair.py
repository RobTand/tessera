#!/usr/bin/env python
"""Issue #75's fair pair: the trailing refit's OBJECTIVE swapped, same pass count.

#75's screen took the served default's codes (LDLQ 1.0/32 + four ``h^1.0``
refits) and ran *one more* full-H Gauss-Seidel refit on them, and read that
arm level with -- ahead on five of six units of -- the full-H Gauss-Seidel
alternation.  The issue says so itself: that is **not a matched pair**.  The
swap arm ran five refits against the control's four, so "the trailing refit is
where the full-H objective earns its keep" and "a fifth refit helps" are the
same number.  The pair that separates them is

    T R_h T R_h T R_h T R_H   against   T R_h T R_h T R_h T R_h

-- the last refit's objective swapped at the SAME pass count.  ``encode_unit``
could not express that when #75 was written; ``refit_metric_trailing`` (added
for this issue) is the missing half, so the pair is runnable and this script
runs it.

**Why the pair is matched, and how the run proves it rather than asserting it.**
Passes 1-3 are identical calls in both arms, and pass 4's trellis runs against
the plane pass 3's refit left, BEFORE pass 4's refit.  So every trellis pass is
identical and only the final scale plane can differ: the two arms must come out
with **byte-identical codes** and **byte-identical blob length**, differing in
the scale plane alone.  The run asserts both, and prints each arm's
``refit_diagnostics`` schedule (``metric_ndim`` and ``gauss_seidel`` per refit
call) so the reader can see ``1,1,1,2`` against ``1,1,1,1`` instead of trusting
a flag name.

**Arms** (LDLQ 1.0/32, ``scale_refit=4``, E2M1x2 ``q256=896`` -- the 4-bit
route's TCQ cap wire, whose plane is LUT16):

* ``A``      -- the served default, ``refit_metric=h^1.0``.  Run FIRST and
  again LAST in one process: an arm-to-arm gap below the control's own spread
  is not a result, and the two must be the same reconstruction.
* ``B-Jac``  -- ``A``'s schedule with the trailing refit under the full H,
  stepped in parallel.  The single-lever half of the pair.
* ``B-GS``   -- the same with the trailing refit swept Gauss-Seidel.  This is
  the arm #75's screen approximates; it changes objective AND optimiser on the
  trailing pass, which is why ``B-Jac`` is carried beside it.
* ``C-GS``   -- ``refit_metric=H`` with the sweep on EVERY pass: #35's promoted
  alternation, the incumbent candidate the pair is measured against.
* ``C-Jac``  -- the same alternation stepped in parallel.
* ``B-GS+CL``/``C-GS+CL`` -- those two with #50's **coupled landing**
  (``refit_coupled_landing``), on the trailing pass and on every pass
  respectively.  #50's own receipt could only replay the landing at frozen
  codes, so these are the arms that show what a re-assignment does to the
  codes the NEXT trellis pass sees.

**Landings, and which numbers a gate may read.**  Since #85
``tessera.control.assert_plane_promotion`` refuses a per-plane promotion whose
ratios were taken at any landing but the wire, because the arms *reorder* when
the landing is removed.  Every arm here is therefore run at
``lut_landing("table")`` -- the wire, ``LUT_LANDING_WIRE``, the state every
encode runs in -- and those are the only numbers offered to the gate.
``--landings table none`` additionally re-runs the pair with the sixteen-entry
landing removed entirely (``none``, the continuous per-block optimum, **not a
plane and not a wire**) purely to report whether this pair reorders the way
#35's did.  Off-wire rows are labelled ``[NOT A WIRE]`` in the log and carry
``serialisable: false`` in the JSON.

*The coupled landing is on master and this script runs it.*  That was not
true when this docstring was first written -- ``refit_coupled_landing`` then
lived only on the session-kill snapshot branch ``claude/ts-50-lut-landing`` --
and the sentence outlived the rescue: the mechanism landed on master with
#105, ``B_GS_CL``/``C_GS_CL`` below pass the flag, and the run reproduces
#75's screen conditions rather than only approximating them.  Every arm is
still scored at the landing that ships (``table``), including the coupled
ones: the coupled landing re-assigns blocks among the SAME sixteen E4M3 table
entries, so it moves no byte of the wire and is not an off-wire read.

**Scored** ``out`` = held-out activation-space relative error (the deciding
column, and a screen -- nothing here is served); ``plain`` = weight-space
relative error; ``hfit`` = ``sqrt(E H E^T / W H W^T)`` on the fit rows, the
quadratic every refit here is monotone in.  fp32 throughout.

Two populations, one process each, the same axis:

    # six dense Qwen3-0.6B units -- #75's own set
    PYTHONPATH=src python experiments/refit_trailing_pair.py --population qwen \
        --out experiments/results/refit_trailing_pair_qwen.json

    # the six GLM-5.3-Flash experts -- the population the h^1.0 default was
    # chosen on, and the gate any move of it has to clear
    PYTHONPATH=src:experiments:/home/rob/prismaquant \
      python experiments/refit_trailing_pair.py --population glm \
        --out experiments/results/refit_trailing_pair_glm.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refit_trailing_screen import (                           # noqa: E402
    assert_arm_proofs, assert_screen_receipt)
from tessera.alphabet import SERIALISABLE_GRIDS               # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian  # noqa: E402
from tessera.encode import lut_landing, refit_diagnostics     # noqa: E402
from tessera.encode import LUT_LANDING_WIRE                   # noqa: E402
from tessera.errors import PromotionRefusedError              # noqa: E402
from tessera.export import (                                  # noqa: E402
    DEFAULT_CODE, encode_linear_planes, wire_recipe)
from tessera.manifest import ScalePlaneKind                   # noqa: E402
from tessera.stock import materialize_stock, stock_dequant    # noqa: E402

# The arm names, fixed here so the report and the gate read the same strings.
A_FIRST = "A drift control FIRST [refit h^1.0 x4]"
A_LAST = "A drift control LAST [refit h^1.0 x4]"
B_JAC = "B-Jac  T R_h T R_h T R_h T R_H          (trailing full-H, Jacobi)"
B_GS = "B-GS   T R_h T R_h T R_h T R_H(GS)      (trailing full-H, sweep)"
C_JAC = "C-Jac  T R_H T R_H T R_H T R_H          (full-H every pass, Jacobi)"
C_GS = "C-GS   T R_H T R_H T R_H T R_H(GS)      (full-H every pass, sweep)"
# #50's coupled landing, the half its own receipt left at TBD: the oracle
# replayed the trailing refit at frozen codes, so it could not see what a
# re-assignment does to the codes the NEXT trellis pass sees.  These two arms
# are that, in the encoder, at the wire.
B_GS_CL = "B-GS+CL T R_h T R_h T R_h T R_H(GS,CL)   (trailing full-H, sweep, coupled landing)"
C_GS_CL = "C-GS+CL T R_H T R_H T R_H T R_H(GS,CL)   (full-H every pass, sweep, coupled landing)"


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def geo(units: dict, arm: str, field: str) -> float:
    vals = [units[u][arm][field] for u in units]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


# --------------------------------------------------------------------------
# The two populations.  Each yields ``(name, W, H, X_eval)`` on the device,
# with ``H`` built from rows disjoint from ``X_eval`` in both cases.
# --------------------------------------------------------------------------

def qwen_units(a, dev):
    """#75's own six dense Qwen3-0.6B Linears, exactly as
    ``lut_landing_ceiling.py`` loads them: the captured full Hessian and the
    held-out eval slice of ``capture_h_full.py``."""
    from safetensors import safe_open

    payload = torch.load(a.h, map_location="cpu", weights_only=False)
    acts = torch.load(a.acts, map_location="cpu", weights_only=False)
    Hall, prov = payload["H"], payload["provenance"]
    names = a.units or sorted(acts["x"])
    yield ("provenance", prov)
    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        for name in names:
            if name not in Hall:
                raise SystemExit(f"no Hessian for {name}")
            W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
            yield (name, W, Hall[name].to(dev, torch.float32),
                   acts["x"][name].to(dev, torch.float32))


def glm_units(a, dev):
    """The six GLM-5.3-Flash expert tensors the ``h^1.0`` default was chosen
    on, built the way ``tessera_window_wire.py`` builds them: expert 0's
    ``gate_proj``/``up_proj`` on layers 5/20/42, the Hessian from the fit rows
    and the score on the held-out tail, so no arm is graded on its fit rows."""
    from safetensors import safe_open

    from tessera8_targets import ACT, SRC

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    yield ("provenance", {"source": SRC, "activations": ACT,
                          "eval_rows": a.eval_rows,
                          "note": "H from the fit rows, score on the held-out tail"})
    for layer in a.layers:
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit = xa[:n_fit].contiguous().to(dev)
        x_ev = xa[n_fit:].contiguous().to(dev)
        H = (x_fit.double().T @ x_fit.double()).float() / x_fit.shape[0]
        del x_fit, xa, blob
        for proj in a.projs:
            wname = (f"model.language_model.layers.{layer}"
                     f".mlp.experts.0.{proj}.weight")
            with safe_open(f"{SRC}/{index[wname]}", framework="pt") as f:
                W = f.get_tensor(wname).contiguous().to(dev).float()
            yield (f"L{layer}.{proj}", W, H, x_ev)
        del H, x_ev
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=("qwen", "glm"), default="qwen")
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--refit", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--landings", nargs="+", default=[LUT_LANDING_WIRE],
                    choices=("table", "grid", "none"),
                    help="'table' is the wire and the only landing a promotion "
                         "gate may read (tessera#85); the others are ceiling "
                         "reads and are labelled [NOT A WIRE]")
    # qwen population
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--h", default="/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--acts", default="/mnt/shared/tessera-runs/ldlq/x_eval_qwen06b.pt")
    ap.add_argument("--units", nargs="*", default=None)
    # glm population
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--out", default="experiments/results/refit_trailing_pair.json")
    a = ap.parse_args()

    grid = grid_by_name(a.grid)
    recipe = wire_recipe(grid, a.q256)
    if recipe.scale_plane is not ScalePlaneKind.LUT:
        raise SystemExit(
            f"{a.grid} q256={a.q256} is a {recipe.scale_plane.name} plane; the "
            "trailing schedule this measures is read by the LUT plane's "
            "per-block refit and by CHANNEL's row refit, but #75's screen and "
            "#35's promotion are both on LUT")
    dev = "cuda"
    out: dict = {"args": vars(a), "units": {}}
    lines: list[str] = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"wire: {grid.name} q256={a.q256} -> body {recipe.body.name} plane "
        f"{recipe.scale_plane.name} span {recipe.span} L={recipe.window_bits}")
    log(f"schedule: scale_refit={a.refit}, LDLQ sigma={a.sigma} block={a.block}, "
        f"landings={a.landings} (wire = {LUT_LANDING_WIRE!r})")

    src = (qwen_units if a.population == "qwen" else glm_units)(a, dev)
    head = next(src)
    out["provenance"] = head[1]
    log(f"provenance: {json.dumps(out['provenance'])[:400]}")

    for name, W, H, X in src:
        Y = X @ W.T
        h = H.diagonal().clone()
        hn = h / h.mean()
        hmetric = hn.pow(a.alpha)
        den_w = W.norm()
        den_h = float(((W * W).sum(0) * hn).sum())
        den_hf = float(((W @ H) * W).sum())
        L = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
        res: dict = {}
        seen: dict = {}
        log(f"\n== {name} {tuple(W.shape)}  eval rows {X.shape[0]}")
        log(f"    {'arm':<52} {'out':>9} {'plain':>9} {'hwt':>9} {'hfit':>9} "
            f"{'bytes':>9} {'s':>6}")

        def run(arm, landing, dup_ok=False, **kw):
            t0 = time.time()
            with lut_landing(landing) as sink, refit_diagnostics() as diag:
                exported, unit, forests = encode_linear_planes(
                    W, grid=grid, q256=a.q256, name=str(name), verify=False,
                    scale_refit=a.refit, ldl=L, ldl_block=a.block, **kw)
            secs = time.time() - t0
            What = sink["work_reconstruction"].to(dev).float()
            r = {"landing": landing, "serialisable": bool(sink["serialisable"]),
                 "bytes": len(exported.blob)}
            if landing == LUT_LANDING_WIRE:
                # The wire is scored the way the wire is scored, and the two
                # must agree: that identity is what licenses reading any
                # off-wire arm off the sink at all.
                wire = stock_dequant(
                    materialize_stock(unit, forests, DEFAULT_CODE)).to(dev).float()
                r["sink_vs_wire_bit_identical"] = bool(torch.equal(wire, What))
                r["sink_vs_wire_rel"] = float((wire - What).norm() / den_w)
                r["codes_sha256"] = sha(unit.codes)
            E = What - W
            r.update({
                "out": float((X @ E.T).norm() / Y.norm()),
                "plain": float(E.norm() / den_w),
                "hweighted": math.sqrt(float(((E * E).sum(0) * hn).sum()) / den_h),
                "hfit": math.sqrt(float(((E @ H) * E).sum()) / den_hf),
                "secs": secs,
            })
            # The schedule, from the refits that actually ran: one record per
            # metric-aware refit call, in order.  ``1,1,1,2`` against
            # ``1,1,1,1`` is the pair; a flag name is not.
            r["schedule"] = [(int(d["metric_ndim"]), bool(d["gauss_seidel"]))
                             for d in diag]
            r["refit"] = [dict(d) for d in diag]
            key = sha(What)
            if key in seen and not dup_ok:
                log(f"    !! IDENTICAL RECONSTRUCTION: {arm!r} == {seen[key]!r} "
                    f"-- that lever did nothing on this unit")
            seen.setdefault(key, arm)
            r["sha256"] = key
            res[arm] = r
            log(f"    {arm:<52} {r['out']:9.5f} {r['plain']:9.5f} "
                f"{r['hweighted']:9.5f} {r['hfit']:9.5f} {r['bytes']:9d} "
                f"{secs:6.1f}"
                + ("" if landing == LUT_LANDING_WIRE else "   [NOT A WIRE]")
                + "  sched " + "".join(
                    f"{n}{'g' if g else ''}" for n, g in r["schedule"]))
            return r

        for landing in a.landings:
            tag = "" if landing == LUT_LANDING_WIRE else f" | landing={landing}"
            run(A_FIRST + tag, landing, dup_ok=True, refit_metric=hmetric)
            run(B_JAC + tag, landing,
                refit_metric=hmetric, refit_metric_trailing=H)
            run(B_GS + tag, landing,
                refit_metric=hmetric, refit_metric_trailing=H,
                refit_gauss_seidel=True)
            run(C_JAC + tag, landing, refit_metric=H)
            run(C_GS + tag, landing, refit_metric=H, refit_gauss_seidel=True)
            run(B_GS_CL + tag, landing,
                refit_metric=hmetric, refit_metric_trailing=H,
                refit_gauss_seidel=True, refit_coupled_landing="trailing")
            run(C_GS_CL + tag, landing, refit_metric=H,
                refit_gauss_seidel=True, refit_coupled_landing="every")
            last = run(A_LAST + tag, landing, dup_ok=True, refit_metric=hmetric)
            first = res[A_FIRST + tag]
            same = first["sha256"] == last["sha256"]
            res[A_FIRST + tag]["drift_control_identical"] = bool(same)
            log(f"    -- drift control [{landing}]: reconstruction "
                f"{'IDENTICAL' if same else 'DIFFERS'}  "
                f"out {last['out'] / first['out'] - 1:+.4%}  "
                f"hfit {last['hfit'] / first['hfit'] - 1:+.4%}")

            # The matched-pair proof.  Inner passes are identical calls, so
            # every trellis pass is identical and only the last scale plane
            # may move: same codes, same blob length, different bytes.
            if landing == LUT_LANDING_WIRE:
                base = res[A_FIRST + tag]
                num = ("before", "stepped", "continuous", "landed", "reverted")
                for arm in (B_JAC + tag, B_GS + tag):
                    m = res[arm]
                    # ``inner_refits_identical`` is the proof, and it compares
                    # NUMBERS, not flags.  ``refit_gauss_seidel=True`` is handed
                    # to the inner passes as well, whose metric is 1-D; there
                    # the sweep is provably the parallel step (the encoder
                    # refuses the flag outright when NO leg couples, for that
                    # reason), so the records must be numerically equal even
                    # though the recorded ``gauss_seidel`` flag differs.
                    m["matched_pair"] = {
                        "codes_identical": m["codes_sha256"] == base["codes_sha256"],
                        "bytes_equal": m["bytes"] == base["bytes"],
                        "plane_moved": m["sha256"] != base["sha256"],
                        "inner_objectives_equal":
                            [n for n, _ in m["schedule"][:-1]]
                            == [n for n, _ in base["schedule"][:-1]],
                        "inner_refits_identical": all(
                            m["refit"][i][k] == base["refit"][i][k]
                            for i in range(len(base["refit"]) - 1) for k in num),
                    }
                    log(f"    -- matched pair {arm.split()[0]:<6}: "
                        + "  ".join(f"{k}={v}" for k, v
                                    in m["matched_pair"].items()))

        out["units"][str(name)] = res
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        del W, H, X, Y, L
        torch.cuda.empty_cache()

    names = list(out["units"])
    arms = list(out["units"][names[0]])
    log(f"\n== geomeans over {len(names)} units  [{a.population}]")
    log(f"    {'arm':<52} {'out':>9} {'hfit':>9} {'out/A':>9} {'hfit/A':>9}")
    gc = geo(out["units"], A_FIRST, "out")
    hc = geo(out["units"], A_FIRST, "hfit")
    for arm in arms:
        g, hh = geo(out["units"], arm, "out"), geo(out["units"], arm, "hfit")
        log(f"    {arm:<52} {g:9.5f} {hh:9.5f} {g / gc:9.4f} {hh / hc:9.4f}"
            + ("" if "landing=" not in arm else "   [NOT A WIRE]"))
    out["geomeans"] = {arm: {f: geo(out["units"], arm, f)
                             for f in ("out", "plain", "hweighted", "hfit")}
                       for arm in arms}
    out["ratios_vs_control"] = {
        arm: {f: geo(out["units"], arm, f) / geo(out["units"], A_FIRST, f)
              for f in ("out", "hfit")} for arm in arms}
    # The per-unit ratios a promotion gate reads, wire landing only.
    out["unit_ratios_wire"] = {
        arm: [out["units"][u][arm]["out"] / out["units"][u][A_FIRST]["out"]
              for u in names]
        for arm in arms if "landing=" not in arm}
    out["log"] = lines
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")

    # The proofs above are recorded and printed; this is where they are
    # BELIEVED (tessera#250).  The same module the promotion gate reads them
    # with reads them here, so a screen whose drift control DIFFERS or whose
    # trailing arm moved a code leaves this process with a nonzero status
    # instead of a JSON that looks like every other one.  The receipt is
    # written first on purpose: a refusal must not take the evidence with it.
    try:
        for arm, reasons in assert_screen_receipt(
                out, name=a.out, where="tessera#75 the trailing-refit pair").items():
            assert_arm_proofs(arm, reasons,
                              where="tessera#75 the trailing-refit pair")
    except PromotionRefusedError as exc:
        raise SystemExit(f"REFUSED: {exc}")


if __name__ == "__main__":
    main()
