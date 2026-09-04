#!/usr/bin/env python
"""Read #75's fair pair through the promotion gate, and print what it says.

``experiments/refit_trailing_pair.py`` produces two JSONs -- the six dense
Qwen3-0.6B units of #75's own table, and the six GLM-5.3-Flash experts the
``h^1.0`` default was chosen on.  This turns them into the five arguments
``tessera.control.assert_plane_promotion`` reads and reports its verdict
**verbatim**, so no one has to decide from a table whether a screen promotes.

The gate's five legs, and what this run can and cannot hand it:

* ``glm_ratio`` -- the GLM six-expert ``out`` geomean against the incumbent.
  Measured here; the bar is ``GLM_GATE`` = 1.00 and no screen overrules it.
* the screen's geomean, and a strict majority of its own units.  Measured here.
* ``served_arm``/``served_kl`` -- **not measured here.**  Nothing in #75 is
  served, so these are ``None`` and the gate refuses on them.  That refusal is
  the correct verdict for a screen and is reported, not worked around.
* ``landing`` -- ``table``, the wire, which is the only landing the run offers
  the gate (tessera#85).  Passing an off-wire ratio here is refused, and this
  script would rather not be able to than be able to.

``--served-bar`` defaults to the LUT plane's incumbent served KL, 0.5310 --
``h^1.0``, the arm every candidate here is a ratio against.  It is quoted from
``assert_plane_promotion``'s own docstring, which records that the 2026-09-02
receipt's 0.640 is the *stock* wire's number and the wrong bar for anything
since.  It is an argument and not a constant because it moves with the
incumbent.

    PYTHONPATH=src python experiments/refit_trailing_pair_gate.py \
        --qwen experiments/results/refit_trailing_pair_qwen.json \
        --glm  experiments/results/refit_trailing_pair_glm.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.control import GLM_GATE, assert_plane_promotion, promotion_block  # noqa: E402
from tessera.encode import LUT_LANDING_WIRE                                    # noqa: E402
from tessera.errors import PromotionRefusedError, TesseraError                 # noqa: E402


def geo(xs) -> float:
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="experiments/results/refit_trailing_pair_qwen.json")
    ap.add_argument("--glm", default="experiments/results/refit_trailing_pair_glm.json")
    ap.add_argument("--field", default="out",
                    help="the deciding column; 'out' is held-out activation space")
    ap.add_argument("--served-bar", type=float, default=0.5310,
                    help="the INCUMBENT's served KL at matched bytes (h^1.0 on "
                         "the LUT plane), not a constant of the gate")
    ap.add_argument("--out", default="experiments/results/refit_trailing_pair_gate.json")
    a = ap.parse_args()

    qwen = json.load(open(a.qwen))
    glm = json.load(open(a.glm))
    control = next(k for k in qwen["units"][next(iter(qwen["units"]))]
                   if k.startswith("A drift control FIRST"))
    arms = [k for k in qwen["units"][next(iter(qwen["units"]))]
            if not k.startswith("A drift control") and "landing=" not in k]

    lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"screen   : {a.qwen}")
    log(f"crosscheck: {a.glm}")
    log(f"field    : {a.field}   landing offered: {LUT_LANDING_WIRE!r} (the wire)")
    log(f"GLM gate : {GLM_GATE:.4g}x   served bar: {a.served_bar:.4g} "
        f"(the incumbent h^1.0's served KL)")
    log()

    verdicts = {}
    for arm in arms:
        qu = qwen["units"]
        gu = glm["units"]
        ratios = [qu[u][arm][a.field] / qu[u][control][a.field] for u in qu]
        glm_ratio = (geo([gu[u][arm][a.field] for u in gu])
                     / geo([gu[u][control][a.field] for u in gu]))
        wins = sum(1 for r in ratios if r < 1)
        log(f"== {arm}")
        log(f"   Qwen six-unit {a.field} geomean {geo(ratios):.4f}x, "
            f"wins {wins}/{len(ratios)}   per-unit "
            + " ".join(f"{r:.4f}" for r in ratios))
        log(f"   GLM six-expert {a.field} geomean {glm_ratio:.4f}x "
            f"against the {GLM_GATE:.4g}x gate: "
            f"{'CLEARS' if glm_ratio <= GLM_GATE else 'FAILS'}")
        rec = {"unit_ratios": ratios, "geomean": geo(ratios), "wins": wins,
               "glm_ratio": glm_ratio}
        try:
            promotion = assert_plane_promotion(
                candidate=arm, served_arm=None, unit_ratios=ratios,
                glm_ratio=glm_ratio, served_kl=None, served_bar=a.served_bar,
                landing=LUT_LANDING_WIRE,
                where="tessera#75 the trailing-refit objective")
            rec["verdict"] = promotion_block(promotion)["verdict"]
            log(f"   assert_plane_promotion: PROMOTED -- "
                f"{rec['verdict']['detail']}")
        except (PromotionRefusedError, TesseraError) as e:
            rec["refused"] = f"{type(e).__name__}: {e}"
            log(f"   assert_plane_promotion: {type(e).__name__}: {e}")
        verdicts[arm] = rec
        log()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"args": vars(a), "control": control, "arms": verdicts, "log": lines},
        indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
