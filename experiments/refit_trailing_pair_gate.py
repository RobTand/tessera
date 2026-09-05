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
* ``served_arm``/``served_kl`` -- **not produced here**, but no longer
  necessarily absent.  With no ``--served`` argument they are ``None`` and the
  gate refuses on them, which is the correct verdict for a screen and is
  reported rather than worked around.  ``--served-arm`` plus a
  ``--served-kl-json`` names ONE arm and reads its KL out of a
  ``prismaquant.kl_compare/2`` receipt, so the number a gate reads is the
  number a serve wrote.  Naming an arm whose KL was measured on some other
  bytes is the failure ``assert_plane_promotion``'s fourth leg exists to
  catch, and it stays the caller's assertion here exactly as it is there --
  which is why the receipt paths land in the output JSON.
* ``landing`` -- ``table``, the wire, which is the only landing the run offers
  the gate (tessera#85).  Passing an off-wire ratio here is refused, and this
  script would rather not be able to than be able to.

**Before any of that: the screen's own recorded proofs** (tessera#250).  The
producer writes, per unit, the evidence that can invalidate its experiment --
the drift control's first/last reconstruction identity, the sink-versus-wire
agreement, and the trailing arms' matched-pair legs -- and this gate read none
of it, so a document whose control DIFFERS or whose trailing arm changed its
packed codes could still print PROMOTED on favourable ratios.
``experiments/refit_trailing_screen.py`` owns that reading for the producer
and for this gate both.  A failed control or a mislabelled landing refuses the
whole document, in **both** populations, before a ratio is computed; a failed
matched-pair leg refuses the arm that claims the pair.  ``plane_moved=false``
is not in that set: an arm whose lever reached nothing is an ineffective arm,
not a broken comparison.

``--served-bar`` defaults to the LUT plane's incumbent served KL, 0.5310 --
``h^1.0``, the arm every candidate here is a ratio against.  It is quoted from
``assert_plane_promotion``'s own docstring, which records that the 2026-09-02
receipt's 0.640 is the *stock* wire's number and the wrong bar for anything
since.  It is an argument and not a constant because it moves with the
incumbent.  ``--served-bar-json`` is the better spelling: point it at the
incumbent's own ``kl_compare`` receipt and the bar is read rather than typed.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refit_trailing_screen import assert_arm_proofs, assert_screen_receipt     # noqa: E402
from tessera.control import GLM_GATE, assert_plane_promotion, promotion_block  # noqa: E402
from tessera.encode import LUT_LANDING_WIRE                                    # noqa: E402
from tessera.errors import PromotionRefusedError, TesseraError                 # noqa: E402


#: The one place this run names itself, so the promotion refusals and the
#: screen-proof refusals below read as one gate rather than two.
WHERE = "tessera#75 the trailing-refit objective"


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
    ap.add_argument("--served-bar-json", default=None,
                    help="read --served-bar out of the incumbent's own "
                         "prismaquant.kl_compare/2 receipt instead of typing it")
    ap.add_argument("--served-arm", default=None,
                    help="the arm the served KL was measured on, as a unique "
                         "substring of one arm's name.  Without it every arm is "
                         "read with served_arm=None, which the gate refuses -- "
                         "the verdict a screen earns")
    ap.add_argument("--served-kl-json", default=None,
                    help="the served arm's prismaquant.kl_compare/2 receipt; its "
                         "all/kl_lower_mean is the served KL")
    ap.add_argument("--served-kl", type=float, default=None,
                    help="the served KL directly, when there is no receipt to read")
    ap.add_argument("--out", default="experiments/results/refit_trailing_pair_gate.json")
    a = ap.parse_args()

    qwen = json.load(open(a.qwen))
    glm = json.load(open(a.glm))

    # The screen's own proofs, BEFORE a single ratio is computed (tessera#250).
    # The producer records what can invalidate its experiment -- the drift
    # control, the sink-versus-wire identity, the matched-pair legs -- and
    # until now nothing read them, so a screen whose control DIFFERS could
    # still print PROMOTED on favourable numbers.  A failed control or a
    # mislabelled landing refuses the whole document here; a failed
    # matched-pair leg refuses the arm that claims the pair, below.
    try:
        proof_failures = {}
        for name, doc in ((a.qwen, qwen), (a.glm, glm)):
            for arm, reasons in assert_screen_receipt(
                    doc, name=name, where=WHERE).items():
                proof_failures.setdefault(arm, ())
                proof_failures[arm] += tuple(reasons)
    except PromotionRefusedError as exc:
        raise SystemExit(f"REFUSED: {exc}")

    def kl_of(path: str) -> float:
        """The ALL-positions lower bound out of a kl_compare receipt.

        Refuses any other schema by name: the field means different things in
        the legacy floor-substituted readout, and a gate that reads whichever
        number is present is the confound it exists to prevent.
        """
        payload = json.load(open(path))
        schema = payload.get("schema")
        if schema != "prismaquant.kl_compare/2":
            raise SystemExit(f"{path}: schema {schema!r}, not prismaquant.kl_compare/2")
        return float(payload["all"]["kl_lower_mean"])

    if a.served_bar_json:
        a.served_bar = kl_of(a.served_bar_json)
    if a.served_kl_json:
        if a.served_kl is not None:
            raise SystemExit("give --served-kl or --served-kl-json, not both")
        a.served_kl = kl_of(a.served_kl_json)
    if (a.served_arm is None) != (a.served_kl is None):
        raise SystemExit(
            "a served KL and the arm it measures travel together: one without "
            "the other is exactly the mislabelling the fourth gate leg refuses")
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
        f"(the incumbent h^1.0's served KL"
        + (f", read from {a.served_bar_json}" if a.served_bar_json else "") + ")")
    log()

    # One arm may carry a served number, and it is matched by name so a typo
    # cannot silently hand the gate the wrong arm's KL.
    served_key = None
    if a.served_arm is not None:
        hits = [k for k in arms if a.served_arm in k]
        if len(hits) != 1:
            raise SystemExit(
                f"--served-arm {a.served_arm!r} matches {len(hits)} arms "
                f"{hits!r}; it must name exactly one")
        served_key = hits[0]
        log(f"served   : {served_key}")
        log(f"           KL {a.served_kl:.6g}"
            + (f" from {a.served_kl_json}" if a.served_kl_json else ""))
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
               "glm_ratio": glm_ratio,
               "served_kl": a.served_kl if arm == served_key else None,
               "served_kl_receipt": a.served_kl_json if arm == served_key else None,
               "screen_proof_failures": list(proof_failures.get(arm, ()))}
        try:
            assert_arm_proofs(arm, proof_failures.get(arm, ()), where=WHERE)
            served = arm if arm == served_key else None
            promotion = assert_plane_promotion(
                candidate=arm, served_arm=served, unit_ratios=ratios,
                glm_ratio=glm_ratio,
                served_kl=a.served_kl if served else None,
                served_bar=a.served_bar,
                landing=LUT_LANDING_WIRE,
                where=WHERE)
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
