#!/usr/bin/env python
"""Put the tessera#60 LDLQ-block evidence in front of the promotion gate.

Nothing here decides anything.  It assembles the record --- the dense
weight-space unit set, the GLM cross-check, and the served pair --- from the
files each leg was actually written to, hands it to
``tessera.control.assert_plane_promotion``, and prints whatever the gate says,
including the refusal.  A refusal IS the output when a leg is missing; that is
the point of running it rather than reading the numbers oneself.

Three things it will not do, because each of them is a way to make a gate
agree with you:

* It does not hardcode a ratio.  ``unit_ratios`` is read out of the sweep's own
  JSON as ``candidate.out / incumbent.out`` per unit, so the numbers the gate
  sees are the numbers the sweep wrote.
* It does not default the ``served_bar``.  The bar is the INCUMBENT's own
  served KL from its own ``kl_compare`` payload, passed in; there is no value
  to fall back to if that file is absent (see ``assert_plane_promotion``).
* It does not invent the GLM leg.  Without ``--glm-candidate``/``--glm-incumbent``
  the record is reported as incomplete and the gate is not called at all,
  rather than called with a flattering 1.0.

The drift control is reported beside the served pair and is not an input to the
gate: the gate asks whether the candidate beat the incumbent, and the control
says whether the two servings that produced those numbers were comparable.  A
reader needs both and the gate only reads one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.control import assert_plane_promotion, promotion_block  # noqa: E402
from tessera.errors import PromotionRefusedError  # noqa: E402


def geomean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def dense_ratios(path: Path, candidate: int, incumbent: int, field: str):
    """Per-unit candidate/incumbent from ``dense4_reach_sweep.py``'s output.

    The arm whose ``block`` equals the asked-for value is the arm, whatever it
    is labelled: the sweep writes its drift controls under names
    (``control FIRST b32``) that also carry ``block: 32``, and picking by label
    would silently read a control as the incumbent.
    """
    doc = json.loads(path.read_text())
    names, ratios, rows = [], [], []
    for unit in doc["units"]:
        arms = unit["arms"]
        def pick(block):
            hits = {k: v for k, v in arms.items() if v.get("block") == block
                    and not k.startswith("control ")}
            if len(hits) != 1:
                raise SystemExit(
                    f"{unit['name']}: expected exactly one non-control arm at "
                    f"block {block}, found {sorted(hits)}")
            return next(iter(hits.values()))
        cand, inc = pick(candidate), pick(incumbent)
        if abs(cand["bpp"] - inc["bpp"]) > 5e-4:
            raise SystemExit(
                f"{unit['name']}: the two arms are not byte-matched "
                f"({cand['bpp']} vs {inc['bpp']} bpp) -- a block is a schedule "
                "and must not move the wire")
        names.append(unit["name"])
        ratios.append(cand[field] / inc[field])
        rows.append((unit["name"], inc[field], cand[field], cand[field] / inc[field],
                     unit.get("drift_same")))
    return names, ratios, rows


def glm_ratio(candidate: Path, incumbent: Path, arm: str, field: str):
    """Geomean of candidate/incumbent over the shared experts of one arm."""
    c = json.loads(candidate.read_text())["experts"]
    i = json.loads(incumbent.read_text())["experts"]
    shared = sorted(set(c) & set(i))
    if not shared:
        raise SystemExit("the two GLM payloads share no expert")
    rows = [(e, i[e][arm][field], c[e][arm][field], c[e][arm][field] / i[e][arm][field])
            for e in shared]
    return geomean([r[3] for r in rows]), rows


def served(path: Path):
    doc = json.loads(path.read_text())
    return doc["all"]["kl_lower_mean"], doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense-sweep", type=Path, required=True)
    ap.add_argument("--candidate-block", type=int, default=4)
    ap.add_argument("--incumbent-block", type=int, default=32)
    ap.add_argument("--field", default="out", help="the held-out error column")
    ap.add_argument("--glm-candidate", type=Path, default=None)
    ap.add_argument("--glm-incumbent", type=Path, default=None)
    ap.add_argument("--glm-arm", default="E2M1x2 TCQ q896 (exporter default) LDLQ1.0 refit-h^1.0")
    ap.add_argument("--kl-candidate", type=Path, default=None, help="kl_compare json for the candidate arm")
    ap.add_argument("--kl-incumbent", type=Path, default=None, help="kl_compare json for the re-run default")
    ap.add_argument("--kl-drift", type=Path, default=None, help="kl_compare json for the SECOND default serve")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    candidate = f"ldlq_block={args.candidate_block}"
    names, ratios, rows = dense_ratios(args.dense_sweep, args.candidate_block,
                                       args.incumbent_block, args.field)
    print(f"== dense weight-space unit set ({args.field}), "
          f"b{args.candidate_block} over b{args.incumbent_block}")
    for name, inc, cand, ratio, drift in rows:
        print(f"  {name:<42} {inc:.6f} -> {cand:.6f}   {ratio:.4f}x"
              f"   drift_control_same={drift}")
    print(f"  geomean {geomean(ratios):.4f}x over {len(ratios)} units, "
          f"{sum(1 for r in ratios if r < 1)} wins")

    record = {
        "schema": "tessera.ldlq_block_promotion_record/1",
        "candidate": candidate,
        "incumbent": f"ldlq_block={args.incumbent_block}",
        "dense": {"sweep": str(args.dense_sweep), "field": args.field,
                  "units": names, "unit_ratios": ratios,
                  "geomean": geomean(ratios)},
    }

    glm = None
    if args.glm_candidate and args.glm_incumbent:
        glm, glm_rows = glm_ratio(args.glm_candidate, args.glm_incumbent,
                                  args.glm_arm, args.field)
        print(f"\n== GLM cross-check, arm {args.glm_arm!r}")
        for e, inc, cand, ratio in glm_rows:
            print(f"  {e:<20} {inc:.6f} -> {cand:.6f}   {ratio:.4f}x")
        print(f"  geomean {glm:.4f}x over {len(glm_rows)} experts")
        record["glm"] = {"candidate": str(args.glm_candidate),
                         "incumbent": str(args.glm_incumbent),
                         "arm": args.glm_arm, "experts": len(glm_rows),
                         "ratios": [r[3] for r in glm_rows], "geomean": glm}

    served_kl = served_bar = None
    if args.kl_candidate:
        served_kl, _ = served(args.kl_candidate)
    if args.kl_incumbent:
        served_bar, _ = served(args.kl_incumbent)
    if args.kl_drift and args.kl_incumbent:
        drift, _ = served(args.kl_drift)
        print(f"\n== served, dense Qwen3-0.6B, KL-vs-BF16 (kl_lower_mean)")
        print(f"  incumbent b{args.incumbent_block} (first)  {served_bar:.6f}")
        print(f"  candidate b{args.candidate_block}          "
              f"{served_kl if served_kl is None else f'{served_kl:.6f}'}")
        print(f"  incumbent b{args.incumbent_block} (last)   {drift:.6f}"
              f"   drift {abs(drift - served_bar):.2e}")
        record["served"] = {"incumbent_first": served_bar, "candidate": served_kl,
                            "incumbent_last": drift,
                            "drift_abs": abs(drift - served_bar)}
    elif served_kl is not None or served_bar is not None:
        record["served"] = {"incumbent_first": served_bar, "candidate": served_kl}

    print("\n== tessera.control.assert_plane_promotion")
    # Two inputs have no honest stand-in, and both would flatter the gate if
    # one were invented: a missing GLM ratio is not 1.0, and a missing bar is
    # not infinity.  ``served_kl`` is different -- ``None`` there is a state
    # the gate has an opinion about ("a screen is not a result"), so it is
    # passed through and the refusal is the gate's, not this script's.
    missing = [n for n, v in (("--glm-candidate/--glm-incumbent", glm),
                              ("--kl-incumbent (the served_bar)", served_bar))
               if v is None]
    if missing:
        record["gate"] = {"called": False, "reason":
                          f"no {' and no '.join(missing)}: that leg has no input, and "
                          "a placeholder would be the flattering default "
                          "assert_plane_promotion exists to refuse"}
        print("  NOT CALLED: " + record["gate"]["reason"])
    else:
        try:
            promotion = assert_plane_promotion(
                candidate=candidate,
                served_arm=candidate if served_kl is not None else None,
                unit_ratios=ratios,
                glm_ratio=glm,
                served_kl=served_kl,
                served_bar=served_bar,
                where=f"tessera#60 {candidate} on the LUT plane",
            )
            record["gate"] = {"called": True, "promoted": True,
                              "block": promotion_block(promotion)}
            print("  PROMOTED: " + record["gate"]["block"]["verdict"]["detail"])
        except PromotionRefusedError as exc:
            record["gate"] = {"called": True, "promoted": False,
                              "refusal": str(exc),
                              "refusal_type": type(exc).__name__}
            print(f"  {type(exc).__name__}: {exc}")

    if args.out:
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
