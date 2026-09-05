#!/usr/bin/env python
"""The byte-matched uniform arm, built and judged (tessera#3).

A candidate on Tessera's rate axis makes a claim no other check in the pipeline
tests: that *choosing* rungs beats spending the same bytes at one rung.  On
2026-09-02 that claim was false by 2.00x served while the bytes were exact to
the unit (196/196), the census was 112/112 on the declared route, and the
surrogate scored the losing moves a 1.30x win
(``docs/measurements/tessera-allocated-served-2026-09-02.md``).  Two serves
caught it and nothing cheaper did.

Two subcommands, one per half of the gate:

``plan``
    Given a candidate ``--plan-json`` and the model's shapes, write the
    byte-matched uniform control plan the exporter can build, and a report
    naming the rung, the slack and which arm is the fatter one.  The byte match
    is **asserted** before the plan is written: a control that misses the
    candidate's bytes by more than a control may is refused here rather than
    discovered after two serves.

``verify``
    Given the two *exported* checkpoints, assert the same match on the bytes
    that actually shipped -- each manifest's ``wire_bytes``, the quantity
    ``check_wire_against_plan.py`` compares against the allocator's charge --
    and, with both KLs, print and write the verdict the shipcard carries:
    *this candidate beat / did not beat its uniform control by X at matched
    bytes*.

Neither subcommand serves anything.  The price of this gate is a second export
and a second serve, and that price is the argument of #3: the alternative was
shipping an allocation 2x worse than the thing it replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.control import (  # noqa: E402
    DEFAULT_MAX_RELATIVE_SLACK,
    assert_byte_matched,
    bits_from_manifest,
    control_block,
    require_kl,
    uniform_control,
    units_from_plan,
)
from tessera.errors import ControlNotByteMatchedError, TesseraError  # noqa: E402


def read_shapes(args) -> dict:
    """``{tensor: (rows, columns)}`` from a shapes JSON or from the model itself."""
    if args.shapes_json is not None:
        raw = json.loads(args.shapes_json.read_text())
        return {name: tuple(int(v) for v in shape) for name, shape in raw.items()}
    if args.model is None:
        raise SystemExit("pass --model (the source checkpoint) or --shapes-json")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plan_from_layer_config import body_weights

    return body_weights(args.model)


def cmd_plan(args) -> int:
    plan = json.loads(args.plan.read_text())
    shapes = read_shapes(args)
    units = units_from_plan(plan, shapes)
    try:
        control = uniform_control(
            units, grid=args.grid, rule=args.rule,
            max_relative_slack=Fraction(args.max_relative_slack).limit_denominator(10**9),
        )
    except ControlNotByteMatchedError as exc:
        loose = uniform_control(
            units, grid=args.grid, rule=args.rule, assert_match=False,
            max_relative_slack=Fraction(args.max_relative_slack).limit_denominator(10**9),
        )
        print(f"REFUSED: {exc}")
        print(f"  nearest rung {loose.grid} R{loose.q256}: "
              f"{int(loose.match.control_bits)} bits against "
              f"{int(loose.match.candidate_bits)}; the axis's local step here is "
              f"{loose.bracket['quantum_bits']} bits")
        return 2

    match = control.match
    print(f"candidate  {int(match.candidate_bits)} bits = {float(match.candidate_bpp):.9f} bpp "
          f"over {match.varying_params} quantized params")
    print(f"control    {control.grid} R{control.q256}  {int(match.control_bits)} bits = "
          f"{float(match.control_bpp):.9f} bpp   [{control.rule}]")
    print(f"slack      {int(match.slack_bits):+d} bits "
          f"({float(match.relative_slack) * 1e6:.1f} ppm; the {match.fatter_arm} arm is fatter)")
    print(f"searched   {control.searched[0]}..{control.searched[1]}, "
          f"{control.legal_rungs} legal rungs; bracket {control.bracket}")
    print(f"units      {len(control.tessera_units)} Tessera, "
          f"{len(control.units) - len(control.tessera_units)} BF16 carried through unchanged")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(control.plan, indent=2, sort_keys=True))
        print(f"  -> {args.out}  (the control's --plan-json)")
    block = control_block(control, candidate_label=args.candidate_label)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(block, indent=2))
        print(f"  -> {args.report}  (the control block, unserved)")
    return 0


def cmd_verify(args) -> int:
    candidate_bits, candidate_units = bits_from_manifest(args.candidate)
    control_bits, control_units = bits_from_manifest(args.control)
    shared = sorted(set(candidate_units) & set(control_units))
    only_candidate = sorted(set(candidate_units) - set(control_units))
    only_control = sorted(set(control_units) - set(candidate_units))
    print(f"candidate  {args.candidate}: {len(candidate_units)} tensors, "
          f"{int(candidate_bits)} bits")
    print(f"control    {args.control}: {len(control_units)} tensors, "
          f"{int(control_bits)} bits")
    if only_candidate or only_control:
        print(f"  TENSOR SETS DIFFER: {len(only_candidate)} only in the candidate, "
              f"{len(only_control)} only in the control; first few "
              f"{only_candidate[:3]} / {only_control[:3]}")
        return 2
    params = args.params
    if params is None:
        print("  (no --params given; bpp figures are omitted, the match is on bits)")
        params = 1
    try:
        match = assert_byte_matched(
            candidate_bits, control_bits, params,
            max_relative_slack=Fraction(args.max_relative_slack).limit_denominator(10**9),
            require_no_larger=args.require_no_larger,
            where=f"{args.control} against {args.candidate}",
        )
    except ControlNotByteMatchedError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"slack      {int(match.slack_bits):+d} bits "
          f"({float(match.relative_slack) * 1e6:.1f} ppm; the {match.fatter_arm} arm is fatter) "
          f"-- BYTE MATCHED over {len(shared)} tensors")

    if args.candidate_kl is None or args.control_kl is None:
        print("VERDICT: not measured -- the bytes match; neither arm's KL was given")
        return 0
    # ``type=float`` accepts "-1", "nan" and "inf" by name, and a negative
    # "KL" sorts below every control's: this printed BEAT and wrote
    # beat_control=true.  The domain is the one ``control_block`` holds its
    # own two KLs to (tessera#225).
    try:
        candidate_kl = require_kl(args.candidate_kl, field="candidate_kl",
                                  where="--candidate-kl")
        control_kl = require_kl(args.control_kl, field="control_kl",
                                where="--control-kl")
    except TesseraError as exc:
        print(f"REFUSED: {exc}")
        return 2
    ratio = candidate_kl / control_kl if control_kl else float("inf")
    beat = candidate_kl < control_kl
    print(f"VERDICT: {args.candidate_label} {candidate_kl:.6g} against the uniform "
          f"control {control_kl:.6g} ({ratio:.4g}x) -- "
          f"{'BEAT' if beat else 'DID NOT BEAT'} its byte-matched control")
    if args.report is not None:
        match_json = match.to_json()
        if args.params is None:
            # The denominator was a placeholder, and the terminal has already
            # said the bpp figures are omitted.  A receipt may not carry what
            # the operator was told was not measured: at params=1 the
            # candidate's whole bit total was written as its bpp.
            for key in ("varying_params", "candidate_bpp", "control_bpp"):
                match_json[key] = None
        report = {
            "schema": "tessera.uniform_control.v1",
            "candidate_label": args.candidate_label,
            "candidate_checkpoint": str(args.candidate),
            "control_checkpoint": str(args.control),
            "match": match_json,
            "verdict": {
                "metric": args.metric,
                "measured": True,
                "candidate": candidate_kl,
                "control": control_kl,
                "candidate_over_control": ratio,
                "beat_control": beat,
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"  -> {args.report}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="build the control plan for a candidate --plan-json")
    plan.add_argument("plan", type=Path, help="the candidate's exporter --plan-json")
    plan.add_argument("--model", type=Path, default=None,
                      help="the source checkpoint, read for tensor shapes")
    plan.add_argument("--shapes-json", type=Path, default=None,
                      help='{"tensor": [rows, columns]} instead of reading the model')
    plan.add_argument("--out", type=Path, default=None,
                      help="write the control's --plan-json here")
    plan.add_argument("--report", type=Path, default=None,
                      help="write the (unserved) control block here")
    plan.add_argument("--grid", default=None,
                      help="the family to build the control on; required when the candidate "
                           "spans more than one")
    plan.add_argument("--rule", choices=("nearest", "no_larger"), default="nearest")
    plan.add_argument("--max-relative-slack", type=float,
                      default=float(DEFAULT_MAX_RELATIVE_SLACK),
                      help="how far apart in bytes two arms may be and still be a control "
                           f"(default {float(DEFAULT_MAX_RELATIVE_SLACK)})")
    plan.add_argument("--candidate-label", default="allocated")
    plan.set_defaults(func=cmd_plan)

    verify = sub.add_parser(
        "verify", help="assert the match on two EXPORTED checkpoints, and record the verdict")
    verify.add_argument("candidate", type=Path, help="the candidate checkpoint directory")
    verify.add_argument("control", type=Path, help="the uniform control checkpoint directory")
    verify.add_argument("--params", type=int, default=None,
                        help="quantizable parameter count, for the bpp figures")
    verify.add_argument("--candidate-kl", type=float, default=None)
    verify.add_argument("--control-kl", type=float, default=None)
    verify.add_argument("--metric", default="kl_vs_bf16")
    verify.add_argument("--candidate-label", default="allocated")
    verify.add_argument("--require-no-larger", action="store_true",
                        help="also refuse a control that outweighs the candidate")
    verify.add_argument("--max-relative-slack", type=float,
                        default=float(DEFAULT_MAX_RELATIVE_SLACK))
    verify.add_argument("--report", type=Path, default=None)
    verify.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
