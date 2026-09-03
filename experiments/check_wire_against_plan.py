#!/usr/bin/env python
"""Do the exported bytes weigh what the allocator charged for them?

``plan_from_layer_config.py`` writes a sidecar whose per-unit
``prismaquant_charged_bits`` comes from PrismaQuant's own accountant
(``tessera_formats.artifact_bpp``); ``export_tessera_serving.py`` writes a
manifest whose per-role ``wire_bytes`` is what the encoder actually emitted.
Rendering identity means those two are the same number for every unit -- not
close, equal -- because the allocator spent a byte budget in the first currency
and the serve reads the second.

Exits non-zero on any disagreement, naming the units.

Three outcomes, not two.  A plan that does not PRICE a unit is silent about
it, which is a different thing from a plan that prices it differently, and the
operator's next move differs: re-run the plan with the allocator attached
against fix the encoder/allocator drift.  ``missing`` and ``extra`` stay
disagreements -- the plan and the export make contradictory claims about which
units exist -- but ``unpriced`` gets its own bucket and its own verdict word.

usage: check_wire_against_plan.py <plan.json.provenance.json> <checkpoint-dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("provenance", type=Path, help="the plan's .provenance.json sidecar")
    ap.add_argument("checkpoint", type=Path, help="the exported Tessera checkpoint directory")
    ap.add_argument("--quiet", action="store_true", help="print only the verdict and any mismatch")
    args = ap.parse_args(argv)

    plan = json.loads(args.provenance.read_text())
    manifest = json.loads((args.checkpoint / "tessera_serving_manifest.json").read_text())

    charged = {u["tensor"]: u for u in plan["units"]}
    emitted = {r["tensor"]: r for m in manifest["modules"].values() for r in m["roles"]}

    missing = sorted(set(charged) - set(emitted))
    extra = sorted(set(emitted) - set(charged))
    rows, bad, unpriced = [], [], []
    for tensor in sorted(set(charged) & set(emitted)):
        want = charged[tensor]
        got = emitted[tensor]
        # ``is None``, not truthiness: the writer emits None or [num, den]
        # (plan_from_layer_config.py:437), so a [0, 1] charge -- a truthy list
        # holding a falsy Fraction -- would otherwise be read as unpriced.
        exact = want["prismaquant_charged_bits_exact"]
        want_bits = None if exact is None else Fraction(*exact)
        got_bits = Fraction(got["wire_bytes"] * 8)
        if want_bits is None:
            # The plan says nothing about this unit's price, so there is
            # nothing here to agree or disagree with.  Reporting it as a
            # MISMATCH would name the wire bits as the offender when the wire
            # is the only side that spoke.
            unpriced.append((tensor, got_bits, got["q256"]))
            rows.append((tensor, want, got, want_bits, got_bits, None))
            continue
        ok = (want_bits == got_bits
              and want["q256"] == got["q256"] and want["grid"] == got["grid"]
              and want["rows"] == got["rows"] and want["columns"] == got["cols"])
        rows.append((tensor, want, got, want_bits, got_bits, ok))
        if not ok:
            bad.append((tensor, want_bits, got_bits, want["q256"], got["q256"]))

    if not args.quiet:
        print(f"{'tensor':46s} {'rung':>6s} {'shape':>14s} {'charged bits':>14s} {'wire bits':>14s}")
        for tensor, want, got, want_bits, got_bits, ok in rows:
            shape = f"{want['rows']}x{want['columns']}"
            wb = "n/a" if want_bits is None else str(want_bits)
            note = {True: "", False: "  MISMATCH", None: "  UNPRICED"}[ok]
            print(f"{tensor:46s} {'R' + str(got['q256']):>6s} {shape:>14s} "
                  f"{wb:>14s} {str(got_bits):>14s} {note}")

    # Both totals and the denominator run over the SAME rows.  Summing charged
    # bits over the priced units and wire bits over every unit compared a
    # subset against the whole and guaranteed the equality below would fail for
    # a reason that has nothing to do with pricing -- and the bpp lines divided
    # a subset's bits by every unit's params, which is low by construction.
    priced = [r for r in rows if r[3] is not None]
    total_charged = sum((r[3] for r in priced), Fraction(0))
    total_wire = sum((r[4] for r in priced), Fraction(0))
    params = sum(r[1]["params"] for r in priced)
    scope = " (priced units only)" if unpriced else ""
    print()
    print(f"units compared        {len(rows)}"
          + (f" ({len(priced)} priced, {len(unpriced)} unpriced)" if unpriced else ""))
    if params:
        print(f"charged (PrismaQuant) {total_charged} bits"
              f" = {float(total_charged / params):.9f} bpp{scope}")
        print(f"emitted (wire)        {total_wire} bits"
              f" = {float(total_wire / params):.9f} bpp{scope}")
    else:
        print("charged (PrismaQuant) no priced unit to total")
        print("emitted (wire)        no priced unit to total")
    whole = " (whole checkpoint)" if unpriced else ""
    print(f"manifest wire_bpp     {manifest['totals']['wire_bpp']:.9f}{whole}")
    if missing:
        print(f"IN THE PLAN, NOT EXPORTED ({len(missing)}): {missing[:5]}")
    if extra:
        print(f"EXPORTED, NOT IN THE PLAN ({len(extra)}): {extra[:5]}")
    if bad:
        print(f"PER-UNIT MISMATCH ({len(bad)}):")
        for tensor, want_bits, got_bits, wq, gq in bad[:20]:
            print(f"  {tensor}: charged {want_bits} R{wq} vs wire {got_bits} R{gq}")
    if unpriced:
        print(f"UNPRICED IN THE PLAN ({len(unpriced)}):")
        for tensor, got_bits, gq in unpriced[:20]:
            print(f"  {tensor}: plan charges nothing; wire {got_bits} R{gq}")

    # `missing` and `extra` stay disagreements: there the plan and the export
    # make contradictory claims about which units exist.  `unpriced` is the
    # plan declining to make a claim at all, so it is neither agreement nor
    # disagreement -- but it is not a pass either, because identity was not
    # demonstrated for those units.  Hence three words and two exit codes.
    disagrees = bool(bad or missing or extra) or total_charged != total_wire
    if disagrees:
        verdict = "DISAGREEMENT"
    elif unpriced:
        verdict = ("UNPRICED IN THE PLAN: the priced units agree; "
                   "identity is not shown for the rest")
    else:
        verdict = "the bytes served are the bytes priced"
    print("VERDICT:", verdict)
    return 0 if not disagrees and not unpriced else 1


if __name__ == "__main__":
    sys.exit(main())
