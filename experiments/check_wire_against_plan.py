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
    rows, bad = [], []
    for tensor in sorted(set(charged) & set(emitted)):
        want = charged[tensor]
        got = emitted[tensor]
        want_bits = (Fraction(*want["prismaquant_charged_bits_exact"])
                     if want["prismaquant_charged_bits_exact"] else None)
        got_bits = Fraction(got["wire_bytes"] * 8)
        ok = (want_bits is not None and want_bits == got_bits
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
            print(f"{tensor:46s} {'R' + str(got['q256']):>6s} {shape:>14s} "
                  f"{wb:>14s} {str(got_bits):>14s} {'' if ok else '  MISMATCH'}")

    total_charged = sum((r[3] for r in rows if r[3] is not None), Fraction(0))
    total_wire = sum((r[4] for r in rows), Fraction(0))
    params = sum(r[1]["params"] for r in rows)
    print()
    print(f"units compared        {len(rows)}")
    print(f"charged (PrismaQuant) {total_charged} bits = {float(total_charged / params):.9f} bpp")
    print(f"emitted (wire)        {total_wire} bits = {float(total_wire / params):.9f} bpp")
    print(f"manifest wire_bpp     {manifest['totals']['wire_bpp']:.9f}")
    if missing:
        print(f"IN THE PLAN, NOT EXPORTED ({len(missing)}): {missing[:5]}")
    if extra:
        print(f"EXPORTED, NOT IN THE PLAN ({len(extra)}): {extra[:5]}")
    if bad:
        print(f"PER-UNIT MISMATCH ({len(bad)}):")
        for tensor, want_bits, got_bits, wq, gq in bad[:20]:
            print(f"  {tensor}: charged {want_bits} R{wq} vs wire {got_bits} R{gq}")
    verdict = not (bad or missing or extra) and total_charged == total_wire
    print("VERDICT:", "the bytes served are the bytes priced" if verdict else "DISAGREEMENT")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
