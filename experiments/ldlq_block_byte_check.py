#!/usr/bin/env python
"""Assert that two arms of the LDLQ-block A/B differ in the block and nothing else.

An LDLQ block is a SCHEDULE: it changes the order columns are compensated in,
not what the wire holds, so two arms must agree on every byte count and
disagree on every quantized tensor.  **This script checks the first half
only.**  Both failures are worth naming, because each alone is its own bug:

* same counts, same tensors -> the flag did not reach the encoder, and the
  A/B is one arm served twice.  This has happened in this repo (a dropped
  kwarg is a silent no-op) and is why the sweep harness prints
  ``!! IDENTICAL BYTES``.  **Not detectable here**: nothing below opens a
  tensor.  ``compare_stock_checkpoints.py`` is what answers it, and it has to
  be run separately -- an arm that passes this script has not been shown to
  differ from its control.
* different counts -> the block moved the wire, which it must not, and the
  served comparison is not at matched bytes.  That is what is checked here.

The encoder settings are compared field by field so that a difference in
anything but ``ldlq_block`` is named rather than summarised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COUNTS = ("quantized_params", "wire_bytes", "wire_bpp", "on_disk_bytes",
          "resident_mode_bytes", "checkpoint_bytes", "modules", "units")


def manifest(wire: Path) -> dict:
    return json.loads((wire / "tessera_serving_manifest.json").read_text())


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("incumbent_wire", type=Path)
    ap.add_argument("candidate_wire", type=Path)
    ap.add_argument("--expect-differs", default="ldlq_block",
                    help="the one encoder setting the two arms may disagree on")
    args = ap.parse_args()

    a, b = manifest(args.incumbent_wire), manifest(args.candidate_wire)
    bad = []

    print("== byte counts")
    for key in COUNTS:
        x, y = a["totals"].get(key), b["totals"].get(key)
        same = x == y
        print(f"  {key:<22} {x!r:<22} {y!r:<22} {'same' if same else 'DIFFER'}")
        if not same:
            bad.append(f"{key} moved: {x} -> {y}; a block is a schedule and must not move the wire")

    print("\n== twin checkpoint size on disk")
    for wire in (args.incumbent_wire, args.candidate_wire):
        twin = json.loads((wire / "tessera_serving_manifest.json").read_text())["stock_twin"]
        size = sum(p.stat().st_size for p in sorted(Path(twin).glob("*.safetensors")))
        print(f"  {twin}  {size}")

    print("\n== encoder settings, field by field")
    fa, fb = {}, {}
    flatten("", a["activation_aware"], fa)
    flatten("", b["activation_aware"], fb)
    for key in sorted(set(fa) | set(fb)):
        x, y = fa.get(key), fb.get(key)
        if x == y:
            continue
        expected = key.split(".")[-1] == args.expect_differs
        print(f"  {key:<28} {x!r} -> {y!r}   {'THE ARM' if expected else 'UNEXPECTED'}")
        if not expected:
            bad.append(f"{key} differs ({x!r} -> {y!r}) but only {args.expect_differs} may")
    if fa.get(args.expect_differs) == fb.get(args.expect_differs):
        bad.append(
            f"{args.expect_differs} is {fa.get(args.expect_differs)!r} on BOTH arms: "
            "the flag never reached the encoder and this is one arm exported twice")
    else:
        print(f"  -> one arm changed, and it is {args.expect_differs}")

    print()
    for line in bad:
        print(f"REFUSED: {line}")
    if not bad:
        print("OK: identical byte counts, one differing encoder setting "
              f"({args.expect_differs}).  Tensor-level difference is "
              "compare_stock_checkpoints.py's job and is run separately.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
