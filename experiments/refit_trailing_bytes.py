#!/usr/bin/env python
"""The matched pair, read off two exported checkpoints instead of asserted.

tessera#75's pair is ``T R_h T R_h T R_h T R_H`` against
``T R_h T R_h T R_h T R_h``: passes 1-3 are identical calls, and pass 4's
trellis runs against the plane pass 3's refit left -- BEFORE pass 4's refit.
So every trellis pass is identical in the two arms and only the last scale
plane can differ.  On the stock twin that separates cleanly: ``weight_packed``
carries the codes and must be **identical on every unit**, ``weight_scale``
carries the plane and must **move**.  A run where the packed tensors differ is
not this pair, whatever the flags said; a run where the scales do not move is a
flag that reached nothing.

``experiments/refit_trailing_pair.py`` proved the relation on six units inside
one process.  This proves it on the 196 units of the artifact that gets served,
across two separate exports -- which is the statement a served A/B needs, since
"identical bytes" there means the checkpoints, not a helper's tensors.

The twin carries the *codes* claim.  ``--wire-a/--wire-b`` carry the *length*
claim, off the two exports' own totals: #75 says the swap costs no bytes, and
that is a statement about the Tessera wire the twin is a materialisation of,
not about the twin.  Both are recorded, and a length that moved fails the same
way a moved code does.

    PYTHONPATH=src python experiments/refit_trailing_bytes.py A_DIR B_DIR \
        --wire-a A_TESSERA --wire-b B_TESSERA \
        --out experiments/results/refit_trailing_bytes.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors import safe_open

SUFFIXES = (".weight", ".weight_scale", ".weight_packed",
            ".weight_global_scale", ".input_global_scale")


def load(path: Path) -> "dict[str, torch.Tensor]":
    out: dict = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                if name.endswith(SUFFIXES):
                    out[name] = handle.get_tensor(name)
    return out


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wire-a", default=None,
                    help="A's Tessera export dir, for the wire-length claim")
    ap.add_argument("--wire-b", default=None,
                    help="B's Tessera export dir, for the wire-length claim")
    args = ap.parse_args()

    ta, tb = load(Path(args.a)), load(Path(args.b))
    shared = sorted(set(ta) & set(tb))
    by_suffix: dict = collections.defaultdict(
        lambda: {"same": 0, "different": 0, "names_different": []})
    for name in shared:
        suffix = next(s for s in SUFFIXES if name.endswith(s))
        x, y = raw(ta[name]), raw(tb[name])
        same = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
        by_suffix[suffix]["same" if same else "different"] += 1
        if not same and len(by_suffix[suffix]["names_different"]) < 4:
            by_suffix[suffix]["names_different"].append(name)

    wire = None
    if args.wire_a and args.wire_b:
        wire = {}
        for side, path in (("a", args.wire_a), ("b", args.wire_b)):
            man = json.loads(
                (Path(path) / "tessera_serving_manifest.json").read_text())
            totals = man.get("totals", man)
            wire[side] = {k: totals.get(k) for k in
                          ("wire_bytes", "on_disk_bytes", "units", "modules")}
        wire["wire_bytes_equal"] = (
            wire["a"]["wire_bytes"] is not None
            and wire["a"]["wire_bytes"] == wire["b"]["wire_bytes"])

    record = {
        "a": args.a, "b": args.b,
        "tensors_only_in_a": sorted(set(ta) - set(tb)),
        "tensors_only_in_b": sorted(set(tb) - set(ta)),
        "by_suffix": {k: dict(v) for k, v in sorted(by_suffix.items())},
        "wire": wire,
    }
    packed = by_suffix[".weight_packed"]
    scale = by_suffix[".weight_scale"]
    record["codes_identical_on_every_unit"] = (
        packed["different"] == 0 and packed["same"] > 0)
    record["the_plane_moved"] = scale["different"] > 0
    record["verdict"] = (
        "the matched pair" if record["codes_identical_on_every_unit"]
        and record["the_plane_moved"] and not record["tensors_only_in_a"]
        and not record["tensors_only_in_b"]
        and (wire is None or wire["wire_bytes_equal"])
        else "NOT the matched pair")

    print(json.dumps(record, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=1))
        print(f"wrote {args.out}")
    return 0 if record["verdict"] == "the matched pair" else 1


if __name__ == "__main__":
    raise SystemExit(main())
