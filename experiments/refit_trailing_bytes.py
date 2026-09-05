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

**Everything else this loads is held identical, and that is now decided rather
than merely recorded** (tessera#248).  ``CHANGE_POLICY`` below states, for
every suffix the loader reads, whether this intervention may move it: the
weight plane's block and global scales may (that is the treatment), the packed
codes and the BF16 source/passthrough ``.weight`` tensors and the A4
``.input_global_scale`` activation quantizer may not.  Before #248 the verdict
consulted the codes, the plane, the tensor-name sets and the wire totals only,
so a B side whose BF16 weight or whose input scale had also moved -- same name,
same shape, same dtype, same byte count -- still returned 0 and
``verdict: "the matched pair"``, with the extra difference sitting under
``by_suffix`` for a human to notice.  A served KL over that pair isolates
nothing.

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

# What this intervention is allowed to move, stated once and totally.
#
# Loading a tensor and recording its difference is not a check: before
# tessera#248 the verdict consulted the packed codes, the scale plane, the
# tensor-name sets and the wire totals, so a B side whose BF16 passthrough
# weight or whose A4 activation scale had ALSO moved was still certified "the
# matched pair" -- with the extra difference sitting in `by_suffix` for a human
# to notice.  A served KL over that pair cannot isolate the trailing
# weight-scale objective, because it is not the only thing that changed.
#
# So every suffix the loader reads carries a policy, and the policy is checked.
# ``MUST_MOVE`` is the leg ``the_plane_moved`` already held and ``MAY_MOVE`` is
# the one thing that needs no check; what was missing is ``IDENTICAL``, and
# stating all three is what makes the set total rather than a pair of habits:
IDENTICAL = "identical"    # the trellis output: identical on every unit or it
                           # is not this pair, whatever the flags said
MUST_MOVE = "must move"    # the intervention itself: a flag that reached
                           # nothing is not a treatment
MAY_MOVE = "may move"      # part of the weight plane the refit rewrites
CHANGE_POLICY = {
    # The BF16 source and passthrough weights: the trailing refit rewrites a
    # scale plane, never a weight.  One of these moving is a second treatment.
    ".weight": IDENTICAL,
    ".weight_packed": IDENTICAL,
    # The weight plane the trailing refit exists to move, block and global.
    ".weight_scale": MUST_MOVE,
    ".weight_global_scale": MAY_MOVE,
    # The activation quantizer, held fixed by the experiment's design: the two
    # arms serve under the same static A4 input scales.
    ".input_global_scale": IDENTICAL,
}
_UNPOLICED = set(SUFFIXES) - set(CHANGE_POLICY)
if _UNPOLICED:                                    # a loaded tensor with no rule
    raise SystemExit(
        f"refit_trailing_bytes: {sorted(_UNPOLICED)} are loaded and compared "
        "but no CHANGE_POLICY says whether this intervention may move them; "
        "a tensor without a policy is a tensor the verdict cannot see")


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

    # Every suffix the loader reads is decided, not just the two the verdict
    # used to read (tessera#248).  `by_suffix` is a defaultdict, so a suffix
    # absent from both exports reads zero/zero -- absence is not a difference.
    immutable_changed = {
        suffix: by_suffix[suffix]["names_different"]
        for suffix, policy in sorted(CHANGE_POLICY.items())
        if policy is IDENTICAL and by_suffix[suffix]["different"] > 0
    }
    record = {
        "a": args.a, "b": args.b,
        "tensors_only_in_a": sorted(set(ta) - set(tb)),
        "tensors_only_in_b": sorted(set(tb) - set(ta)),
        "by_suffix": {k: dict(v) for k, v in sorted(by_suffix.items())},
        "change_policy": dict(sorted(CHANGE_POLICY.items())),
        "wire": wire,
    }
    packed = by_suffix[".weight_packed"]
    scale = by_suffix[".weight_scale"]
    record["codes_identical_on_every_unit"] = (
        packed["different"] == 0 and packed["same"] > 0)
    record["the_plane_moved"] = scale["different"] > 0
    # The names, so a refusal says WHICH tensor moved rather than only that one
    # did.  `names_different` caps at four per suffix; the counts beside it in
    # `by_suffix` are the whole population.
    record["immutable_changed"] = immutable_changed
    record["immutable_tensors_identical"] = not immutable_changed
    record["verdict"] = (
        "the matched pair" if record["codes_identical_on_every_unit"]
        and record["the_plane_moved"] and record["immutable_tensors_identical"]
        and not record["tensors_only_in_a"]
        and not record["tensors_only_in_b"]
        and (wire is None or wire["wire_bytes_equal"])
        else "NOT the matched pair")
    if immutable_changed:
        print("NOT the matched pair: these tensors are held identical by this "
              "intervention and moved --")
        for suffix, names in immutable_changed.items():
            print(f"  {suffix}: {by_suffix[suffix]['different']} changed, "
                  f"e.g. {names}")

    print(json.dumps(record, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=1))
        print(f"wrote {args.out}")
    return 0 if record["verdict"] == "the matched pair" else 1


if __name__ == "__main__":
    raise SystemExit(main())
