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

**Everything else the pair carries is held identical, and that is now decided
rather than merely recorded** (tessera#248).  ``CHANGE_POLICY`` below states,
for every tensor, whether this intervention may move it: the weight plane's
block and global scales may (that is the treatment), the packed codes and the
BF16 source/passthrough ``.weight`` tensors and the A4 ``.input_global_scale``
activation quantizer may not.  Before #248 the verdict consulted the codes, the
plane, the tensor-name sets and the wire totals only, so a B side whose BF16
weight or whose input scale had also moved -- same name, same shape, same
dtype, same byte count -- still returned 0 and ``verdict: "the matched pair"``,
with the extra difference sitting under ``by_suffix`` for a human to notice.  A
served KL over that pair isolates nothing.

**And the policy is total over the checkpoint, not over a roster**
(tessera#278).  #248's totality check compared ``CHANGE_POLICY`` against the
five suffixes ``load`` chose to read, so it could not fire: a ``.bias``, a
``.k_scale`` or anything else the exporter writes was never loaded, appeared in
no count, in no ``tensors_only_in_*`` set and in no verdict, and a pair whose B
side had also moved a bias still exited 0 as "the matched pair".  ``load`` now
reads every tensor, and one whose suffix no ``CHANGE_POLICY`` entry governs
refuses the pair by name -- a tensor without a policy is a tensor the verdict
cannot see, and identical-today is not a policy.  Qwen3-0.6B's W4A4 twins carry
nothing outside the five, so no receipt taken before this moves; a Qwen2-family
model with q/k/v biases, or a W8A8 twin, is the input that broke it.

    PYTHONPATH=src python experiments/refit_trailing_bytes.py A_DIR B_DIR \
        --wire-a A_TESSERA --wire-b B_TESSERA \
        --out experiments/results/refit_trailing_bytes.json

**The exit status says which of three things happened** (tessera#269), because
one caller's expected reading is the refusal: ``refit_trailing_serve.sh
compare-drift`` runs this tool to report that the 2026-09-02 bytes are NOT the
matched pair, so it cannot read a nonzero status as "the tool failed" and it
cannot read a zero one as "it ran".  Until #269 a computed
``verdict: "NOT the matched pair"`` and an uncaught exception both exited 1,
and the two are not the same event: one has a receipt and one has none.

===== ==================================================================
``0`` ``EXIT_MATCHED`` -- a verdict was computed: **the matched pair**.
``3`` ``EXIT_NOT_MATCHED`` -- a verdict was computed: **NOT the matched
      pair**.  The receipt at ``--out`` is written first and describes it.
``1`` ``EXIT_TOOL_FAILED`` -- **no verdict**: an uncaught exception, or one
      of this module's ``SystemExit`` refusals.  Nothing was decided, and
      whatever sits at ``--out`` was left there by an earlier run.
===== ==================================================================
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors import safe_open

# The exit-code contract, named once here because a second process reads it:
# ``refit_trailing_serve.sh compare-drift`` accepts EXIT_MATCHED and
# EXIT_NOT_MATCHED as readings of its stage and refuses anything else.  A shell
# script cannot import a Python constant, so those two numbers are written out
# there as well, and `tests/test_refit_trailing_bytes.py` pins the two lists
# against each other.  See the module docstring for what each one means.
EXIT_MATCHED = 0
EXIT_NOT_MATCHED = 3
EXIT_TOOL_FAILED = 1   # what Python already returns for an uncaught exception
                       # and for ``raise SystemExit("message")``; named so the
                       # third outcome is stated rather than implied.

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
# So every tensor carries a policy, and the policy is checked.  ``MUST_MOVE``
# is the leg ``the_plane_moved`` already held and ``MAY_MOVE`` is the one thing
# that needs no check; what was missing is ``IDENTICAL``, and stating all three
# is what makes the set total rather than a pair of habits.
#
# **Total over the checkpoint, not over a roster** (tessera#278).  #248's fix
# was total over the five suffixes the loader chose to read, which left the
# same door one over: a ``.bias``, a ``.k_scale`` or any other tensor the
# exporter writes was not loaded at all, so it was in no count, in no
# ``tensors_only_in_*`` set and in no verdict -- and a pair whose B side had
# also moved a bias still certified as "the matched pair" and exited 0.  There
# is no second roster now: ``load`` reads every tensor, this dict decides, and
# a tensor no entry governs REFUSES the pair by name (``EXIT_TOOL_FAILED``:
# nothing was decided) rather than defaulting to identical.  Identical today is
# not a policy -- the check cannot say whether this intervention may move a
# tensor nobody wrote a rule for, and guessing is what #248 and #278 both are.
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


def policy_suffix(name: str) -> "str | None":
    """The ``CHANGE_POLICY`` entry that governs ``name``, or None.

    Longest match wins, so a rule for a narrower suffix cannot be shadowed by
    a broader one.  None is a refusal and never a default: see
    ``refuse_unpoliced``.
    """
    governing = [s for s in CHANGE_POLICY if name.endswith(s)]
    return max(governing, key=len) if governing else None


def refuse_unpoliced(names) -> None:
    """The one home of "a tensor without a policy is a tensor the verdict
    cannot see".  It used to guard the source (``SUFFIXES`` against
    ``CHANGE_POLICY``, where it could not fire) and now guards the artifact,
    which is where the tensors actually are (tessera#278)."""
    unpoliced = sorted({n for n in names if policy_suffix(n) is None})
    if unpoliced:
        shown = unpoliced[:4] + (["..."] if len(unpoliced) > 4 else [])
        raise SystemExit(
            f"refit_trailing_bytes: this pair carries {len(unpoliced)} "
            f"tensor(s) no CHANGE_POLICY entry governs -- {shown}; a tensor "
            "without a policy is a tensor the verdict cannot see, so this "
            "pair is not decided.  Say whether this intervention may move "
            "that suffix (IDENTICAL, MUST_MOVE or MAY_MOVE) and run again")


def load(path: Path) -> "dict[str, torch.Tensor]":
    """Every tensor in the export, not a chosen roster of them: a tensor that
    is never read is one no policy can refuse (tessera#278)."""
    out: dict = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                out[name] = handle.get_tensor(name)
    return out


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def main() -> int:
    """Compare the two exports and return the exit code the docstring states:
    ``EXIT_MATCHED`` or ``EXIT_NOT_MATCHED`` for a verdict, and never for
    anything else -- a run that reached no verdict raises instead, which is
    ``EXIT_TOOL_FAILED``."""
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
    # An absent, empty or unreadable export dir globs to nothing and raises
    # nothing, so before tessera#269 it fell through to a "NOT the matched
    # pair" verdict computed over zero tensors -- and that verdict is exactly
    # the reading `compare-drift` accepts.  A comparison that compared nothing
    # is the tool failing, and it says which side.
    for side, given, loaded in (("a", args.a, ta), ("b", args.b, tb)):
        if not loaded:
            raise SystemExit(
                f"refit_trailing_bytes: {side} ({given}) holds no *.safetensors "
                "tensor at all; an absent or unreadable export compares "
                "nothing, which is this tool failing and not a verdict about "
                "a pair")
    # Every tensor either side carries is governed or the pair is refused, so
    # the verdict below is a claim about the whole pair (tessera#278).
    refuse_unpoliced((*ta, *tb))
    shared = sorted(set(ta) & set(tb))
    by_suffix: dict = collections.defaultdict(
        lambda: {"same": 0, "different": 0, "names_different": []})
    for name in shared:
        suffix = policy_suffix(name)
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

    # Every tensor the pair carries is decided, not just the two suffixes the
    # verdict used to read (tessera#248, #278).  `by_suffix` is a defaultdict,
    # so a suffix absent from both exports reads zero/zero -- absence is not a
    # difference.
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
    # A verdict, either way -- and the receipt above describes it.  The tool
    # failing to reach one is EXIT_TOOL_FAILED and never returned from here.
    return (EXIT_MATCHED if record["verdict"] == "the matched pair"
            else EXIT_NOT_MATCHED)


if __name__ == "__main__":
    raise SystemExit(main())
