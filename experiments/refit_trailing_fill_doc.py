#!/usr/bin/env python3
"""Fill tessera#75's served page from the receipts the pool action wrote.

Every number here is read out of a file the run produced; nothing is typed --
which is the point: the page and the receipts cannot drift apart if the page is
generated from them.  It also refuses a KL receipt that is not
``prismaquant.kl_compare/2``, and it says so in the page when the pair check
FAILED, because a page that quietly omits the failing arm is worse than none.

Run from the worktree root once ``/mnt/shared/tessera-runs/refit-trailing/DONE``
exists:

    python experiments/refit_trailing_fill_doc.py

then read the diff before committing it.
"""
import json
import sys
from pathlib import Path

R = Path("/mnt/shared/tessera-runs/refit-trailing")
DOC = Path("docs/measurements/tessera-refit-trailing-served-2026-09-04.md")

pair = json.loads(Path("experiments/results/refit_trailing_bytes.json").read_text())
kl_a = json.loads((R / "kl_a4h1.json").read_text())
kl_b = json.loads((R / "kl_bjac.json").read_text())
gate = json.loads(
    Path("experiments/results/refit_trailing_pair_gate_served.json").read_text())

for name, k in (("A", kl_a), ("B", kl_b)):
    if k.get("schema") != "prismaquant.kl_compare/2":
        sys.exit(f"{name}: not a kl_compare/2 receipt: {k.get('schema')!r}")
a = kl_a["all"]["kl_lower_mean"]
b = kl_b["all"]["kl_lower_mean"]

sfx = pair["by_suffix"]
packed, scale = sfx[".weight_packed"], sfx[".weight_scale"]
pair_md = f"""`experiments/results/refit_trailing_bytes.json`, `a4h1` against `bjac`,
both built by this checkout on 2026-09-04.

| | `a4h1` (A) vs `bjac` (B) |
|---|---|
| `.weight_packed` | {packed['same']} same, **{packed['different']} different** |
| `.weight_scale` | {scale['same']} same, {scale['different']} different |
| `wire_bytes` | {'equal' if pair['wire']['wire_bytes_equal'] else 'DIFFERENT'} ({pair['wire']['a']['wire_bytes']} / {pair['wire']['b']['wire_bytes']}) |
| codes identical on every unit | {pair['codes_identical_on_every_unit']} |
| the plane moved | {pair['the_plane_moved']} |
| verdict | **{pair['verdict']}** |
"""
if not pair["codes_identical_on_every_unit"]:
    pair_md += (
        "\n**The codes moved.** The pair theory says they cannot: the encoder's"
        " loop is trellis-then-refit, so the trailing objective enters after"
        " the last trellis pass. Same-day arms whose codes differ mean the"
        " encoder is not deterministic, and the action stopped before either"
        " serve rather than serve two treatments. That is the finding.\n")

served_md = f"""Both receipts are `prismaquant.kl_compare/2`, both against the same
teacher, the same corpus, the same pinned image, eager, one box, back to back.

| arm | served KL vs BF16 (`all.kl_lower_mean`) | receipt |
|---|---|---|
| A `a4h1` (control) | **{a:.6g}** | `{R / 'kl_a4h1.json'}` |
| B `bjac` (trailing full `H`) | **{b:.6g}** | `{R / 'kl_bjac.json'}` |
| B/A | **{b / a:.4f}x** | |

{'B is lower.' if b < a else 'B is not lower.'} The screen said 0.9191x in
held-out activation-space relative error on six units; served, on all 196, the
ratio is {b / a:.4f}x. A screen and a serve are different measurements and the
served one is the one that counts.
"""

log = gate.get("log") or []
arm = next((k for k in gate["arms"] if gate["arms"][k].get("served_kl") is not None),
           None)
verbatim = "\n".join(log)
gate_md = f"""`experiments/results/refit_trailing_pair_gate_served.json`, run with
`--served-arm B-Jac --served-kl-json …/kl_bjac.json --served-bar-json
…/kl_a4h1.json`. A separate file from the merged
`refit_trailing_pair_gate.json`, which is the screen's own verdict and stays
the record of what a screen earns.

The arm carrying the served number is `{arm}`. What
`tessera.control.assert_plane_promotion` said, verbatim:

```
{verbatim}
```
"""

s = DOC.read_text()


def swap(head, body):
    global s
    at = s.index(head)
    end = s.index("\n## ", at + len(head))
    s = s[:at] + head + "\n\n" + body.rstrip() + "\n" + s[end:]


swap("## The matched pair, on the artifacts\n", pair_md)
swap("## Served\n", served_md)
swap("## The gate, verbatim\n", gate_md)

banner_end = s.index("**What this is.**")
s = ("# The trailing refit's objective, served (2026-09-04)\n\n"
     "**Status: measured. The pair check, both served KLs and the gate's "
     "verdict below were produced by one pool action "
     "(`experiments/refit_trailing_run_all.sh`, key `aadd46b6525d…`) on sparky, "
     "which exports the control, proves the pair on all 196 units, and serves "
     "only if the pair check passes.**\n\n" + s[banner_end:])
DOC.write_text(s)
print(f"A {a!r}\nB {b!r}\nB/A {b / a}\npair {pair['verdict']}\narm {arm}")
print("filled", DOC)
