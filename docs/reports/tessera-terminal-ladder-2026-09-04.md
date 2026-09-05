# The truncation-terminal ladder on an encode: what stops it, and what it would buy

**Date:** 2026-09-04. **Issue:** tessera#144. **Verdict:** the ladder is
unreachable on every artifact this tree writes, and making it reachable is a
wire change (two, in fact) before it is a reader change. That is Rob's
decision; this report prices it and does not make it.

Every number below was produced on `4f2f95a` plus this branch, on CPU, with
`/tmp/.../fleet/ladder_count.py` and `ladder_rows.py` (16-row toy units, seed
0) and is pinned by
`tests/test_audit_container_accounting.py::test_no_shorter_terminal_survives_the_wire_on_an_encode`.
The one quality number quoted is someone else's screen and is labelled as such.

## 1. What is true

**The encoder writes one terminal, and #163 already says so** (`7180f35`:
`errors.py`, `container.py`, `planes.py`, `layout.py`, schema §3c, README).
`build_unit_artifact` is the only encode-side terminal write site
(`grep -rn "terminals=(" src/` -> `unit_artifact.py:586` and the docstring in
`errors.py` that quotes it). `artifact.py`, which the audit cited as the
ladder's only producer, was deleted in `b504d70` as unreachable. So the
"prose in four modules" half of the issue was closed before this branch.

**The count.** Twelve `(grid, rung)` points through the exporter's own path
(`export.encode_linear_planes`, default arguments, so `wire_recipe` chose the
body and plane): E2M1 at q256 256/512/768, E2M1x2 at 384/640/896 (1024 is
refused above the cap), E4M3 at 768/1024/1536/1792, BF16 at 1024/2048.

| grid | rungs | body | plane | planes with elements | COMPLETION | RELEASE | shorter terminals the layout could declare | of which decode to weights |
|---|---|---|---|---|---|---|---|---|
| E2M1 | 256, 512, 768 | TCQ | LUT | ALPHABET, DESCENDANT, BODY, SCALE_REFINE | 0 | 0 | 4 | 0 |
| E2M1x2 | 384, 640 | WINDOW | LUT | ALPHABET, BODY, SCALE_REFINE | 0 | 0 | 3 | 0 |
| E2M1x2 | 896 (cap) | TCQ | LUT | ALPHABET, DESCENDANT, BODY, SCALE_REFINE | 0 | 0 | 4 | 0 |
| E4M3 | 768 .. 1792 | WINDOW | CHANNEL | ALPHABET, BODY, DIAG_SV | 0 | 0 | 3 | 0 |
| BF16 | 1024, 2048 | WINDOW | CHANNEL | ALPHABET, BODY, DIAG_SV | 0 | 0 | 3 | 0 |

"Shorter terminals the layout could declare" counts the cuts
`Manifest._validate_terminal_prefixes` would accept on a one-superblock unit:
each present plane can be left empty (a whole-plane cut), and BODY can be cut
at a superblock boundary. Every one of them drops a plane nothing decodes
without -- the alphabet, the body, or the scale index -- or drops whole
columns. None is a rate rung. The reason is the first row of every table:
`encode_linear_planes` defaults to `completion=0` and `released_positions=0`
(`export.py:1291`, `:1403`), and a window body -- every E2M1x2 sub-cap, E4M3
and BF16 rung -- refuses any other completion (`export.py:1399`). Release is
additionally refused at arity > 1 (`encode.py:2819`) and is 0 on every
exporter path. **The exporter's default terminal has 0 COMPLETION and 0
RELEASE elements on all four serialisable grids.**

**Where a completion axis exists at all.** Only E2M1 (arity 1, TCQ body)
below the cap, and only when a caller passes `completion=None` (full depth)
or a positive depth explicitly. E2M1 is in no live menu
(`glm-expert-menu-nvfp4-is-dominated`: the GLM menu is {3.5, 4.0, 8.0} =
E2M1x2 sub-cap window, E2M1x2 cap TCQ, E4M3 window) and is "unmeasured under
the window body" (`export.py:982`). On that encode three shorter terminals
were built and offered to the manifest, in the classes the schema names:

| terminal | columns | plane | manifest | reader |
|---|---|---|---|---|
| depth 0 of 2 (`t-c0`) | 256 or 512 | LUT | REFUSED: `not a prefix: SCALE_REFINE carries 512 elements after an earlier plane was left incomplete` | -- |
| depth 1 of 2 (`t-c1`) | 256 (one superblock) | LUT | REFUSED: `cuts COMPLETION at 4096 elements, which is not a per-superblock quota boundary of [0, 8192]` | -- |
| depth 1 of 2 (`t-c1`) | 512 (two superblocks) | LUT | REFUSED: `not a prefix: SCALE_REFINE carries 512 elements ...` (the cut at 8192 *is* a boundary; the prefix rule fires next) | -- |
| T-po2 (base only, completion 0, no refine) | 512 | S6B | ACCEPTED, +57 manifest bytes, 1812 of 4116 region bytes | FAILS: `GrammarError: need 2048 bits for 512 elements of 4 bits, the plane holds 0` |
| T-C3 (base + full completion, no refine) | 512 | S6B | ACCEPTED, +58 manifest bytes, 3860 of 4116 | FAILS: same line |

So the obstacles, in the order they are met:

1. **D5 plane order.** `ALPHABET, DESCENDANT, BODY, SCALE_BASE, COMPLETION,
   DIAG_SU, DIAG_SV, SCALE_REFINE, RELEASE` was forced by the S6b terminal
   classes, where the refinement plane was optional. On the LUT plane the
   `SCALE_REFINE` slot *is* the scale index and nothing decodes without it;
   on CHANNEL the row scale rides `DIAG_SV`, also after COMPLETION. No
   terminal can shorten the completion axis and keep the scales.
2. **COMPLETION is cut by superblock, not by depth.** The plane packs one
   `c`-bit word per position (`pack_body(unit.completion_bits, widths)`), and
   a shallower reading is the top bits of each word --
   `decode.reconstruct_unit` shifts them (`decode.py:893`). A byte prefix of
   the plane is superblock 0 at full depth, which is not a rate rung of
   anything. The embedded rate axis is real (`tests/test_completion_rate_axis.py`)
   and it is not a prefix property of the wire.
3. **The reader sizes SCALE_BASE / SCALE_REFINE / BODY from the geometry**
   (`unit_artifact.py:869`, `:876`, `:765`), which is what the audit named. It is
   reached only by the S6B classes, which no default recipe writes.

## 2. What the ladder would buy

The reading a shorter terminal would carry already exists:
`read_unit_artifact` / `reconstruct_unit(..., completion=c)` yields every
shallower depth from a *full* artifact. The ladder adds exactly one thing:
**fewer bytes on disk for that reading**, at +57-58 manifest bytes per extra
terminal record (a 32-byte digest plus the count array).

What the shallower reading is worth was measured once, in weight space, on a
real GLM-5.3 expert (`experiments/tessera_embedded_ladder.py`; result file
outside every checkout, `/mnt/shared/codex-ts87-verify-07c2290/experiments/results/tessera_embedded_ladder.json`;
not re-run here; a screen, not a served number), under the E2M1x2 **TCQ**
body the recipe table has since left below the cap: a truncated reading costs
**1.03-1.28x** rel_err over an encode made at that depth (exactly 1.000x at
the written depth), and on E4M3 the same trick collapses to **8.74x** because
the scalar forest's subtrees span 370:1 in magnitude. Choosing the rung to
cover a band costs up to 2.15x at the low end. So even where the axis exists
the honest byte-saving is bounded by "one encode covering [R/2, 3.5] bpp at
<=1.33x" on the 4-bit grid and nothing on the 8-bit one.

On the recipe table as written today the ladder buys **zero**: no default
rung has a completion axis, and the one grid that could carry one is in no
menu.

## 3. The decision (Rob's)

| option | what changes | what it costs | what it buys |
|---|---|---|---|
| **(a) retire the ladder** | `Manifest.terminals` stays a tuple (wire unchanged, minor unchanged); `layout.build_terminal`'s per-terminal digest, `container.parse`'s terminal match and `tests/conftest.py::make_artifact`'s three rungs become pins of a capability nobody plans; the schema's §3c and D5 rationale are reworded as history | prose plus the removal of `tests/test_review_1a.py`'s F8/F9 truncation tests or their re-labelling as layout-only | an honest release note: "Tessera artifacts have one legal length" |
| **(b) plan it** | (i) reorder D5 so the scale index / row scale precede COMPLETION; (ii) lay COMPLETION out by depth level (a plane per level, or bit-plane packing) so a rung is a byte prefix; (iii) make `parse_unit_artifact` read every plane at the terminal's counts; (iv) have the exporter write the ladder | every manifest byte moves (schema minor bump, `encoder_profile_id` unchanged but every fixture digest changes); the exporter must stop defaulting `completion=0` on TCQ; the kernel lane and `lane_planes` re-index `plane_elements` | bytes-on-disk for shallower readings on the 4-bit TCQ body only, at the 1.03-1.28x penalty above -- and nothing on the window bodies every live rung uses |
| **(c) leave it** | nothing | the prose says "one legal length", the pin says why; the capability stays in the layout at no byte cost | nothing, and no release claim about truncation |

Nothing here is priced against a served KL; the ladder has no served
number and this report does not manufacture one. If (b) is ever chosen, the
first measurement owed is a served A/B of "truncated reading of a deep
encode" against "an encode at that rate", on a rung the recipe table
actually writes -- which today means bringing the TCQ body back below the
cap first.
