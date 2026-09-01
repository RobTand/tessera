# Tessera handover, 2026-09-01 evening

Written on 2026-09-01 by the session that was asked to take a handover of this
name and found that no such file existed. Everything below was re-verified
against the code and the running processes, not inherited from prose. Where a
previous claim failed verification, this document says so and gives the check.

## Read this first

The handover you were promised was never written. The previous session left
three docs and three running jobs instead. The docs disagree with each other and
with `HEAD`, because two of them were written before the commits that refuted
them.

Trust order for this project, highest first: the running code, then the
measured bytes on disk, then this file, then `docs/status/`, then
`docs/measurements/`.

## Jobs that were running when this file was written

Three processes from the previous session were still running, unattended. None
of them were started by the session that wrote this file.

| PID | Job | Started | State at handover |
|---|---|---|---|
| 2428662 | `export_glm53_tessera.py --shards 1-60` -> `...-partA` | 07:13 | 22 of 60 shards, about 3.5 min/shard |
| (other box) | shards 61-120 -> `...-partB` | 07:15 | 29 of 60 shards |
| 2530035 | `glm53_full_body_cost.py --menu NVFP4,FP8_E4M3 --a-side none` | 08:22 | 78 units |
| 2532256 | `glm53_full_body_cost.py --menu TESSERA_E4M3_K1_R{896,1024,1152,1280} --a-side none` | 08:25 | 20 units |

Both boxes write to `/mnt/shared/models/`. The two halves merge with the script
from `a5a6ebe`. The merge step must reconcile the final byte count against the
156.743 GiB allocation from `555de50` and explain any difference; do not assume
the difference is the vision and MTP handling without checking.

## The export is valid, and that was measured

The export started at 07:13. Three encoder and decoder fixes landed after it:
`a96064b` (07:43), `eec18ba` (07:55). The commit messages assert that top-rung
bytes are unaffected. That assertion was load-bearing on 26 GB of already-written
shards and was written by the session under audit, so it was re-checked from
scratch:

1. Every one of the 37,694 tensors in
   `glm53_tessera_plan.json` is at `q256=896`, which is `E2M1x2`'s cap. At the
   cap `completion_capacity` is 0, so all three bugs are structurally inert.
2. A worktree at `552c53a` — the commit the export process actually loaded —
   encodes four shapes at `q256=896` to bytes that are sha-identical to `HEAD`.
3. Three units read out of `model-00003-of-00120.safetensors` in `partA`,
   written at 07:25 by pre-fix code, are sha-identical to a `HEAD` re-encode of
   the same source tensors.

Check 3 is the one that matters: it compares bytes on disk against current code,
not code against code. **Let both exports finish.**

## The trellis grids and their data ranges hold up

Rob raised the concern that the previous session was unaware of the trellis
format specifications and their data ranges. The concern does not reproduce at
the encoder level. Four checks, each citing code:

- `alphabet.py::GROUP_SCALED_SOURCE` models the source the alphabet sees as
  amax-normalised and **bounded by the grid peak**, not as a Gaussian of any
  sigma. Its docstring names the exact failure mode the concern describes and
  records it as found and fixed: "That is the whole of TESSERA-8's sub-cap
  collapse."
- `encode.py:353` threads `peak=max(abs(v) for v in grid.values)` from the grid
  into the scale packer, giving 6.0 for `E2M1` and 448.0 for `E4M3`. The 6.0 in
  `_pack_scales`'s signature is a parameter default that no caller takes.
- The scale granularity agrees with the source model. `_pack_scales` refines per
  half of 16 weights, and `GROUP_SCALED_SOURCE` uses a group of 16.
- `build_forest` does not assert which partition suits a log-spaced grid. It
  builds both the contiguous and the mass-balanced partition, scores both
  against the source, and takes the cheaper one.

Measured anchors confirm it. At rate 1, `E4M3`'s anchors are the values
±128 and ±288 against a peak of 448, which is 0.29 and 0.64 of range; `E2M1`'s
are ±1.5 and ±4 against a peak of 6.0, which is 0.25 and 0.67. The two ladders
track within 2 to 4 percent at matched bits per parameter, which is what the
rate grid shows.

## Where the real defect is: serving claims without attestation

The uncommitted change to `experiments/glm53_full_body_cost.py` put four
`TESSERA_E4M3` rungs on the GLM allocator menu and described their serving
behaviour in a comment. Two claims in it were invented:

- `E4M3 -> FP8_E4M3 (sm89+, W8A8)`. No runtime published this. The only
  materialiser in `src/tessera/` is `decode.py::materialize_nvfp4`.
  `tessera/fp8.py` is the S6b scale-word codec, not a weight-tensor path, so no
  `E4M3` artifact can become a stock tensor.
- "AQUA prices A4 at ~11.7x A8 per unit." The string `11.7` occurs exactly once
  across both repositories: in that comment.

This is what CLAUDE.md principle 14 forbids. The claims are withdrawn in the
working tree rather than softened, and the comment now records the gap instead.

Pricing an unbacked rung is still legitimate under principle 1: an allocator
that wants an unbacked route is reporting a serving gap. The gate belongs at
export and allocation, not at measurement, so **the two scoring runs were left
alone.** Anything that consumes `glm53_full_body_cost_T8.json` must mark the
`E4M3` rungs unexportable.

## The uncommitted calculator change is a no-op with a false rationale

The working tree carried a change to `calculator.py::terminal_rate` whose
comment claimed the function "returned the family's cap at every rung". It does
not, and it did not:

- `terminal_rate` at `552c53a`, at committed `HEAD`, and with the change applied
  all return identical values.
- A sweep of 520 configurations — caps 3 and 7, arities 1 and 2, every rung at
  stride 64, completion depths 0, 1, 2 and full, with and without the refinement
  plane — found zero differences between committed `HEAD` and the change.

`build_terminal` recomputes the exact byte count from `spec`, so the plane
extent never reaches the returned value. The code change is kept because passing
`spec` removes a latent divergence between the two accountants, and the comment
now says that instead of claiming a fix.

Against measured bytes, `terminal_rate` is correct: 22 of 22 arity-1 rungs match
the built artifacts, across `E2M1_K1` from 1.5 to 3.5 bpp and `E4M3_K1` from 1.5
to 7.5 bpp.

## One footgun worth knowing

`terminal_rate` takes `q256` per **code**; `encode_linear`, the exporter, and
prismaquant's `artifact_bpp` take it per **position**. At arity 2 the two differ
by a factor of 2, and passing the exporter's 896 to the calculator returns 2.25
against a true 4.00 — a 44 percent underprice, in the direction that busts a
byte budget silently.

This is known and guarded, not live.
`tests/test_tessera_formats.py::test_the_bpp_formula_agrees_with_tesseras_exact_byte_accountant`
applies the arity factor explicitly and checks both arities; the `arity != 1`
skip that once hid it is gone. The guard passes, 60 of 60, against the working
tree. The parameter name is still the same on both sides of the boundary.

## What is stale in the other documents

`docs/status/2026-09-01-where-tessera-stands.md` was written at 05:51 and is
refuted by commits made at 07:43 and 08:08. Its section "The rate axis is two
points, not a band" is false on `HEAD`, and so is the claim that the
serialisable set is two sizes with "nothing above 4.0". The corrected picture:

| Family | Ladder, at completion 0 | Top rung |
|---|---|---|
| `E2M1_K1` | 1.5 to 3.5 bpp | 3.5000 |
| `E2M1_K2` | 1.0 to 4.0 bpp | 4.0000 |
| `E4M3_K1` | 1.5 to 7.5 bpp | 7.5000 |

The `E4M3` ladder saturates at a relative error of 0.0397 from 6.5 bpp upward,
which is the floor of `E4M3` as a reconstruction alphabet, not a coder deficit.

`docs/measurements/tessera-rate-ceiling-2026-09-01.md` already carries a
superseding banner and is safe to read.

Still true, and re-verified: prismaquant cannot build or export a Tessera
allocation. `export_native_compressed.py` has zero Tessera references,
`tessera_allocator.py` sets `producer_eligible: False` in five places, and
`tessera_render.py:69` sets `_TESSERA_SERVING_LANE_EXISTS = False`. The export
now running is Tessera's own exporter writing `.tessera` blobs into safetensors,
which is a different thing from a served artifact.

## What to do next

1. Let the two exports finish, then run the merge from `a5a6ebe` and reconcile
   the byte count against 156.743 GiB.
2. Let the two scoring runs finish. Mark the `E4M3` rungs unexportable wherever
   their numbers are consumed.
3. Decide the `E4M3` serving question before any allocation spends those rungs:
   either build the materialiser and attest the route, or drop them from the
   producer menu.
4. The standing quality blocker is unchanged. Tessera loses 1.72x to EXL3 at
   matched bits per weight on GLM routed experts, and the gap is a
   rate-distortion slope, not a missing calibration pass.
