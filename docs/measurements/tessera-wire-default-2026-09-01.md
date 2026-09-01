# The new default wire, built: span-2 trellis over a LUT scale plane

**Date:** 2026-09-01 (late evening). **Script:**
`experiments/tessera_wire_default_check.py` → `experiments/results/tessera_wire_default_check.{json,log}`.
**Encoder:** the production `encode_unit` in `src/tessera/`, not an
experiment harness. **Tensors:** six GLM-5.3-Flash routed experts (L5/L20/L42
gate/up, 2048×4096), full width. **Eval:** the last 1024 rows of the 16-doc
pread capture (`glm53-bf16-pread-capture-1469b9b-20260901`), the served
NVFP4 activation quantiser fitted on the other rows. **Metric:** output-space
weight leg `‖x_ev·(Ŵ−W)ᵀ‖/‖x_ev·Wᵀ‖`; W4A4 = both legs served;
EXL3@A4 = √(0.05653² + act²) per tensor.

## What changed on the wire

Schema minor 1 (`docs/schema/prismaquant.tessera.v1.md` §1a):

- **Span-2 trellis** (Wei 1987 multidimensional partition): one select bit
  per two trellis positions; the second position stores a 2-bit subset label
  ahead of its point bits; the first position's label is derived. Body rate
  at the E2M1×2 cap: 3.75 b/wt instead of 3.5.
- **LUT scale plane:** the SCALE_BASE plane is gone; the SCALE_REFINE nibble
  per 16 weights indexes a per-unit table of sixteen distinct E4M3 bytes
  carried in the manifest, times a power-of-two fp32 global. 0.25 bpp instead
  of 0.5. Decode is still E2M1 codes × one E4M3 per 16 × fp32 global: the
  NVFP4 tile, unchanged.
- Same 4.0 bpp at `E2M1_K2_R896`. Arity-1 families gain 0.25 bpp per rung
  (the stored labels cost more than the plane saves at one weight per
  position).

Both fields travel in the manifest and are bound into `encoder_profile_id`.
Span-1 S6b units still serialise as minor-0 artifacts, byte-identical to the
2026-09-01 export (checked on two of its units, one of them the 317 MB
`lm_head`).

## Result (mean over six tensors)

| arm | bpp | out-space weight leg | vs today's default | W4A4 | vs EXL3@A4 | encode s |
|---|---:|---:|---:|---:|---:|---:|
| span 1, S6b, refit 4 (today's default, `61df165`) | 4.00 | 0.09055 | 1.000× | 0.12507 | 1.205× | 1.5 |
| **span 2, LUT, refit 4 (new default)** | **4.00** | **0.08044** | **1.125×** | **0.11802** | **1.137×** | 2.1 |
| span 2, LUT, refit 6 | 4.00 | 0.08044 | 1.125× | 0.11802 | 1.137× | 3.0 |
| span 1, LUT, refit 4 (plane alone) | 3.75 | 0.08915 | 1.015× | 0.12408 | 1.195× | 2.2 |
| span 2, S6b, refit 4 (trellis alone) | 4.25 | 0.08221 | 1.101× | 0.11920 | 1.148× | 1.3 |
| span 1, S6b, refit 0 (the 151 GiB export) | 4.00 | 0.09819 | 0.922× | 0.13063 | 1.258× | 0.4 |

Per tensor the new default is 1.121–1.129× over today's default; no tensor
below 1.12×.

## Three facts

1. **The cold start converges.** The experiment warm-started the LUT from a
   full per-16 refit encode (1.111× vs the flat-E4M3 reference, ≈1.13×
   composed vs S6b). Production cold-starts the table from the amax targets
   and alternates four times: 1.125× measured, and two more rounds change
   the fourth digit. No warm pass is needed; encode cost is +40%.
2. **Attribution.** The plane alone is 1.015× at 3.75 bpp (it is the
   flat-E4M3 vs S6b gap, cashed for free); the trellis alone is 1.101× at
   4.25. The two compose to 1.125× at 4.0 because the plane's saving pays for
   the trellis's bits.
3. **The export on disk is 1.22× behind the new default** (refit 0, span 1,
   S6b). Re-draining is the whole gain; the merge would be a new artifact.

## Evidence class

Six tensors, output-space weight leg with the served activation quantiser,
1024 held-out rows: a screen (principle 3). Default-on is authorised for an
unshipped format ("not opt-in. default."); promotion needs the served A/B on
the served-KL harness, queued behind the kernel lane's span-2 decode.

## What is not in this number

- **LDLQ.** Out-of-document σ=1.0 is 1.08–1.11× on top of the refit plane
  (`tessera_ldlq_generalisation.json`: 1.081×/1.105× on the two held-out
  folds, 1.52× on the adjacent-halves control that flattered every earlier
  figure); it stacks with per-32/L=2 at 1.119–1.143×. Not wired into the
  encoder; needs the ≥64k-token capture first.
- **The kernel lane.** `pack_kernel_planes` refuses span 2; the Triton
  decode reads one select bit and R−1 point bits per position. The span-2
  decode (one select per pair, a stored label, a derived label) is the next
  item. The LUT plane needs no kernel change: it materialises to the same
  per-16 E4M3 bytes the kernel already reads.
