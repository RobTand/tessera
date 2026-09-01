# Tessera-4 under the FP4-native constraint: every lever, priced on the real plane

**Date:** 2026-09-01. **Commits:** `cf82b00` (scale refit, default-on),
`61df165` (trailing refit, default four passes). **Scripts:**
`experiments/tessera_fp4_native_levers.py` (the battery),
`experiments/tessera_rank1_plane_multidim.py`,
`experiments/tessera_plane_alternatives.py`; every table below is rendered by
`experiments/render_fp4_native_tables.py` from the JSONs in
`experiments/results/` — no number here is typed by hand except the schedule
and encode-cost tables, which say so.

## The constraint

Rob, 2026-09-01: *"We want a format that can natively use nvidia's 4-bit
hardware."* At the instruction level Blackwell's FP4 MMA consumes **E2M1 bit
patterns × one E4M3 scale per 16 consecutive K elements**. Tessera-4's decoded
tile already satisfies it: the codes are E2M1 and the S6b two-tier plane
(E8M0 per 32 + a 4-bit `(d, m)` refinement per 16) decodes to
`2^(E−127+d)(1+m/8)`, which is E4M3-representable and is relabelled exactly by
`wire.nvfp4_scale_bytes` behind a power-of-two global. The measured per-tensor
scale range on these experts is 2.85–3.32 octaves, inside E4M3's window.

So the question is not "can Tessera run on the FP4 path" but "which levers
survive it". **Out:** the learned codebook (`7562fcb`, kernel-lane research —
arbitrary floats in the tile). **In:** the trellis and its partition, every
choice of scale *value*, encoder-side search, and anything the exporter can
price exactly as the MMA will multiply it.

## Setup

Six GLM-5.3-Flash routed experts (layers 5/20/42, `gate_proj`/`up_proj`,
2048×4096), the K-grouped S6b plane the artifact ships (every earlier
experiment used an N-grouped fp32 amax plane — self-consistent, not the
format). Activation cache: 256 tokens per layer, the first 128 fit any
activation-aware arm, the second 128 are held out and score everything.
Columns: weight-space relative error; **output-space weight leg** on the
held-out tokens (the ranking column); W4A4 composite under the served
activation quantiser (`nvfp4_activation_qdq_served` + the global-scale MSE
grid from `/home/rob/prismaquant`); and the ratio to **EXL3@A4 projected**:
EXL3 K=4's weight leg 0.05653 (`exl3-head-to-head-2026-09-01.md`) added in
quadrature to the served activation leg 0.08572 → 0.10288.

The battery's Viterbi starts from state 0, as the decoder replays. A first
pass of the harness started free (`metric = zeros`), found unencodable paths
and read ~0.3% SSE optimistic on every arm, uniformly; that run is kept as
`*_freestart.json`. Pinned to the shipping encoder: harness SSE vs
`encode_unit` on L5 gate, refit 0: 37.5662 vs 37.5661; refit 3: 31.7257 vs
31.7246 (equal-cost ties account for the rest).

## P — the plane's rule and format

| arm | bpp | weight-space | out-space weight leg | vs baseline | min–max | W4A4 (served A4) | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| artifact plane (`_pack_scales`, floor mantissa) | 4.000 | 0.10782 | 0.09736 | 1.000× | 1.000–1.000 | 0.12934 | 1.253× |
| headroom 0.97 (best of the sweep) | 4.000 | 0.10731 | 0.09673 | 1.007× | 1.003–1.021 | 0.12905 | 1.249× |
| nearest mantissa, same words | 4.000 | 0.10357 | 0.09335 | 1.043× | 1.041–1.045 | 0.12659 | 1.226× |
| flat E4M3 per 16 (same 0.5 bpp) | 4.000 | 0.10384 | 0.09355 | 1.042× | 1.034–1.069 | 0.12671 | 1.226× |
| E8M0 per 32 only (best threshold) | 3.750 | 0.12874 | 0.11654 | 0.837× | 0.819–0.863 | 0.14500 | 1.403× |

The artifact's plane rounds the mantissa *down* to keep the group inside range
and clips ~6% of elements; nearest rounding is worth 4.3% by itself. The global
headroom multiplier (0.89–1.20 swept) stays closed. The E8M0-only plane
(MXFP4's) is 0.84× — the refinement bits are load-bearing.

## R — refitting the plane's values

| arm | bpp | weight-space | out-space weight leg | vs baseline | min–max | W4A4 (served A4) | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| artifact plane | 4.000 | 0.10782 | 0.09736 | 1.000× | 1.000–1.000 | 0.12934 | 1.253× |
| LS refit ×1 → S6b | 4.000 | 0.10137 | 0.09161 | 1.063× | 1.058–1.069 | 0.12519 | 1.212× |
| LS refit ×3 → S6b (`cf82b00` default) | 4.000 | 0.09948 | 0.08997 | 1.082× | 1.076–1.089 | 0.12404 | 1.201× |
| LS refit ×5 → S6b | 4.000 | 0.09919 | 0.08969 | 1.086× | 1.079–1.093 | 0.12385 | 1.199× |
| LS refit ×5 → fp32 (unrepresentable) | — | 0.09536 | 0.08615 | 1.130× | 1.123–1.138 | 0.12139 | 1.176× |
| H16-weighted LS ×3 → S6b, σ=0.1 | 4.000 | 0.09981 | 0.08926 | 1.091× | 1.082–1.101 | 0.12352 | 1.196× |
| LS refit ×3 → flat E4M3 | 4.000 | 0.09820 | 0.08846 | 1.101× | 1.093–1.108 | 0.12292 | 1.190× |
| H16-weighted LS ×3 → flat E4M3, σ=0.1 | 4.000 | 0.09857 | 0.08804 | 1.106× | 1.101–1.116 | 0.12266 | 1.188× |
| LS refit ×3 → E8M0 only | 3.750 | 0.12874 | 0.11654 | 0.837× | 0.819–0.863 | 0.14500 | 1.403× |

In this table "×k" is the `cf82b00` schedule, k refits *between* k+1 trellis
passes (`(TR)^k T`). Per half the least-squares scale for the codes just chosen
is `<w,u>/<u,u>`, landed on the nearest S6b word by exact per-group SSE
arithmetic (monotone by construction, `_refit_scales`); the trellis then
re-runs on the new plane. The fp32 row is the ceiling the E4M3 mantissa costs
against (1.130× vs 1.086×). Weighting the LS by the 16×16 diagonal Hessian
block is +1%, needs activations, not wired.

### The schedule: end on a refit

Measured after the battery on the same six tensors and scorer (ad-hoc run,
the encoder itself at each schedule; ratios are the output-space weight leg
over the amax plane):

| schedule | Viterbi passes | vs amax plane |
|---|---:|---:|
| `T` (amax plane) | 1 | 1.000× |
| `TR` | 1 | 1.044× |
| `TRT` | 2 | 1.063× |
| `TRTR` | 2 | 1.072× |
| `TRTRT` | 3 | 1.077× |
| `TRTRTR` | 3 | 1.080× |
| `TRTRTRT` (`cf82b00` default) | 4 | 1.082× |
| **`TRTRTRTR` (`61df165` default)** | 4 | **1.084×** |

A trailing refit costs no Viterbi and is monotone, so at every pass count
ending on R beats ending on T. `61df165` makes `scale_refit=k` mean k passes
and k refits with the last one trailing; `scale_refit=0` is the amax plane
byte for byte and `scale_refit=1` is the free 4.4%.

**Encode cost** on the real export path (one 2048×4096 expert, concurrent jobs
inflating absolutes; the ratio is the claim): refit 0 / 1 / 3 = 2.25 / 4.61 /
9.11 s. The default is ~4× the amax plane's encode time. The merged
151.487 GiB GLM export was built at refit 0; shipping the refit means
re-draining.

## M — half the redundancy (Wei multidimensional partition)

| arm | bpp | weight-space | out-space weight leg | vs baseline | min–max | W4A4 (served A4) | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| artifact plane | 4.000 | 0.10782 | 0.09736 | 1.000× | 1.000–1.000 | 0.12934 | 1.253× |
| Wei L=2 on S6b (3.75 payload) | 4.250 | 0.09939 | 0.08968 | 1.086× | 1.082–1.090 | 0.12372 | 1.198× |
| Wei L=2 on S6b + H16-LS ×1 | 4.250 | 0.09212 | 0.08254 | 1.181× | 1.170–1.199 | 0.11876 | 1.150× |
| Wei L=2 on E8M0-only (4.0 bpp) | 4.000 | 0.11934 | 0.10834 | 0.901× | 0.879–0.927 | 0.13860 | 1.341× |

## Deleting the plane and spending its bits on L

`tessera_rank1_plane_multidim.py`, six LS iterations on every arm. The rank-1
field is `r[n]·c[b]` over (row, 16-column block), materialised to per-16 E4M3
at load and priced as materialised.

| arm | bpp | weight-space | out-space weight leg | vs baseline | min–max | W4A4 (served A4) | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S6b plane + LS ×6, L=1 (the shipping plane) | 4.000 | 0.09916 | 0.08966 | 1.000× | 1.000–1.000 | 0.12382 | 1.199× |
| S6b plane + LS ×6, L=2 | 4.250 | 0.08996 | 0.08127 | 1.104× | 1.098–1.114 | 0.11791 | 1.142× |
| S6b plane + LS ×6, L=4 | 4.375 | 0.08809 | 0.07964 | 1.127× | 1.121–1.136 | 0.11681 | 1.131× |
| S6b plane + LS ×6, L=8 | 4.438 | 0.08799 | 0.07946 | 1.129× | 1.123–1.135 | 0.11663 | 1.130× |
| rank-1 field (row × 16-col block), amax start, L=1 | 3.504 | 0.14209 | 0.13015 | 0.687× | 0.645–0.722 | 0.15476 | 1.502× |
| rank-1 field + LS ×6, L=1 | 3.504 | 0.12818 | 0.11616 | 0.772× | 0.756–0.797 | 0.14401 | 1.395× |
| rank-1 field + LS ×6, L=2 | 3.754 | 0.12099 | 0.10975 | 0.817× | 0.799–0.852 | 0.13891 | 1.346× |
| rank-1 field + LS ×6, L=4 | 3.879 | 0.12013 | 0.10911 | 0.822× | 0.807–0.855 | 0.13839 | 1.340× |
| rank-1 field + LS ×6, L=8 | 3.942 | 0.11942 | 0.10840 | 0.828× | 0.809–0.858 | 0.13790 | 1.335× |

The plane's 0.5 bpp buys per-column magnitude structure the rank-1 field
cannot carry: even with the saved bits re-spent on L=8 at 3.94 bpp it is
0.83×. Together with the E8M0-only rows above, "free the plane's bits for
rate" is dead in every FP4-native form. Wei L=2 on the real plane is the one
place extra rate can be spent (E2M1×2 caps the per-position rate at 3.5):
+0.25 bpp for 1.104×. *A first run of this script read 0.68–0.78× for the
refit arms — it started 6× too large (κ lacked `/PEAK`) and rounded through a
borrowed E4M3 clamp; void, archived as `*_void-bad-start.json`.*

## Plane alternatives at exactly 4.0 bpp, and the LDLQ stack

`tessera_plane_alternatives.py`, three LS iterations. `W4A4 vs` is the
composite ratio against the shipping plane.

| arm | bpp | weight-space | out-space weight leg | vs S6b+LS | act leg | W4A4 (served) | W4A4 vs | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S6b plane + LS ×3, L=1 | 4.000 | 0.09948 | 0.08997 | 1.000× | 0.08572 | 0.12404 | 1.000× | 1.201× |
| S6b plane + LS ×3, L=2 | 4.250 | 0.09019 | 0.08144 | 1.105× | 0.08572 | 0.11802 | 1.051× | 1.143× |
| E4M3 per-32 dup + LS ×3, L=1 | 3.750 | 0.10557 | 0.09515 | 0.945× | 0.08572 | 0.12776 | 0.971× | 1.237× |
| **E4M3 per-32 dup + LS ×3, L=2** | **4.000** | 0.09541 | 0.08588 | **1.047×** | 0.08572 | 0.12108 | 1.024× | 1.173× |
| full-LDLQ σ=1.0 + in-block LS ×3 (S6b) | 4.000 | 0.10325 | 0.07998 | 1.134× | 0.08572 | 0.11697 | 1.064× | 1.131× |
| full-LDLQ σ=1.0 + in-block LS ×3 (flat E4M3) | 4.000 | 0.10205 | 0.07892 | 1.149× | 0.08572 | 0.11632 | 1.070× | 1.125× |
| rank-1 per-column (fold) + LS ×3, L=1 | 3.512 | 0.13900 | 0.12668 | 0.710× | 0.08579 | 0.15222 | 0.814× | 1.475× |
| rank-1 per-column (fold) + LS ×3, L=2 | 3.762 | 0.13309 | 0.12144 | 0.741× | 0.08575 | 0.14789 | 0.838× | 1.433× |
| rank-1 per-column (fold) + LS ×3, L=8 | 3.949 | 0.13149 | 0.11982 | 0.751× | 0.08580 | 0.14658 | 0.846× | 1.420× |

Two things here are new. **One E4M3 scale per 32 (0.25 bpp, duplicated to the
two per-16 slots the MMA reads) plus Wei L=2** is exactly 4.0 bpp and beats the
shipping encoder by 4.7% at the same size — a wire change (plane container +
trellis partition), no kernel change, and every decoded tile is still E2M1 ×
per-16 E4M3. **The LDLQ stack** (σ=1.0, then the refit inside each 32-block)
is 1.134× on S6b and 1.149× on the flat E4M3 plane — the refit and the
compensation stack, mostly. The per-column rank-1 fold (a per-column scale
folded into the rows, the overhead-budget experiment's "rank-1 ≈ block plane")
is 0.71–0.75× once it has to be carried by a per-16 hardware scale.

## F — error feedback, and the regulariser

| arm | bpp | weight-space | out-space weight leg | vs baseline | min–max | W4A4 (served A4) | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| artifact plane | 4.000 | 0.10782 | 0.09736 | 1.000× | 1.000–1.000 | 0.12934 | 1.253× |
| group-LDLQ (32×32 blocks) σ=1.0 ×2 + H16-LS, S6b | 4.000 | 0.10287 | 0.08825 | 1.106× | 1.085–1.144 | 0.12267 | 1.187× |
| group-LDLQ σ=1.0 ×2 + H16-LS, flat E4M3 | 4.000 | 0.10099 | 0.08671 | 1.126× | 1.105–1.163 | 0.12164 | 1.177× |
| full-LDLQ σ=0.025 (EXL3's), S6b | 4.000 | 0.11738 | 0.09075 | 1.083× | 1.016–1.179 | 0.12443 | 1.203× |
| full-LDLQ σ=0.1, S6b | 4.000 | 0.11517 | 0.08899 | 1.104× | 1.035–1.202 | 0.12313 | 1.190× |
| full-LDLQ σ=1.0, S6b | 4.000 | 0.11151 | 0.08637 | 1.137× | 1.068–1.236 | 0.12122 | 1.172× |
| full-LDLQ σ=1.0, flat E4M3 | 4.000 | 0.10769 | 0.08346 | 1.178× | 1.106–1.285 | 0.11958 | 1.156× |

The activation-aware encoder doc inherited EXL3's `σ_reg = 0.025` and measured
1.088×; the regulariser, not the block size, was the unswept knob. Weight-space
error gets *worse* under compensation (the signature of trading weight error
for output error along the input Hessian). Group-local LDLQ (32×32 diagonal
blocks, all groups in parallel) gets ~1.10× — the gain is in the off-diagonal
structure.

**Caveat that stands on every LDLQ row:** the cache holds 256 tokens per
layer, fit and eval are adjacent halves of one probe, and the Hessian has rank
≤128 over 4096 features. Shared low-rank structure flatters generalisation.
These rows are a screen. The unblock is running: a 16-document × 512-token
capture of the same experts on lina (`glm53-bf16-pread-capture-1469b9b-20260901`,
rows in document order) so fit and eval can be split by whole documents.

## What this adds up to

| | vs EXL3@A4 | status |
|---|---:|---|
| artifact plane (floor mantissa, refit 0) | 1.253× | what the 151.487 GiB export was built with |
| LS refit, trailing, four passes (`61df165`) | ~1.199× | **default**; no wire change |
| + Wei L=2 (+0.25 bpp) | 1.142× | wire change to the trellis |
| per-32 E4M3 dup + L=2 at 4.0 bpp | 1.173× | wire change, same size |
| + full-LDLQ σ=1.0 (S6b / flat E4M3) | 1.131× / 1.125× | screen until the multi-document capture scores it |

Every row is still an FP4-native tile. The alphabet-shape lever
(`tessera-alphabet-shape-not-spacing`, 1.111×) is the only one this constraint
removes, and the plane VALUES lever recovers most of it.
