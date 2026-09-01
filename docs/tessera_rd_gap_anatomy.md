# Where Tessera's 4-bit error actually goes

*Measured 2026-09-01 on GLM-5.3-Flash expert weights (layers 20, 42 x gate/up,
2048x1024 slices), group-32 amax scaling, rung 7. Raw numbers in
`experiments/results/tessera_{redundancy_exchange,overhead_budget,
free_codebook_trellis,memory_and_codebook}.json`.*

## The budget at 4.0 bpp

    Tessera rung 7    3.500 payload + 0.500 scale planes   = 4.000 bpp
    EXL3 K=4          4.000 payload + 0.0117 rank-1 diag   = 4.0117 bpp

Two overhead terms of almost identical size, not one. The redundancy bit gets
the attention because `cap = payload_bits - 1` is written into the grammar; the
scale planes are the same 0.5 bpp and had never been priced.

## What each term is worth

| term | measured | verdict |
|---|---|---|
| the redundancy bit (trellis vs NN, matched rate, at the cap) | **1.134x** | fairly priced; 0.5 bpp is worth 1.161x at the RD slope, so it roughly breaks even |
| group-32 scale planes vs a rank-1 row x col fit | **1.140x** for 0.477 bpp | also fairly priced (1.155x at the slope). Not a lever. |
| trellis memory 6 -> 10 | **1.010x** | free in bits AND in decode, and nearly empty. Closed. |
| tensor-product grid vs a free 256-point 2D codebook | **1.106x** at identical bpp | **the live lever**, replicated |
| rotation (Hadamard, in or out) | 1.005x, and *negative* on the grid arms | dead, third independent confirmation |

## The redundancy bit CAN be split -- just not by changing the code

`trellis.py` states the construction: `|A_R| = 2^(R+1)` anchors, four subsets, a
rate-1/2 code, `R = 1 + (R-1)`. Every member of the per-position Ungerboeck
family has the same property -- a rate-2/3 code gives eight subsets of
`2^(R-2)` and spends `2 + (R-2) = R` on a `2^(R+1)` book. Exactly one bit, at
every rate. **Puncturing the convolutional code is genuinely closed.**

*An earlier revision of this file concluded from that proof that fractional
redundancy needs a different format. That was wrong, and the proof does not
support it: it constrains the code, not the partition.*

Multidimensional set-partitioning (Wei 1987, the construction V.34 modems use
to pay half a bit of constellation expansion instead of a whole one) splits the
bit with the alphabet fixed. Group L positions, label the super-symbol by the
SUM of its positions' subset labels mod 4, and spend one conv-code bit on the
super-label:

    bits/super-symbol = 1 + 2(L-1) + 6L   over 2L weights
    payload/weight    = (8L - 1) / 2L = 4 - 1/(2L)

Same asymptote as k-tupling, at a completely different price. k-tupling reaches
`4 - 1/k` by building a `G^k` codebook, so the Viterbi scores 16x more anchors
per step. This reaches `4 - 1/(2L)` with the codebook untouched at 256, because
the min-plus convolution over Z/4 is associative: L positions combine in a
binary tree of 4x4 minima, **linear in L**.

Measured, and the test is the frontier rather than the baseline -- L=2 spends
0.25 bpp more, so beating rung 7 outright would prove nothing. Against the
log-linear interpolation between rung 7 and the zero-redundancy NN point:

| L | payload | bpp | E2M1x2 | vs frontier | Lloyd | vs frontier |
|---|---|---|---|---|---|---|
| 1 | 3.5000 | 4.000 | 0.10812 | - | 0.09732 | - |
| 2 | 3.7500 | 4.250 | 0.09901 | **1.032x above** | 0.08530 | **1.036x above** |
| 4 | 3.8750 | 4.375 | 0.09688 | 1.026x above | 0.08165 | 1.031x above |
| 8 | 3.9375 | 4.438 | 0.09664 | 1.014x above | 0.08060 | 1.020x above |
| 0 | 4.0000 | 4.500 | 0.09663 | - | 0.08022 | - |

Every L is above the frontier, so these are real rungs, not re-labelled ones --
and they land in 4.0-4.5 bpp, a range where the E2M1x2 family previously had
nothing at all. The gain is modest (1.03x) and decays with L exactly as the
shaping gain vanishes with the redundancy, which is the consistency check: at
L=8 the trellis has all but converged on the untrellised NN point.

Separately, `tessera_redundancy_exchange.py` sweeps p = 3..8 and the trellis
only beats nearest-neighbour at p = 7. Below the cap the forest's kd-tree
anchors lose to medoid selection by 1.11-1.21x -- a defect of the sub-cap rungs,
which the allocator does use.

## The product grid is the one real lever

`tuple_grid(E2M1_GRID, 2)` crosses sixteen scalar levels with themselves, so it
spends codes on the corners of a square while the weight density is a round
blob. Re-spacing the scalar ladder cannot fix that, which is why the earlier
learned-values experiment came back 1.006x and then inverted to 0.990x -- it
learned the levels and kept the cross.

A free per-tensor Lloyd codebook, partitioned by Ungerboeck's *criterion*
(greedy 4-colouring: each centroid into the subset whose nearest same-subset
member is furthest) rather than his *formula*, is worth **1.106x at identical
bpp** with the greedy partition still crude.

**Replicated**, which for this lever is not optional -- the learned-*values*
lever measured 1.006x on one tensor and inverted to 0.990x across eight, and a
mean over a few slices is a screen. 33 expert tensors at **full width** (11
layers spread 3..43 x gate/up/down): min 1.091x, median 1.107x, max 1.116x,
geometric mean 1.106x, spread 2.3%, and **zero tensors where the free codebook
loses**. Flat across projections -- gate 1.108x, up 1.107x, down 1.105x -- so it
holds on `down_proj`, whose post-SwiGLU input statistics differ from the rest
and which nothing else in this document tested.

It changes no plane and no width -- the payload still carries codes and the
decode is still a 256-entry table lookup -- but it **does** change the
container, and an earlier revision of this file wrongly said it did not. A
shipped artifact names its grid by digest (`"name": "E2M1x2", "digest":
"514a72..."`) and the reader rebuilds the values from that name; it stores no
values. A per-tensor codebook must be carried in the artifact and read by the
decoder: a new field and a `container_version` bump. The partition is wire on
the same argument -- the decoder replays the subset structure, so `colour4`'s
greedy colouring would have to become a specified, deterministic object rather
than a heuristic. Storage is cheap (256 x 2 x fp16 = 8 KB per tensor, ~0.0005
bpp); the container decision is not, and it is unmade.

**Readiness:** none of this exists in `tessera/` -- every number above came from
a standalone Viterbi in `experiments/`, validated against `encode_unit` to
0.48%. On the promotion ladder this is Research. It reaches Candidate only after
the encoder change, the container decision, and a serving lane, and Tessera's
`route_status` is still `unbacked`.

## The comparison to EXL3 is not format-vs-format

`experiments/exl3_rate_sweep.py:82` fits EXL3 with a real Hessian
(`H = x_fit.T @ x_fit`) and LDL-ordered error compensation, and scores
`(x @ w_hat - ref).norm() / ref.norm()` -- **held-out activation error on a
compensated fit**. Tessera's arms are **uncompensated weight error**.
`exl3_arm_glm_experts_v2.py:35` records this as "asymmetry, stated not netted".

So the standing 1.72x/1.90x is a mechanism gap as much as a format gap, and the
sanity check agrees: EXL3's 0.05653 sits *below* the i.i.d. Gaussian
rate-distortion bound at 4 bits (2^-4 = 0.0625), which a memoryless coder of any
alphabet cannot do. Error feedback against a Hessian can, because it is no
longer coding the weights -- it is coding the output.

Tessera has no error compensation at all. Its Viterbi runs down rows within a
column; GPTQ propagates across columns. The axes are orthogonal, so the trellis
drops into the GPTQ loop unchanged -- and it works, but only under heavy
damping:

| arm (cols=64, cond(H)=1.8e2) | held-out | in-sample | gap |
|---|---|---|---|
| E2M1x2 uncompensated | 0.10828 | 0.10810 | 1.00 |
| E2M1x2 GPTQ damp=0.01 | 0.11575 | 0.08810 | **1.31** |
| E2M1x2 GPTQ damp=0.1 | 0.11081 | 0.08849 | 1.25 |
| E2M1x2 GPTQ damp=1.0 | **0.10313** | 0.09351 | 1.10 |
| Lloyd GPTQ damp=1.0 | **0.09251** | 0.08342 | 1.11 |

damp=1.0 is the house default (`PRISMAQUANT_GPTQ_DAMP`, sweep off since
2026-06-12) and it is the only setting that generalises. At 0.01 the fit
improves 23% in-sample while *degrading* 7% held-out.

**And that reframes the target.** The probe's cached activations hold 256
tokens. `exl3_rate_sweep.py` splits them 128/128 and builds a Hessian over 4096
input features -- rank 128 of 4096, minimum eigenvalue negative, condition
number 9.0e7 at 1024 features before damping. EXL3 was fitted through that with
`sigma_reg 0.025`, in the regularisation range where our own compensation
overfits by 1.25-1.32x, and scored on 128 "held-out" tokens that are adjacent
tokens of the same short probe. The 1.72x/1.90x standing gap is therefore **not
a measured format gap** and should not be used as a target until it is re-run on
calibration with more tokens than features. That is the blocked item; nothing in
this repo currently has enough activations to do it.

## What is closed

Puncturing, fractional redundancy, higher-rate convolutional codes, trellis
memory, rotation, and learned scalar spacings. Do not re-open without new
evidence; each has a measurement above.

Added 2026-09-01 under the FP4-native constraint
(`docs/measurements/tessera-fp4-native-levers-2026-09-01.md`): deleting or
thinning the scale plane (rank-1 field 0.77–0.83×, E8M0-only 0.84×), the
global headroom multiplier, group-local LDLQ. What opened: the plane's
VALUES (LS refit, 1.084×, default-on), Wei L=2 (+0.25 bpp, 1.104×), and the
LDLQ regulariser (σ=1.0, 1.137× — a screen).


## Serving cost of a per-unit grid, and why it is now 0.047%

The kernel lane was already grid-agnostic -- `build_tuple_value_lut` reads
reconstructions out of `grid.vector`, never an E2M1 table, and no path in this
repo has ever used a hardware FP4 conversion instruction. So a learned codebook
costs nothing arithmetically: same table shape, same load count, same inner
loop, different numbers.

The cost was memory. That fused table folds a **shared** structure -- which
anchor a `(window, point)` lands on -- together with a **per-unit** meaning --
what that anchor reconstructs to. While the grid is global the fold is free,
because every unit at a given rate shares one 64 KB table. Give each unit its
own grid and it becomes per-unit: 37,694 x 64 KB = **2.301 GiB** resident, 1.52%
of the body, spent buying back bits the format had just saved.

`build_tuple_index_lut` and `build_anchor_values` are that table split along the
seam, and the split is exact because the fused form is now *defined* as their
composition (`test_the_split_lookup_composes_back_to_the_fused_table` pins it):

| | shared | per unit | 37,694 units |
|---|---|---|---|
| fused | - | 64 KB | 2.301 GiB |
| split | 16 KB | 2 KB | **73.6 MiB** |

**2.229 GiB saved, 1.52% -> 0.047% of the body**, for one extra dependent load
per output row -- and `window * POINTS + pt` does not depend on `a`, so that
first load is per *code* and the arity rows of a code hit one address. The hot
working set also falls from 64 KB to 18 KB, which should help the cache rather
than hurt it, though that is a prediction and not a measurement.

**Still open:** the scalar lane (`build_value_lut`, `tessera_gemm`) folds the
same seam and would need the same split before a learned *scalar* grid ships.
Nothing needs it today, since the shipped family is arity 2.

**What a learned grid does foreclose:** putting Tessera codes in an NVFP4
container and letting a stock vLLM kernel dequantize them. A stock FP4 kernel
cannot read a learned table. That route was never built (`route_status:
unbacked`) and it was never attractive -- materialising to NVFP4 costs 0.5 bpp
of resident footprint, and plain nearest-neighbour on the grid at 4.5 bpp
(0.09663) already beats Tessera's trellis at 4.0 (0.10812), so anything that
materialises to a wider format erases the format's whole reason to exist. But it
does commit Tessera to shipping its own kernel permanently.
