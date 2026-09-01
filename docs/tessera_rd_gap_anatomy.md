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
member is furthest) rather than his *formula*, is worth **1.116x at identical
bpp** with the greedy partition still crude. It changes no plane, no width and
no container: the payload still carries codes, the decode is still a 256-entry
table lookup. It costs 256 x 2 x fp16 = 8 KB per tensor, ~0.0005 bpp.

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
