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
| tensor-product grid vs a free 256-point 2D codebook | **1.116x** at identical bpp | **the live lever** |
| rotation (Hadamard, in or out) | 1.005x, and *negative* on the grid arms | dead, third independent confirmation |

## The redundancy bit cannot be punctured

`trellis.py` states the construction: `|A_R| = 2^(R+1)` anchors, four subsets, a
rate-1/2 code, `R = 1 + (R-1)`. Every member of the Ungerboeck family has the
same property -- a rate-2/3 code gives eight subsets of `2^(R-2)` and spends
`2 + (R-2) = R` on a `2^(R+1)` book. **Exactly one bit, at every code rate.**
Fractional redundancy is not reachable by changing the convolutional code; it
needs the alphabet to stop being fixed, which is a different format.

Confirmed empirically: `tessera_redundancy_exchange.py` sweeps p = 3..8 and the
trellis only beats nearest-neighbour at p = 7. Below the cap the forest's
kd-tree anchors lose to medoid selection by 1.11-1.21x -- a separate defect of
the sub-cap rungs, which the allocator does use.

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
column; GPTQ propagates across columns. The axes are orthogonal, so the two
compose directly: quantise column j with the existing Viterbi, then push the
residual into columns > j through the LDL factor, which is exactly EXL3's own
structure. That is the next lever, and it is bigger than anything in the table
above.

## What is closed

Puncturing, fractional redundancy, higher-rate convolutional codes, trellis
memory, rotation, and learned scalar spacings. Do not re-open without new
evidence; each has a measurement above.
