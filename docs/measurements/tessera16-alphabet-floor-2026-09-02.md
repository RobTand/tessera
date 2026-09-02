# Is a 16-bit alphabet worth building?  The window body's floor over a BF16 grid

**Date** 2026-09-02 · **Repo** `/home/rob/tessera`, encoder at `795137c`
("start every row inside the body's reach", 10:49) — **nothing under
`src/tessera/` changed while any run below was in flight**, which matters
because other agents were committing elsewhere in the tree, so `HEAD` moved and
the encoder commit is the one to quote.  `tests/test_channel_plane.py
tests/test_window_body.py tests/test_scale_refit.py`: **53 passed** on that
encoder.  · **Script**
`experiments/tessera16_alphabet_floor.py` · **Status** measurement, no library
change.

The question is the floor, not a design: **what error does the window body
reach when its alphabet is BF16 (E8M7) instead of E4M3**, at R = 4…8 bits per
weight, against E4M3 at the same rates and against EXL3 K = 4, 5, 6, 8 on the
same rows.  A 16-bit alphabet is not serialisable on today's wire — the
ALPHABET plane is one byte per table entry and `SERIALISABLE_GRIDS` is closed
(E2M1, E2M1², E4M3) — so every BF16 arm here is measured **in memory** and
priced as if the table carried a second byte.  That pricing is the honest one
for the A16 question the allocator would ask.

## Verdict

**Below R = 5 the alphabet is not the constraint** — the BF16 window matches
E4M3 to within 1% on six GLM experts *and* on 196 dense Qwen Linears, and pays
0.016–0.058 bpp of extra table for the privilege, so a 16-bit alphabet is worth
nothing at 4 bits.  **From R = 6 the E4M3 table binds**: 202 distinct values at
L = 14 is a 7.66-bit ceiling, the window body saturates against it (0.02309 →
0.02218 → 0.02217 on L5.gate_proj), and at 8 bits it ties or loses to plain
per-channel FP8 rounding — while the identical body over a BF16 grid keeps
halving per bit, lands on EXL3 K8 on both R = 8 tensors (0.00512 vs 0.00511,
0.00479 vs 0.00479), and beats FP8 RTN by 3.0× at 8 bits on dense Qwen *in the
H-weighted weight-space census — a screen, not a served number*.
**So yes, build it — but only for the 6–8 bit rungs of the A8 and A16 lanes**,
knowing it is a wire change rather than a lever (the ALPHABET plane is one byte
per entry, so 256 codes is today's ceiling and `SERIALISABLE_GRIDS` is closed),
that L = 16 is a separate and much smaller axis worth ~10–20% at R = 8, and
that folding the result into a bf16 tile hands ≈0.0015 of relative output error
straight back — 16% of the win at R = 7, 28% at EXL3-K8 quality.

## Environment and exact commands

```
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
export PYTHONPATH=src:experiments:/home/rob/prismaquant
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python

# D: table shape, saturation, snap-vs-RNE
$PY experiments/tessera16_alphabet_floor.py --stage checks --window-bits 12 14 16 \
    --out experiments/results/tessera16_alphabet_floor_checks.json

# A + B: six GLM experts, R = 4..8 (deep column, one tensor at R=8).
#   NOTE: as written this runs six tensors x two grids at R=8, ~11 h.  It was
#   stopped by hand once L5.up_proj began, so `_glm.json` holds one expert and
#   carries no `summary` block.  Re-run with `--layers 5 --projs gate_proj`
#   for the same rows in ~1 h.
$PY experiments/tessera16_alphabet_floor.py --stage glm \
    --out experiments/results/tessera16_alphabet_floor_glm.json
# A + B: six GLM experts, R = 4..7 (the six-tensor geomean)
$PY experiments/tessera16_alphabet_floor.py --stage glm \
    --rungs 1024 1280 1536 1792 --no-parity \
    --out experiments/results/tessera16_alphabet_floor_glm47.json

# C: Qwen3-0.6B census, 196 Linears at R=4, layer 2 also at R=8
$PY experiments/tessera16_alphabet_floor.py --stage dense \
    --rungs 1024 2048 --rungs-all 1024 --subset-layer 2 \
    --out experiments/results/tessera16_alphabet_floor_dense.json

# D: is the BF16 alphabet scale-free?
$PY experiments/tessera16_alphabet_floor.py --stage sigma \
    --sigmas 0.25 1.0 4.0 --rungs 1024 \
    --out experiments/results/tessera16_alphabet_floor_sigma.json
$PY experiments/tessera16_alphabet_floor.py --stage sigma \
    --sigmas 4.0 --bf16-exp -27 4 --rungs 1024 \
    --out experiments/results/tessera16_alphabet_floor_sigma_wide.json

# C's pricing (every arm at its true bytes, table included) and the
# 196-Linear row-RMS census
$PY experiments/tessera16_alphabet_floor.py --stage price \
    --dense-json experiments/results/tessera16_alphabet_floor_dense.json \
    --out experiments/results/tessera16_alphabet_floor_price.json

# the second R = 8 GLM point, at the far end of the network
$PY experiments/tessera16_alphabet_floor.py --stage glm \
    --layers 42 --projs up_proj --rungs 2048 --no-parity \
    --out experiments/results/tessera16_alphabet_floor_glm_r8_L42.json
```

**Where each ran.**  The deep R = 8 column (`_glm.json`) and the dense census
(`_dense.json`, `_price.json`) ran on **sparky** (GB10, the repo checkout).  The
six-tensor R = 4…7 run (`_glm47.json`) and the second R = 8 point
(`_glm_r8_L42.json`) ran on **sparklina** from an rsync of the same working
tree, writing to `/mnt/shared/tessera-runs/alphabet/` and copied back into
`experiments/results/`.  Both boxes agree — see the cross-box check in "Scope".

**Result files**, all under `experiments/results/` with a `.log` beside each:
`..._glm.json` (deep, R = 4…8, L5.gate_proj) · `..._glm47.json` (six tensors,
R = 4…7) · `..._glm_r8_L42.json` (second R = 8 point) · `..._dense.json` and
`..._price.json` (part C) · `..._checks.json` (part D tables) ·
`..._sigma.json` and `..._sigma_wide.json` (scale-freeness).

## What is being compared

Every arm below is the **same body and the same plane**: `BodyKind.WINDOW`,
span 1, `ScalePlaneKind.CHANNEL`, L = 14, `scale_refit = 4`,
`trellis_weighting = "scale"` — the shipping `E4M3_RECIPE`.  The only thing
that changes is the alphabet the window table snaps to:

| grid | codes | how it is built |
|---|---|---|
| E4M3 | 256 | the serialisable wire grid, `channel_sigma = default_channel_sigma(E4M3_GRID)` |
| BF16 | 8192 | `bf16_grid()` in the experiment script — a 32-binade window of E8M7 (`±(1+m/128)·2^e`, e ∈ [−29, 2]), `channel_sigma = 1.0` |

The BF16 grid is built **in the experiment, not in `src/tessera/alphabet.py`**:
no library file was touched, and `tests/test_channel_plane.py
tests/test_window_body.py tests/test_scale_refit.py` were re-run on the working
tree — **53 passed**.

Metrics are the house ones on the last 1024 rows of the captured expert
activations (held out from the fit rows):
`wt = ‖Ŵ−W‖/‖W‖`, `out = ‖X(Ŵ−W)ᵀ‖/‖XWᵀ‖`, plus `a4` (served NVFP4 activation
qdq) and `a8` (vLLM FP8 dynamic per-token qdq).

**Comparator provenance.**  The EXL3 references are the pre-computed
reconstructions at `/home/rob/dq-runs/exl3-ref/L{layer}_{proj}_K{K}.pt`, priced
`K + 0.0117` bpw, and reproduce `experiments/results/tessera8_targets.json`'s
arithmetic means over the same six tensors: **K4 0.067875, K5 0.034549, K6
0.017662, K8 0.004785** (`out`).  The FP8 RTN per-channel LS-refit floor is
0.018869 at 8.008 bpp on the same rows.  The E4M3 wire comparator is
`experiments/results/tessera_window_wire_e4m3_reach.json` (CHANNEL plane,
L = 14) — the older `tessera_window_wire_e4m3.json` is the **LUT-plane L = 12**
wire and is a different object.  The reach json has no q1024 rung, so its own
rungs q960 and q1216 are re-run here as reproduction arms and match **to the
digit** (L5.gate_proj: wt 0.09085 / out 0.08747 at q960, wt 0.04687 /
out 0.04518 at q1216).  Those two rungs are also the drift detector: any
relaunch that does not print them is running a different encoder.

## Pricing

Every arm carries its true bytes.  The E4M3 arms are priced at the exporter's
own blob length (`8·len(blob)/numel`), which already contains the payload, the
CHANNEL plane (one fp16 word per row + the fp32 global) and the 2^L-byte
ALPHABET plane.  The BF16 arms are priced at the same blob **plus one extra
byte per table entry** (2^14 · 8 / numel), i.e. the table doubles from 16 KiB to
32 KiB.  On a 2048×4096 GLM expert that surcharge is **0.0156 bpp**; on a
1024×1024 Qwen Linear it is **0.125 bpp**, which matters and is carried.

## A. The floor, six GLM experts

Arithmetic mean of `out` over the six routed-expert projections (layers 5, 20,
42 × gate_proj, up_proj), the aggregation `tessera8_targets.json` uses.  The
EXL3 and FP8 references reproduce that file **to six digits** — K4 0.067875,
K5 0.034549, K6 0.017662, K8 0.004785, FP8 RTN 0.018869 — so these rows are on
the same rails as every earlier Tessera measurement.

| arm | bpp | wt | out | a4 | a8 |
|---|---|---|---|---|---|
| EXL3 K=4 | 4.012 | 0.08630 | 0.06787 | 0.10989 | 0.07200 |
| EXL3 K=5 | 5.012 | 0.04387 | 0.03455 | 0.09319 | 0.04210 |
| EXL3 K=6 | 6.012 | 0.02246 | 0.01766 | 0.08835 | 0.02987 |
| EXL3 K=8 | 8.012 | 0.00608 | 0.00478 | 0.08670 | 0.02456 |
| FP8 RTN per-channel, LS-refit | 8.008 | 0.02084 | 0.01887 | 0.08859 | 0.03061 |
| E4M3 window R=4 | 4.020 | 0.06940 | 0.06316 | 0.10703 | 0.06759 |
| E4M3 window R=5 | 5.020 | 0.03580 | 0.03266 | 0.09249 | 0.04058 |
| E4M3 window R=6 | 6.020 | 0.02323 | 0.02120 | 0.08913 | 0.03209 |
| E4M3 window R=7 | 7.020 | 0.02234 | 0.02039 | 0.08893 | 0.03156 |
| BF16 window R=4 | 4.036 | 0.06924 | 0.06322 | 0.10710 | 0.06765 |
| BF16 window R=5 | 5.036 | 0.03560 | 0.03241 | 0.09240 | 0.04037 |
| BF16 window R=6 | 6.036 | 0.01851 | 0.01686 | 0.08819 | 0.02941 |
| BF16 window R=7 | 7.036 | 0.00968 | 0.00880 | 0.08701 | 0.02565 |

**The ratio table — this is the answer to the question.**  Lower is better;
`< 1` means the Tessera window beats EXL3 at that rate.

| R | E4M3 `out` | BF16 `out` | BF16 gain | EXL3 at K=R | E4M3 / EXL3 | BF16 / EXL3 |
|---|---|---|---|---|---|---|
| 4 | 0.06316 | 0.06322 | 1.00× | 0.06787 | **0.931×** | **0.931×** |
| 5 | 0.03266 | 0.03241 | 1.01× | 0.03455 | **0.945×** | **0.938×** |
| 6 | 0.02120 | 0.01686 | 1.26× | 0.01766 | 1.200× | **0.955×** |
| 7 | 0.02039 | 0.00880 | 2.32× | 0.00919† | 2.218× | **0.958×** |

† No K7 reconstruction exists on disk (`/home/rob/dq-runs/exl3-ref` has K ∈
{2,3,4,5,6,8}); 0.00919 is the geometric interpolation of K6 and K8 and is the
one number in this document that is not measured.

**Per tensor**, `out`, so the aggregate above can be audited.  The six-tensor
*geometric* mean is given alongside; the tables above use the arithmetic mean
because that is the aggregation `tessera8_targets.json` uses, and the two agree
to within 0.7% on every row.

| tensor | grid | R=4 | R=5 | R=6 | R=7 |
|---|---|---|---|---|---|
| L5.gate_proj | E4M3 | 0.06689 | 0.03497 | 0.02309 | 0.02218 |
| L5.gate_proj | BF16 | 0.06692 | 0.03445 | 0.01789 | 0.00937 |
| L5.up_proj | E4M3 | 0.06933 | 0.03589 | 0.02379 | 0.02286 |
| L5.up_proj | BF16 | 0.06949 | 0.03556 | 0.01842 | 0.00963 |
| L20.gate_proj | E4M3 | 0.06383 | 0.03284 | 0.02099 | 0.02004 |
| L20.gate_proj | BF16 | 0.06391 | 0.03286 | 0.01707 | 0.00887 |
| L20.up_proj | E4M3 | 0.06752 | 0.03487 | 0.02298 | 0.02220 |
| L20.up_proj | BF16 | 0.06756 | 0.03450 | 0.01809 | 0.00940 |
| L42.gate_proj | E4M3 | 0.04847 | 0.02480 | 0.01564 | 0.01507 |
| L42.gate_proj | BF16 | 0.04823 | 0.02484 | 0.01291 | 0.00680 |
| L42.up_proj | E4M3 | 0.06292 | 0.03260 | 0.02072 | 0.01998 |
| L42.up_proj | BF16 | 0.06323 | 0.03225 | 0.01679 | 0.00876 |
| **geomean (6)** | **E4M3** | **0.06273** | **0.03242** | **0.02100** | **0.02020** |
| **geomean (6)** | **BF16** | **0.06278** | **0.03219** | **0.01675** | **0.00875** |

The sign of the answer is the same on every one of the six: E4M3 and BF16 are
within 0.5% at R = 4–5, and BF16 is 1.21–1.28× better at R = 6 and 2.13–2.37×
better at R = 7.  No tensor carries the aggregate.

**Read it this way** — remembering that `out` is a calibration-activation
weight-space proxy, not served KL, and that `tessera-first-served-kl-loses-to-nvfp4`
records an out-space ordering inverting on the serve.  Tessera's window body
already beats EXL3 by ~0.93–0.95× at 4 and 5 bits.  On an E4M3 alphabet that advantage **inverts at 6 bits**
(1.20× behind) and collapses at 7.  On a BF16 alphabet it **holds at 0.93–
0.96× across the whole range**.  A 16-bit alphabet is not a new capability; it
is what stops the capability Tessera already has from expiring at 5 bits.

## A2. The same thing one tensor deep, carried to R = 8 (L5.gate_proj, 2048×4096)

R = 8 is expensive enough that only two tensors were carried there — this one
at every rate, and L42.up_proj at R = 8 alone (next section); the reason
is in "What was not run" below.  Note that L5.gate_proj is a tensor EXL3 finds
hard (its K6 is 0.01877 against the six-tensor 0.01766), so read the ratios
above, not these, for the comparison.

| arm | bpp | wt | out | a4 | a8 |
|---|---|---|---|---|---|
| EXL3 K=4 | 4.012 | 0.08816 | 0.07237 | 0.11671 | 0.07672 |
| EXL3 K=5 | 5.012 | 0.04482 | 0.03674 | 0.09877 | 0.04470 |
| EXL3 K=6 | 6.012 | 0.02294 | 0.01877 | 0.09360 | 0.03166 |
| EXL3 K=8 | 8.012 | 0.00623 | 0.00511 | 0.09184 | 0.02602 |
| FP8 RTN per-channel, LS-refit | 8.008 | 0.01908 | 0.01844 | 0.09352 | 0.03147 |
| E4M3 window R=4 | 4.020 | 0.06966 | 0.06689 | 0.11328 | 0.07158 |
| E4M3 window R=5 | 5.020 | 0.03618 | 0.03497 | 0.09806 | 0.04328 |
| E4M3 window R=6 | 6.020 | 0.02394 | 0.02309 | 0.09454 | 0.03439 |
| E4M3 window R=7 | 7.020 | 0.02302 | 0.02218 | 0.09432 | 0.03379 |
| E4M3 window R=8 | 8.020 | 0.02299 | 0.02217 | 0.09432 | 0.03378 |
| BF16 window R=4 | 4.036 | 0.06942 | 0.06692 | 0.11334 | 0.07159 |
| BF16 window R=5 | 5.036 | 0.03572 | 0.03445 | 0.09785 | 0.04286 |
| BF16 window R=6 | 6.036 | 0.01859 | 0.01789 | 0.09342 | 0.03115 |
| BF16 window R=7 | 7.036 | 0.00973 | 0.00937 | 0.09218 | 0.02716 |
| BF16 window R=8 | 8.036 | 0.00531 | 0.00512 | 0.09183 | 0.02602 |

**At 8 bits the BF16 window lands level with EXL3 K8** — 0.00512 against
0.00511 at 8.036 vs 8.012 bpp — while the E4M3 window sits at 0.02217, **4.33×
worse**, and even plain per-channel E4M3 RTN beats the E4M3 window by 1.20×.

**E4M3 saturates; BF16 does not.**  From R = 6 the E4M3 window stops paying for
rate — 0.02309 → 0.02218 → 0.02217 — and its asymptote, 0.0222, is **1.20×
worse than plain per-channel E4M3 RTN at the same 8 bits** (0.01844).  At 8 bits
on an E4M3 alphabet the window body is the wrong tool: it spends a whole byte a
weight to land a fifth worse than rounding.  The BF16 window over the same
body, same plane, same L keeps halving — 0.03445 → 0.01789 → 0.00937 → 0.00512,
factors of 1.93, 1.91 and 1.83 per bit — with no floor reached at R = 8.

The BF16 advantage as a ratio of `out`: **1.00× at R=4, 1.02× at R=5, 1.29× at
R=6, 2.37× at R=7, 4.33× at R=8.**  Below R = 5 the alphabet is not the
constraint and the BF16 table's extra bytes buy nothing.

## R = 8, both tensors: the BF16 window lands on EXL3 K8

R = 8 was measured on two GLM experts at opposite ends of the network.  They
agree, and they agree with EXL3 K8 to the digit.

| tensor | E4M3 window R=8 | BF16 window R=8 | EXL3 K=8 | FP8 RTN | BF16 vs E4M3 | BF16 vs EXL3 |
|---|---|---|---|---|---|---|
| L5.gate_proj | 0.02217 | **0.00512** | 0.00511 | 0.01844 | 4.33× | 1.002× |
| L42.up_proj | 0.01995 | **0.00479** | 0.00479 | 0.01999 | 4.17× | 1.000× |

bpp: BF16 window 8.036, E4M3 window 8.020, EXL3 8.012, FP8 RTN 8.008 — matched
to within 0.03 bpp.

On both tensors the E4M3 window at 8 bits is at or behind plain per-channel FP8
rounding (1.20× behind on L5, level on L42), and the BF16 window over the
identical body reaches EXL3's 8-bit reconstruction quality.  The 4.2–4.3× is
the alphabet, and nothing else changed.

## C. Qwen3-0.6B census, 196 Linears, H-weighted

A second model, a dense one, with the diagonal-H weighting the census uses.
The three anchors reproduce exactly, so the arms sit on the census's own rails:

| arm | bpp | plain geomean | H-weighted | anchor |
|---|---|---|---|---|
| NVFP4 GPTQ+JSO (production) | 4.500 | 0.0941 | **0.0955** | 0.0955 ✔ |
| E4M3 window R=4 (the shipping wire, reach fix) | 4.071 | 0.0704 | **0.0765** | 0.0765 ✔ |
| FP8 RTN per-channel | 8.025 | 0.0264 | **0.0262** | 0.0262 ✔ |
| BF16 window R=4 | 4.129 | 0.0701 | 0.0762 | — |
| BF16 window R=4, folded to bf16 | 4.129 | 0.0702 | 0.0762 | — |

**At 4 bits the alphabet buys 1.004× for 0.058 bpp.**  That is the same answer
the GLM experts gave (1.00×), now on a dense model with 196 tensors behind it.
Below R = 5 a 16-bit alphabet is not worth its table bytes.

The rate-8 arms ran on one layer's seven Linears; the reason is mechanical, not
a judgement about which tensors matter, and every arm is re-priced over exactly
those seven.  Layer 2 was chosen a priori from
`tessera-dense-outlier-mechanism`'s pointer at the rows that overrun the window
table's reach.  On the R = 4 arms it turns out to land **slightly easier than
the census average** (E4M3 0.0738 vs 0.0765; FP8 RTN 0.0238 vs 0.0262), so the
R = 8 subset is not a cherry-pick in either direction.

| arm | bpp | plain | H-weighted |
|---|---|---|---|
| NVFP4 GPTQ+JSO | 4.500 | 0.0933 | 0.0866 |
| E4M3 window R=4 | 4.071 | 0.0705 | 0.0738 |
| BF16 window R=4 | 4.129 | 0.0703 | 0.0735 |
| FP8 RTN per-channel | 8.025 | 0.0264 | 0.0238 |
| E4M3 window R=8 | 8.071 | 0.0264 | 0.0234 |
| **BF16 window R=8** | 8.129 | **0.0062** | **0.0079** |
| BF16 window R=8, folded to bf16 | 8.129 | 0.0064 | 0.0080 |

At 8 bits the E4M3 window lands on 0.0234 — a dead heat with plain per-channel
FP8 rounding (0.0238) for the same bytes, which is another way of saying the
E4M3 alphabet gives the trellis nothing to do at 8 bits.  The BF16 window at the
same rate reaches **0.0079: 3.0× better than FP8 RTN and 3.0× better than the
E4M3 window in the H-weighted weight-space census**, for 0.10 bpp more than RTN.
That is the clearest *screen* signal in the document that the A8 lane is where a
16-bit alphabet would pay on a dense model — **and it is a screen, not a served
result.**  This repository has watched a weight-space margin invert on the way to
the serve more than once: `tessera-serialisable-grid-costs-quality` (weight-space
1.16× → 0.90× once NVFP4 is priced W4A4 as it serves),
`tessera-first-served-kl-loses-to-nvfp4`, and — on this exact pairing —
`tessera8-targets`, where every Tessera-8 win was on Gaussian-input GLM experts
and dense Qwen lost 23× to FP8 RTN when served.  The dense-outlier memo puts the
same E4M3-vs-FP8 gap at 2.9× in weight space and 7.4× served.  Nothing here
promotes anything; it sizes a floor.

Two scope notes.  The table surcharge is **0.058 bpp** on these 1–3 M-parameter
Qwen Linears against 0.016 bpp on an 8 M-parameter GLM expert, so the smaller
the tensor the worse the trade — on a 1024×1024 Linear the BF16 table alone is
0.25 bpp.  And the bf16 fold costs 1.3% here (0.0079 → 0.0080), consistent with
part B's floor.

**The per-row scale is doing much more work on dense than on GLM.**  Row RMS
spread over the 196 Linears: **median 5.95×, max 42.3×, min 1.63×**, against
1.27× on the GLM expert.  A wider alphabet does not remove the CHANNEL plane's
job, and the reach clamp (4.00 σ on BF16 vs 4.077 σ on E4M3) is unchanged, so
`tessera-dense-outlier-mechanism`'s reach-aware per-row start stays load-bearing
either way.

## B. The fold cost: what an A16 tile pays for having no scale tensor

An A16 tile carries no scale tensor, so the decoded weight is
`bf16(code × row_scale)` rather than the fp32 product.  Every arm above is
therefore also scored after folding to bf16 (`out_bf16`) and to fp16
(`out_fp16`).  Subtracting in quadrature isolates the rounding:
`fold = √(out_fold² − out²)`.

L5.gate_proj, full precision from the JSON:

| arm | out (no fold) | fold→bf16 | as % of out | fold→fp16 | as % |
|---|---|---|---|---|---|
| EXL3 K=4 | 0.072366 | 0.001611 | 2.2% | 0.000349 | 0.5% |
| EXL3 K=6 | 0.018773 | 0.001516 | 8.1% | 0.000170 | 0.9% |
| EXL3 K=8 | 0.005108 | 0.001457 | **28.5%** | 0.000188 | 3.7% |
| FP8 RTN | 0.018441 | 0.001140 | 6.2% | 0.000165 | 0.9% |
| E4M3 window R=7 | 0.022179 | 0.002229 | 10.0% | 0.000504 | 2.3% |
| BF16 window R=4 | 0.066915 | 0.001498 | 2.2% | 0.000221 | 0.3% |
| BF16 window R=5 | 0.034449 | 0.001432 | 4.2% | 0.000000 | 0.0% |
| BF16 window R=6 | 0.017890 | 0.001406 | 7.9% | 0.000000 | 0.0% |
| BF16 window R=7 | 0.009371 | 0.001492 | **15.9%** | 0.000000 | 0.0% |

**The bf16 fold is a floor, not a tax.**  It is 0.0011–0.0022 in absolute
relative-output error on *every* arm — EXL3, RTN, both window grids, every rate
— because it is a property of bf16's 7-bit mantissa and the activations, not of
the weights.  Its *share* therefore grows as the coding error shrinks: 2% at
R = 4, 16% at R = 7, 28% on EXL3 K8.  Any A16 tile that folds to bf16 cannot
go below ≈0.0015 on these rows however many weight bits it is given, which in
this regime costs roughly 0.4 bit of rate at R = 7–8.

**Not folding is nearly free, and folding to fp16 is nearly free.**  The
brief's alternative — an fp16 or fp32 per-row scale applied in fp32 by a
streamed decode — is the `out` column itself, i.e. the fold cost disappears.
An fp16 fold costs 0.0000–0.0005 (0–3.7%), an order of magnitude below bf16.
The conclusion is narrow and useful: **if a 16-bit tile is built, do not make
it a bf16 tile with the scale folded in.**  Keep the row scale and apply it in
fp32, or fold to fp16.

## D. Why: the table's distinct values, not the table's size

`--stage checks` counts what the window table can actually emit.

| grid | L=12 | L=14 | L=16 | codes available |
|---|---|---|---|---|
| E4M3 | 168 | 202 | 236 | 256 |
| BF16 | 1546 | 2116 | 2680 | 8192 |

At L = 14 the E4M3 table holds **202 distinct values** — log₂ 202 = **7.66
bits** is the most information the body can carry per position on that alphabet
however large R grows, and that is the saturation.  BF16 holds 2116 distinct
values, **11.05 bits**, which does not bind through R = 8.  Note that L is
almost irrelevant to this axis: L = 16 moves E4M3 from 7.66 to 7.88 bits and
BF16 from 11.05 to 11.39.  **Widening the alphabet is what breaks the floor;
lengthening the table is not.**

Two things a BF16 alphabet does *not* change:

- **Reach.**  The L = 14 table reaches 4.000 σ on BF16 against 4.077 σ on E4M3
  — the same clamp, doing the same work.  BF16 is therefore **not** a fix for
  the dense-Qwen outlier rows; the reach-aware per-row start still is.
- **Snap exactness.**  Snapping the Gaussian quantiles onto the BF16 grid
  differs from plain bf16 round-to-nearest-even on 30 of 16384 entries at
  L = 14 (max |d| 1.6e-2), so "the table is bf16" is true to within a handful
  of entries, and the arms above use the snapped table.

**Is the alphabet scale-free?**  Yes, and the check has a trap in it.  On
L5.gate_proj at R = 4 the BF16 arm gives wt **0.069416** at `channel_sigma` of
0.25, 1.0 **and** 4.0 — bit-identical, because a power-of-two rescaling is
exact in E8M7.  The first run showed σ = 4.0 degrading to 0.089717; that was
the *experiment's* 32-binade grid window (peak 7.97) failing to hold ±16σ, not
the alphabet.  Re-run on a shifted window (`--bf16-exp -27 4`, peak 31.9) it is
identical again.  σ = 1.0 is therefore a placement, not a fit.

**Is the per-row scale still doing work?**  On this tensor, barely: the LS
refit buys 0.069416 vs 0.069561, **1.002×**, against the 1.084× it is worth on
the E4M3 wire — a fine alphabet leaves the row scale little to fix.  The row
RMS spread is 1.27× max/min (1.07× p99/p1), so there is little for a per-row
scale to equalise on a GLM expert in the first place.  Scope: one tensor,
R = 4.

## L = 16: not run, and answered from the data

The brief asked for L = 16 at R = 6 and 8 "if it is cheap".  It is not: a new
`SIZE = 65536` constexpr forces fresh Triton compiles of the fused window
Viterbi, on top of a rate-8 encode that measured **3247 s of wall clock** for
one 2048×4096 tensor under contention (33 s at R=7, 16 s at R=6, 5 s at R=4).
That is a measured wall-clock figure, **not a profile**; ruling out the column
batch width (see below) is a reading of the code, not a finding about where the
time goes.

It can be answered from what was measured, and the answer is that **L and the
alphabet are two different axes**:

- **The saturation axis is the alphabet.**  L = 16 moves the E4M3 table from
  202 to 236 distinct values — 7.66 to 7.88 bits — which does not lift a floor
  that sits at 7.66.  A wider table cannot cure a narrow alphabet.
- **The shaping axis is L − R, and it *is* binding.**  Against the `2^-R`
  reference the BF16 window's `wt` widens monotonically with rate —
  **1.108× (R=4), 1.139×, 1.184×, 1.239× (R=7)**, and 1.360× at R=8 on the deep
  tensor — consistent with running out of shaping bits (untested here, since
  L = 16 was not run; the one measured L datum on record is L = 12 → 14 at
  R = 4, worth ~4.6% of `out`, memory `tessera-window-body`), since L − R
  falls from 10 to 6.  So L = 16 would help, at high R only, and the trend
  bounds the gain: restoring R=8 to R=7's ratio is 1.10×, restoring it to
  R=4's is 1.22×.  **Between roughly 10% and 20% in `wt` at R = 8, and zero at
  R = 4.**

That gain is second-order next to the 4.33× the alphabet is worth at the same
rate, which is why the alphabet is the thing to build first.

## What this means for `W(n≤A) A{4,8,16}`

The allocator's question is not "which weight coder wins" but "at each
activation width, where do weight bits stop paying".  That is a read of the
`out` / `a8` / `a4` columns above, with the activation floor backed out of
EXL3 K8 in quadrature: **A4 floor 0.08657, A8 floor 0.02409, A16 floor 0**.

| arm | bpp | A16 (`out`) | A8 (`a8`) | over A8 floor | A4 (`a4`) | over A4 floor |
|---|---|---|---|---|---|---|
| E4M3 window R=4 | 4.020 | 0.06316 | 0.06759 | 181% | 0.10703 | 23.6% |
| E4M3 window R=6 | 6.020 | 0.02120 | 0.03209 | 33.2% | 0.08913 | 3.0% |
| E4M3 window R=7 | 7.020 | 0.02039 | 0.03156 | **31.0%** | 0.08893 | 2.7% |
| BF16 window R=4 | 4.036 | 0.06322 | 0.06765 | 181% | 0.10710 | 23.7% |
| BF16 window R=5 | 5.036 | 0.03241 | 0.04037 | 67.6% | 0.09240 | 6.7% |
| BF16 window R=6 | 6.036 | 0.01686 | 0.02941 | 22.1% | 0.08819 | 1.9% |
| BF16 window R=7 | 7.036 | 0.00880 | 0.02565 | **6.5%** | 0.08701 | 0.5% |
| EXL3 K=8 | 8.012 | 0.00478 | 0.02456 | 2.0% | 0.08670 | 0.2% |
| FP8 RTN | 8.008 | 0.01887 | 0.03061 | 27.1% | 0.08859 | 2.3% |

- **A4 needs nothing from a 16-bit alphabet.**  By R = 6 every arm is within
  3% of the A4 floor; the activation leg owns the error, as
  `tessera-w4a4-changes-the-lever` already found.  A BF16 alphabet under A4 is
  0.016 bpp spent for nothing.
- **A8 is where the E4M3 alphabet costs real quality.**  The E4M3 window
  plateaus **31% over the A8 floor and cannot get closer at any rate**, while
  the BF16 window reaches 6.5% at R = 7 — and 22% at R = 6, already better
  than E4M3's best.  That gap is the whole case for the wider alphabet.
- **A16 is where it compounds**: `out` halves per bit on BF16 through R = 8 and
  is flat on E4M3 past R = 6.

## Scope, and what was not run

**Unmet against the brief, stated plainly:**

- **R = 8 on GLM is two tensors, not six.**  L5.gate_proj (both grids) and
  L42.up_proj.  The measured reason is 3247 s of wall clock per 2048×4096
  rate-8 encode under contention; six tensors × two grids is ~11 h.  E4M3
  moved 0.00001 from R=7 to R=8, and BF16's R=6→7→8 ratios (1.91, 1.83) are
  regular, so the six-tensor R=8 row would be an interpolation of rows that are
  already measured.  It is cheap to add later on an idle box.
- **L = 16 was not run.**  Answered from the data instead; see above.
- **No K7 EXL3 reference exists**, so the R = 7 comparison uses a geometric
  interpolation of K6 and K8.
- **The sigma / refit / row-RMS findings are one GLM tensor at R = 4.**

**Scope that limits the conclusion, not just the coverage:**

- **Nothing in this document is a served number.**  Every metric here is
  weight-space or calibration-activation error (`wt`, `out`, `a4`, `a8`, the
  H-weighted census).  The brief asked for the floor, and a floor is a screen by
  construction.  Under this repo's own promotion rule none of these arms is
  promoted by anything written here; served KL on the pinned runtime is what
  would decide, and `tessera8-targets`, `tessera-first-served-kl-loses-to-nvfp4`
  and `tessera-serialisable-grid-costs-quality` all record weight-space margins
  that inverted on the way there.

- Every GLM number is on **routed-expert projections with Gaussian-ish
  inputs**, the regime where memory records that Tessera looks best.  Part C is
  the dense counterweight.
- The BF16 arms are **in-memory only**.  The wire cannot carry them: the
  ALPHABET plane is one byte per table entry, so 256 codes is the serialisable
  ceiling and `SERIALISABLE_GRIDS` is closed.  A 16-bit alphabet is a **wire
  change**, not a lever — that is the cost side of this measurement.
- The in-memory path is nonetheless the wire path: `encode_unit` was checked
  bit-identical to `encode_linear_planes` on L5.gate_proj at q1024 L=14
  (`parity.equal = true`, max abs diff 0).
- **Cross-box identity holds for both grids.**  The six-tensor run executed on
  sparklina and reproduced sparky's L5.gate_proj rows to five digits on E4M3
  (0.08747 / 0.04518 / 0.06689) **and on BF16** (0.06692 / 0.03445 / 0.01789 /
  0.00937).  The BF16 grid construction and its `cdist` snap are deterministic
  across machines.

