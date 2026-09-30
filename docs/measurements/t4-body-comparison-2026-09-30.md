# T-4 body comparison: span-2 TCQ against the window body over LUT16

**Date:** 2026-09-30 · **Issue:** #750 · **Harness:** `experiments/t4_code/`
on branch `claude/t4-routed-e2m1`. The accuracy runs are at `b6f93198d2`
(`t4_code_compare.py`, `decode_cost.cu`); the tables (`summarize.py`) and the
A-side timing (`aside_cost.py`) are on the branch head · **Image:** `localhost/prismaquant/spark-vllm-nccl230:nightly-20260929`
(`sha256:5be13705…`) · **Runner:** PrismaBuild, GB10 (sparky), one GPU per
action, not a timing run · **Actions:** `64f0e529` (routed experts, 621 s),
`2a355306` (dense MLP and shared expert, 569 s), `b4e94f7c` (MLA attention,
423 s), all rc 0 · **Raw JSON:**
`/mnt/shared/tessera-measurements/t4-code-20260930/run1/{experts,dense,attn}/`
· **Encoder digest:** `8e53c800858d3a6c`

## The question

The FP4 block-scaled MMA on sm_121
(`mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…e2m1.e2m1.f32.ue4m3`)
takes the T-4 activation contract `e2m1_group16_ue4m3_static` natively. Its
weight operand is E2M1 codes over a per-16 UE4M3 block scale. A T-4 wire must
therefore decode to E2M1 symbols plus a group-scale plane; a CHANNEL-plane
window (the T-8 wire) cannot feed it. Two bodies already do this over the LUT16
plane (a 4-bit index per 16 weights into a per-unit 16-entry E4M3 table,
0.25 bits per weight):

- **(a) span-2 TCQ** (`TCQ_RECIPE`), the served routed wire. It spends a 0.25
  bpw label plane on top of the LUT16 plane. One code rate per unit: q256 128
  to 896 in steps of 128 (the cap is `payload_bits - 1` = 7 bits per tuple).
- **(b) the window body** (`E2M1X2_SUBCAP_RECIPE`: window, span 1, LUT16,
  L=12). No label plane, so it is matched to (a) at q256 + 64. Per-column
  integer rates 1 to 8 per tuple, fractional rungs from two adjacent-rate run
  tables, and it reaches q256 1024 (the whole E2M1x2 grid, 4.0 body bits).

Question: under the mandatory group-scale plane, which body spends the E2M1
symbol bits best from 1.0 to 4.0 bpw, and what does each cost to decode?

## Method

- **Tensors (19).** Routed experts: expert 0 gate/up/down at layers 5, 20 and
  42 (2048×4096, 4096×2048). Dense MLP: layer 0 gate/up (first 2048 rows) and
  down (first 1024 rows). Shared expert: layer 10 gate/up/down. MLA layer 3:
  `q_a_proj`, `kv_a_proj_with_mqa`, `q_b_proj` (first 4096 rows), `o_proj`
  (first 1024 rows).
- **Encoder.** The production encoder (`encode_linear`, `scale_refit` 4) to
  bytes, then `read_unit_artifact`. bpp is priced at the bytes written. No
  activation weighting and no LDLQ on either body.
- **Activations.** The pread capture
  (`glm53-bf16-pread-capture-1469b9b-20260901`). The last 1024 rows are held
  out for evaluation; the remaining rows fit the static global.
  Proxies, named in every JSON row:
  - routed `down_proj` reads the expert's own SwiGLU (clamp 10) of the
    captured MoE input through its BF16 gate/up, over every captured token;
  - `q_a` and `kv_a` read the layer-2 captured attention input;
  - `q_b` reads that input through layer 3's `q_a_proj` and `q_a_layernorm`;
  - `o_proj` has no capture and is weight-only.
- **Legs** (relative errors, geometric mean over tensors):
  - `wt`: Frobenius weight error.
  - `out`: `||X dWᵀ|| / ||X Wᵀ||`, the activation-weighted error (H = XᵀX).
  - `a4s`: executed W4A4 with the static per-module global (2688 / fit amax)
    through vLLM's `scaled_fp4_quant`, the served contract.
  - `a4d`: the same with a per-token dynamic global, through a torch
    reference quantiser.
  - `a4s_ref`: the static global through that reference, so static and dynamic
    share one quantiser.

These are proxy metrics on single tensors, not the served KL. They rank the two
bodies; they do not certify an artifact.

## Accuracy at matched bytes (all 19 tensors)

`out` and `a4s` are over the 18 tensors with activations.

| bpp (TCQ/win) | TCQ q256 | win q256 | wt TCQ | wt win | win/TCQ | out TCQ | out win | win/TCQ | a4s TCQ | a4s win | win/TCQ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.001/1.005 | 128 | 192 | 0.7670 | 0.6026 | **0.786** | 0.7505 | 0.5598 | **0.746** | 0.7523 | 0.5637 | **0.749** |
| 1.501/1.505 | 256 | 320 | 0.4906 | 0.4272 | **0.871** | 0.4233 | 0.3750 | **0.886** | 0.4290 | 0.3819 | **0.890** |
| 2.001/2.005 | 384 | 448 | 0.3796 | 0.3029 | **0.798** | 0.3179 | 0.2545 | **0.801** | 0.3262 | 0.2650 | **0.813** |
| 2.501/2.505 | 512 | 576 | 0.2871 | 0.2157 | **0.751** | 0.2338 | 0.1765 | **0.755** | 0.2452 | 0.1916 | **0.781** |
| 3.001/3.005 | 640 | 704 | 0.1992 | 0.1552 | **0.779** | 0.1594 | 0.1245 | **0.781** | 0.1758 | 0.1451 | **0.825** |
| 3.501/3.505 | 768 | 832 | 0.1505 | 0.1157 | **0.769** | 0.1194 | 0.0919 | **0.769** | 0.1408 | 0.1183 | **0.841** |
| 4.001/4.005 | 896 | 960 | 0.0894 | 0.0947 | 1.059 | 0.0701 | 0.0745 | 1.064 | 0.1023 | 0.1054 | 1.030 |

By structure class, window/TCQ on `a4s`:

| class | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|---|
| routed expert (9) | 0.777 | 0.890 | 0.816 | 0.780 | 0.825 | 0.839 | 1.029 |
| dense MLP (3) | 0.816 | 1.008 | 0.859 | 0.830 | 0.870 | 0.882 | 1.038 |
| shared expert (3) | 0.691 | 0.862 | 0.796 | 0.763 | 0.809 | 0.831 | 1.026 |
| attention (3 with activations; `wt` over 4) | 0.670 | 0.812 | 0.776 | 0.757 | 0.800 | 0.817 | 1.029 |

- **Below the cap (1.0 to 3.5 bpp) the window body wins everywhere but one
  cell.** The executed error is 11% to 25% lower; the weight error is 13% to
  25% lower. The exception is dense MLP at 1.5 bpp, where `out` is 0.7% higher
  (0.3649 against 0.3623) and `a4s` 0.8% higher.
- **At the cap (4.0 bpp) TCQ wins.** The weight error is 5.9% lower, and the
  executed error is 3.0% lower. This matches the earlier on-the-wire result
  (`docs/tessera-one-format.md` §4: 1.170× against 1.244× EXL3).
- **The A side compresses the gap.** At 3.5 to 4.0 bpp the static NVFP4
  activation error (about 0.09 relative) is the larger term, so the executed
  ratios sit closer to 1 than the weight ratios.

Other arms (all 19 tensors):

| arm | bpp | wt | out | a4s |
|---|---|---|---|---|
| TCQ q896 | 4.001 | 0.0894 | 0.0701 | 0.1023 |
| window L=12 q1024 | 4.255 | 0.0886 | 0.0696 | 0.1020 |
| window L=14 q960 | 4.019 | 0.0925 | 0.0727 | 0.1042 |
| window L=12 q960 | 4.005 | 0.0947 | 0.0745 | 0.1054 |
| window L=14 q576 | 2.519 | 0.2106 | 0.1717 | 0.1871 |
| window L=12 q576 | 2.505 | 0.2157 | 0.1765 | 0.1916 |
| NVFP4 RTN (reference) | 4.500 | 0.0922 | 0.0723 | 0.1039 |

- **q1024 matches TCQ at the cap for 0.25 more bpw.** The window's top rung
  (the whole grid) reaches TCQ q896's executed error at +0.254 bpp.
- **L=14 beats L=12 per bit.** It lowers every leg by 2.3% to 2.4% for
  +0.014 bpp. The local slope of the L=12 curve buys about 1% for that many
  bits at 2.5 bpp and 0.6% at 4.0, so the net gain is 1.4% to 1.7%. L=14 still
  trails TCQ at 4.0 bpp by 1.9% executed.
- **Both bodies at 4.0 bpp are within 1.5% of NVFP4 RTN at 4.5 bpp** on the
  executed error (TCQ 1.5% better, window 1.4% worse).

## A-side global: static against dynamic per token

Same fit and eval rows, same reference quantiser.

- **Activation quantisation error.** Static and dynamic agree within 0.5% on 16
  of 18 inputs (within 0.1% on 12). The layer-0 dense gate/up input is the
  largest gap: 0.09213 static against 0.09130 dynamic (0.9%).
- **Executed output error** (`a4s_ref / a4d`, window L=12 q704): from 0.9979 to
  1.0039. Dynamic is at most 0.4% better, and on 3 tensors it is slightly
  worse.
- **Saturation under the static global.** At most 3 of 1024 eval tokens exceed
  the fit amax, and at most 6 FP8 block scales saturate (L5 expert input).
- **Reference quantiser against vLLM** at the static global: scales equal on
  every tensor; E2M1 codes equal on 99.69% to 100%.

Caveat: the static global is fit on the same capture's other rows. A served
calibration set drawn from different text may see more tokens above its amax
than this split does.

The cost side (the extra row-amax reduction and a per-row epilogue) is timed
by `aside_cost.py`, PB `f8f09d49`, measurement mode on sparklina. A dynamic
global also changes the attested contract string (`…_static`), so it is a
contract decision, not a kernel option.

**Recommendation.** On accuracy alone, a per-token dynamic global does not
justify changing `e2m1_group16_ue4m3_static`: it buys at most 0.4% of executed
error. The cost measurement can only add to the case against it. The contract
decision belongs to the #750 lead.

## Decode cost per weight

`decode_cost.cu` decodes each body in register form straight into the FP4
MMA's B fragments. Both bodies do the same LUT16 scale and nibble-placement
work. Static SASS count (nvcc 13.0.88, `-arch=sm_121a -cubin`, the decode loop):

| body | instructions per weight | rates |
|---|---|---|
| window | 6.8 | every rate 1 to 8, at L=12 and L=14 |
| span-2 TCQ | 9.8 | rates 2 to 7 |
| span-2 TCQ | 8.0 | rate 1 (no point plane) |

- **The count does not depend on rate** within either body, for one-run
  (single-rate) tables. One kernel templated on rate covers every one-run
  rate of either body, and no one-run rate is intrinsically dearer to decode.
  Rate differences there come from bytes.
- **Two-run tables are not measured.** Every matched-byte window arm in the
  accuracy table except q1024 is a two-run table (`rate_set(q256 / 128)` =
  `[r, r+1]`), so the accuracy and the decode-cost evidence cover different
  rungs. On the E4M3 lane, two-run launches pay for the descriptor ring and a
  per-warp rate branch (#694 measured 1.31× to 1.49× when the rate was a
  runtime switch). The two-run cost in T-4's register form is the first thing
  the fused kernel's geometry sweep under the #750 protocol must answer. If
  that sweep excludes two-run tables, the comparison the allocator faces is
  window one-run rungs (q256 = 128·r, 0.25 + r/2 bpw) against TCQ at the
  nearest bytes, not the matched pairs above.
- **The window decodes in 30% fewer instructions** than TCQ at rates 2 to 7.
- Measured cycles per weight and power are timed by the same PB action
  `f8f09d49`; this section is updated when it lands.

## The FP4 MMA's scale-register layout (pinned)

`experiments/t4_code/fp4_mma_layout.cu`, PB `bd932440` (GB10, rc 0, raw JSON
`/mnt/shared/tessera-measurements/t4-code-20260930/fp4-layout1/`). The fused
lane issues `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…ue4m3`
from inline PTX; nothing else in the tree runs it below Triton's
`tl.dot_scaled`.

- **Discovery.** Each experiment doubles one byte of one lane's scale
  register and reads which output rows (SFA) or columns (SFB) change, one k16
  group at a time. With the thread and byte selectors at 0 (lane = 4g + t):
  - SFA: lane (g, t=0) byte b scales row g, k16 group b; lane (g, t=1) byte b
    scales row g + 8, group b. Lanes t = 2, 3 are ignored.
  - SFB: lane (g, t=0) byte b scales column g, group b. Lanes t = 1, 2, 3 are
    ignored.
  - No anomalies; every (row, group) and (column, group) is covered once.
- **Verification.** 4096 random trials (524,288 outputs): every E2M1 code,
  UE4M3 scales from 0.25 to 4. All partial sums are exact in fp32, and all
  outputs equal the double-precision reference bit for bit. That confirms the
  A, B and D fragment layouts (the sm80 int4 `m16n8k64` ones) as well. It also
  means the lane's oracle can demand bitwise equality wherever sums are exact.
- **Build flag.** Compile with `-gencode arch=compute_121a,code=sm_121a`.
  With nvcc 13.0, `-arch=sm_121a` also embeds `compute_121` PTX, and ptxas
  refuses the block-scaled MMA there.

## Options for the T-4 wire

1. **Window body at every rung (q256 128 to 1024), TCQ retired from T-4.**
   - One decoder family: an E2M1 library in the fused window lane, with the FP4
     MMA.
   - Below the cap the executed error is 11% to 25% lower at matched bytes.
   - At exactly 4.0 bpp it is 3.0% worse executed. The allocator can buy
     parity at q1024 (+0.25 bpw on that unit) or accept the 3%.
   - The allowable set is the window grammar's: run tables `[1]`, `[1,2]` …
     `[7,8]`, `[8]` over q256 128 to 1024, evaluated as
     `rate_set(q256 * 2 / 256)`. At step 1 there is no gap between adjacent
     supported rungs, so nothing to cost in GB, unless the geometry sweep
     excludes two-run tables. Then the gap is one integer rate, q256 128
     (0.5 bpw, about 19 GB on GLM-5.3's 304.5 B routed parameters).
2. **Window body below the cap, TCQ at q896.**
   - Keeps the 3.0% at the one rung.
   - Needs a second fused decoder family (span-2 TCQ, 9.8 instructions per
     weight) and a second attested wire for that rung.
   - Window rungs between 896 and about 1000 would then be dominated on
     accuracy by TCQ q896 at fewer bytes.

The measurements favour option 1: one kernel family, the cheaper decode, and
a large win at 6 of 7 rates against a 3% loss at one. Choosing it changes the
served wire for routed T-4 (today `served_recipe` promotes every routed E2M1
rung to TCQ, and contract v47 attests q896 on TCQ), so the choice belongs to
the #750 lead.

## Limitations

- Proxy metrics on single tensors; one expert per layer; rows sliced on the
  large dense and attention tensors. Not a served KL.
- Unweighted encodes on both bodies. An H-weighted or LDLQ encode was not
  compared.
- The routed `down_proj`, MLA, and `q_b` activations are proxies (named
  above); `o_proj` is weight-only.
