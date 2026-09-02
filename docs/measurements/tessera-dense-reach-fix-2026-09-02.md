# The reach-aware row start: dense Qwen from KL 0.470 to 0.151 at the same wire (2026-09-02)

**Claim.** The E4M3/CHANNEL/window wire's failure on dense Qwen3-0.6B was the
encoder's source model, not the format: the 2^14 Gaussian window table reaches
384 grid units = 4.08 sigma0 (sigma0 = `default_channel_sigma(E4M3)` = 94.2),
and every row whose largest weight exceeds 4.08 row-RMS clips there. A Gaussian
row of width 1024 does so 4.5% of the time; Qwen3-0.6B's rows do so 25.5% of
the time (87,870 of 344,064 rows; 59% of `down_proj` rows, 48% of `o_proj`,
17-26% of the rest, up to 30 sigma), and the clipped entries sit in the
Hessian-dominant columns. `initial_channel_scale(work, sigma, reach=...)` now
starts every row whose max would land past the body's reach at the sigma that
puts it exactly on the reach; rows inside the reach are the plain RMS start
byte for byte. The CHANNEL plane already stores one fp16 scale per row, so the
wire, the table, the decoder, the kernel and Gridbook's lane are untouched.

## Served (the gate)

Vanilla vLLM 0.28 (`vllm/vllm-openai:latest`), the compressed-tensors
per-channel FP8 materialisation of the wire (the stock twin the exporter writes
from the same units), corpus `corpus_qwen_n8_s512.json`, teacher
`qwen_teacher_bf16_v028`, KL-vs-BF16 over top-1024 support (lower bound), the
same instrument as every row of `tessera-stock-lane-served-2026-09-02.md`.

| arm | bpp (wire / resident) | KL all | KL confident | top-1 |
|---|---|---|---|---|
| Tessera-8 E4M3 window L=14, pre-fix (`qwen_stock_tessera-e8`) | 4.07 / 8.0 | 0.4699 | 0.3894 | 63.2% |
| **Tessera-8, reach-aware start** (`qwen_stock_tessera-e8-reach`) | 4.07 / 8.0 | **0.1512** | **0.1005** | **78.1%** |
| production NVFP4 GPTQ+JSO (W4A4), stock-lane receipt | 4.5 / 4.5 | 0.511 | | 62.6% |
| FP8 RTN per-channel (W8A8), stock-lane receipt | 8.0 / 8.0 | 0.0205 | | 91.2% |

3.1x lower KL at the same bytes; against production NVFP4 at 4.5 bpp the
4.07-bpp wire is now 3.4x better instead of 8%; against FP8 RTN at 8.0 bpp it
is 7.4x behind instead of 23x. Serve log `/home/rob/tessera-runs/gbfam/serve_e8-reach-twin.log`,
compares `kl_e8-reach-twin_vs_teacher.json` and `kl_e8-old_vs_teacher.json`
(the old dump re-compared against the same teacher in the same minute: 0.4699).

## The mechanism, on the worst unit (`model.layers.2.mlp.down_proj`, 1024x3072)

H-weighted RMS relative error `sqrt(sum_j h_j sum_i e_ij^2 / sum_j h_j sum_i w_ij^2)`
with `h` the render-activation second moments (`/home/rob/tessera-runs/stock/h_diag.pt`,
max/median 2.2e6, top-4 columns 96% of the mass) and plain RMS relative error.
`experiments/dense_spread_fix.py`:

| arm | plain | H-weighted | the 0.582 weight decodes to |
|---|---|---|---|
| A production Tessera-8, 4.07 bpp | 0.0838 | 0.4102 | 0.3395 |
| C global `channel_sigma` 20.5 | 0.0844 | 0.4296 | 0.3281 |
| B per-row reach-aware start | 0.0738 | 0.0578 | 0.5764 |
| production NVFP4 GPTQ+JSO, 4.5 bpp | 0.0925 | 0.0533 | 0.5898 |

C fails because the four-pass least-squares refit is amax-blind and re-inflates
a globally lowered sigma back to the clip; B keeps the max in range through the
refit. That one clipped entry was 99% of its column's SSE and 42% of the
module's H-weighted SSE (the other outlier columns clip in other rows).

## Census, 196 tensors (`experiments/dense_spread_census.py`, `results/dense_spread_census.json`)

H-weighted geomean (plain geomean in brackets), `B<nv` = tensors on which the
arm beats production NVFP4:

| role | Tessera-8 pre-fix | reach-aware | NVFP4 GPTQ+JSO 4.5 | FP8 RTN 8.0 | B<nv / A<nv |
|---|---|---|---|---|---|
| down_proj (28) | 0.0997 | 0.0761 | 0.0925 | 0.0258 | 27 / 16 |
| k_proj (28) | 0.1108 | 0.0794 | 0.0952 | 0.0259 | 27 / 3 |
| all 196 | 0.0872 [0.0742] | 0.0765 [0.0704] | 0.0955 [0.0941] | 0.0262 | 192 / 147 |

The four remaining losses are within 8% (layer-2 down_proj 1.08x, layer-1 k_proj 1.06x).

## Where h enters (`experiments/dense_h_alpha_unit.py`)

A per-input-column weight on the window Viterbi's branch metric is a no-op by
construction: the trellis runs down each column with columns independent, so a
per-column factor cannot move the path (identical to four digits at every
exponent). h enters through the row-scale refit and through cross-column
feedback. Refit weighted by h^alpha on top of B, plain / H-weighted: alpha 0.5
0.0740/0.0565; 0.75 0.0899/0.0476 (beats NVFP4 on both); 1.0, the explicit
diagonal-H objective, 0.3738/0.0282 at a 5x plain cost that served KL would have
to adjudicate. The exponent is a screen knob; the principled form is the full-H
objective, LDLQ on the window body, unmeasured there.

## GLM regression check (six experts, held-out rows, `tessera_window_wire.py --grids E4M3 --window-bits 14 --exl3 4 5 --no-tcq`)

`results/tessera_window_wire_e4m3_reach.json` against the identical run with the
reach disabled (`tessera_window_wire_e4m3_noreach.json`; EXL3 references
identical across the two runs): output-space geomean 0.998x at q960 and 0.998x
at q1216, weight-space 0.999x / 1.0005x, no tensor moved more than 0.5% either
way. Gaussian rows of width 4096 exceed the reach 17% of the time, so a sixth
of GLM rows changed and the wire did not move. The default holds on both models.

## Bytes

The fresh export is deterministic (two encodes of the worst unit are identical,
1,592,000 bytes) but no longer equals the pre-fix artifact: a changed row scale
re-routes every column's Viterbi path, so most codes and, through the refit,
most row scales differ (layer-0 q_proj 58% of code bytes, 98% of scales). The
`compare_stock_checkpoints.py` identity in the FP8-lane receipt is a statement
about the pre-fix encoder. Checkpoints: `/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-{gridbook,stock-twin}`.

## Open

Rotation (R1 folded into weights) would also disperse the tails, but on the
weight leg it lands on the same uniform-error floor this start reaches for free
and cannot touch `down_proj`'s input second moments; its value is the W4A4
route's activation leg, a per-model measured checkpoint transform, not a format
mode. Next on the encoder: the h-weighted refit and LDLQ on the window body under
served KL; the E2M1x2 sub-cap window arm on the NVFP4 route carries the same
CHANNEL-free plane and is unaffected.
