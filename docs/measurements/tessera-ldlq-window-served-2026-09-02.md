# LDLQ and an H-weighted refit on the window body: served KL 0.1512 -> 0.1046 (2026-09-02)

**Claim.** Two encoder-side levers on the FP8 route's window body — cross-column
error feedback (LDLQ) and a row-scale refit run under the input Hessian's exact
quadratic instead of the plain squared error — take Qwen3-0.6B's served
KL-vs-BF16 from **0.1512 to 0.1046** at the *same* 4.07-bpp wire, with top-1
agreement 78.06% -> 81.51%, and move six GLM experts' held-out out-space error
to **0.932x**. Neither lever changes a byte of the wire's grammar: the same
decoder, the same kernel, the same Gridbook lane, the same bpp. Both gates pass,
so the recipe is now what an exporter applies **when it is handed a Hessian**
(`export.DEFAULT_LDLQ_SIGMA/_BLOCK/_REFIT_OBJECTIVE` = 1.0 / 32 / `"hessian"`).
A weights-only encode is byte for byte what it was.

Commit: see `git log` for this file. Tests: `tests/test_ldlq_window.py` (10),
full suite 549 passed.

## The two levers

`compensate.py` already said where the coupling is: the trellis runs *down* each
column (over output features, which the loss says are independent) and treats
columns (input features, which the loss couples through every off-diagonal of
`H`) as independent. A per-column weight on the branch metric is a provable
no-op — a positive per-column scalar moves no per-column argmin — so the only
thing available is error feedback, and the only place `h` can enter is the
scale plane's refit.

* **LDLQ** (`encode_unit(ldl=..., ldl_block=...)`). The unit-block-lower factor
  of the regularised Hessian (`compensate.block_ldl`); column blocks are
  quantised last to first and each block's reconstruction residual is pushed
  into the blocks not yet quantised. It is a **schedule, not a second encoder**:
  the Viterbi carries no state across columns, so a block is exactly the columns
  of a whole-matrix pass restricted to that range. `test_block_diagonal_ldlq_is_the_plain_pass`
  pins that — a factor with no off-diagonal blocks reproduces the ordinary pass
  bit for bit (codes, row scales, sse) at block 32, 64 and 128.
  The CHANNEL plane is what makes it possible at all: a block plane's scales are
  fit to within-row column spans and would have to be scheduled with the blocks,
  so `encode_unit` refuses LDLQ under one rather than silently fitting a span to
  a target the next block has not produced.
* **The refit metric** (`refit_metric=`). With the codes `u` fixed, row `r`'s
  true proxy loss `(w_r - s u_r) H (w_r - s u_r)^T` is a scalar quadratic in the
  row scale, `A = u_r H u_r^T`, `B = w_r H u_r^T`, optimum `B/A` — the same
  closed form `refit_channel_scale` already solved for `H = I`. So the exact
  objective costs one matmul and has **no exponent to choose**; `h^alpha` is
  available as the diagonal special case and was measured beside it. The
  monotone guard (keep the landed word only where it lowers the row's error) is
  evaluated under whichever metric was asked for.
* **A reach floor** (`refit_reach_floor=`) was built and measured because LDLQ
  can push a compensated target past the body's reach the way the raw weights
  did before the reach-aware start. It loses; see below. It stays a flag.

## The Hessian, and what it may not have seen

`experiments/capture_h_full.py`, fp32 accumulation of `x^T x` per Linear from a
BF16 HF forward with hooks.

| field | value |
|---|---|
| source | wikitext-2-raw-v1 **train** (local `datasets` cache), 295,562 chars |
| text sha256 | `a5c5fd091a3486361e71eae1132fff141aeadd0ae51ebf688da4661752f853d3` |
| fit slice | 16,384 tokens, `seqlen` 512, ids sha256 `229c6f72307f7050…` |
| eval slice | 8,192 tokens from the next offset, ids sha256 `30ec7255e6934172…` |
| Linears | 196 (every `*_proj` under `model.layers`), 2.1 GB |
| diag max/median | median 552, max 2,085,591 |

The served KL corpus is `corpus_qwen_n8_s512.json` — wikitext-2 **test**
(`source_sha256 076d33efc447…`). Train and test are disjoint splits, so no arm
was fit on text it is graded on. The weight-space sweep is scored on the **eval**
slice's activations, disjoint from the fit slice the Hessian and the refit were
built from: the Gridbook LDLQ measurement that regressed every rung had a
hold-out half of which was its own fit rows, and that is the mistake this split
exists to not repeat.

`h_diag.pt` (the diagonal already on disk) was captured on the Qwen model card
plus a Tessera doc — also disjoint from the corpus, but only a diagonal, and
only ~4k tokens. It is not used here.

## Weight space: the sweep (`experiments/ldlq_window_sweep.py`)

Six dense Qwen3-0.6B Linears, the shipping FP8 wire (`wire_recipe(E4M3, 1024)`:
E4M3 grid, window body L=14, CHANNEL plane, span 1, four refit passes,
reach-aware start), materialised the way the stock twin materialises it. Score =
**out-space on the held-out eval rows**, `||X_ev (W - What)^T|| / ||X_ev W^T||`;
plain weight error and diagonal-h-weighted error alongside, because the levers
move them in opposite directions and quoting one alone chooses the answer.

Regulariser and block grid, out-space geomean over the six
(`results/tessera_ldlq_window_sweep.json`; the brief's {1, 3, 10} plus 0.3 to see
which side of the curve we were on, and blocks 32 and 128):

| sigma \ block | 32 | 128 |
|---|---|---|
| 0.3 | 0.8078x | 0.7929x |
| **1.0** | **0.7912x** | 0.8004x |
| 3.0 | 0.8077x | 0.8132x |
| 10.0 | 0.8297x | 0.8394x |

The curve is shallow and 1.0/32 is the best *common* setting — per-unit bests
disagree (three units prefer 0.3), and an exporter picks one for the whole
checkpoint, so 1.0/32 carries everything downstream.

Arms at that fixed pair (`results/tessera_ldlq_window_pair.json`):

| arm | out (geomean) | vs baseline |
|---|---|---|
| baseline (no LDLQ, plain refit) | 0.05305 | 1.0000x |
| refit h^0.5 only | 0.04886 | 0.9210x |
| LDLQ 1.0/32 | 0.04197 | 0.7912x |
| refit h^1.0 only | 0.04109 | 0.7745x |
| LDLQ 1.0/32 + reach floor | 0.03933 | 0.7414x |
| refit full-H only | 0.03920 | 0.7389x |
| LDLQ 1.0/32 + refit h^0.5 | 0.03748 | 0.7065x |
| LDLQ 1.0/32 + refit full-H + reach floor | 0.03599 | 0.6785x |
| LDLQ 1.0/32 + refit h^1.0 | 0.03382 | 0.6376x |
| **LDLQ 1.0/32 + refit full-H** | **0.03173** | **0.5982x** |

The exact full-H refit beats every diagonal power, which is the point of having
a closed form. The reach floor helps LDLQ alone (0.791 -> 0.741) and *costs*
once the refit is H-weighted (0.598 -> 0.679): the H-weighted refit deliberately
lowers scales onto the loud columns, and the floor forbids exactly that. It is
not the default.

Per unit, out-space, and the price in plain weight error:

| unit | baseline out | LDLQ | LDLQ+H | baseline plain | LDLQ+H plain | s: base / LDLQ / LDLQ+H |
|---|---|---|---|---|---|---|
| `layers.0.self_attn.q_proj` | 0.04021 | 0.03256 | 0.03087 | 0.06918 | 0.07185 | 4.6 / 4.5 / 4.7 |
| `layers.1.self_attn.k_proj` | 0.06135 | 0.04059 | 0.03251 | 0.06852 | 0.07197 | 2.0 / 3.1 / 2.9 |
| `layers.2.mlp.down_proj` | 0.02484 | 0.02057 | 0.00791 | 0.07382 | **0.35189** | 4.7 / 7.1 / 7.2 |
| `layers.13.mlp.down_proj` | 0.07651 | 0.06430 | 0.06413 | 0.07402 | 0.07592 | 4.8 / 8.5 / 11.1 |
| `layers.14.mlp.gate_proj` | 0.06521 | 0.05666 | 0.03642 | 0.06676 | 0.06948 | 6.3 / 5.6 / 5.3 |
| `layers.27.self_attn.o_proj` | 0.07287 | 0.05518 | 0.05504 | 0.07025 | 0.07286 | 2.9 / 4.9 / 5.5 |

The plain-error blow-up is confined to one unit and is the H-weighted refit
doing what it was asked: layer-2 `down_proj`'s Hessian diagonal has a
max/median of 2.2e6 and its top four columns carry 96% of the mass, so the
least-squares row scale under `H` lands on those columns and lets the rest go —
plain error 4.8x worse, out-space error 3.1x better. Every other unit moves less
than 5% in plain error. This is precisely the trade a weight-space number cannot
adjudicate, which is why the gate below is served KL.

Encode cost: 1.4x geomean per unit at block 32 (0.90x to 2.32x); whole-checkpoint
exports on an idle sparklina were 1288 s (LDLQ) and 1092 s (LDLQ + refit) against
861 s for a flags-off export on a contended sparky, so the honest statement is
"same order, roughly 1.3-1.5x", not a clean ratio — the boxes and the contention
differ. `experiments/results/tessera_ldlq_window_{sweep,pair}.log` carry the
per-unit seconds, which are the paired numbers.

## Served (the gate)

Vanilla vLLM `vllm/vllm-openai:latest`, RepoDigest
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` (the
same manifest on both boxes), `--enforce-eager`, port 8001, gpu-memory-utilisation
0.30. The served object is the **stock twin** — the compressed-tensors
per-channel FP8 materialisation of the same wires, which vanilla vLLM loads with
no plugin. Corpus `corpus_qwen_n8_s512.json`, 4088 scored positions,
KL-vs-BF16 over top-1024 support (a lower bound), teacher
`qwen_teacher_bf16_v028` — the same instrument as every row of
`tessera-stock-lane-served-2026-09-02.md` and `tessera-dense-reach-fix-2026-09-02.md`.

| arm | bpp (wire / resident) | KL all | KL confident | top-1 |
|---|---|---|---|---|
| baseline: reach-aware start, no LDLQ (`qwen_ldlq_base`) | 4.071 / 8.025 | 0.151163 | 0.100486 | 78.06% |
| **+ LDLQ sigma=1.0 block=32** (`qwen_ldlq_ldlq`) | 4.071 / 8.025 | **0.112864** | 0.080449 | 81.60% |
| **+ LDLQ + full-H row-scale refit** (`qwen_ldlq_ldlqH`) | 4.071 / 8.025 | **0.104577** | **0.072552** | 81.51% |
| production NVFP4 GPTQ+JSO (W4A4), stock-lane receipt | 4.5 / 4.5 | 0.511 | | 62.6% |
| FP8 RTN per-channel (W8A8), stock-lane receipt | 8.0 / 8.0 | 0.0205 | | 91.2% |

**Tail checked, because a lower mean KL can hide a heavier tail** (the
H-weighted refit deliberately lets some rows' plain error rise to serve loud
columns, so this is the shape to look for). It does not: per-position `kl_lower`
improves monotonically at every quantile reported.

| arm | mean | p99 | max |
|---|---|---|---|
| baseline | 0.151163 | 0.924707 | 3.612462 |
| + LDLQ | 0.112864 | 0.847693 | 3.545269 |
| + LDLQ + full-H refit | **0.104577** | **0.756224** | **2.295350** |

(`jq '.all | {kl_lower_mean, kl_lower_p99, kl_lower_max}'` over
`kl_base_vs_lina.json`, `kl_ldlq.json`, `kl_ldlqH.json`.)

**0.692x of the baseline's KL at identical bytes** (LDLQ alone 0.747x) -- both
arms 4.071 wire / **8.025 resident**, so that ratio is the two levers and
nothing else. **The comparisons to the other two encoders cross a residency and
are not 4-bit wins:** this arm serves W8A8 at 8.025 resident. Against
production NVFP4 GPTQ+JSO at 4.5 wire / 4.5 resident it is 4.9x better (was
3.4x) *at 8.025 resident against 4.5, and on a W8A8 A-side against a W4A4
one*;
against FP8 RTN at 8.0 / 8.0 -- the comparator at this arm's own residency --
it is still 5.1x behind (was 7.4x). At equal residency Tessera loses on both
legs on this model: 5.1x on the 8-bit leg, and 1.254x on the 4-bit leg where
the E2M1x2 wire serves at 4.5 resident (0.640 vs 0.511, W4A4,
`tessera-stock-lane-served-2026-09-02.md`). What these two levers move is the
wire at a fixed residency, which is the claim this receipt makes. The margin is 31%, not 2%, so the "one corpus draw, 4088
positions" caveat does not decide this one.

Three controls, because the arms were dumped on a second box (sparklina) after
the coordinator moved GPU work there:

* **The flags-off export byte-reproduces the shipped artifact.**
  `compare_stock_checkpoints.py` on the shipped
  `qwen3-0.6b-tessera-e4m3-reach-stock-twin` against a fresh flags-off export
  from this branch: **392 shared quantized tensors, 392 identical, 0 different.**
  The plumbing did not move the baseline.
* **The baseline re-served on the new box is the number on record.** 0.151163 /
  78.06% — identical to six digits to the sparky measurement in the reach-fix
  receipt. Baseline and arms therefore share a box.
* **The two boxes are bit-identical on this instrument.** A BF16 teacher dumped
  on sparklina (`qwen_teacher_bf16_lina`) has `ids` and `lps` arrays
  `np.array_equal` to sparky's `qwen_teacher_bf16_v028`, and every arm scores the
  same against either. There is no cross-box term in these numbers.

Compares: `/mnt/shared/tessera-runs/ldlq/kl_{base,ldlq,ldlqH}_vs_{lina,sparky}.json`;
serve logs `serve_qwen_ldlq_*.log` in the same directory.

## GLM cross-check (`experiments/tessera_window_wire.py --ldlq-sigma 1.0 --ldlq-block 32 --refit-metric hessian`)

Six GLM-5.3-Flash expert tensors (L5/L20/L42 x gate/up), the true wire through
`encode_linear` -> bytes -> `read_unit_artifact`, `H = x_fit^T x_fit` from the
same fit rows every other fitted quantity in that harness uses, scored on the
held-out eval rows. Out-space geomean (`results/tessera_ldlq_glm_window.json`):

| arm | q960 (3.770 bpp) | q1216 (4.770 bpp) |
|---|---|---|
| window L=14 | 0.08203 | 0.04211 |
| + refit full-H | 0.9915x | 0.9932x |
| **+ LDLQ 1.0/32** | **0.9321x** | **0.9355x** |
| + LDLQ + refit full-H | 0.9321x | 0.9414x |
| vs EXL3 K=4 (LDLQ, W4A16) | 1.218x -> **1.135x** with LDLQ | 0.625x -> 0.585x |

No regression anywhere: the GLM gate was "geomean <= 1.00x vs LDLQ-off" and the
measurement is 0.932x (q960) / 0.941x (q1216) for the default recipe, i.e.
LDLQ + full-H refit; LDLQ alone is 0.932x / 0.936x. The full-H refit is worth
~1% here and ~26% on dense Qwen — the asymmetry is the Hessian's, not the
encoder's: GLM expert inputs are near-Gaussian
(`tessera-w4a4-changes-the-lever`), Qwen's dense rows are not.

## The gate, and what "default" means

| gate | required | measured | verdict |
|---|---|---|---|
| served KL on Qwen | strictly better than 0.1512 | 0.1046 (0.692x) | **pass** |
| GLM six-expert geomean | <= 1.00x vs LDLQ-off | 0.932x / 0.941x (q960 / q1216) | **pass** |

Both pass, so the recipe is default — but "default" here cannot mean "always
on", because `encode_unit` cannot invent a Hessian. The default is the *answer to
having one*: `export.DEFAULT_LDLQ_SIGMA = 1.0`, `DEFAULT_LDLQ_BLOCK = 32`,
`DEFAULT_REFIT_OBJECTIVE = "hessian"`, applied by
`experiments/export_gridbook_tessera.py` whenever `--hessian` is supplied and
overridable per run. The encoder's own `ldl`/`refit_metric` stay `None`, so a
weights-only encode is byte for byte the artifact it always was
(`test_a_weights_only_encode_is_untouched_by_the_defaults`), and the three
constants are pinned by a test that names this receipt.

**Provenance cost.** An activation-aware encode is not reproducible from the
weights alone. Every exporter writes an `activation_aware` block — into the
Gridbook manifest and into `tessera_config.json` — carrying the settings *and*
the Hessian file's own provenance (source, text sha, fit token count, ids sha),
so a replay knows what it needs and a merge can refuse parts built against
different activations. The block is built in one place,
`ActivationSource.config_block`, so the manifest and the config cannot disagree
about which capture shaped the bytes.

## Scope, and what is not measured

* Every served number *here* is Qwen3-0.6B, dense, E4M3/CHANNEL/window L=14 at
  4.07 bpp, one corpus draw. ~~The E2M1x2 sub-cap window arm (the NVFP4 route)
  carries no CHANNEL plane and was not touched; LDLQ under a block plane is
  refused by `encode_unit`, not merely untested.~~ **Closed 2026-09-02:** the
  refusal's stated reason was wrong — the plane is read once per pass, before
  the block loop, and refit once after it, so the schedule and the plane never
  interleave on any plane kind. Both levers now run on the LUT plane, and the
  4-bit route's own served receipt is
  `tessera-ldlq-lut-plane-served-2026-09-02.md`.
* The GLM leg is weight/out-space only. There is no served GLM A/B of these
  levers.
* The Hessian's token budget (16k) was not swept. `tessera_ldlq_token_scaling`
  measured that axis for the TCQ wire, not this one.
* The reach floor is implemented, measured and **not** default. It is the right
  thing under LDLQ alone and the wrong thing under an H-weighted refit; nothing
  here establishes which way it goes at other rungs.
* Encode-time numbers are contended on the baseline side. A clean paired
  whole-checkpoint timing on one idle box is not done.
* ~~**The default lives only in `experiments/export_gridbook_tessera.py`.**~~
  **Closed 2026-09-02** (follow-up commit): the recipe now lives in
  `tessera.export.ActivationSource`, which `export_checkpoint`,
  `export_checkpoint_streaming` and the experiment driver all take and all
  call `for_unit` on, so the library and the script cannot carry two copies of
  it. The exported config records an `activation_aware` block naming the
  capture (`text_sha256` / `fit_tokens` / `fit_ids_sha256`);
  `merge_tessera_parts.check_configs` compares every field of it and refuses
  parts built against different Hessians, at different settings, or one aware
  and one not; `encode_settings_from_config` refuses to replay an
  activation-aware config at all. `tests/test_merge_guard.py` gives each field
  its own failing case and asserts every guarded path resolves in a config the
  exporter actually wrote — the check that the earlier 8/13 vacuity lacked.
* ~~**Both levers are CHANNEL-plane only, and now say so.**~~ **Superseded
  2026-09-02.** `refit_metric` and `refit_reach_floor` were being *silently
  dropped* under a block plane or at `scale_refit=0`, and `encode_unit` began
  refusing both. The refusals are now scoped to what is actually unimplemented:
  `refit_metric` runs on CHANNEL and on LUT (`_refit_scales_lut_metric`) and is
  refused only under S6b; `refit_reach_floor` stays CHANNEL-only because it
  raises a *row* scale and a block scale already tracks its own sixteen
  weights. `export_glm53_tessera.py` regained `--hessian` with the LUT
  implementation.

**Fable consultations:** none. The advisor call settled the one design question
(LDLQ inside `encode_unit` as a block-sequential schedule sharing one plane,
rather than the standalone `compensate.compensated_targets`, which is unsound
under a row scale fit over all columns), and the row-scale optimum is a closed
form, so no sub-question was hard enough to escalate.

## Files

Code: `src/tessera/encode.py` (`encode_unit(ldl, ldl_block, refit_metric,
refit_reach_floor)`), `src/tessera/scale_channel.py` (`refit_channel_scale(metric,
floor)`, `land_at_least`), `src/tessera/export.py` (the three constants and the
pass-through), `experiments/capture_h_full.py`, `experiments/ldlq_window_sweep.py`,
`experiments/export_gridbook_tessera.py` (`--hessian` and friends),
`experiments/tessera_window_wire.py` (the GLM arms), `tests/test_ldlq_window.py`.

Artifacts (shared, both boxes): `/mnt/shared/tessera-runs/ldlq/` —
`h_full_qwen06b.pt`, `x_eval_qwen06b.pt`, `{base,ldlq,ldlqH}-stock-twin`,
`{ldlq,ldlqH}-gridbook`, the chain scripts and every log.

Also fixed on the way: `experiments/serve_and_dump_kl.sh` ran `docker rm -f` on a
hardcoded container name *before* taking the serve lock, so a second worker's
run killed the serve that held the lock and then waited on it. The name is now
`TESSERA_KL_NAME` and the removal happens inside the lock.
