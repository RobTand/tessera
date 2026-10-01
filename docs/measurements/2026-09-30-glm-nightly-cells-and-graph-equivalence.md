# GLM on the vLLM nightly: eager cells, and why no graph serve is eager's

Status: measured 2026-09-30 on sparky (GB10, sm_121), TP 1; the drafter's TP 2
acceptance on both Sparks. Refs tessera#702, tessera#695.

This page records two results on the image the GLM-5.3 release serves on:

1. **Contract v48 mints eight eager cells on the nightly image.** Until v48
   every GLM cell named image `f8dbe1a0`, so a serve on the nightly resolved
   every Tessera module to no cell. One eager route census of u1 stub B on the
   nightly covers all 21 Tessera modules in both phases.
2. **No CUDA-graph serve of GLM-5.3 on this image computes what eager
   computes.** vLLM, not Tessera, takes a different branch in each of two
   places. Because of this, v48 claims no `compiled` scope, and the compiled
   half of tessera#702 is blocked on a design decision, described in
   [What this blocks](#what-this-blocks).

## The image

| Field | Value |
|---|---|
| Image | `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a` (eugr nightly 155ce16b plus the nccl230 layer; the U4 stack's pin) |
| vLLM | `0.30.1rc1.dev336+gaf5b4857e.d20260929` |
| torch | `2.13.0+cu130` |
| FlashInfer | 0.7.0 |
| `v1/worker/gpu/model_runner.py` sha256 | `217e4b87c0bd22c4e98640775d7b4e3349000f79d37f9f7d3e569768786c3a7f` |
| `v1/worker/gpu/cudagraph_utils.py` sha256 | `cc090e6749029baaa5055135215acc112ce034a32fc1e886f0f02054ad488b58` |
| Attention backend | `FLASHINFER_MLA_SPARSE_SM120` (the image's own choice; no `--attention-backend`, GLM53 NoPE plugin off) |

A newer eugr nightly exists:
`eugr/spark-vllm@sha256:e813795a18ea115211fd46b4fbbbb0be5e76d49d26c8f4b8c0763ac5b9f9f07c`
(vLLM `0.30.1rc1.dev420+gd71f66260.d20260930`, FlashInfer 0.7.1, the same
torch). Its runner, `cudagraph_utils.py`, both GLM indexer files and
`model_states/default.py` are byte-identical to the pinned image's, so cause 2
below holds there by the same source. Its sparse MLA backend file differs
(`flashinfer_mla_sparse_sm120.py`, sha256
`d6344e4410db65814638f5c830ef3f9d12d64cdb4bf40e6af37326910f1dd0c9`), and no
graph arm ran on it. Its load smoke is in [Newer nightly](#newer-nightly).

## The eager cells

The census ran `experiments/routed_fused_census.sh` from a clean checkout of
master `7d5ea711cb`, TP 1, resident, eager, with the release serve's settings:
`fp8_ds_mla` KV, triton MoE backend, FlashInfer autotune off,
`TESSERA_RESEARCH_GLM53_NOPE=0`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, and
`--require-lane tessera_routed_fused_mma_e4m3 --require-lane
tessera_routed_fused_value --expect-modules 21`. Driver:
`experiments/graph_attest_nightly/census-nightly.sh`.

Result: verdict `served`, `problems: []`, 21 modules in both phases, both
required lanes engaged. Every module recorded exactly the launch pair the
`f8dbe1a0` E4M3-instruction receipt of the same stub recorded
(`experiments/results/glm53_u1_stub_b_e4m3mma_tp1_eager_census.json`):

| Scope | Modules | Launch | Rungs (q256) |
|---|---|---|---|
| E4M3 dense | 8 | `tessera::fused_window_dense` / `native_fused_window_dense_e4m3mma` | 832, 960, 1024, 1088 |
| BF16 dense | 8 | `tessera::fused_window_dense` / `native_fused_window_dense_folded` | 832, 880, 960, 1024, 1088 |
| E4M3 routed | 4 | `FusedRoutedWindowMoE.__call__` / `native_routed_fused_window_e4m3mma` | 896, 928, 1024, 1088 |
| BF16 routed | 1 | `FusedRoutedWindowMoE.__call__` / `native_routed_fused_window_folded` | 1024 |

Contract v48 mints eight cells, one per scope and regime, each on exactly
these rungs, eager only, with the launches its `f8dbe1a0` twin names (the
validator derives the same set). Their ids carry the derived runtime suffix,
because the `f8dbe1a0` cells hold the bare scope ids. The release artifact
carries q256 1024 in all four scopes.

Receipt: `experiments/results/glm53_u1_stub_b_nightly_tp1_eager_census.json`
(sha256 `a5f1a4a1...`) with its serve log beside it; checkpoint config
`experiments/results/glm53_u1_stub_b_config.json` (sha256 `499bbad8...`).
`tests/test_glm_nightly_census_cells.py` replays it against the packaged table.

## Graph serves against eager

### Method

Each arm is one serve of stub B (u1, 8 layers, no MTP) on the pinned image,
TP 1, `max_num_seqs 8`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, the release serve's
flags. Each serve runs the tessera#508 equality suite twice
(`experiments/glm53_508_graph_qual/equal-508.py`, second pass `<arm>-r2`):
greedy completions with top-20 logprobs for batch 1 at prompt lengths
1..2000, batches 2..8 admitted paused, and two back-to-back repeats; 48
choices per pass. A graph choice is a **member** when its token ids and every
top-20 list equal the same choice of some eager run of the same batch. A
patched runner hook (`experiments/glm53_508_graph_qual/digest/usercustomize.py`,
`GA_DISPATCH_LOG`) records the graphs the runner captured and every step's
replay size, so a result is read only if graphs actually replayed. Driver:
`experiments/graph_attest_nightly/drive.sh`; arm scripts beside it.

### Results

Each arm ran the suite twice; both passes agree in every column. Members are
out of 48 choices. "Changed" counts choices whose generated tokens differ from
the nearest eager outcome. "Worst" is the largest top-20 logprob departure on
the nearest eager outcome's shared prefix, in nats. "First departure" is the
earliest decode step (and context length) at which any choice departs.

| Arm | Execution | `compilation_config` | Graphs captured | Members | Changed | Worst | First departure |
|---|---|---|---|---|---|---|---|
| sbE1 | `--enforce-eager` | none | none | 48 | 0 | 0 | none |
| sbE2 | `--enforce-eager` | none | none | 48 | 0 | 0 | none |
| sbG1 | graphs | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | 1, 2, 4, 8 | 0 | 33 | 0.768 | step 0 |
| sbG2 | graphs | FULL_DECODE_ONLY, sizes 1..8 | 1..8 | 0 | 33 | 0.768 | step 0 |
| sbP1 | graphs | `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` (vLLM runs FULL_DECODE_ONLY) | 1, 2, 4, 8 | 0 | 33 | 0.768 | step 0 |
| sbG3 | graphs | `{"mode":"NONE"}`, FULL_DECODE_ONLY, sizes 1..8 | 1..8 | 0 | 30 | 0.612 | step 1, context 5 |
| sbG4 | graphs | default mode, `custom_ops ['all']`, `--kernel-config` IR priority `['vllm_c', 'native']` for both norms, FULL_DECODE_ONLY, sizes 1..8 | 1..8 | 0 | 30 | 0.612 | step 1, context 5 |
| sbE6 | `--enforce-eager`, `max_model_len 2048` | none | none | 48 | 0 | 0 | none |
| sbG6 | graphs, `max_model_len 2048` | `{"mode":"NONE"}`, FULL_DECODE_ONLY, sizes 1..8 | 1..8 | **48** | 0 | 0 | none |

sbE6 is judged against sbE1 and sbE2, sbG6 against sbE6, sbE1 and sbE2
(48/48 against sbE6 alone as well). Every graph arm replayed every captured size (FULL replays per size in the
dispatch log: size 1 808, size 8 44 to 180), so no result is read off a
serve whose graphs never ran. Common serve settings: `max_model_len 4096`,
`max_num_batched_tokens 2048`, chunked prefill, no prefix caching,
`--kernel-config '{"enable_flashinfer_autotune":false}'`, KV 4 GiB.

The eager serves reproduce each other exactly (48/48 across serves), so the
pool is stable and any departure is the graph serve's.

### Cause 1: vLLM's compile mode switches the operator implementations

With `enforce_eager=False` vLLM defaults to `mode: VLLM_COMPILE`. GLM-5.3
(`Glm5Next`) is not a torch-compiled model on this image, but the mode still
resolves `custom_ops` to `['none']` and the `rms_norm` and
`fused_add_rms_norm` IR ops to `native`. Eager resolves `['all']` and
`['vllm_c', 'native']`. So the graph serve runs other norm kernels everywhere,
prefill included: sbG1, sbG2 and sbP1 differ from the first sampled token
(step 0). Capturing every size (sbG2) changes nothing, so padding is not the
cause. `FULL_AND_PIECEWISE` (sbP1) is overridden to `FULL_DECODE_ONLY` by
vLLM, because the model provides neither a compiled submodule nor breakable
graphs, and behaves exactly like sbG1.

`mode: NONE` (sbG3) removes this cause: the engine log resolves
`custom_ops ['all']` and `rms_norm ['vllm_c', ...]`, and the prefill token of
every choice equals eager's.

sbG4 isolates it: the default compile mode with `custom_ops ['all']` and the
eager IR op priority for both norms returns, response for response, exactly
sbG3's tokens and top-20 logprobs (all 20 response files identical). The mode
itself changes nothing for this model; the operator resolution is the whole
of cause 1.

### Cause 2: the capture freezes the GLM indexer's long-context branch

With cause 1 removed, sbG3 still leaves every choice a non-member. The
departure starts at a decode step whose context is five tokens or more, never
before: batch 1 with a one-token prompt matches eager through step 3 and
departs at step 4 (context 5); a two-token prompt departs at step 3 (context
5); every longer prompt departs at step 1.

The GLM indexer takes a host-side branch on the batch's `max_seq_len`:

- `models/glm5next/common/sparse_indexer.py:124`
  (`_fill_short_decode_causal_indices`): when `max_seq_len <= index_topk`
  (2048 for GLM-5.3), fill each decode row with the causal indices
  `0..position` and skip the indexer's logits and top-k.
- `models/glm5next/nvidia/sparse_indexer.py:499` calls it before the logits
  and top-k path.
- `v1/worker/gpu/model_states/default.py:199`: a FULL capture builds its
  metadata with `max_seq_len = self.max_model_len`, "so the graph is valid at
  any replay".

At `max_model_len 4096` the capture sees `4096 > 2048`, so every captured
graph holds the logits and top-k path, and every replay runs it. Eager runs
the causal fill whenever the batch's longest sequence is at most 2048 tokens,
which is every step of this suite. Both paths attend to every token of a
short context. They differ in order: the top-k path selects `index_kpool = 4`
token pools by score and expands them, while the fill lists tokens in causal
order. The sparse MLA kernel then reduces in a different order. With one pool
(context at most 4) the orders coincide, which is exactly where sbG3 still
matches. The effect is a reordered sum over the same token set, not a lost
token. It still moves outputs: logprobs depart by up to 0.61 nats on the
nearest eager outcome's shared prefix, and 30 of the 48 choices change at
least one generated token (the one-token prompt keeps eager's tokens through
step 13 while its logprobs have already moved). Stub B is eight layers of a
larger model and amplifies small differences; the size of the effect on the
full model is not measured here.

File digests (vLLM package paths, read from the pinned image):

| File | sha256 |
|---|---|
| `models/glm5next/common/sparse_indexer.py` | `a3ab1edda8490b8e21c1c240e07e8c8fcd0bb34246a9ed1f64acfe067d15067c` |
| `models/glm5next/nvidia/sparse_indexer.py` | `549f94234e44000c0b9995745328fd262b7ff69e62c6089ad049f65a7573914d` |
| `v1/worker/gpu/model_states/default.py` | `f1d34d5c8c03be6e5afec8c2a24480ec259ca7c87392462f4e52d6c4eb775733` |
| `v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | `102ca08793d567f95598eefeb34c3f6ec50b3b9d704f162d1402b97b12b5777b` |

**The discriminating pair confirms it.** The mechanism predicts that a
capture at `max_model_len 2048` (so the capture's `max_seq_len <= index_topk`)
takes the causal fill and matches eager. sbG6 is sbG3 with only
`max_model_len` changed from 4096 to 2048: 48 of 48 choices are members, in
both passes, with every captured size replayed. Its eager control sbE6 is
48/48 against the 4096 eager serves, so eager does not depend on
`max_model_len` here and the pools are the same object. On this image a
FULL_DECODE_ONLY graph serve of GLM-5.3 with `mode NONE` is eager-equivalent
exactly when `max_model_len <= index_topk` (2048), which excludes every
release configuration.

**Smallest reproducer:** stub B, TP 1, `mode NONE`,
`cudagraph_mode FULL_DECODE_ONLY`, capture sizes 1..8, `max_model_len 4096`;
one request, prompt of one token, greedy, 32 tokens. Step 4 (context 5 =
`index_kpool + 1`) departs from eager at every capture size the suite
replays.

### Tessera's launches under graphs

A compiled route census of the sbG3 configuration
(`census-nightly.sh --compiled`) recorded the same 21 launch pairs as the
eager census, in both phases. The census tool refused it:
`compiled records must be shape-polymorphic (M*)`. The tool reads the last
Python-executed record of each module; under a `mode NONE` graph serve no
Python runs at replay, so the records are capture-time and prefill shapes, not
`M*` traces. The tool can attest a torch-compiled serve, not a CUDA-graph
serve of an uncompiled model. This is recorded as a tool limit, not worked
around.

### Drafter (MTP) under graphs

Method: stub B with its MTP layer (`stub-B-mtp1`, draft checkpoint
`stub-B-mtp1-draft`), speculative config
`{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}`, the same
image, flags and equality suite as the arms above, TP 1. Tessera at
`f8fb860450`, whose `src/` is identical to tessera#752's head (`3bd3c8f6da`,
merged as `7d5ea711cb`). Arms ran from
`experiments/graph_attest_nightly/plan-mtp-smoke.txt` through `drive.sh`.

| Arm | Execution | `compilation_config` | Graphs captured (target and draft) | Members | Changed | Worst | First departure |
|---|---|---|---|---|---|---|---|
| mtE1 | `--enforce-eager` | none | none | 48 | 0 | 0 | none |
| mtE2 | `--enforce-eager` | none | none | 48 | 0 | 0 | none |
| mtG1 | graphs | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | 2, 4, 8, 16 | 0 | 33 | 0.768 | step 0 |
| mtG3 | graphs | `{"mode":"NONE"}`, FULL_DECODE_ONLY, sizes 1..16 | 2, 4, ..., 16 | 0 | 24 | 0.570 | step 1, context 5 |

mtE1 and mtE2 are judged against each other, the graph arms against both.
Each arm in the table ran the suite twice, and both passes agree in every
column.

Not yet run: the drafter's discriminating pair at `max_model_len 2048` (mtE6
eager, mtG6 `mode NONE` graphs), the MTP counterpart of sbE6 and sbG6. Both
are in `plan-mtp-smoke.txt` and wait for a sparky slot.

- **Speculative decoding does not stop vLLM from capturing.** The runner
  captured two managers, `ModelCudaGraphManager` (target) and
  `SpeculatorCudaGraphManager` (draft), at the same sizes, and replayed both at
  every captured size with equal counts (mtG3: size 2 794 replays, sizes 4 to
  16 40 to 50 each). With `k = 1` each decode request carries two tokens, so
  only even sizes are captured: mtG3 asked for 1..16 and got 2..16.
- **The drafter adds no third cause.** mtG1 changes prefill tokens, as sbG1
  does (cause 1). mtG3 keeps every prefill token and first departs at context
  5, as sbG3 does (cause 2). Greedy speculative decoding emits the target's
  tokens, so the departures are the target's arithmetic.
- **Acceptance on the stub is not a quality signal.** mtE1 accepted 56 of 2362
  drafts; mtG1 60 of 2356; mtG3 50 of 2368. A drafter on the 8-layer stub is
  not expected to predict the stub's tokens, so these counts only show that
  the drafter ran.

**Draft vocabulary interception, measured at stub scope.** A hook in the arm
(`usercustomize.py`, `[t695] drafter load peak`) logs the torch allocator's
peak across the draft model's load. mtE1 (interception on) goes from
21.457 GiB to a peak of 25.316 GiB. mtN1 is mtE1's tree with
`install_for_current_config` returning at entry (the draft rename kept); it
peaks at 26.498 GiB. The difference, 1.182 GiB, is the draft embedding the
interception avoids (154880 x 4096 bf16 = 1.18 GiB). The
[vocabulary lifetime page](mtp-draft-vocabulary-lifetime-2026-09-30.md) found
no difference in vLLM's own load figures, which cannot see this transient;
the allocator peak does. mtN1 is 48/48 against mtE1 and mtE2, so the
interception changes no output. The TP 2 full-model load peak is not
measured.

**TP 2, the release artifact.** Window `u4-A8SE752-20260930T2257Z` served the
A8S release artifact on both Sparks, TP 2, eager, Tessera `7d5ea711cb`, with
MTP `k = 1`:

- MTP acceptance (a screen, not the ship metric): 1665 of 2023 drafts
  accepted, rate 0.823, mean accepted length 1.823, 64 prompts.
- The draft route census was refused by the census tool itself: `draft census
  requires the stock Glm5NextMTP model class`. The tool admitted only the eugr
  image's draft class module; the nightly's stock class lives in another
  module and carries no source-name mapper. tessera#769 records it;
  tessera#773 admits the nightly's class through its digest-recognized
  interface and in-code rename.
- No graph serve with MTP ran at TP 2. Graph-vs-eager decode speed and TTFT
  with MTP are not measured.

## What this blocks

The contract cannot truthfully publish a GLM-5.3 cell as `compiled` on this
image at a release-sized `max_model_len`: no graph configuration reproduces
eager's arithmetic there, and the reason is a vLLM branch Tessera does not
own. The options are a design decision and are listed in the coordinator
mail, not taken here:

- a compiled scope whose receipt names the measured vLLM divergence (the
  same tokens reduced in another order at contexts of at most 2048), with
  the release's quality gates measured on the graph serve itself;
- a vLLM change, upstream or as a runtime plugin, so a FULL capture keeps
  the short-context branch where eager takes it (for example one graph per
  size and branch, chosen by the step's `max_seq_len`), then the suite
  again;
- eager-only for GLM-5.3 on this stack.

Tessera does not patch vLLM's eager path to manufacture equality: that would
change the eager baseline every cell and KL receipt was measured on.

## Newer nightly

Not yet run. The load smoke of
`eugr/spark-vllm@sha256:e813795a...` (smE1 eager, smG3 `mode NONE` graphs at
`max_model_len 4096`, stub B, TP 1) is in `plan-mtp-smoke.txt` and waits for a
sparky slot. The sources behind cause 2 are byte-identical to the pinned
image's (see [The image](#the-image)), so moving the U4 stack to it would not
remove that cause; nothing on this page was measured on it.

## Measured and not measured

Measured: the eager census; equality of the arms in both tables; the replayed
graph sizes, for the target and the draft; the causes above, from the engine
logs and the cited sources; the drafter load peak with and without the
vocabulary interception at stub scope; MTP acceptance of the A8S release
artifact at TP 2, eager.

Not measured:

- the drafter's `max_model_len 2048` pair (mtE6, mtG6);
- the newer nightly's load smoke (smE1, smG3);
- graph-vs-eager speed on this stack, with or without MTP (not taken while no
  graph configuration is eager-equivalent; the TP 2 graphs+MTP latency leg did
  not run);
- CUDA-graph capture time and pool memory at the release's `max_num_seqs` and
  TP 2;
- any TP 2 graph arm;
- the TP 2 draft route census on the nightly (refused by the tool; tessera#773
  fixes the tool, and no served census has run with it);
- the TP 2 full-model drafter load peak;
- quality (KL) of a graph serve against BF16.

## Receipts

Committed: `experiments/results/glm53_u1_stub_b_nightly_tp1_eager_census.json`
and `.log`; the scripts under `experiments/graph_attest_nightly/`.

Raw, with `SHA256SUMS`, under
`/mnt/shared/tessera-measurements/graph-attest-20260930/`: `stub-B/` (every
arm's serve receipts, dispatch logs, engine logs, `summary-*.json`),
`census/` (eager and compiled censuses), `vllm-sources/` (the cited
files) and `mtp/` (the drafter arms' receipts and engine logs,
`summary-mtE1.json`, the plan and the drive log).

The TP 2 window's evidence (`mtp-acceptance.json`, `census.log`,
`census-wrapper.json`, `run-summary.json`) is under
`/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/A8SE752-2c-nightly-20260930-r2/run/2c-evidence/`.
