# GLM on the vLLM nightly: eager cells, and why no graph serve is eager's

Status: measured 2026-09-30 on sparky (GB10, sm_121), TP 1. Refs tessera#702,
tessera#695.

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
`model_states/default.py` are byte-identical to the pinned image's, so the
graph finding below applies to it unchanged. Its load smoke is in
[Newer nightly](#newer-nightly).

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

Every graph arm replayed every captured size (FULL replays per size in the
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

**Not yet run: the discriminating pair.** The mechanism predicts that a
capture at `max_model_len 2048` (so `max_seq_len <= index_topk`) takes the
causal fill and matches eager. Arms sbE6 and sbG6 (the sbG3 configuration at
`max_model_len 2048`, and its eager control) are queued behind other GPU work
on sparky. Until they run, cause 2 rests on the cited source and on the
departure pattern above (context 5, never before), not on a controlled
experiment. A 2048-token context is not a release configuration either way.

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

Not yet run. The drafter arms (tessera#695: stub with MTP, eager against
FULL_DECODE_ONLY graphs, capture sizes up to `max_num_seqs * (1 + k)`) are
queued behind the arms above. TP 2 drafter arms need both Sparks and are not
scheduled.

## What this blocks

The contract cannot truthfully publish a GLM-5.3 cell as `compiled` on this
image at a release-sized `max_model_len`: no graph configuration reproduces
eager's arithmetic there, and the reason is a vLLM branch Tessera does not
own. The options are a design decision and are listed in the coordinator
mail, not taken here:

- a compiled scope whose receipt names the attributed vLLM divergence, with
  the release's quality gates measured on the graph serve itself;
- an upstream vLLM change so the capture does not freeze the long-context
  branch;
- eager-only for GLM-5.3 on this stack.

Tessera does not patch vLLM's eager path to manufacture equality: that would
change the eager baseline every cell and KL receipt was measured on.

## Newer nightly

Not yet run. A load smoke of
`eugr/spark-vllm@sha256:e813795a...` is queued. Its sources that decide the
graph finding are byte-identical to the pinned image's (see
[The image](#the-image)), so moving the U4 stack to it would not change the
result on this page.

## Measured and not measured

Measured: the eager census; equality of the arms in the table; the replayed
graph sizes; the causes above, from the engine logs and the cited sources.

Not measured: the discriminating pair at `max_model_len 2048`; the drafter
arms; the newer nightly's load smoke; graph-vs-eager speed on this stack (not taken while no graph
configuration is eager-equivalent); CUDA-graph capture time and pool memory at
the release's `max_num_seqs` and TP 2; any TP 2 arm (both Sparks are needed,
and no window was requested); quality (KL) of a graph serve against BF16.

## Receipts

Committed: `experiments/results/glm53_u1_stub_b_nightly_tp1_eager_census.json`
and `.log`; the scripts under `experiments/graph_attest_nightly/`.

Raw, with `SHA256SUMS`, under
`/mnt/shared/tessera-measurements/graph-attest-20260930/`: `stub-B/` (every
arm's serve receipts, dispatch logs, engine logs, `summary-*.json`),
`census/` (eager and compiled censuses) and `vllm-sources/` (the cited
files).
