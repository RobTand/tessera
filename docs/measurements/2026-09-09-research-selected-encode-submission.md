# Sizing the research selected encode, 2026-09-09

Issue #442, bullet 1. This is a submission record, not a result. It states what
the two GPU jobs demand and where each number came from, so the jobs can be
submitted from a paste once the hold on GPU launch lifts.

## Where the numbers come from

The measured input is the 09-07 export's own PB receipt, action key
`c6de6cb4d34e7a41937c732fcb9839756cd681dd16f253f9d8bc42f4262defff`. That run is
the closest ancestor of this driver: same model, same 33.5 GB Hessian, same
exporter, on `sparklina` under tag `gb10`.

- `detail.resource_telemetry.memory_peak_bytes` 74346086400, so 69.2 GiB peak
  against a declared `resources.mem_gb` 96 on a box whose cap is 104.
- `resource_scope.memory_max_bytes` 103079215104, which is 96 GiB exactly. The
  declaration is a kernel ceiling, not a hint: the job is killed at it.
- `resources` cpu 4, gpu 1; `gpu_admission.gpu_memory_budget_bytes` 17179869184.
- `detail.elapsed_s` 202.17, `execution_timeout_ceiling_s` 7200 at the time.

That run passed `--cached-units` **and** `--hessian`
(`full-model-original-r1024/export-command.json`), so the 69.2 GiB peak already
carries the whole-file Hessian load. `src/tessera/export.py` loads it with
`torch.load(..., weights_only=False)` and has no lazy path for a `.pt`, so the
resident cost is the same whether the plan holds 2142 units or 11. A `--layers`
smoke is cheaper in time, not in memory.

What the 09-07 run did **not** do is enter the encoder, so its 16 GiB GPU budget
is not a measurement of one. The encoder-profile job is, and it measured a peak
rather than only declaring a cap. Its encoder phase records
`after.cuda_peak_reserved` 11423186944, so 10.64 GiB reserved
(`encoder-profile-original382/profile-01/joined_campaign_batch.json`; the input
preparation phase before it reached 400556032, so 0.37 GiB). Scope, from that
run's own README: one joined batch of all 32 layer-9 `w2` projections at
E4M3/q896, through `_measure_anchor_batch` into `encode_linears`, on the frozen
`382a1a97` producer. Its PB row declared `gpu_memory_gib` 48, which is 4.5x what
it used.

Two consequences for these two jobs.

The GPU cap here is 32 GiB, three times the measured peak, not the 48 the
profile declared. The headroom is wider than that ratio says, because the
profile's batch is wider than this job's. The launcher execs
`experiments/full_model_research_selected_checkpoint.py`, which imports
`export_tessera_serving` and runs it as a subprocess (`:20`, `:107`); that
script encodes through `encode_linear_planes`, at two call sites, and both pass
exactly one weight per call. The expert path slices one expert's one projection
out of the packed shard tensor with `packed_expert_weight` and frees it after
(`export_tessera_serving.py:2013`). The dense path passes a row slice of one
Linear, narrower than a whole Linear when a unit is partitioned (`:2104`).
Neither reaches the 32-unit joined batch the profile measured through
`encode_linears`, which has no caller under `src/tessera` at all. So the
10.64 GiB bounds this job's encoder peak rather than describing it, and the cap
is priced against the bound because a per-unit peak on this tree has never been
recorded.

`mem_gb` stays 96, and now it is a sum of two measurements rather than a copied
declaration. GB10 is unified memory, so a CUDA reservation is charged to the
same cgroup as the host allocation: 69.2 GiB of cached intake and whole-file
Hessian, plus 10.64 GiB of encoder reserve, is roughly 80 GiB against a 96 GiB
ceiling. The headroom is about 16 GiB. Declaring 104 would take the whole box
for a margin the measurements do not ask for.

Encode time comes from the anchors run, `native-moe-wires-r1024/complete.json`:
96 units in 222.7 s wall, so 2.32 s per unit end to end. 2142 units is roughly
83 minutes, and the trimmed `--layers 3` plan of 106 units is roughly 4 minutes
of encode on top of a load the 09-07 wall clock puts near 200 s.

Both gb10 workers publish `timeout_ceiling_s` 86400, so neither request clamps.

## The two jobs

Order matters. The smoke runs first because the fresh expert encode has no
receipt at any scale: both 09-07 exports passed `--cached-units`, so the
exporter's own expert encode has never run on this model. A `--layers 3` plan
reaches both encode paths, dense and routed, and refuses if it reaches only one.

Smoke, roughly 10 minutes:

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --demand gpu=1,mem_gb=96 --gpu-memory-gb 32 --cpus 4 \
  --tag gb10 --priority -10 --timeout-s 1800 \
  -- python3 experiments/run_full_model_research_selected_checkpoint.py \
     --out /mnt/shared/tessera-measurements/research-selected-20260909/smoke-layers3 \
     --execution-json /control/experiments/research_selected_moe_lfm_tp1.json \
     --layers 3
```

Full encode, roughly 85 minutes, submitted only after the smoke's
`export-proof.json` reads back:

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --demand gpu=1,mem_gb=96 --gpu-memory-gb 32 --cpus 4 \
  --tag gb10 --priority -10 --timeout-s 7200 \
  -- python3 experiments/run_full_model_research_selected_checkpoint.py \
     --out /mnt/shared/tessera-measurements/research-selected-20260909/full-tp1 \
     --execution-json /control/experiments/research_selected_moe_lfm_tp1.json
```

`--out` sits under `/mnt/shared` because that bind and the read-only `/control`
checkout are the only two the launcher mounts.

## What the launcher hands the container

`experiments/checkout_runtime_identity.py` drops to uid 1000 before it execs the
driver, and the stock vLLM image leaves `HOME=/root`. The launcher's own
`mkdir` runs as that same uid, so the directories it creates are writable after
the drop: every host file the 09-07 launcher wrote under
`full-model-original-r1024` is owned 1000:1000, `launch.json` and
`export-command.json` and `export-proof.json` among them. That is the half of
the HOME fix that is easy to get wrong, because a root-owned cache directory
denies the same way an unset HOME does. Every JIT cache this
encode fills expands a home relative default: Triton writes `$TRITON_CACHE_DIR`
or `$HOME/.triton`, and `src/tessera/kernel_window_gemv.py:196` reads
`$TORCH_EXTENSIONS_DIR` or `~/tmp/torch-ext-gemv`. Unset, the first kernel
compile is a `mkdir` denial under `/root` that surfaces as a build failure
rather than as a permission message. The launcher creates `home`,
`triton-cache` and `torch-extensions` on the host, so they exist with the right
owner before the container opens, binds their parent into the container, and
records all four paths in `launch.json`.

They are **not** under `--out`. `--out` is on the shared mount, and a JIT cache
there is its own recorded failure: a build that takes a file lock on NFS can
hang on the baton, and the kernels it holds are valid only for the box that
compiled them. `--jit-root` defaults to `~/tessera-runs/jit` on the worker and
the launcher refuses a root under `/mnt/shared`, because the natural thing to
write is `ROOT / name` and that is the wrong answer. Each job gets its own
subdirectory, so no run inherits another tree's compiled kernels.

The encoder takes the fused Triton path whenever CUDA and triton are both
present (`src/tessera/window_viterbi.py:206`), so this is on the path of every
run, not a corner. No prior job in this container shape reached the encoder:
both 09-07 exports were cached intake.

## Limits

Every number above was measured, and none of them was measured on this job. The
69.2 GiB is a cached intake on this model; the 10.64 GiB is a 32-unit encoder
batch at a different rung on the frozen producer tree, so it bounds a per-unit
export loop rather than describing one; the 2.32 s per unit is the anchors run. Their sum is an estimate of a combination nothing has run. The
smoke exists because that is the one direction sizing cannot cover, which is
whether the fresh encode runs at all, and its own telemetry replaces both
borrowed numbers before the full encode is submitted.
