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
- `resources` cpu 4, gpu 1; `gpu_admission.gpu_memory_budget_bytes` 17179869184.
- `detail.elapsed_s` 202.17, `execution_timeout_ceiling_s` 7200 at the time.

That run passed `--cached-units` **and** `--hessian`
(`full-model-original-r1024/export-command.json`), so the 69.2 GiB peak already
carries the whole-file Hessian load. `src/tessera/export.py` loads it with
`torch.load(..., weights_only=False)` and has no lazy path for a `.pt`, so the
resident cost is the same whether the plan holds 2142 units or 11. A `--layers`
smoke is cheaper in time, not in memory.

What the 09-07 run did **not** do is enter the encoder, so its 16 GiB GPU budget
is not a measurement of one. The encoder-profile job on the same tree declared
`gpu_memory_gib` 48 against `total_memory_gib` 50
(`encoder-profile-original382/profile-submit.jsonl`), and that is the number
carried here.

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
  --demand gpu=1,mem_gb=96 --gpu-memory-gb 48 --cpus 4 \
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
  --demand gpu=1,mem_gb=96 --gpu-memory-gb 48 --cpus 4 \
  --tag gb10 --priority -10 --timeout-s 7200 \
  -- python3 experiments/run_full_model_research_selected_checkpoint.py \
     --out /mnt/shared/tessera-measurements/research-selected-20260909/full-tp1 \
     --execution-json /control/experiments/research_selected_moe_lfm_tp1.json
```

`--out` sits under `/mnt/shared` because that bind and the read-only `/control`
checkout are the only two the launcher mounts.

## What the launcher hands the container

`experiments/checkout_runtime_identity.py` drops to uid 1000 before it execs the
driver, and the stock vLLM image leaves `HOME=/root`. Every JIT cache this
encode fills expands a home relative default: Triton writes `$TRITON_CACHE_DIR`
or `$HOME/.triton`, and `src/tessera/kernel_window_gemv.py:196` reads
`$TORCH_EXTENSIONS_DIR` or `~/tmp/torch-ext-gemv`. Unset, the first kernel
compile is a `mkdir` denial under `/root` that surfaces as a build failure
rather than as a permission message. The launcher creates `home`,
`triton-cache` and `torch-extensions` under `--out` on the host, so they exist
with the right owner before the container opens, and records the three paths in
`launch.json`.

The encoder takes the fused Triton path whenever CUDA and triton are both
present (`src/tessera/window_viterbi.py:206`), so this is on the path of every
run, not a corner. No prior job in this container shape reached the encoder:
both 09-07 exports were cached intake.

## Limits

Nothing here is measured. The two commands are sized from an ancestor run and a
profile job, and the smoke exists because the sizing could be wrong in the one
direction sizing cannot cover, which is whether the fresh encode runs at all.
