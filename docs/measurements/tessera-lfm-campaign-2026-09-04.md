# Full LFM MoE campaign: identities and completed preflights

## 1. Scope at 19:35 UTC

This is an **in-flight campaign record**, not a served result. At this point
one encoder partition is active and the second is queued behind the existing
Sparklina reservation. No full-model student load, census, or KL has completed.
No positive routed-MoE contract cell is claimed by this record.

The uncut source is `/mnt/shared/models/LFM2.5-8B-A1B-BF16`. Its 24 layers
contain 22 routed stacks of 32 experts, with three projection containers per
expert. The common explicit plan quantizes all routed stacks at E4M3
`q256=1024`, leaves dense weights BF16, and leaves routers on the exporter's
mandatory passthrough path. The population is checked against that plan and
the source, not accepted from a hardcoded expected count.

Both partitions use the immutable source commit
`5c56b7e5ad777136661cc854f2576772b7bf3274` and image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
The image is vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, not the existing
dense contract's vLLM 0.28.0 image. Issue #126 separately makes that scope
machine-readable before any MoE promotion.

Campaign root: `/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2`.
Its `plan.json` SHA256 is
`2cf926e1061b2ade55383e08eeabbc00d3cf5825e4f5defe437c64a626405504`.

| partition | PrismaBuild action | physical reservation | status at this section |
|---|---|---|---|
| 0/2 | `d7fd29d4f73f2ae3ca665a5c65ca4beba0b52da4b1c74982949f55b9347aa1ec` | Sparky, gpu=2, mem_gb=32, cpu=2 | active |
| 1/2 | `80fcb8d1cddb6059c795f5205860c430375728365ee3b11e24b7ea4a5d833817` | gx10-6b77, gpu=3, mem_gb=32, cpu=2 | queued, not yet claimed |

Each action additionally caps its container at 32 GiB memory with no additional
swap allowance, records power and host-memory telemetry every ten seconds,
and has an outer two-hour timeout. Guarded output namespaces stop deployed
PrismaBuild retries from repeating a failed encode. Both halves must retain
this exact source snapshot even while the final serving controller changes.
The earlier `astra-campaign-r1` attempt refused a router plan entry before
encoding; it is retained as a failed attempt, not reused or hidden.

The bounded one-expert profile and its actual CUDA memory observations are in
`tessera-lfm-encode-preflight-2026-09-04.md`. They are throughput and memory
measurements, not served quality evidence.

## 2. Reused teacher verified before the student serve

PrismaBuild action
`db05c6cd514f71ff48fb1ea4a33cef8fb517bfd2ac5267953637031fcf10ccce`
completed on dl380g10 with worker return code zero. It checked all four exact
file hashes below, then reran `experiments/kl_reference_usable.py` on the
teacher/corpus pair. Receipt CAS:
`eef10494c2b48f70ea659f52322c55f8a4768f2fa78793f1df954dafdee8bb4d`.
This was CPU-only reference validation, not a new GPU serve.

Teacher directory:
`/mnt/shared/tessera-runs/ts5/lfm25/teacher-eugr-0281rc1-r3`.

| file | SHA256 |
|---|---|
| `teacher_bf16.json.npz` | `a72b54e5f87026acca5aab748460e6bf47934a04693945e5c497558a55608248` |
| `teacher_bf16.build.json` | `b1048b955666f753de22a0ce6817dce843988867c8229fab983419ec5c0d20a5` |
| `teacher_bf16.meta.json` | `67c7dac89a439a485370367daec92d5677b5008b9700bd6cd4b81ab0eccf0158` |
| `/mnt/shared/tessera-runs/ts5/lfm25/teacher-gate/corpus_n8_s512.json` | `2fdd48eeab69109c6222ef2f857815d2b35d5422747815c495e0467712751d44` |

The unchanged reference gate reports 4,096 positions, returned K=1,025,
mean retained support mass 0.9904, 1,968 confident positions, next-token
top-1 accuracy 36.33%, true token present in support 100%, and median
true-token rank 2. It is usable as a reference. The original dump's build
sidecar is complete, eager, and names the same exact EUGR image above; its
metadata names the full BF16 source, the LFM tokenizer, prefill regime, and
the corpus contract. The student must match those axes.

## 3. Integrated sidecar preflight tests

The duplicate-shard-owner refusal and the explicit-plan population refusal
were integrated together at `82bf28a`, then tested through PrismaBuild action
`b3e8f8a1685e1c89b786be6ce952e60ccdf2b753a982dcf94ff503d860100773`.
Worker return code zero; test return code zero; `test_ts5_sidecar_check.py`
reports **9 passed in 1.04 seconds**. Receipt CAS:
`8881ee3e1c8d86f11f8e5125482d94f45267098a79d6b1828d21e2c86a7f03be`.

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Red-before-green receipts for the individual changes are recorded in
`moe-sidecar-duplicate-shards-2026-09-04.md` and
`tessera-explicit-plan-coverage-2026-09-04.md`. The eventual merger must run
from the integrated controller containing those guards; the older frozen
encoder outputs remain compatible. A loadable merged config is published
only after exact source/plan/code/runtime identity, complete source ownership,
shard hashes and tensor-header coverage, and explicit-plan obligations pass.
