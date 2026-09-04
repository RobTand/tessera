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

## 4. Half0 completed; half1 claimed without GPU overlap

Sparklina claimed action
`80fcb8d1cddb6059c795f5205860c430375728365ee3b11e24b7ea4a5d833817`
at 19:45:05 UTC after the #113 reservation released. Its PB snapshot is
`524d56f4827769265a109145103805d5af9e99f9`; half0's snapshot is
`96bad5276ff8785ad989886b7849688fd55b3af4`. These are independently generated
PB snapshot commits of the same frozen source tree, not two evolving encoder
branches. The checked merger compares their sealed code/source/plan/runtime
identities rather than assuming the short Git stamp is shared.

Half0 finished with worker return code zero and a published result:
`bc0016e36d5a139184239a590dc16a2b8d686df35febf80ae4d1cfab3a682772`.
The exporter reports 2,358 seconds, including its final write; PB reports
2,435 seconds for the whole invocation. It encoded all 1,056 projection
containers in its 11 planned stacks and wrote 1,158 output tensors. Its
partial checkpoint has no loadable `config.json`, by design.

| half0 quantity | measured value |
|---|---:|
| quantized parameters | 3,875,536,896 |
| wire bytes | 1,959,034,880 |
| wire bits per quantized parameter | 4.043898809523809 |
| container bytes | 1,959,821,590 |
| resident-mode prepared bytes | 3,883,466,752 |
| BF16 passthrough bytes | 953,739,904 |
| checkpoint bytes, including passthrough | 2,913,700,766 |

Independent CPU verification action
`396bb6ef48c969efd7100f59d6b5ed0f85a87e2f2b690a42e5f3bdd916c146fb`
read and hashed the completed shard, matched the manifest's sealed output
hash, and summarized its immutable telemetry log. Its successful worker
result is in CAS
`df2df4df7b43147e64fbe18e362e57c824d47f4cd1fe69405a0cdd270cf6ad16`.
The code identity sealed into the partition is
`a57ccfd37af16e82f99b172abda46c87652a855eb53f7cd78ac0326b4116b524`.

| half0 file | independently checked SHA256 |
|---|---|
| `model.safetensors` | `f3504c20e11188b0705556e5daee473788db66b02bd86083f0631fe6b16f9f82` |
| `tessera_serving_manifest.json` | `55d774d0cc512faa3ac46c310f043b783a742f02c2125ac834fd7513bb2a3102` |
| `tessera_part_config.json` | `dc1d84f140429a1102a0f02416b60758445e9dd24c1e7e3074fe0ea0f736a3d7` |
| `model.safetensors.index.json` | `ded483bd7d3b9dbfea37df9c25cc87210b99abc44111c2d9fec93dea032302fe` |

The 243 telemetry samples span 15:12:00–15:52:26 EDT, 2,426 seconds, with a
maximum sample gap of 11 seconds. Maximum sampled board power was 72.89 W;
minimum host MemAvailable was 111,953,124 KiB; swap use was 364 KiB at both
ends. Trapezoidal integration of `power.draw` over that interval gives
157,440.135 joules. This is gross sampled board energy including startup,
without idle-power subtraction, not a paired efficiency comparison. The
telemetry SHA256 is
`2b9c17b7cda875025b739168b2f2b6453ad15bd12157c29b472b7d25a5ade779`.

**Pool outcome caveat, not concealed as a clean client exit.** The verifier's
original worker returned zero after 7.4 seconds and published the result
above, but a live-claim/reaper race had already launched three guarded retries
within about 1.5 seconds. The submitting client first returned the retry's
`mkdir: File exists` failure. The authoritative `done/` record retains the
successful original `worker_detail.returncode=0`; both terminal records exist.
This is filed as PrismaBuild #36 with a read-only worker investigating; no
PrismaBuild code, queue record, or active encoder reservation was changed.
An earlier verifier attempt used the wrong partial-manifest key and failed
before producing a result; the corrected verifier explicitly reads
`export_partition.identity`, not the merged manifest's `export_identity`.

At this section's completion, half1 is still encoding. There is still no
full-model merge, student serve, or positive MoE contract promotion to report.
