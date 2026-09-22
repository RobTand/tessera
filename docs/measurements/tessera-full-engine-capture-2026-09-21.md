# Full-engine resource capture on master, 2026-09-21

Status: **partial.** The capture did not complete. This document records what was
measured, the two blockers that stopped it, and what remains before tessera#399 can
close.

Scope: one served Tessera artifact (Qwen3-0.6B, mixed three-family, tp1, eager,
`resident`), producer source at master `affae68ed`, runtime
`vllm/vllm-openai@sha256:61fc8a89…`. Nothing here is a price: no fixed-resource term, no
timing term and no admission verdict moves.

Receipts: `/mnt/shared/tessera-runs/receipts/399-qwen3-0.6b-20260921/`.

## Summary

The 2026-09-19 landscape on tessera#399 expected the software to be complete and only the
measurement to remain. That holds for the **derivation layer** and not for the **capture
driver**:

1. tessera#557 fixed the **exporter**, so the artifact had to be re-exported before
   `worker_startup` could close. Re-exported; the manifest gains exactly the 20,484 bytes
   the 2026-09-18 refusal named, and all 228 checkpoint tensors are byte-identical.
2. Re-deriving the 2026-09-18 capture on master leaves `unclassified_allocation_count` at
   358. That count does not fall to re-derivation alone.
3. The D37 consumer on PrismaQuant `origin/main` refuses every v2 report master produces,
   on an unregistered observation name.
4. **Blocker A**, fixed here: the `"unset"` allocator binding PR #565 introduced reached
   the capture container verbatim and aborted torch before any engine started.
5. **Blocker B**, open at the time of writing: `experiments/step4_capture_driver.py`
   imports an API that `37e89f576` retired on 2026-09-16.

## The artifact re-export

tessera#557 (`c38730701`, "price every resident row in the export manifest") changed
`experiments/export_tessera_serving.py`, `src/tessera/serving_parts.py` and
`src/tessera/decode.py` — the **producer** of the manifest, not the consumer of it. The
2026-09-18 capture ran on `qwen3-0.6b-mixed3-l0-v2`, written 2026-09-17 under contract
v29, whose manifest predates that change. Re-deriving a report over it can therefore never
close the dense startup check, however current the derivation code is.

Re-exported on master as `qwen3-0.6b-mixed3-l0-v3` (PB action `1b8ccafcdb48`, sparky,
257 s, 196/196 units), from the same plan and input scales as v2:

| module | family | v2 | v3 | delta |
|---|---|---|---|---|
| `model.layers.0.self_attn.qkv_proj` | `TESSERA_BF16` | 8,388,608 | 8,404,992 | +16,384 (`row_scale`) |
| `model.layers.0.mlp.gate_up_proj` | `TESSERA_NVFP4` | 3,538,944 | 3,543,044 | +4,100 |
| `totals.resident_mode_bytes` | — | 443,179,008 | 443,199,492 | **+20,484** |

The NVFP4 delta is 4 bytes of `trellis_input_global_scale` plus 4,096 bytes of memoised
trellis tables, as `serving_parts.dense_resident_bytes_resident_mode` and
`decode.replay_table_bytes` price them. +20,484 is exactly the figure the 2026-09-18
`worker_startup` refusal named.

Nothing else moved. `experiments/export_byte_identity.py` hashed every tensor of both
checkpoints independently (PB action `4b6a1acc623d`, dl380g10, 11 s):

```
"tensors_a": 228, "tensors_b": 228,
"only_in_a": [], "only_in_b": [], "differing": [], "identical": 228
```

Manifest members outside the resident figures are equal too; only `git`, `written` and
`serving_gate.contract_version` (29 → 33) differ. v3 is a drop-in for v2.

## Re-deriving the 2026-09-18 capture on master

`experiments/report_full_engine_resources.py` at `affae68ed`, replayed over the
2026-09-18 capture (PB action `443cbe43b48a`, dl380g10, 655 s). Output:
`report-20260918-rederived-on-master/`.

| field | 2026-09-18 report | master re-derivation |
|---|---|---|
| schema | `tessera.full_engine_resource_report.v2` | same |
| `derived.admission.verdict` | refused | refused |
| `worker_startup` | refused | refused, identical string |
| `history_join`, `external_closure`, `provenance_admission`, `cache_capacity`, `timing_partition` | closed | closed |
| `unclassified_allocation_count` | 358 | 358 |
| `uncharged_allocation_count` | 0 | 0 |
| `reserved_peak_bytes` | field absent | present, null |
| `reservation_slack_peak_bytes` | field absent | present, null |
| `reservation_witness` | field absent | present, null |

Two results worth stating plainly, because both were expected to go the other way:

- **The 358 unclassified allocations do not resolve by re-derivation.** `9b6705a9c` sets
  the boundary-ledger version from the ownership derivation having run; it does not
  reclassify a site. The reclassification needs
  `report_full_engine_resources.py --boundary-classification`, whose input
  `experiments/full_engine_boundary_classification.py` builds from **two** captures under
  two per-Linear assignments and which refuses a pair that ran one assignment twice.
  Without it every census-shared site stays `pending_548`. No substitution capture exists.
  A fresh capture may still shrink the count — `89cc53ad2` changed capture-side route
  telemetry — but re-derivation alone does not, and that is measured rather than inferred.
- `reserved_peak_bytes` is null because the 2026-09-18 worker predates `1ca98ac6b`, which
  samples the allocator's reserved extent beside the startup allocation. Only a new
  capture can make it non-null, which is what #558 expects of it.

## Blocker A: the "unset" allocator binding aborted the container

PR #565 (`c8d300478`) made the serving configuration bind `PYTORCH_CUDA_ALLOC_CONF`,
because a reservation witness transfers to a serve only under an equal allocator segment
policy, and it spelled "no policy" as the explicit string `"unset"`.
`capture_full_engine_resources.prepare` pops the key when the binding is that sentinel, so
the worker never sees it.

`experiments/step4_capture_launch.py` sits one level further out and is what starts the
container. It copied `config["environment"]` into `docker run --env` verbatim, so the
container's own interpreter did receive `PYTORCH_CUDA_ALLOC_CONF=unset` — and torch parses
that variable at `libc10_cuda.so` load time, before any of this repository's code runs:

```
terminate called after throwing an instance of 'c10::Error'
  what():  i < config_.size() INTERNAL ASSERT FAILED at
  "c10/core/AllocatorConfig.h":124 ... Index out of bounds in ConfigTokenizer
```

Measured on sparklina, PB action `a8b77498d17f`, preflight returncode 133, no engine
started and nothing in the ledger to say why. An explicit "no policy" binding became a
process abort.

Fixed as `step4_capture_launch.bound_container_environment`, which delegates to
`require_allocator_policy` so the grammar keeps one home rather than being restated. The
launch summary now records the binding the run was taken under. Red first on PB x86:
action `8fd4d88c3f05` (3 failed, 3 passed); green `9c2f3994a7ad` (6 passed). The refused
attempt is kept at `capture-preflight-attempt1-refused-alloc-conf-sentinel/`.

Only this launcher copies the environment block verbatim today, but all five shipped
serving configurations stamp `"unset"`, so any future launcher that does the same is a
latent abort. Noted on tessera#558.

## Blocker B: the step-4 driver targets a retired decoder

With blocker A fixed the capture reached the driver and refused (PB action
`51b34146a60b`, preflight returncode 3):

```
ImportError: cannot import name 'NVFP4_MODULE_PREFIX' from 'tessera.serving.ext'
```

`37e89f576` ("Retire the A4 whole-weight expansion (ops.py + span-2 CUDA decoder)",
2026-09-16, contract v29 → v30, an ancestor of `affae68ed`) deleted `serving/ops.py`,
`serving/csrc/tessera_nvfp4.cu`, the NVFP4 entry of `ext.NATIVE_EXTENSIONS`,
`ext.NVFP4_MODULE_PREFIX` and `ext.require_tessera_ext`.
`experiments/step4_capture_driver.py:81` and
`experiments/step4_route_qualification.py:55` still import that API.

Nothing caught it because the 2026-09-18 run installed
`producer-source-0f98fc010-b399` — a **pre-retirement** `src/` seal with only
`experiments/` overlaid. The step-4 driver has never run against a post-retirement `src/`.

### What the retirement did to the proof

The preflight exists to close one hazard, in the driver's own words: a container that
cannot compile the decode extension does not fail, because the route substitutes a
pure-torch decoder and serves. The serve is numerically identical, so no quality check
notices; a resource capture measures allocation, and the two decoders do not allocate
alike.

On master, for this artifact in `resident` mode, that hazard is no longer present on the
dense path, and the code says so directly:

| family | modules | prepared by | decoder stamped | on a failure |
|---|---|---|---|---|
| `TESSERA_FP8` | 110 | `native_window.prepare_dense_native_module` (`fp8_route.py:385`) | `native_window_gemm` (`fp8_route.py:77`) | raises (`fp8_route.py:425`) |
| `TESSERA_BF16` | 1 | same (`bf16_route.py:846`) | `native_window_gemm` (`bf16_route.py:114`) | raises (`bf16_route.py:877`) |
| `TESSERA_NVFP4` | 1 | `kernel_a4.a4_span2_gemm` (`nvfp4_route.py:40`) | `native_span2_gemm` (`nvfp4_route.py:273`) | — |

`fp8_route.apply` is explicit: it refuses "to fall back to a materialised weight path this
build no longer wires". The compute behind `native_window` is
`tessera.window_gemm`, which is Triton (`window_gemm.py:51-63`), so it compiles or raises;
it is not an optional `cpp_extension`. The one entry left in `ext.NATIVE_EXTENSIONS`, the
window GEMV (`tessera_window_gemv`, `when_unavailable.resident = substituted/torch_window`),
is loaded by `fp8_gemv` and `bf16_route.prepare_bf16_gemv` — the GEMV lane, not the dense
path these three routes take.

So the library leg of the qualification has no subject on this artifact's resident dense
path, and the dispatch leg becomes the whole proof: every dense FP8 and BF16 dispatch must
name `native_window_gemm`, the NVFP4 dispatch must name `native_span2_gemm`, and no
dispatch may name `torch_window` or `torch_materialize_stock`.

## Blocker C: the D37 consumer refuses every v2 report master produces

PrismaQuant's `consume_full_engine_resource_report` at PQ `origin/main` (`2a47a93bdc`),
run read-only over the master re-derivation (report sha256
`f30dd55d9b1c7c10fea3026d95675e8acf9c2d0db990e973c480414b49537e76`; verdict beside it as
`pq-verdict.json`):

```
RuntimePriceError: report observations: observation 'allocator_config' is not a
registered observation ([... 'reservation_slack', ...]); a producer that adds an
observation registers its name and its shape check here first
```

Tessera PR #565 (`6e30ddbe3`) added `observations.allocator_config`
(`full_engine_resource_partition.py:1097, 1151`). PrismaQuant registered `reservation_slack`
from that same PR (`prismaquant/full_engine_resource_report.py:239`) but not
`allocator_config`, and `_observations` refuses an unregistered name — the rule working,
not failing.

The refusal is on the **name**, not the value: `allocator_config` is `null` in this report
and is still refused. So the consumer leg of tessera#399's acceptance cannot pass today for
any master-produced report, including the one a qualifying capture will produce — and a
capture under the 2026-09-21 configuration binds `"unset"`, so the field will be populated
rather than null. The fix is one registration plus its shape check on the PrismaQuant side;
recorded on RobTand/prismaquant#718 rather than taken up here.

## Frozen inputs

| input | value |
|---|---|
| configuration | `stage/control/qwen3_0.6b_tessera_full_engine_mixed3_20260921_attested.json`, sha256 `e8de9ce100534b981baf12991ed885622e074ebed40058ed4c745adff1667463` |
| producer source | `producer-source-affae68ed`, plain `git archive affae68ed`, 1,406 files, no overlay |
| runtime image | `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` (vLLM 0.28.0, `g2cf0a6915`, 4,829 core files unchanged across plugin install) |
| artifact | `/mnt/shared/tessera-runs/allocated/qwen3-0.6b-mixed3-l0-v3` |
| observer build, collector, workspaces, calibration | unchanged from 2026-09-18 |
| allocator segment policy | `PYTORCH_CUDA_ALLOC_CONF` bound `"unset"` |

## How the work was admitted

Every test, export and capture ran through PrismaBuild. Two submission choices are worth
recording so they are not re-litigated:

- **No `--measurement`.** A pool measurement implicitly retains the submitting host, and
  the coordinator runs on sparky while the capture belongs on sparklina. `--exclusive`
  supplies the property the measurement needs — the whole GPU capacity of one box — without
  the host-retention constraint.
- **No `--container-image`.** It would bind the image digest into the action key, but it
  also requires the `container-image-v1` worker capability. The launcher performs the
  stronger, in-band check anyway: `serving/runtime_image.require_pinned` refuses unless the
  pinned reference appears in the resolved image's `RepoDigests`, and the result is
  recorded in each phase's `runtime-image-declaration.json`.

The capture also now carries its CPU reservation into the container. PrismaBuild's
execution policy requires the granted mask to be preserved inside containers, and an
observation taken on an oversubscribed box is not the observation the configuration
describes. `step4_capture_launch` reads `os.sched_getaffinity(0)` once and spells it
`--cpuset-cpus`, following `experiments/run_glm_native_construction.py` and the measurement
in `experiments/container_limits_probe.sh` that a CFS quota changes no CPU count a library
can read. The admitted runs received 8 of sparklina's performance cores.

## PrismaBuild actions

| action | what | host | result |
|---|---|---|---|
| `443cbe43b48a` | master re-derivation over the 2026-09-18 capture | dl380g10 | executed, 655 s |
| `1b8ccafcdb48` | v3 re-export | sparky | executed, 257 s |
| `4b6a1acc623d` | v2/v3 per-tensor byte identity | dl380g10 | 228/228 identical, 11 s |
| `b2acb6d3983a` / `bdb9a909df15` | cpuset test, red / green | dl380g10 | 3 failed / 3 passed |
| `a8b77498d17f` | capture attempt 1 | sparklina | preflight rc 133 (blocker A) |
| `8fd4d88c3f05` / `9c2f3994a7ad` | allocator-binding test, red / green | dl380g10 | 3 failed 3 passed / 6 passed |
| `51b34146a60b` | capture attempt 2 | sparklina | preflight rc 3 (blocker B) |

The D37 consumer leg ran on the coordinator, read-only, under `CUDA_VISIBLE_DEVICES=""`:
it opens two JSON documents and allocates nothing on a device.

## What remains before tessera#399 can close

1. Rewire the step-4 preflight and route qualification onto master's real inventory
   (blocker B), and re-run the four capture phases.
2. Read the resulting `report.json`: the six-domain table,
   `unclassified_allocation_count`, `uncharged_allocation_count`, `reserved_peak_bytes`,
   `reservation_slack_peak_bytes`, `derived.admission.verdict`.
3. The 358 unclassified allocations need a **second** capture under a different per-Linear
   assignment, and the `--boundary-classification` join built from the pair. That is not in
   this run's scope and is the most likely reason a first master capture still refuses.
4. Register `allocator_config` in PrismaQuant's consumer (blocker C) before the D37 leg
   can read any master report.
