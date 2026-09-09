# Explicit packed checkpoint execution — 2026-09-08

The bounded native GLM controls selected `ResearchSelectedMoeConfig` through
Python. A normal exported checkpoint could not make that selection:
`TesseraConfig.get_quant_method` constructed the ordinary materialized FP8
owner. The change tracked in tessera#430 and tessera#431 carries an explicit
research execution declaration from producer input to checkpoint loading.
It does not establish full-model loadability, memory fit, quality or speed.

## Input and boundaries

The standalone exporter accepts `--research-selected-moe-json PATH` with this
closed, versioned object (the chunk bound below is an example, not a measured
optimum):

```json
{
  "schema": "tessera.research_selected_moe.v1",
  "max_experts_per_chunk": 8,
  "decode_backend": "torch",
  "expected_tensor_parallel_size": 2
}
```

The shared parser is in `src/tessera/moe_execution.py`. The exporter reads the
input bytes once and binds the literal UTF-8 text, SHA-256 and parsed object
into partition identity and the manifest. The checkpoint carries the parsed
object at `quantization_config.research_selected_moe`. Partition assembly
requires the config, manifest and sealed identity to agree. Explicit choices
also enter compile identity. Omitting the declaration preserves the ordinary
owner and its compile identity.

The declaration requires resident mode and declared routed E4M3 targets.
Existing eager, backend and topology checks still apply. This does not change
wire bytes, source passthrough, the family menu, native-cell qualification or
`TESSERA_SERVE_MODE=resident|streamed`. Export residency estimates still describe
the ordinary materialized representation; they are not evidence of packed-owner
fit. Full-model generation and peak residency must be measured separately.

Producer and serving identity are separate contracts. Cached-unit
`encoder_source_sha256` hashes producer code extensions, excluding JSON;
contract-only publication therefore need not invalidate priced wires when code
is unchanged. Export partition identity includes the runtime contract and
requires homogeneous parts. Serving consumes the exported wire contract; it
does not require the producer source hash to equal its own package hash. This
bridge changes producer Python code, so a new producer freeze must include it
before pricing; these statements do not authorize relabelling old receipts.

## Validation

All runs below used PrismaBuild on dl380g10, Python 3.14 and torch 2.11.0+cpu.
No CUDA device was visible. Native threads were bounded to one per test worker.
The focused runs reserved four CPU cores and 8 GiB; the selected compatibility
suite used the supported 16-shard fanout, two workers and 6 GiB per shard,
with preferred physical CPU assignments and no overflow cores.

Evidence root:
`/mnt/shared/tessera-measurements/glm-delivery-readiness-20260908/`.

| Run | Actual result | Terminal/action evidence |
| --- | --- | --- |
| Checkpoint regressions on base `b9cda5b03e` | 24 failed, 8 passed | `bridge-before-audit.json`, action `b9d1e3d02721d4e4af6e624c894744915452e8291522fa09aead1b37fd6e278f` |
| Export regressions on base | 3 failed: missing CLI input | `export-before-audit.json`, action `6dbf07122ed3faf0bf3940eeaa5ec3795ea4123d0a5740c982c9cb1c793d6196` |
| Merge/compile regressions on base | 4 failed: three missing carrier refusals and absent execution identity | `parts-before-audit.json`, action `9e7fc72930dd84f3f85fce024e606f09fd9add04f2674a9d4bc155d79972c06f4` |
| First fixed focused run | 40 passed, zero skips or uncollected modules | `bridge-after-01-audit.json`, action `b87c72fc7dda44b409b10f319e0e086f9fa8049d5c97f7247de56a5cca214b32` |
| All 209 files selected by the repository impact tool | 3,664 passed, 602 skipped, 28 warnings, zero uncollected modules; 16/16 actions exited 0 | `impacted-pb.json`, `impacted-pb.log`, `impacted-audit.json`, `impacted-source-audit.json` |

The regressions exercise dispatch, malformed input, exact snapshot preservation,
real miniature expert-wire equality, direct and partitioned export, carrier
tampering, and compile identity. The impact selection was
`python3 tools/impacted_tests.py --ref b9cda5b03efe0b9a44aadaf1c565fe2cb9c23032...HEAD --json`.
Its exact file population is retained in `impacted-tests.json` and the submitted
commands are in the terminal audit. The first malformed pbtest invocation was
refused before queueing and retained as `impacted-pb-submission-refused.log`.

All 16 terminal records, published payload SHA-256/size, source-bundle SHA-256/size
and population SHA-256 were checked. Each population verified the same effective
source hash, `9481259696b7b3de6cafddffc3b89d84b2bff63520cb27b24d6bbeb8f2bae8df`.
The tested tree differs from code commit `f3b6953f0c5821ab2108b6db9942f9a2e0179e91`
only in its verified PB closure metadata and refreshed issue snapshot.

The 602 skips cover CUDA paths, box-specific model/serving artifacts, absent
vLLM/Transformers dependencies and one unsupported E2M1 reader-range case.
Verbatim reasons are retained in `impacted-source-audit.json`. CPU passes do not
cover those surfaces. There is no performance claim or new GPU measurement in
this report. Full-engine TP=2 load/generation, useful quality, prefill performance
and peak residency remain measured gates before any qualification or shipping
claim.

## Review corrections and final targeted validation

GitHub's dependency-free CI at code commit `f3b6953f0c` reported 2 failed,
1,589 passed and 98 skipped: the issue snapshot lacked newly filed tessera#430,
and the pure parser test imported the legacy PyTorch-dependent reexport.
`ci-before-test-split.log` retains that result. The issue snapshot was refreshed;
commit `68088925d9` moved only the reexport assertion into the existing
PyTorch-gated suite, keeping incompatible-family validation dependency-free.
PB action `d5be9ff1aaf8313e634cd25d18714d9dc7c5bc67cb7b35c3560f1908affae81f`
then passed 14 pure checks and 45 torch checks. Its scoped pure environment
contains pytest 9.0.2/pytest-xdist 3.8.0 and no torch. An earlier setup attempt
failed because Python's `ensurepip` was absent; the repaired setup uses
`venv --without-pip` and the existing scoped pip's `--python` option.

Review found a separate typed-carrier defect: Python equality treats
`True == 1` and `2.0 == 2`. Raw input was strictly parsed, but config copies
inside records and partition carriers could bypass the same grammar.
PB action `081d142e9a3de56a7da9e0a31971215ae8189680b517c6c575cdd0bfe29b82bf`
reproduced four failures, including an actual exported partition accepting
a floating-point field. Commit `07ad344c32` validates the carried config and
every config/manifest/identity carrier with the existing strict parsers before
comparison. The positive export test checks both integer fields in all three
carriers, with unchanged successful wire-byte and merge checks.

Final PB action
`efd8a1b3de48347ee310ae79b3078232329edbfc6540fe7a0e47c5a121496db2`
exited 0 after compiling all touched Python modules/tests. Its dependency-free
parser/parts/issue checks passed 45 tests, with two missing-torch skips;
the surface recorder lists 79 globally excluded torch modules. Its torch
parser/parts/dispatch/export checks passed 98 tests, with five GPU-encoder
skips and zero uncollected modules. Full terminal records, payload and source
bundle SHA-256/size checks, and population hashes are in
`typed-carrier-before-audit.json` and `typed-carrier-after-audit.json`.
The earlier 209-file result applies to `f3b6953f0c`; the final targeted run
validates the subsequent test split and strict-carrier correction.
