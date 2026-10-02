# #808 source-read selection census (blocked draft)

This is a bounded CPU selector change, not external-origin or serving acceptance.
Refs #808; #769 nightly census and #790 host-I/O acceptance remain held separately.
Astra owns review/merge and the external-origin follow-on. No runtime, wire, pin,
export, default or numerical behavior changes are included.

## Delivered rule and limitation

Consumer scanning is separated from dependency classification first. The selector
then computes source-execution capability to a fixed point over recognized calls
to top-level functions, using its existing ambiguity-preserving module resolver.
Relative imports, assignment aliases and re-exports participate; shadowed aliases
retain possible execution instead of proving it ceased. Mere import presence does
not promote a reader. Exact reads still use the existing root-bounded resolver;
unknown and named-but-refused reads retain their existing distinct fallback rules.

Capability is **not provenance**. Neither a generic module parameter nor an import
absent from this graph proves external origin. The three banked `foreign_runtime`
fixtures expect narrowing from precisely that unsupported assumption; they remain
unchanged for review, not xfailed, and are not established production contracts.
The real leaf/conftest/box-artifact selectivity ratchets also remain unchanged.
No successful qualification is claimed in this census; exact observed populations
and immutable action/log/source identities belong to the accompanying PR report.

The concrete blocker is `src/tessera/serving/glm53_prefill.py:793-802`: generic
`module.__file__` bytes feed imported `rebuild_method`. Production installation
calls it at line 845 using `modules[0]` after runtime interface matching.
`tests/test_glm53_prefill.py:854,874,882,894` also pass runtime-created modules,
not exclusively external imports. Separately, the retained selection diagnostic
forces full for `tests/conftest.py`. No predecessor artifact yet joins these two
observations: a GLM53-to-config-to-conftest attribution is unverified. A diagnostic
must identify the actual uncertainty seed and resolved-file path before any
source-origin redesign is priced. Caller-wide external proof also cannot be
inferred from the production intention.
Serving signatures/reads and the selectivity assertions must not be silently
changed to conceal this blocker. Future options are a separately approved narrow
caller/provenance contract or a separately priced source-interface change; this
draft implements neither and is not merge-ready.

## Complete tracked module-file lead inventory

At base `4e5674d8b0c7b413dfdfe5829f261c92c6897748`, AST inspection found 70
module `.__file__` / literal `getattr(..., '__file__')` occurrences across 39
tracked Python files (58 direct, 12 getattr). Bare current-file `__file__` path
construction is not counted. These are leads, **not 70 defects**. Source bodies
were inspected to separate reads from metadata, binary probes and argv. This
inventory does not assert complete arbitrary-Python read discovery.

| File(s) and lead lines | Actual semantics; disposition / owner |
|---|---|
| `_pb_native_moe_measure/run_native_child.py:18,28,51` | Package/loaded-origin census plus package file identity; native measurement owner, unchanged |
| `experiments/bench_native_operator.py:499,500,505,514` | Binary-library hashes and runtime-contract bytes, not Python execution from those reads; native measurement owner |
| `experiments/bench_routed_load.py:671,672,789` | Checkout-path guard and printed identity; distinct metadata |
| `experiments/checkout_runtime_identity.py:110` | Checkout-root assertion; distinct identity |
| `experiments/full_engine_worker.py:239,266,288` (two at 288) | Package file hashes and loaded-origin census; native measurement owner |
| `experiments/full_model_original_wire_checkpoint.py:92`, `experiments/full_model_research_selected_checkpoint.py:107`, `experiments/original_wire_checkpoint.py:71` | Exporter subprocess argv, an explicitly unsupported source-execution route; no export migration |
| `experiments/measure_glm_native_execution.py:124`, `experiments/qualify_native_artifact_operator.py:74` | Neighbor resource-analysis source hashes, not parse/exec of those bytes; native measurement owner |
| `experiments/moe_greedy_smoke.py:296,326` | Path identity guard and metadata; no read/execute seam |
| `experiments/original_wire_generation.py:43,55,74` (two at 74) | Package source manifest hashes and loaded-origin identity; native measurement owner |
| `experiments/qualify_reader_namespace.py:102` | Loaded namespace-root assertion; distinct identity |
| `experiments/routed_load_memory_probe.py:156,262`, `experiments/t8r_speed/bench_t8r.py:437` | Printed measurement metadata; unchanged |
| `experiments/t8r_speed/bench_geometry.py:90` | cuobjdump binary-resource argv; GPU qualification held, not a Python read |
| `experiments/tessera385_bench.py:60,248,250` | Package file identity, metadata and Git cwd; measurement owner |
| `experiments/tessera486_fit_lut_stats.py:127,138` | Printed identity; separate runpy call already has its own loader classification |
| `src/tessera/historical_producer.py:115` | Namespace-root guard; no new historical producer behavior |
| `src/tessera/serving/ext.py:551` | Header-directory discovery from torch install; not Python source execution |
| `src/tessera/serving/glm53_nope.py:63,73` | Package-relative pinned stock-source hashes; generic external-origin inference not implemented; serving owner |
| `src/tessera/serving/glm53_prefill.py:289,793` | Generic source hash and read-to-imported-executor boundary; relation to the conftest selectivity block unverified, Astra follow-on |
| `src/tessera/serving/mtp_draft_lifetime.py:148` | Generic digest-only read, no Python execution of these bytes; retain data-reader semantics, serving owner |
| `tests/test_audit_byte_baseline.py:103,110,112` (two at 110) | In-tree source mutation/exec and synthetic module metadata; conservative selection, no wire changes |
| `tests/test_audit_container_accounting.py:117,202`, `tests/test_audit_sec2.py:222`, `tests/test_compensate.py:83` | Package/text assertion reads; existing plain-reader rules |
| `tests/test_compiled_census_expectations.py:17`, `tests/test_dense_prefill_cross_check_merge.py:294`, `tests/test_nope_execution_identity.py:27` | Python parse/exec or AST inspection; existing source-execution rules |
| `tests/test_full_engine_worker.py:532`, `tests/test_mtp_draft_lifetime.py:438` | Mutate synthetic module metadata for refusal tests, not reads themselves |
| `tests/test_mtp_draft_shards.py:193,197,229,242` | Synthetic module origins plus digest reads; test fixture semantics |
| `tests/test_native_a4_serving.py:151,153`, `tests/test_native_operator_receipt.py:820,822` | Source text assertions / neighbor resource hashes; native gates unchanged, not executed by this inventory |
| `tests/test_serving_dispatch.py:110,121`, `tests/test_serving_moe_dispatch.py:328,339` | Stub-vs-real module cleanup identity; not source reads |
| `tools/tessera_construction_census.py:359`, `tools/tessera_route_census.py:1958` | Runtime version/path/Git identity; #769 nightly census still unmeasured |

## Separate held acceptance

- #769: CPU module/mapper fix already landed in PR773. Actual compatible nightly
  draft-route census, approved TP2 `--only-2c`, remains coordinator/native-owner work.
- #790: CPU CLI/cache prerequisite already landed in PR810. Full matched host-I/O
  acceptance with real input, parent/in-process profiling and both-box telemetry
  remains unmeasured; no performance claim is made here.
- PB#811/#1360: production scratch finalizer belongs to pb-sched, not this cohort.
- Central integration/full-suite and independent exact-head review belong to Astra.
