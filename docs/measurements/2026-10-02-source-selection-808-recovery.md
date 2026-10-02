# Bounded source-selection recovery for #808

Tested implementation: `96f170c65bfa4b7ce2a9b60d94d30f84477ba824`.
This records targeted CPU qualification, not serving, CUDA, performance or
full-suite acceptance. Astra retains independent review and merge authority.

The dependency graph retains its conservative union for changed files. Only
UNKNOWN propagation can exclude a source-proved guarded re-export edge. Named
direct calls, namespace/object escapes and known helper effects are derived
from the existing lexical owner and ambiguity-preserving resolver. GLM53 source
origin remains UNKNOWN. No serving code, wire, runtime contract or pin changed
in this recovery.

The frozen `202d1f07` causal probe `f50acdcbcfe6` showed that removing only
`slicing → layout` eliminated every uncertainty-to-conftest path across all 82
recorded seeds. This established a static graph cut, not runtime call reachability.
The final annotated graph keeps that edge for changed-file selection, records
its guarded exclusion and predecessor witness separately, and still records
GLM53 among unresolved source loaders.

## Actual final populations

| PB action | Population | Device and limits |
|---|---|---|
| `d496f002faae6ae8d8f7bce0d6151bea7f3459ffb63e8d05f124eff7d82aa876` | 314 passed, 0 failed, 1 module-level skip; eight files, eight workers | sparklina, CPU Torch 2.11.0, no CUDA allocations; skip: `tessera.kernel imports Triton (CUDA-only)` |
| `e3ea0542be43b4827b0e6f02d6b8833d5eaac5faf5f47902b3efd94740bfe106` | 32 passed, 0 failed, 0 skipped; only the missing shape-guard file, four workers | dl380g10, CPU Torch 2.11.0, no CUDA allocations; x86 dependency constraint supplies the existing Triton environment |

Both actions ended with exit 0 and successful CAS receipts. Controller and
worker source spans agree within each action. Both snapshots match every
tracked path/mode/blob of the tested implementation, plus their independently
verified sealed closure member. Raw source hashes differ; no automatic metadata
exclusion or source-equivalence result is claimed. The ARM population reports
`not_collected=[]` despite the module-level skip: its 32 shape cases were absent,
not passed. They were executed only by the second action. Together these are
346 distinct passing cases, including all 55 new cases and the three unchanged
leaf, shared-conftest and box-artifact ratchets. Seven touched/prior-PR Python
files compiled inside the first action.

The affected-family command was:

```text
python -m pytest -q tests/test_impacted_tests.py tests/test_source_dependencies.py tests/test_source_execution_helpers.py tests/test_source_execution_refusals.py tests/test_guarded_reexports.py tests/test_impacted_manual_gate.py tests/test_kernel_shape_guards.py tests/test_glm53_prefill.py -n 8 --dist worksteal --durations 10 --basetemp <action-owned sibling>/pytest --surface-json .pytest_cache/ts811-final.json
```

All execution used the published PrismaBuild client, bounded native threads and
PB affinity. Evidence, exact sealed commands, immutable log hashes, CAS checks,
populations, worker shares and source-tree comparisons are retained on sparky
and dl380g10 under
`/home/rob/tmp/codex-campaign-takeover-20261002/ts811-evidence/`.
The final stdout log is under PB's
`attempts/d496f002faae6ae8d8f7bce0d6151bea7f3459ffb63e8d05f124eff7d82aa876/740e4da8ef7ac393bccec92c6a38d124dd2130db3a0ab0bad468ca96ebdf871e/00000001.stdout.9d3401fee6dd9cb9e8211a58a1405f4be106f07e713a90f3ea63d44a53b22ec9.log`.

## Causal failures retained

Verified new RED on frozen implementation: `07f573c9a4a0` — 5 failed, 36
passing controls. The first portable attempt `80cfc8c8d4c8` had unknown worker
source attribution and is retained without qualification credit. Additional
verified prototype regressions were banked before repair: `a6ff9316224c` — two
source-helper failures; `094e2a955ccc` — seven escapes plus changed-file union;
`3af5c5cbdff1` — two reflective namespace accesses; `4513663da8c8` — two known
namespace-helper effects. The intermediate combined action `850580f526c0`
remains FAILED: 333 passed / one unchanged shared-conftest ratchet failed.
None is relabeled successful.

The analyzer remains bounded recognition of supported imports and top-level
helper calls. Arbitrary callbacks, returned callables, class-method flows and
external-source provenance are not certified. #769 nightly census, #790 host-I/O
acceptance, integration and serving qualification remain separate.
