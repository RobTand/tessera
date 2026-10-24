# mHC dispatch installation interface validation

Scope: CPU-only installation safety for the existing default-off fused mHC
candidate (Refs #783). No kernel, runtime pin, adoption or execution policy
changes. Acceptance tests were not edited.

The installer now binds each dispatch helper's inspected signature to the
wrapper's positional argument count before rebinding the model method. It
neither executes the helpers nor catches runtime launch failures. Required
extra arguments and uninspectable signatures decline; compatible optional
extensions remain accepted in development and certified modes. Development
still bypasses historical source hashing; certified mode still requires it.

## Observed validation

Checkout HEAD before edits: `169554c4e80a8ceb7c49f07c7ce5e999860dad6b`.
All test runs below used the x86 arm, CPU torch, no CUDA device, zero device
allocations, and no skips or uncollected modules.

Pre-fix command:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_mhc_fusion.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: six incompatible-helper/mode cases failed, 34 passed in 10.24 s.
Each failed at `assert mf.install_mhc_fusion(other_config) is False` because
installation returned True for a required keyword-only extension.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4e9fc75e47d2016e94fd983821d99a95462651b033a75a97e61e374157f26eda","size":25087}},"operation":"510f1df80f90da5abb04aea35b98486d439ef30c1bbe35201f917fa0ae31762e","receipt":"1d8ab2a301b87ef74dec2996ae3720b525cdbb40eb7b6934e7edd4c1703d6857"}]

Post-fix command (acceptance plus mandatory guards):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_mhc_fusion.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: 252 passed in 148.57 s. This proves CPU interface behavior, not native
numerics, serving quality or performance.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9cd6c217d0fb57c8c068d428251d78e83d4d671e8841e942584d23eff289730e","size":1865}},"operation":"e2bbf1cd51725899e05ab6277c28a3582592e5b8c72fd53b0e9e2568f32926e6","receipt":"40c0eae906ad3459d0e46667236472319fbeb2445dd18f5bd94f3d13391c66a1"}]

`python3 tools/impacted_tests.py --ref 17b3fdb396410698c9a142615345f046d2f94a5d`
returned `narrowed` with 496 selected tests (3 changed files at selection time).
That conservative selection was not executed. Full integration remains the
coordinator's once-only check. `git diff --check` passed after implementation.

## O2/O3 remain unmet

The inspected existing design (`docs/design/mhc-fusion-783.md`) records the
corrected-source native integer-view probe PB `403c665b2ebd`, kernel SHA256
`0ebf0f43304f01439204855a6bfa6ae7e7e6b20aee6fd05814aa428978c29a30`.
This is a documentation reference, not an independently verified raw receipt
in this work. The eight random/halfway native tests have a failed containing
action; do not promote them to a fully passing action.

The existing measurement document `2026-10-04-mhc-fused-783.md` explicitly
leaves served E2E, quality/KL and energy acceptance unmeasured. Historical
microbenchmarks are not corrected-source timing. The runtime contract retains
an empty evidence list and default off. No inspected record establishes the
required current-source paired per-rank L8192/L512 profiler traces, stock/fused
call and SP-shape attribution, measured DRAM traffic, both-Spark aligned
Netdata against 140 W, work/J, matched served TP2 cross-check, and KL packet.
No speed or release-qualification claim is made.

Missing legs must remain with the existing campaign/qualification owner until
the recorded PrismaBuild qualification/adoption/fresh exact-tuple decision.
No GPU campaign, external runner invocation, environment repair or new
adoption was performed or authorized here. Before any later authorized GPU
run, its owner must record the decision, CPU insufficiency, smallest case and
expected minutes (target about ten before a sweep).

The supplied open-list title for issue number 1491 cannot establish overlap. The local
`docs/issues-snapshot.json` instead associates that number with Stage A
shared-expert resource charging, so that stale snapshot cannot authenticate
the current matched-TP2 lever/source either. No packet was shared and no
parallel measurement campaign was opened on the basis of the generic title.
