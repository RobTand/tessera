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

## Qualification-custody follow-up

At checkout `8db522bbecbbd35df9b4c570e8256ddcb8eed557`, a read-only
inspection independently recovered the existing native probe at
`/mnt/shared/tessera-measurements/mhc-fusion-783/takeover-894146476-20261006/capacity16/native/mhc_fused_probe.json`.
Its SHA256 is
`4bcf075576ef3021a13018dc1c5ef6618231cc2d4a70d9072b8d6f1df446abd5`.
The receipt records PB action
`403c665b2ebd7844542efbfbbfebb14ed32c858019b2ffe86b404eb86a2fbdf4`,
source checkout `2cbc4e274fb6c04c393fdc51ed5d57eac277a5d5`, build ID
`54fffa63e0f7c4d2dcbb0e1d4eae3dcffee83115c20e14585b35d29afb55ded9`,
and GB10 execution on sparklina. Its recorded CUDA source SHA256 matches
this checkout's independently calculated
`0ebf0f43304f01439204855a6bfa6ae7e7e6b20aee6fd05814aa428978c29a30`.
The JSON records 60 cases and `bitwise.passed: true`, including
`residual_cur`, `post_mix`, `comb_mix`, and `layer_input` comparisons,
stock/fused split metadata, determinism and tile comparisons. This upgrades
the native probe reference above to an inspected raw result, not to a new
execution or whole-checkout serving qualification. Existing numerical
assertions remain unchanged.

The probe ran `--parts bitwise` and includes historical Netdata charts and
an NVML sampler declaration. Those are not paired stock/fused, both-Spark
energy evidence; their presence does not establish work/J or a speed delta.
No current-source performance or KL acceptance follows from this receipt.

### Review repair: admission is not qualification

Reinspection at `0ae947fa407037f23b19fef1f5e8731cff2bd6e4` confirms both
review findings: **O2 is unmet; O3 is only partially substantiated and remains
unmet.** The following additional read-only checks narrow the handoff; they
do not complete either objective or authorize execution.

- The owning mHC declaration is `src/tessera/serving/ext.py`'s
  `STOCK_KERNEL_OVERRIDES`, mirrored in
  `src/tessera/serving/runtime_contract.json`. Both retain default `off`,
  `required_identity: bitwise_vs_stock` and `evidence: []`. The declared
  evidence-entry fields are `{gate, receipt}`. No evidence entry was added:
  the existing native result is not a full serving/adoption decision.
- The native probe's retained fleet outcome is
  `/mnt/shared/prismabuild-fleet/pb-queue/done/403c665b2ebd7844542efbfbbfebb14ed32c858019b2ffe86b404eb86a2fbdf4.json`
  (SHA256 `90aea86d6d8da795f69c57e67368bda6927b0b8ffdfacfc7a73b192f691a6827`).
  It records `prismaquant.prismabuild.pool_outcome.v1`, status `executed`,
  return code 0, but admission flags `measurement: false` and
  `exclusive: false`. Its telemetry uses a **100 W** power reference, not
  the acceptance envelope of 140 W. Do not convert this correctness action
  into a paired performance or energy result by relabeling it.
- That outcome's published result-log blob,
  `/mnt/shared/prismabuild-fleet/cas/blobs/3e/3ef6f44c0b58ed88a40b75b630fe25b14b77371f47d57743664c533274b24424`,
  independently hashes to its filename (13,623 bytes) and records 60
  bitwise-case lines followed by `done bitwise passed`. This corroborates
  the native result above; it supplies no KL or matched served leg.
- In the served PrismaBuild checkout
  `9dce28460e76d4115c925bac8efc4f397160f9cf`,
  `tools/fleet/pbcampaign.py` documents generic action rows (`argv`, `env`,
  `snapshot_ref`, `container_images`, `measurement`, `as_sealed_by`, etc.),
  not a model qualification/adoption decision. It explicitly permits a
  missing receipt to submit an action, so invoking it is not a read-only
  way to recover evidence and was not attempted.

A bounded inspection of fleet rollout, investigation and measurement records,
plus 276 runtime-version and 91 canary metadata documents, did not locate
this candidate's exact-tuple decision. A broader recursive runtime-generation
search timed out; it is not a completed exhaustive check. These records
concern PB infrastructure admission/publication, not an established mHC
qualification. No owner communication or adoption decision was made through
file inspection. The current identity of issue 1491 remains unverified; no
packet was shared with it.

The existing campaign/qualification owner must supply the actual recorded
decision and its owning configuration/row references, bound to the candidate
source, build and serving tuple, before missing execution can resume. An
admitted pytest action, an empty contract evidence list, or failure to locate
a receipt cannot substitute for that decision. This is the same custody
dependency tracked in the supplied open list as issue 1521, not a new
measurement campaign or a new duplicate finding.

O2 and the remaining O3 legs are still **unmet**, not waived. Continue custody
through the existing v6 Kernels Lead and qualification owners, with the v6
Review and Integration Lead as integrator; the supplied open list already
tracks qualification custody in issue 1521. The required handoff is the
recorded PrismaBuild decision and source/build/config-bound raw receipts
for native/quality qualification, matched per-rank L8192 chunk and L512 step
profiles (including fallback calls and actual SP shapes), byte/roofline
assumptions, aligned both-Spark Netdata against 140 W, work/J and served TP2
cross-check. The campaign tuple remains TESSERA_E4M3_K1 GLM-5.3-Flash,
L8192 c1 TP2 max_num_batched_tokens=2048. Cache-based byte estimates must
not be presented as measured DRAM traffic. An inconclusive or negative
served result must remain so. No GPU run, new campaign, external runner,
environment repair, adoption or pin change was performed in this follow-up.

### Repair validation (x86 only)

The unchanged O1 acceptance and documentation guards were run after the
repair documentation edit:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_mhc_fusion.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: 252 passed in 152.52 s, no skips or uncollected modules, no CUDA
device and zero device allocations; retained stdout was read back. This
checks O1 regression preservation and documentation guards, not O2/O3.
No GPU check was submitted: KL, paired performance/energy and served checks
remain unavailable pending the existing-owner decision and execution packet.
The once-only full integration suite remains with the coordinator.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a149024e5c90f1fab91c771843f0642455f46ae2fcefa988aeaae28fce4c3149","size":1863}},"operation":"bbeaff02642ea5008b7220cd8638c5f1d949903a7722084615a816da58d6c03f","receipt":"c4550ac25cfe70992e1cbc328327dc3843db52caabf4604704665040aeabce55"}]
