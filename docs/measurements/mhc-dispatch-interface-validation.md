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

The supplied open-list title for Tessera issue number 1491 cannot establish
overlap. The only `1491` entry in `docs/issues-snapshot.json` belongs to
`RobTand/prismaquant`, not Tessera, and concerns Stage A shared-expert resource
charging. It provides no evidence about the live Tessera issue's lever/source.
No packet was shared and no parallel measurement campaign was opened on the
basis of the generic title.

## Review repair disposition

Rechecked at `0ae947fa407037f23b19fef1f5e8731cff2bd6e4` without changing
application code or numerical tests. The native JSON digest and CUDA-source
match reported below were independently reproduced; all 60 cases retain zero
differing words for the four outputs. This remains a `--parts bitwise` result.
A focused read of the five named historical timing packets under
`/mnt/shared/tessera-measurements/mhc-fusion-783/timing-*` found none with the
native packet's CUDA-source hash. In particular, `timing-c52a7602` and
`timing-5e04726c` record `043cce073fef...`, not `0ebf0f43304f...`.
They cannot repair the current-source performance gap.

The review's O2/O3 failures are confirmed: no reusable complete packet or
recorded execution decision was established by this recheck. Documentation
and CPU tests cannot supply those missing measurements or authorize them.
The accepted O1 implementation is preserved; **O2/O3 are not repaired**.
Reuse existing custody issue 1521 rather than filing a duplicate or starting
an unauthorized qualification campaign.

Revalidated O1 and documentation guards on x86 after the review-disposition
edit:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_mhc_fusion.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: 252 passed in 138.31 s; no skips, uncollected modules, CUDA device
or device allocations. Retained stdout was read back and verified. No new
GPU/native/serving check was submitted; this run does not satisfy O2/O3.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2d2046ab67dd3ec2076d7b7210f942225bc00a05d9f15c5919586d2b9ea001c1","size":1859}},"operation":"d8e9368b68b19788ec9d71d5f1857664f998ae20899b9ff8302daa8ea866c69b","receipt":"b5dcd74c1c124649c3bde61de27fe82c6facec22c8032a201662f77fc4c52091"}]

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

A bounded search of the served PrismaBuild checkout
`9dce28460e76d4115c925bac8efc4f397160f9cf` found no literal `mhc`
references. Listings of the fleet evidence and results directories exposed
no named mHC decision packet. This is not an exhaustive proof of absence:
no recorded qualification/adoption/fresh exact-tuple decision was located
in the inspected material. No owner communication or adoption decision was
made through these file inspections. The current identity of issue 1491
remains unverified; no packet was shared with it.

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

Documentation guards were run on x86:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed after the custody documentation edit: 212 passed in 111.08 s, no skips or uncollected modules, no CUDA
device and zero device allocations. This validates the documentation guards,
not O2/O3. No GPU check was submitted; the missing external qualification
legs require the recorded owner decision before execution.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"48dd0318b3727c8f65a77e215932d0a9e0227fbb432678cb0cd103acdcc39cf9","size":1814}},"operation":"0bd24dca4d5e762ef457999f898e6f40895b3f3a40c174a4c4c21879d0125c75","receipt":"af993503d02f339616c2ad636ad810dfb013b12275e5f88f87c22a2cb4be6ade"}]
