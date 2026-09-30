# GLM MTP vocabulary lifetime — CPU screen, tessera#645

Base: `1381c3b7166c3cba996217d97fc37d1ff4d30213`.
This is small-shape allocation/lifetime evidence, **not** full GLM, GPU,
watchdog, served-quality, image-digest or MTP-acceptance qualification.
The historical approximately 1.18 GiB/rank is not a new measurement. The
existing conservative per-rank duplicate fit allowance remains unchanged.

## Boundary

`TesseraConfig.get_quant_method` installs the production hook during target
construction only for Tessera GLM MTP. Eight inspected runtime source digests
and affected load signatures guard installation. Sources are the issue's
cached GLM-image `mtp-census/vllm-src` and direct vLLM dependency sources at
`fd4a15126`; no installed dependency or image was edited.

Context-local constructor bindings return parameter-free vocabulary modules
only while an eligible draft loads. Exact per-layer embedding and head input
names are omitted from draft loading, not norm or decoder inputs. Stock
sharing must replace every placeholder with the exact target module before
return. The target is not reachable through draft loading/post-processing.
PP other than 1, TP other than 1/2, LoRA, quantized vocabulary, incompatible
shape/dtype/device/layout or unmatched runtime source/signature refuse.
Unrelated loads/threads retain original construction; exception and nested
load contexts restore. There is no cache-clear workaround or math change.

## Measured CPU pair

`tests/test_mtp_draft_lifetime.py` exercises the production config installation
and both stock loader interfaces through CPU stand-ins. Within each process,
the legacy unwrapped loader and hooked loader run under Torch's CPU memory
profiler. Incremental live high-water sums chronological self-memory events;
weak references independently observe duplicate storage at a later load
workspace. Target storage, module identity, values and embedding/logit products
are checked after load. TP2 is rank-local shape simulation, not two-device TP.

| Interface / population | Incremental CPU peak before → after | Duplicate live at workspace before → after |
|---|---:|---:|
| V1 and V2, TP1, `[64,8]` float32 | 8256 → 4160 bytes | 4096 → 0 bytes |
| V1 and V2, TP2 rank 0, `[32,8]` float32 | 6208 → 4160 bytes | 2048 → 0 bytes |
| V1 and V2, TP2 rank 1, `[32,8]` float32 | 6208 → 4160 bytes | 2048 → 0 bytes |

These bytes exclude the already-built target. They must not be extrapolated
into a full-model/GB10 memory peak. Runtime source acceptance in these fixtures
is stubbed deliberately; real source identity is independently fail-closed,
and in-image compatibility remains unmeasured.

## PrismaBuild receipts

All actions are portable x86 CPU-only, priority -10, native threads 1,
`PYTHONPATH=src:experiments`, explicit
`TMPDIR=/home/rob/tmp/claude-campaign-20260926/tmp`. Interpreter path:
`/home/rob/venvs/pq-pb059953bc-tessera-a5f3b232/bin/python`; the worker reports
Python 3.14.4, Torch 2.11.0+cpu, pytest 9.1.1, xdist 3.8.0 (the historical
interpreter path now resolves to this environment). CPU affinity was retained.

- Initial RED, before production correction:
  `4ee13dcdf21366228007113f7eb8e4f34fe75f124b61e1f7af54a8f4556d09d0`,
  terminal failed, rc 1: 8 failed / 0 passed / 0 skipped / 0 uncollected.
  Genuine assertion: `assert 4096 == 0`, "draft vocabulary duplicates survive
  through load peak" at the original test line 186.
- Expanded RED on all final cases, using baseline config restored **inside
  the admitted snapshot only**:
  `82a6428cd3f9c4efb08c089f8e13bb6a7376404b23ee5bcfd60df8655b24fe3e`,
  terminal failed, rc 1: 19 failed / 0 passed / 0 skipped / 0 uncollected.
  Lifetime assertions, missing installation assertions and `DID NOT RAISE`
  refusal assertions fail, not tools or fixtures. Failed actions publish no
  successful CAS receipt; canonical failure records and log hashes retain RED.
- Paired GREEN:
  `c9ddcfbd2f57e7259853b49a83b794f7073205690d3b244f76418088fde39388`,
  terminal executed, rc 0: **19 passed / 0 failed / 0 skipped / 0 uncollected**,
  `-n 2 --dist worksteal --durations=10 -raP`, 2 CPUs / 4 GiB, timeout 300 s.
  Snapshot `7c0a5e8520b03b1ff43f26d771ded963bb66cbc2`.
  CAS receipt SHA-256
  `201a34fd75ddf67ef76cf2d4bb8f018946ac7ffb04219956c1c4697984a39851`;
  output SHA-256
  `61b205d91b9e1628cee87db8795463664259cc117aee3d318329c3560353ee25`.
  Population SHA-256
  `5a4098914c0a87a50f8de9bdee235888bbcfb1499b4bb3007d4add5f82ea8473`.
- Compile and first selector:
  `b6ab382f36e5f9163dd8bf5bacd4875ac4a98d883d8d01d4ef38fc5b644f25bc`,
  terminal executed, rc 0, 1 CPU / 2 GiB, timeout 300 s.
  `python -m compileall -q` covered config, lifetime hook, serving_parts and
  the test module. First selector returned FULL solely for unverified
  generated `.pbrun-closure.9ca32bb411332c71.json`; this result is retained.
  Receipt `422b0a5eca627a463cf2396549dc2acbfac26ec017b154e94bab6b56de90d36e`;
  output `32e999330f7827a69fa282904c2aafd29bdfff6262492f25c2b445474e0ae4ca`.
- Verified selector:
  `91d3940766c5fd3f111d8dd6473ce8b7b39d36f7dc11439d4ad40d1e3075c785`,
  terminal executed, rc 0. The existing `TESSERA_SOURCE_VERIFIER` points to
  `/usr/bin/python3 /mnt/shared/prismabuild-fleet/repo/tools/pbsnapshot.py verify`;
  no verifier was mocked or guard weakened. Exact base `...HEAD` returns
  **narrowed, 325 files, no forces_full or unreadable sources**.
  Receipt `08098a7d34dc412762fadcc4f146198704c08ccb75ea1f9a755bf79507340f17`;
  output `39b7a7d2d56a045ec182cc1ca7042d987696b1c98704894caf27642f3d157ef8`.
- Selected-population validation is **PENDING at this checkpoint**:
  `3e77a80192b9b006519c1808d55ec84db2f6bcdb4128b24f69b2f277c1fec081`,
  2 CPUs / 8 GiB, timeout 1500 s, xdist worksteal 2, durations 25. It recomputes
  the verified selection before pytest. pbrun's 600 s observation and a
  subsequent 600 s pbwait both expired with client rc 75; the action was
  claimed by dl380g10, not canceled or resubmitted. These client statuses are
  **not a terminal test verdict**. Parent owns canonical same-key completion,
  any narrow failure investigation and the independent review gate.

Literal paired population: `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`;
`0 test(s) skipped, 0 module(s) not collected`; `0 test(s) allocated on the device`.
There are no skip reasons in the paired run. This population does not certify
CUDA, forward kernels, the serving image or served quality.

## Durable locations

Coordinator logs and selection JSON:
`/home/rob/tmp/claude-campaign-20260926/tmp/645-receipts/`
(`red.log`, `red-expanded.log`, `final-paired.log`, `compile-selector.log`,
`selector-verified.log`, `selection.json`, `selected.log`, `selected-wait.log`).
Canonical terminal records:
`/mnt/shared/prismabuild-fleet/pb-queue/{failed,done}/<full-action-key>.json`.
CAS result blobs: `/mnt/shared/prismabuild-fleet/cas/blobs/<first-two>/<output-sha256>`.
Worker-local paired profiles on dl380g10:
`/home/rob/tmp/claude-campaign-20260926/tmp/645-receipts/final-paired/`,
`v{1,2}-tp{1,2}-rank{0,1}.json` plus `-before.trace.json`/`-after.trace.json`
(for the six valid combinations above). Population is worker-local
`final-paired-surface.json`, not a file present in the coordinator checkout.
The pending selected action writes fresh `selected-profiles/`, prints every
profile file's SHA-256/path into its CAS output and writes `selected-surface.json`.

No pins, release artifacts, image bytes, kernels, wire bytes or numerical
paths were changed. Unchanged host diagnostics in config and serving_parts
include absent optional vLLM imports and pre-existing typing/style findings;
new hook/tests were diagnostic-clean. Independent review and the pending
selected population remain gates; this receipt does not mark an image ready.

## Follow-up: canonical timeout and bounded diagnosis

Independent read-only review of `6fd95a4e8e9ce61ee52284d54cdad30c0c0e211e`
found no actionable code defect, but blocked merge on broad validation. Review
SHA-256 `f960e9554e225e3153ceedfbc223fd10435e83331973a08747e30e9cfb41c862`,
preserved at the coordinator's `tmp/p2p3/tessera/tessera-645-review1.md`.

The earlier selected action
`3e77a80192b9b006519c1808d55ec84db2f6bcdb4128b24f69b2f277c1fec081`
is now confirmed **terminal TIMEOUT**, elapsed 1508.3761541843414 s,
returncode null. It collected 7093 items and printed progress/failure markers,
but published **no final counts, surface or successful CAS receipt**. Progress
is not a pass/fail population or an exact failure count. Its canonical failure
record SHA-256 is
`d182bd0c04e496e84cdffa7f8ed8ceaca2245eac1b2ae34364398d08b5daa3fd`;
stdout SHA-256
`cd4a6373766455cca25e34ba8f8d88cc10259ae965bb3477d40f539b82cbb36b`.
The record is `pb-queue/failed/<key>.json`; retained stdout is
`pb-queue/attempts/<key>/41a55809194978779344ac8809274597a280c5c121070926be3d89491f161ac9/00000001.stdout.cd4a6373766455cca25e34ba8f8d88cc10259ae965bb3477d40f539b82cbb36b.log`.

To obtain an actual earliest failure rather than infer one from markers,
a **diagnostic-only** action recomputes the exact verified selection and runs
all selected files with `pytest -x -n 2 --dist worksteal --durations=25 -ra
--tb=short`. There is no hand-deselection or broad-acceptance claim.
Key: `e9e8a2f80463f702fb31dc38231392f61cdf8705abc9456145f51172f1d72eec`;
2 CPUs / 8 GiB, native threads 1, priority -10, CPU x86, timeout 900 s.
Snapshot `be37d26dde2fd57a168c1c666955a482a89a1934`, parent
`6fd95a4e8e9ce61ee52284d54cdad30c0c0e211e`, exact base ref retained.
The 600 s pbrun observation expired **client rc 75**, while a subsequent
single canonical read found it claimed by `dl380g10:2634866:77ec388a`.
No terminal diagnostic outcome or earliest concrete failure trace was available.
It was not canceled, resubmitted or followed by another wait. Parent explicitly
adopts this exact key; final broad acceptance remains the parent's separate gate.

No fixture/behavior correction is justified before a concrete failure is read.
Python sources and all original RED/GREEN/compile evidence remain unchanged.
This follow-up is documentation only. Coordinator diagnostic log:
`/home/rob/tmp/claude-campaign-20260926/tmp/645-receipts/diagnostic-x.log`;
planned worker-local surface `diagnostic-x-surface.json` under the same root
is not claimed to exist yet. Client-window expiry is not a terminal test result.

## Follow-up: concrete non-MTP fake-runtime fixture repair

Both diagnostics have now been consumed as terminal **TIMEOUT**, not GREEN:
`e9e8a2f80463f702fb31dc38231392f61cdf8705abc9456145f51172f1d72eec`
at 904.88920378685 s (stdout SHA-256
`5e4de220868069867d08aa6f652d018658848fad8172bbea7408b3d83e92ec90`),
and the parent's immediate-report diagnostic
`c0b9ab238b2f8b1e9529c35a4cbe1c619a9363397abcf00e32886cc93c027703`
at 904.966596364975 s (stdout SHA-256
`0eaf3c34435c2af8bd6de7dbb3ec4ddbf268a9aeba14411d178b6d991f1b09f2`,
4281 bytes). Returncodes are null; neither supplies final counts, a final
surface or successful CAS receipt.

The latter retained an immediate concrete failure at
`tests/test_serving_moe_dispatch.py::test_checkpoint_reconstruction_selects_existing_packed_owner[torch-1]`:
line 78 calls `get_quant_method`, which reaches the lifetime hook and cannot
import `config` from the fake `vllm`. That file's `_install_vllm_stubs` created
no `vllm.config`, unlike the existing ordinary dispatch fixture. This is an
incomplete test runtime, not evidence that installed stock vLLM lacks config.

Before editing, whole-file PB RED
`4499d6d6b3c7b2af6fecde38e4d267182a37533ca18a06c81d34e9a63f179e8c`
reproduced **19 failed / 30 passed / 0 skipped / 0 uncollected**, terminal
failed rc 1, on snapshot `771ab9c0baf6cbc2e987f155088f719aa6d02fa9`,
parent `c6f3329d546bccf9d391ca581bb892ac5e1b4f61`.
Exact failure: `ImportError: cannot import name 'config' from 'vllm' (unknown location)`
at `mtp_draft_lifetime.py:220`. Canonical terminal SHA-256
`cc5d6b903917f3dc2a2fecaa0e9d017ef15a0ec3e4557f7aeb90826a930282cf`;
stdout SHA-256
`889607f48774bcea2811dd42159a450a9f9669fc8f32f43667697228aae8e698`.

The only executable follow-up change is the fixture's `vllm.config` module
with `get_current_vllm_config_or_none` returning None outside a construction
context, plus its comment. Production dependency imports and source/interface
refusals are unchanged: no missing production dependency is swallowed.

PB GREEN + scoped compile + real-verifier selector:
`18670f65a602ba8434c325d38deeb7d6af13e43c8a306829a3f0339c30b50c96`,
terminal executed rc 0, **68 passed / 0 failed / 0 skipped / 0 uncollected**
(49 MoE dispatch cases plus the 19 lifetime cases). Both test files ran with
`-n 2 --dist worksteal --durations=15 -raP`; 2 CPUs / 4 GiB, timeout 300 s.
`compileall -q` covered config, lifetime hook, serving_parts and both test
files. Exact-base selector with the real source verifier returns **narrowed,
325 files, no forces_full or unreadable sources**; no broad child retry ran.
Snapshot `7ec7a355bc74af32bd405ee940fb42298e50153f`, parent
`c6f3329d546bccf9d391ca581bb892ac5e1b4f61`, base branch bound to
`1381c3b7166c3cba996217d97fc37d1ff4d30213`.

- CAS receipt SHA-256:
  `445344f86aecfc09697edb001d6d9e677c0e3e5cdaa12d5e818b2afd3f76fd90`.
- CAS output SHA-256:
  `4f4b1fd1d49fd7e6e431796d06c1c61872c97bb2715992d3c58c76801868bbd9`.
- Canonical terminal SHA-256:
  `5ac76c4aaa0a5b090d2b33af47cc7b9863b25f415fe7d9ce27afc4948cdec5dc`.
- Canonical stdout SHA-256:
  `973160c0c6633276437e2db40dce7d22210c3488688f1b329dde41eb12849e10`.
- Surface SHA-256:
  `7744333abbe34a2cc201035f91362ec35b4b226cc8bd49020177927bfac1ad6f`.

Literal population remains `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`;
`0 test(s) skipped, 0 module(s) not collected`; `0 test(s) allocated on the device`.
No skip reasons apply. Coordinator logs: `moe-fixture-red.log` and
`moe-fixture-green.log` under the same `645-receipts/` root. Worker-local
fresh paired profiles are in `moe-fixture-green-profiles/`, with population
`moe-fixture-green-surface.json`. Canonical failed/done records and CAS blobs
use the mappings above. All action keys were consumed before this report.

Production Python source is byte-identical to the reviewed `6fd95a4e8e...`;
the new tested source differs only in the non-MTP fake-runtime fixture.
The receipt append is documentation-only relative to that tested snapshot.
Original genuine RED and lifetime profiler evidence are preserved. The
parent still owns final broad validation and any re-review; CPU fixture GREEN
is not CUDA/image/served-quality admission. Host static fake-module findings
were treated as unchanged advisory baseline with explicit supervisor approval;
no diagnostics or runtime guards were suppressed.

## Owner rebase integration onto fresh master

PR744 was reported conflicting at accepted head
`581c89cdcdd00f5267a806f4263e946c188278d1`. The owner fetched and rebased its
single issue commit onto fresh `origin/master`
`4ed2b1cc8d32d72e92ab846fbcbda5eb7e7b1cfc`, retaining the original
`1381c3b7166c3cba996217d97fc37d1ff4d30213` history and separate new base ref
`sol/tessera-645-rebase-base-4ed2b1cc`.

The only conflict was the architecture header: upstream's additive-source-
profiles restamp and #645's lifetime restamp both inserted after the title.
Both blocks were retained verbatim, upstream first. All upstream provenance
and prior measurement history were preserved. No owned Python conflict or
algorithm/default change occurred. Each of the five owned Python paths is
byte-identical to accepted581: config, lifetime hook, serving_parts, lifetime
tests and MoE dispatch fixture. The provisional rebased source was
`e66bfe15009a91d9bf99a34bc4b6667809bc0c6c`.

Bounded new-base integration PB action
`b120ded9c3152fa4002faf4d2572f23dec5491e5795cb5998c5babf6a2a61b89`
finished executed, rc 0, on dl380g10: **68 passed / 0 failed / 0 skipped /
0 uncollected**. The two same MoE/lifetime files ran with `-n 2 --dist
worksteal --durations=15 -raP`; scoped compilation of the five owned Python
files passed. The real executor source verifier was retained. Selector ref
`4ed2b1cc8d32d72e92ab846fbcbda5eb7e7b1cfc...HEAD` returns **narrowed326
files, no forces_full or unreadable sources**. Selection is not execution of
those326 files; no broad child retry was submitted.

- Demand2 CPUs / 4 GiB, priority -10, x86 CPU-only, timeout300 s; explicit
  campaign TMPDIR/source PYTHONPATH, native threads1, PB affinity retained.
- Snapshot `24a56c8fde00fded584b92898670c18fc22e6ccf`, parent provisional
  rebasede66, new base ref bound exactly to4ed2b1cc8d32d72e92ab846fbcbda5eb7e7b1cfc.
- CAS receipt SHA-256
  `4ce481d2d49ca7797d7d280457c3b602cc268230b0029147267054d1ecce961d`.
- CAS output SHA-256
  `33a1ccf9ed746c2766c0cb65f8bc871b619607463be189ca69ac35536e0d6173`.
- Canonical terminal SHA-256
  `ed98eba80a3dd69d367a1e1a03c5d34acd7a256021d6a41c8706e02bdeb2e219`.
- Canonical stdout SHA-256
  `92bacd2d0795f8ea7191f3b3398104384f97b5f7410da480a3bd7f1ac2a68f42`.
- Surface SHA-256
  `278d061206413ba030e3f137e324c2ab6c53913970334598ab5052305448985b`.

Literal population: `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`;
`0 test(s) skipped, 0 module(s) not collected`; `0 test(s) allocated on the device`.
No skip reasons apply. Coordinator log `rebase-4ed2-green.log` and
worker-local population `rebase-4ed2-green-surface.json` use the same
`645-receipts/` root. Snapshot/source/receipt records retain actual identities;
this appended documentation does not change the tested executable source.

The parent's independent fixture/receipt review of581 found no issues,
conditional on broad acceptance; preserved review SHA-256
`fb32052015a30fdcf8a0f23c06ef31117fb4ba68068442b71dca187cf7f583e1`.
The parent reports its separate immutable581 broad job, prefix `b82fa68e4`,
with log `tmp/p2p3/tessera/645-final-validation.log`. The child did not observe,
repeat or cancel that job. Its old-head evidence cannot automatically qualify
this rebase, even though owned Python bytes are unchanged: upstream context
and the new326-file selection differ. Parent owns rebased integration review,
broad acceptance and merge. No image/GPU/full-model admission is implied.

## Profiler cohort-ownership correction after whole-population failure

The parent-owned whole326-file CPU gate on frozen
`ed143f95895eb920734a6c2c86744c54b91b2f0d` completed canonical failed rc 1:
`df254989768ae4b50b15fa8311f601aab4b8b039416369c4e463be6bfc26589f`,
**1 failed / 5623 passed / 1596 skipped / 14 warnings**. The failing V1 TP1
case asserted `(8256 - 3656) == 4096`; its independently observed duplicate
lifetime was still 4096 → 0. This is not evidence of changed serving numerics.
Client trace: `tmp/p2p3/tessera/645-rebased-final.log`; canonical stdout SHA-256
`daed0b5e98e9902f07908cc5aa27166bdca4105ad207b34b6767dffae55fd510`;
terminal SHA-256
`25a7bf39f1b2150f5250d646c1932e6fb9e336456acc749ab62bfa42a9cddaa1`.
That action did not enable Chrome export, so the provenance of its particular
504-byte object cannot be recovered from a Chrome trace. The full surface's
1596 skips and literal reasons remain in its retained parent log, not silently
converted into GPU coverage or into a pass.

Before correcting accounting, a deterministic regression created a cyclic
504-byte CPU tensor inside a **prior** profile, then explicitly GC-freed it
inside the measured draft load. Weakrefs prove its lifetime. Old accounting
reproduced **3656 vs expected4160** (hooked) and **7752 vs expected8256**
(legacy). Genuine PB RED:
`f613409b8dfbb36cc5fd89883d66d8a3593867c3ec9e3d430daee18330215a79`,
canonical failed rc 1, **2 failed / 0 passed / 0 skipped / 0 uncollected**;
snapshot `8ebfed088d37604310dc480d8f983e769a1586dc`, parent ed143.
Terminal SHA-256
`8ebb1ea4258be42b2a1f75fcb4ee6ca9495f0d63f4b62d83bd37204337172cfc`;
stdout SHA-256
`51419802cfcb73c8e239e4c4d9e70132a56707008074718997d64d8d42b55457`.
The earlier cold, unprofiled prior-tensor variant `f479563ea307...` passed;
it was an insufficient reproduction and is **not** claimed as RED evidence.

Raw Chrome instant `[memory]` evidence from RED contains a -504 free at
address368772928 with **no positive birth at that address in the measured
interval**, between a +32 interval birth and the checkpoint allocations.
The raw address359489152 is later freed and reused for the load workspace.
The old sum of function `self_cpu_memory_usage` started at zero, deducted
that external free, and charged aggregate values at function-start times.
The deterministic trace establishes this mechanism and the exact offset;
it does not establish which object caused the original whole-run failure.

Only test profiling/fixtures change. `_owned_cpu_peak` now sweeps chronological
Chrome memory events, owns positive CPU births keyed by process/device/address,
and subtracts only size-matched frees of those live births. Removing a freed
address allows later reuse to start a new lifetime. Unmatched frees are audited
as `ignored_preexisting_cpu_free_bytes`; duplicate births or mismatched sizes
refuse. There is no global-GC predrain in the measurement helper, no process-
memory baseline substitution, and no removal of instrumentation. Full Chrome
trace documents are retained after one Kineto export for accounting and receipts.
**Exact4096/2048 delta and duplicate-lifetime checks are unchanged.** No
production hook/load/share/forward/refusal, source identity or admission changed.

Scoped GREEN + five-file compilation + real-verifier new-base selector:
`12447f647c37e035119656412af2592ad547f05058e9e18a278ac6215876dab9`,
canonical executed rc 0, **70 passed / 0 failed / 0 skipped / 0 uncollected**
(49MoE +19original lifetime +2new GC regressions), worksteal2, durations15,
2CPUs/4GiB, priority-10, timeout300s. Exact4ed2base selector remains
**narrowed326, no forces_full or unreadable sources**; no child full retry.
Snapshot `e5cbe3c4126cbdc9023be6eec5ff2edd662aafce`, parent ed143;
snapshot input SHA-256
`04c2eea5598ab8a69ba76c100d3605d5b0277a0e6880df05427cef29bb91d839`.

- CAS receipt SHA-256
  `293449e291b529b75e4042044c2a6b626d220d46bdf5fe9737dbced12a58228b`.
- CAS output SHA-256
  `f7134c646aafdd52575f149279df9449a43583f4780e8da8c9b81e84f9f98315`.
- Canonical terminal SHA-256
  `092165b54524c8cdafae46969da67dc6e42ddce70e313c1393bae3bea4dbdb23`.
- Canonical stdout SHA-256
  `6d50a9c5a2de1396295d9c81e92de2bbc0cf62c22d207fe04299ea2dcce68206`.
- Surface SHA-256
  `9e5ba8e2579bc4cdb162fd919d6bc568587ea422e7491f2f4c26489134db9e80`.

Both GC regressions audit504 unmatched free bytes and recover exact4160/8256
owned peaks. All original TP1/TP2 identity/value/lifetime/delta checks pass.
Worker-local full traces on dl380g10 under
`tmp/645-receipts/profiler-owned-green-profiles/`:
`preexisting-garbage-False.trace.json` SHA-256
`38ae8627606923f6bb21e3af3db575c7be929b45da16ca7a3ab310c6d2760cf9`,
and `preexisting-garbage-True.trace.json` SHA-256
`e95f10b79b4848a6bc7000c865f983b9d9a14ce9d98c8c1d26e746a004e855fd`.
The six original paired traces and JSON summaries are freshly exported there.
Coordinator logs `profiler-prior-red.log`, `profiler-owned-green.log`; population
is worker-local `profiler-owned-green-surface.json` under the existing root.
Literal scoped population: `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`;
`0 test(s) skipped, 0 module(s) not collected`; `0 test(s) allocated on the device`.
No scoped skip reasons apply; no CUDA or served-quality claim.

Old581 action193's broker-EOF rc125 is not accepted as GREEN merely because
its stdout reported passes (parent filed PB#1378). It is not the deterministic
fixture RED above. All previous histories remain source-bound. Parent owns a
new exact-head whole-population gate and retained review; the scoped70GREEN
is not whole326-file acceptance. No other live branch, release, image, wire,
math, pin, production guard or shared RESULT/PENDING state was changed.

## Final owner integration including merged PR 743

On parent steering the owner fetched master
`138b24e5e2a16308554917486fb856ec48bfb67f` ([PR 743](https://github.com/RobTand/tessera/pull/743) merged) and rebased the
single issue commit. Only the architecture-header conflict required resolution:
upstream #567 pricing and #645 lifetime restamps were both retained verbatim,
alongside the prior source-profile provenance. `serving_parts.py` auto-merged
upstream PR 743 additions; #645's delta there remains docstring-only. Config,
lifetime hook and both test files are byte-identical to reviewed/tested e541.
No new algorithm/default or production edit was introduced. Separate base ref:
`sol/tessera-645-rebase-base-138b24e5`; all historical bases remain recorded.
Provisional rebased/tested source: `3a2ed5936b13cde8ecd7c89b8311f86858ed732d`.

Bounded PB integration `c2cd0ba37b3b54b61483c5e041de37581401189a01c3dc3057c29dc7ab51b062`
completed executed rc 0: **70 passed / 0 failed / 0 skipped / 0 uncollected**,
five-file compile passed, real-verifier fresh-base selector **narrowed327files,
no forces_full or unreadable sources**. Worksteal2/durations15,2CPUs/4GiB,
priority-10,timeout300s; source PYTHONPATH/campaign TMPDIR/native1/affinity retained.
No broad child retry or GPU work. Snapshot `c8c9a920d0b6a13ac738c06daf69de1045fad5c7`,
parent provisional3a2; base ref binds exactly138b24e5e2a16308554917486fb856ec48bfb67f.

- CAS receipt `a5fc9ad9f7e3531c7effc085981c3cfc11cacc8caf5fbbd9bffe5130676009df`.
- CAS output `ed605be0277006fcf7943cd14ab9c788b5d20c4cbae6d889c41ef79964ffdd29`.
- Surface `dfff67d78ccf451abd8d3b0905bfdddc6edc3e38256ae9b26406a79df8a22dc3`.
- Canonical terminal `1258ccddd8613635adbee9c8c59473168a7fede026933cb52f4640addf948631`.
- Canonical stdout `c8f9d8b72684123a6c54cc008dd376ebdf81ba618dd2b850654fc4b31cc3ef71`.

Literal population remains NO CUDA (Torch2.11.0+cpu),0skips/0uncollected/0device
allocations; no skip reasons apply. Coordinator log `rebase-138b-green.log`,
worker-local population `rebase-138b-green-surface.json`, under the same receipt root.
Parent reports reviewer e7db2e56 found no issues in immutable ed143→e541 amendment,
conditional on final full GREEN and rebase equivalence. No full admission is inferred.
Untouched upstream type advisories at `experiments/full_model_original_wire_checkpoint.py:78/84`
were confirmed baseline by parent; no globally type-clean claim or suppression.
After testing only this receipt append changes; parent owns final exact-head327+
broad gate, integration review and merge. Historical RED/GREEN/source bindings
are preserved, not relabeled as whole-package equivalence or new full acceptance.
