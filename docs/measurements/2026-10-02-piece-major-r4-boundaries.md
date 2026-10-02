# Routed R4 resident layout: CPU boundary and compile evidence

Source reviewed and built: `5f366b45d2fc7823d9720a51d6a8875b8d2cea29`,
branch `sol/routed-piece-major-r4-20261002`, based on `82e6740416`.
The CUDA source is `d8691b2a95772e6e880f8658d91a541df85b2e82ca4d7fef859d72db7491324e`.
No GPU correctness, graph, served quality, timing or energy qualification is
established here. The default remains legacy resident order.

## Boundary corrections

The resident intake freezes opt-in/family/reader selection before its callbacks.
Only the E4M3 MMA one-run R4 path places piece-major words. BF16, the alternate
f16 reader and routed opt-out retain legacy words. Actual loader callbacks and
`WindowUnitAxis.put/finish` are exercised on CPU tensors with only the CUDA
repacker stubbed: the stored permutation, start-state values, layout tags,
single word-plane owner and folded BF16 arithmetic are checked through finish.

Dense Triton calls, fused dense preparation, the native dense custom-op owner,
GEMV argument extraction and E2M1 WINDOW readers now refuse unsupported resident
layouts before metadata can be discarded or a reader launched. Tests use exact
refusal reasons and admitted legacy controls, including value rate 8 admission
and rate 9 refusal. They also cover mixed/unknown tags and native-build/opt-out
fallback refusal. The three direct native benchmark callers pass the new layout
ABI argument; the ordinary benchmark records actual stack tags, native selection
and loaded ELF identity.

## PrismaBuild evidence

All tests were CPU-only (`CUDA_VISIBLE_DEVICES=''`) using Python 3.12,
torch `2.11.0+cu130`, pytest and real Triton imports from `pq-cu130`.
Native math threads were one; pytest used `--dist worksteal` where parallel.

| Action | Source/scope | Result |
|---|---|---|
| `89b2983124e623f7a75b9b4ec03e2dad0ae2add244782849bbd76caa2b699868` | Initial production-boundary regressions on the inherited source | 15 failed, 8 passed; five failures were an incorrect new fixture wire-length shape, corrected before the causal comparison below |
| `07668151f2138e71ddf4e55181da4816dea8ba998c2e59b553356e975fab9466` | Frozen pre-fix `aea3db88e1`, same corrected boundary tests | 16 failed, 20 passed; reader refusals absent and changing the environment changes intake placement |
| `73ed5e41589c1528510f5e7b005cd158705c63a31a550572d1124c00cfc2c837` | `4c107769f7`, boundary and retained mapping files, two workers | 45 passed, zero skips, exit 0 |
| `c6180143a74dd9b92335c2008268c4033e214c2713b3c562691011d5e1204b1f` | Import-graph-selected 362 files, four workers | 6421 passed, 2021 skipped, two failures, two collection errors; retained as a red outcome |
| `6608f0bc29f75ca5b77285be843acdc4e835324c52a2b8aee6be9f0dbccc9571` | Pre-fix source: fully admitted E2M1 refusal controls and external pricing fixture | Three expected failures: two readers reach the device-query bomb, and the old PrismaQuant fixture root is absent |
| `14f5c95567a005e930850b24e1e9c6b067ba6f7b337e907148457ff992b68f63` | `5f366b45d2`, affected files and strengthened controls, three workers | 113 passed, 60 CUDA skips, zero errors, exit 0 |

The broad run's layout-identity expectation omitted the new `word_layout`
component and was updated without relaxing the identity check. Its other
failure used the absent default `/home/rob/pq-wt/tessera-continuous`; the
follow-up supplied the existing `TESSERA_PRISMAQUANT_WORKTREE` setting pointing
to `/home/rob/prismaquant` on Sparky. That checkout was `22149e1aa35a6190e2d7925defb085a698a74caa`,
with accountant SHA-256 `11f171accd45868cbea6e2ae3961568734dcb5c1ed2322d1a9f16e7a41f034df`.
The two collection errors were missing `prismabuild`; the follow-up supplied
the published SDK on `PYTHONPATH`. Only the affected files were rerun. Counts
from overlapping runs are not added into a fictional new full-suite result.

The broad run used 1040.56 CPU-seconds in 278.89 seconds, with an 8 GiB scope
peak. Subsequent builds were sequential to preserve the aggregate resource
bound. These are PB resource observations, not quantization performance claims.

## Retained native artifact

The earlier compile `19f42876…` genuinely exited zero, but its ELF was not
ingested and was deleted with its temporary checkout. Its log and receipt are
retained as compile history; its recorded `3ac45a08…` hash is not a usable binary.

One authorized recovery used the existing `experiments/t8r_speed/build_ext.sh`
for `e4m3mma`, CPU 1 / memory 8 GiB / native threads 1, no GPU, in image
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`.
Action `53b3ed1c915b47850e4e28a097858ac22b3e3ae649ccee5167f034f97c705087`
exited zero and also compiled touched Python modules. Its receipt is
`5b2619663f657cbacd276a1d884bb2d78b7025358dbfc147e292b80aa4e58e3b`.
Full CAS manifest/attestation/payload verification passed.

The existing extension directory is now persistent:
`/mnt/shared/astra-resume-20261002/t8_performance/routed-native-d8691b2a-5f366b45`.
The ELF under `tessera_routed_fused_mma_e4m3_sm_121_tessera_guarded_v1/`
has SHA-256 `ac004831299983bf37a9df889ba3bf4d627e133eb0e27ca9589d7f30bc01737c`.
`resource_usage.txt` hashes to `98f21ae317f27544745a9aaf426ceb99d55cd8bcade93be44f607006d0233a2c`;
`sass.txt` hashes to `4cd1a501fbe8df5197f22b9a937c85b4899449b434124497247d082e122d5d7f`.
These actual files were independently hashed after the admitted build.

All six routed R4 PM specializations (modes 0/1/2, widths 64/128) exist. PM and
legacy have equal register counts: width64 uses 96; width128 mode0 uses 121,
and mode1/2 use 122. Each reports STACK/LOCAL zero and 1024 bytes static shared
memory; no LDL/STL instruction was found in the corresponding SASS. Dynamic
shared memory remains sized by the unchanged launch owner. This does not
establish occupancy, latency or numerical correctness on a GPU.

## Commands, receipts and remaining work

Test command family: `pbrun.py --cwd CHECKOUT --cpus N --demand mem_gb=M
--env CUDA_VISIBLE_DEVICES= --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1
--env OPENBLAS_NUM_THREADS=1 -- /home/rob/venvs/pq-cu130/bin/python -m pytest
-n N --dist worksteal --durations=8 -q -ra FILES`. Every exact argv is retained
in `/mnt/shared/prismabuild-fleet/cas/requests/<first-two>/<action>.json`;
terminal logs are under `pb-queue/attempts/<action>/` and canonical successful
receipts under `cas/actions/v3/<first-two>/<action>.json`.

Detailed CPU and ELF manifests, impact selection and the proposed tiny real-wire
protocol are at `/home/rob/tmp/astra-resume-20261002/t8_performance/routed_deep_qa/`.
The initial proposal is an output-bit rejection screen of L10 TP2 rank0 at
M1/2048 using the existing loader/benchmark, with explicit legacy/PM selection
and the same retained ELF. Shared wrapper forwarding and exact root GPU GO
are required. It does not replace the remaining TP cuts, intermediate outputs,
layer45 BF16, remaining M values, graph, quality and matched performance gates.

## Follow-up: bounded output observations and failed execution contract

The first approved action `7c0c5fd30d29738bc17c8e9346ce738941653c1b3c60f21061cee4403af5af98`
failed before device setup: the benchmark's ordinary path refused the A8SE
artifact override. Commit `08828c6154` admits only the exact L10/M1,2048,
outputs-only, non-graph case, with no staged/routing/native override. The
strict historical replay and ordinary default admission remain unchanged.
Actual causal CPU action `726c68b8e6e2ab9f86278a2b350fca81745f2b841b8f34d5ff73710a8892607b`
was 1 failed / 1 passed. Final focused action
`3c1dfa7028fbe86300c7bd66ed7d2a061988b33d3ba8a45ddc22eb4eaa462398`
was 38 passed / zero skips on DL380 CPU, `-n2 --dist worksteal`, 2 CPU / 4 GiB,
native threads 1, portable `pb-cpu` interpreter. Full CAS verification accepted
receipt `ffd2e81d27db2798c4c02815357d25b325399891b80bd81afe0d5f65b957ccbc`.
The intervening `129fbe10...` failed before tests because a shell hid a
nonportable interpreter path from placement; `55ea3a4e...` was 37 passed / one
stale mock-parser fixture failure, corrected in `0d63b68f4b`.

The single corrected GPU action
`affe687f7b7fe6eda700b2ed82964075cb900aa71440a23d54001856a922d83b`
used source `0d63b68f4b964c9640725c04c266792012cc2919`, production overlay
`5f366b45d2`, the retained `ac004831...` ELF and image `5be13705...` on Sparklina.
Both arm programs exited zero. Actual gate/up/down layouts were all legacy
with the toggle off and all piece-major with it on; native flags matched.
Both reported 1,872,827,160 resident bytes. The final output hashes matched:

| M | Shape | Both arms' SHA-256 |
|---|---|---|
| 1 | 1 x 4096 | `f77e4b5afcd4137c51ccc998e8f6e88b24feb871e0464a9f3f533c5f28cc8f84` |
| 2048 | 2048 x 4096 | `ccd39f30c733d7cb1a317b0e3c3b7ca6859e175a94f528c9e0e27b22c774d192` |

**The outer action failed (exit 1), and has no success CAS receipt.** Its final
no-rebuild check caught an ELF timestamp change. The retained `.ninja_log`
shows one linker invocation in the legacy arm; CUDA object bytes and timestamp
did not change, and the ELF hash remained `ac004831...` in both arms and after
the action. This is a failed execution-contract check alongside two limited
numeric observations, not a passed screen or performance qualification.
No timing run or GPU retry followed.

The original before-state, both JSONs, action window, copied Ninja evidence,
and explicit failed-status `OBSERVATION.json` are retained at
`/mnt/shared/astra-resume-20261002/t8_performance/routed-screen-0d63b68f-v2/`.
Both-host telemetry was recovered read-only for that same elapsed window;
`netdata.json` SHA-256 is `76695da04c862923cf4d636d95c5758cf92a7ae4c564e9a153ca428e6cd1b2f8`.
Those observations do not address the subsequently investigated terminal-barrier
race (#855), intermediate outputs, other shapes/routes, serving or performance.

## Common-source continuation

The continuation uses `sol/739-piece-major-common-20261002`, rebased onto
`09cdb22f6cd4a51e7d480531fde82376c785f4f7` (merged MLA plus #855). Duplicate
terminal fixes were dropped, and the canonical version/identity support was
carried from the reviewed standalone branch. The routed CUDA source still
hashes to `c236b7aa340c74c9b965d84133d54c16c1488cecc6781d996ee14cc4d4068522`,
identical to the earlier four-family compile. Its old build paths do not match
the canonical `/tessera/src` and `/ext` namespace, so those ELFs cannot establish
no-rebuild reuse under the new protocol.

The existing benchmark wrapper now honors the same canonical source, extension
and version-metadata mounts as the existing build and test wrappers. Its actual
Docker-argument regression failed before the fix: PB
`3ecfad8584102424bd01ecffef6153ce178914ad6348c7d6626df58dc4b0efb2` recorded
2 failures / 8 passes at the `/tessera/src` mount assertion. The corrected wrapper
and unchanged output-screen argument tests passed PB
`a0628a8108cd15b114c256e3e663aadf4a9c93a913296192c9723aa4ec53aded`: 25 passes,
zero skips or missing collection, two xdist workers, CUDA disabled. This is
wrapper evidence only. It establishes no GPU numeric or performance result.

The existing byte-audit owner now includes a resident matrix row: an actual
encoded E4M3 one-run R4 520x256 unit reaches two tiles and the partial trailing
row. It hashes the serialized unit before/after resident relaying, the original
words, the piece-major words and the restored words. The new corpus regression
failed before the row existed (PB `c648658007a49b9631eba0097ec24363f16777fb0caa528ac46183e3acbafcb4`,
`resident_hashes` absent). PB `fe3a599c7c9dbd3da94d64c39f3646e6d980786f0697909cf762f58db2b74aec`
passed 63 selected CPU checks, zero skips/missing collection, two xdist workers.
The row established equal serialized blobs and an exact word bijection, with
different physical resident order. It measured no GPU decode. The 20.18-second
row ran once; the prior encoder matrices were not redundantly remeasured.

The direct-vLLM input/native extension is reused byte-for-byte from
`731f5caf9e7303318feeaa254d5b4a0f471967ea` at the existing `pb_staged_store.py`
owner, with its direct/native callback tests. Its original causal control was
PB red `e73d2521bc0423099bfdabc4f52d8b30f117a23e769c7cd78d5c8c94c1a8cd02`;
PB `daa406b752cd23ecc0f2b11d6f1b4d2e1c5cda4d62d427f79c933c9328b96d37`
passed 101 CPU checks, zero skips/missing collection, receipt
`925703ae54707baeecde446a981241835f7b89481a87fdf55b34739e041adcce`.
This owner qualification is reused rather than repeated. No paired-K32
kernel, schedule, feature selection or benchmark mode is imported.

One canonical native CPU action,
`293aa99d09279ad065e12df84385509816b5f2b6b43f3cc751d56e227313a1fa`, passed
with CUDA hidden on Sparky: one CPU, 4 GiB, 59.98 seconds, measured cgroup peak
3,840,479,232 bytes. Both PM arms use its ELF
`4d693c2621333eb36b32b21479e0283e95236e8864923fad9d1f15bc1487091d`.
The finalizer retained exact source/object/ELF/recipe bindings and Ninja no-work
state. Full CAS lookup and all 116 frozen source files were checked. This is
compile/reuse evidence, not speed or GPU numeric evidence.

The owned 873-entry readset reuses all 872 entries of the verified sealer
`a01cb090aa43f96220159e92788ae372c161fcb1a27895548f0797b63d2d165a`
(original manifest `ac4606e82384c891114a4836525456c3356f04dc8b6ec4db898949454fb4f2d0`),
including captured routing, and adds only the exact PM ELF. No whole-model
rehash or shared-inventory mutation occurred. Its digest is
`ae1fb40659d2a1b8c372c4609f3256a9fd62980f09e0687bde4042e5225ccf07`.
The packet is under
`/mnt/shared/astra-resume-20261002/t8_performance/piece-major-common-c236b7aa/`.
Numeric GPU execution and measured performance remain pending root GO.

The existing Netdata collector previously overwrote `steady-power` for every
cell, losing all but the last arm's measured window. The actual collector
regression reproduced that loss (PB
`f6f3a705d7a3cf36cdcb42ef5c32521dfcf748f3faac0f66251f6b72d47ac3d6`, one failure).
It now keeps every cell's window for both boxes while retaining the historical
alias. PB `8049cd2c142c94fb7d53e402327d3a1121711022edbfbd106bb99ed2e9d4de72`
passed the control, zero skips/missing collection. This observable defect is
fixed in a separate commit; the control used inert telemetry responses and
establishes no actual coverage or energy claim.

Mode-specific integration passed PB
`ba11d685b547746fe0d7ea7588cc3ed8eb8939aa222f4c38168de6ebf9e7728f`:
75 CPU passes, zero skips/missing collection, two xdist workers. The direct PM
container/canonical namespace control passed PB
`40d20c6a401c4ff9620bb60dda755c20c220729a6bf1c7b812a891b0687f7070`.
The final cold-script check reproduced an import-path defect at the real entry:
PB `98639dd605b454940ee9f12f3c43c065bf93e2448061bf2319c4ef99f2528d49`
failed because running the script does not add the checkout root to sys.path.
Using the existing script-local helper import passed PB
`42c766bd57f1561ea42f5055f16f620b5e9bd2fe06e93fea9f67f3e8912f3771`, one
CPU pass, zero skips/missing collection. The entry refused the missing protocol
before device access. Numeric, timing and profiling execution are still held.

## Frozen observer and cold dependencies

The real production adapter is frozen. The original PM observer tried to set
its `_launch` field; a CPU control using the actual production class reproduced
six failures with `FrozenInstanceError` in PB
`740856dc8360ac3daac92b51de1a349941b3fc9637e651c6f9ab58b8ea885e31`.
The observer now uses the same exact-instance guarded class seam as the paired
owner (`827abb48d787506c4fcc2295a9d0eeda4f0fc48e`), forwards the original calls
and restores the class method in `finally`. It never mutates adapter fields.
The CPU control reaches mode0/1/2 and covers a foreign instance, missing or
duplicate roles, wrong family and forward exceptions.

The direct wrapper reuses the qualified pure-Python runner mounted by
`3c3b73ca49ed9b9731efa8be0fcd9ac646eae580`; its actual source remains the
#855 shipping GPU43 runner. The old PM wrapper failed the mount/PYTHONPATH
control in PB `8c43910d26d0c607c9cfcc53d295cdf627006ad0ff9eb0458eeaba844aadcc17`.
The corrected observer, wrapper and existing argument checks passed PB
`468f8f4551ecce68554092fbc99383cd1ced2c360a9aff267ed639cbfb6e2022`:
33 CPU passes, zero skips/missing collection, two xdist workers, native1.
The cleanup/time/pressure owner remains the existing external
`paired_k32_action.run_direct_arm` and `owned_cleanup`, now bound to the exact
qualified `3c3b73` bytes containing `e3234180a8b5f4d1bd3743ae556d2f18e68e86e1`:
only an exact owned-CID absence is accepted, case is normalized, and cleanup
survives the poll/signal ProcessLookupError race. Its prior focused 50-check PB
qualification is reused, not repeated. No native build or GPU run was made.
The original numeric packet is retained as an unlaunched, superseded input;
updated execution uses a separate v2 protocol and window packet.
