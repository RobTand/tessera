# GLM research TP2 intake: CPU evidence and prepared native gate

2026-09-08, issue #435, PR #436. The serving change is `85f9971f7`;
its separate bounded native observation harness is `fd17e7cfaa97`. The full
artifact's producer remains `07ad344c3275` and is not replaced by this serving
checkout. Native execution has **not yet run** for this change.

Stock vLLM's exact image
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`
constructs the complete model on its target device, loads all weights, then
finalizes all quantized modules. Full padded wire banks on every TP rank therefore
prevent a checkpoint larger than one device from reaching its final rank-local
packed representation. The change makes explicit research TP2 intake validate,
slice and pack each original projection during its loader callback. Original
length/maximum-stride validation remains a finalization gate.

CPU regression before the fix used unchanged base `2890976db` under PrismaBuild:

- `0a817da325653ed690021bcb4d52b5bf9bcfa228349e182c3f2d55116637fd52`:
  4 failed, 0 skipped, 0 uncollected. Both ranks allocated `653808` wire bytes
  where the regression requires zero; each rank had prepared zero projections
  after its first callback where the regression requires one.
- `000ffacf8f39` (full key and snapshot in the evidence below): 3 failed,
  3 passed, 0 skipped, 0 uncollected. The final corrupt/wrong-role callback
  tests reported `DID NOT RAISE`; the old path validated these only at
  finalization. The role-concatenation test reported missing `concatenate`.
  Missing-wire, maximum-stride and duplicate/wrong-parameter/late-load guards
  already passed and remain preserved by the new path.

The import selector returned 209 files. PB's supported 20-shard fanout used two
pytest workers and four GiB per shard, one native thread per worker, x86 placement
and priority −10. All 20 actions exited zero: **3,708 passed, 602 skipped,
0 uncollected** on dl380g10, PyTorch `2.11.0+cpu`. Skips include CUDA, absent
box artifacts and optional packages, including Transformers. They are not
native-serving coverage. Actual CAS payload hashes, terminal summaries and the
single measured source seal
`165155ce52684bda3c60ade02e6e961c4b1849f9b6fb658839a526ae8e41c84f`
were checked across the population.

Final observation-harness CPU guards passed separately: 22 passed, 0 skipped,
0 uncollected, action
`727a0a0218a5b47460b3df5ac9a811044b5bec1d87915153bfb9af9b6ab6c855`,
receipt `bfb1e75fcb35a9c36bad53f5943acfec124cdd5b60daa31130fa24434be12ec1`.
The harness compile checks and refreshed issue-reference checks passed in
`118e592e44385a9da90a58c28f4bf76d43c34e7e33c8c4810b0a73d5a5e70a81`
(3 tests, no skips/uncollected). The observation tests cover measurement guards;
they make no claim about native allocation or collectives.

Evidence root:
`/mnt/shared/tessera-measurements/glm-first-artifact-launch-plan-20260908/`.
`cpu-suite.json` holds commands, actual outputs and receipts;
`cpu-receipt-audit.json` binds all 20 action keys and verified CAS payloads;
`cpu-surface-*.json` holds exact skip reasons and source verification.
`pre-fix-regression.{json,log}` and `pre-fix-extra-regression.{json,log}` preserve
the baseline failures. `launch-plan.md` records the first complete candidate's
155,668,854,528-byte common budget, full 36,423-Linear roster, producer/cache/H
inputs, explicit calibrated cached-export commands and full-checkpoint TP2 gate.

The finite native A/B is prepared in `prepared-native-intake-ab.json`, awaiting
the coordinated two-Spark window. It binds frozen `07ad344c3275` before and
serving/harness `fd17e7cfaa97` after, using the same JSON-selected original
288-expert owner and actual stock loader callbacks. Both ranks run NCCL TP2;
trained gate/shared-expert execution and the actual final collective are checked
against an independent stock TP2 reference. Torch traces cover construction and
the first three projection callbacks; allocator counters cover the complete
owner before its independent stock oracle is allocated. Both-box Netdata is
required for each arm. Each rank has four CPUs, 48 GiB and a 600-second bound.

The bounded fixture repeats original projection containers across experts.
It can establish the changed intake's native ownership and collective behavior,
not diverse-expert quality or full-engine fit. The complete 45-layer artifact
and its two-rank memory/KV/generation gate remain required. No runtime cell,
producer pin, serving default or quality claim is promoted by these CPU results.


## Native A/B completed, 2026-09-08 23:49 UTC

This entry supersedes the pending native status above. Both arms completed on
Sparky and Sparklina in the pinned stock vLLM image, before at 23:41:38–23:43:31
UTC and after at 23:48:01–23:49:50 UTC. Every launcher and container exited zero,
without timeout or OOM; the four owned containers were removed and their absence
was independently checked. A first launch failed before Python/CUDA because
Docker could not traverse the evidence directory under root-squashed NFS. The
bounded retry changed traversal/read modes on owned inputs only; source and
request bytes, CPU/memory limits and timeout remained identical. Both initial
failure records and `retry-native-intake-ab-02.json` are retained.

The same checkpoint-JSON-selected owner, three original projection containers
repeated across 288 experts, 864 callbacks, trained router/shared expert and
input-file hash were used on each rank. Memory counters cover construction,
loading and finalization before allocating the independent stock oracle:

| Quantity, bytes | Before rank 0 | After rank 0 | Before rank 1 | After rank 1 |
| --- | ---: | ---: | ---: | ---: |
| Construction wire banks | 1,831,476,672 | 0 | 1,831,476,672 | 0 |
| Peak CUDA allocated | 3,592,044,032 | 1,876,871,680 | 3,610,852,864 | 1,895,746,048 |
| Peak CUDA reserved | 4,594,860,032 | 2,696,937,472 | 4,611,637,248 | 2,715,811,840 |
| Final packed owner | 943,423,488 | 943,423,488 | 943,423,488 | 943,423,488 |
| Prepared projections after callbacks | 0 | 864 | 0 | 864 |

Decode (one token), all-expert routing (36 tokens), clamp stress (four tokens)
and empty input all produced finite direct outputs with maximum absolute error
zero against the independent stock reference. Selected tiles/scales were exact.
Every nonempty case also traversed both the stock runner and GLM wrapper, with
one quant-method call, one shared-expert call and one actual final NCCL TP2
all-reduce per entry point. Trained gate outputs and selected routing agreed
between ranks, and their outputs agreed exactly with the stock TP2 oracle.

All four Torch traces contain actual CUDA kernels and CPU operations. They
cover construction and the first three callbacks; complete-intake allocator
counters supply the memory comparison. Both-box Netdata retains raw CPU,
power, memory and pressure series for each arm. Mean intake GPU power was
18.26/14.04 W before and 16.64/13.28 W after on Sparky/Sparklina, far below the
approximately 140 W envelope. This was one cold A/B measuring ownership and
functional behavior; it does not qualify throughput or GPU saturation. No
speed or work-per-joule ranking is claimed.

The sealed plan mistakenly supplied raw 75-file source-tree hashes in fields
named `expected_installed_package_source_sha256`. The sealed archives' own
`pyproject.toml` excludes five `tessera._dev` files. The audit derived the exact
70-file wheel projection and verified every installed byte and loaded origin
on both ranks. Actual installed source hashes were
`7d1e3b011eaba8d3a779656e4e3cfda9d9c3db41b7cf7d35aedb3d8b094f99df`
before and
`ea04dd407ac3c0b4eccc3b9d8ad0f05751bba63abe00387dc98bb1cbe30bf96c`
after. The original plan and producer-tree seal remain unchanged; the correction
is explicit in `native-package-projection-audit.json`.

[Native measurement bindings](glm-rank-local-intake-native-20260908.json)
retain the exact source/archive identities, rank receipt/profile hashes,
functional results, allocator values, Netdata windows/statistics and hashes of
the raw evidence. The native after-source is **fd17e7cfaa97**, not a later head.
This remains one repeated-expert fixture, not diverse-expert quality, a complete
45-layer engine load, KV-capacity proof or full-model generation. It promotes no
runtime cell, producer pin or default.

## Separate device-lifecycle correction

Review found that capturing the construction device could prepare CPU owners
when stock vLLM uses an explicit load device. Commit `0a3dff7e5` resolves the
live loader anchor, preserves the ordinary finalizer's promotion to the current
CUDA device, and refuses a device change after accepting the first projection.
The regression stops at the parser's device handoff before any CUDA allocation.
On prior code, PB action `982ad1822674` failed with
`device(type='cpu') != device(type='cuda', index=1)`; the targeted corrected
run `7b25b5953f14` passed 48 tests on dl380g10 with PyTorch `2.11.0+cpu`, zero
skips, zero uncollected modules and zero device allocations. Its CAS payload was
verified against receipt
`02be32aa14dba977ed937eabe1b145870e8ec6d230d02b35716d31348339f959`.
`device-regression-before.{json,log}` and `device-targeted-after.{json,log}`
retain the exact action keys, snapshot bindings, outputs and exit statuses.
The additional device-change guard is covered by a CPU refusal test. Native
CPU/meta-to-CUDA loading was not measured. The canonical CUDA construction path
used by the native A/B keeps the same parser, shard and packing calls; the
later source is not relabeled as that native measurement.


The device correction's refreshed import selector returned 210 files. All 20
PB fanout actions exited zero: **3,715 passed, 602 skipped, zero uncollected**
on dl380g10, PyTorch `2.11.0+cpu`, two pytest workers per shard under
`--dist worksteal`, four GiB per shard, one native thread per worker and priority
−10. No test allocated on CUDA. Skips include CUDA paths, absent box artifacts
and optional packages (including Transformers); exact reasons are retained in
`device-cpu-receipt-audit.json` and the `device-cpu-surface-*.json` populations.
Every CAS payload and terminal result was checked, all worker/controller source
identities agreed, and all 20 populations measured effective source
`5e7f0aa8a92eed36e97e88ed610dd990ed6212be1f24cf89baebe961c59349de`
with parent `0a3dff7e5`. `device-cpu-command.json` records the exact invocation;
`device-cpu-suite.json` retains all shard commands/results and durations. Later
edits in this report and architecture are documentation only.
