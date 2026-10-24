# Issue 1358: partial provenance recovery, not TP2 readiness

## Disposition

This is an incomplete implementation handoff for tessera#1358. No repaired export,
compatibility qualification, GPU serve, READY event or successful request was
produced in this attempt. The required `issue1358-tp2-readiness.json` is deliberately
absent: creating an acceptance-shaped receipt would misrepresent missing evidence.
No exporter, reader, application configuration or protected acceptance test was
changed. Original artifacts were not mutated. No speculative setting was removed.

Compact original evidence is retained in [issue1358-recovery/inventory.json](issue1358-recovery/inventory.json).
Each entry binds a repository-relative path and SHA-256, with its historical source
location, byte count and extraction scope. This is an inventory of **recovered
historical evidence**, not the generated-file inventory required by O3.

## Newly recovered producer reconciliation

The original export's fleet request was recovered verbatim from
`cas/requests/ad/ad87b7c860f116d50cc63af767aa9524cc5dfa22221eaa46c56ab7b43a6f846b.json`.
The matching `pb-queue/done` record reports `executed`, return code 0 and elapsed
2912.352590560913 seconds. A compact selected-field projection binds the full
record's SHA-256; it omits bulk progress output and resource-scope bearer tokens.
The raw fleet record remains the authority to authenticate, not this projection.

The request and completion record agree on:

- checkout snapshot commit `3725c1b111f73475b479662aac7ef4f1aabf6504`;
- snapshot parent `af7a86d43da3487179b7d16606ef0f5a3046d73c`;
- source snapshot input SHA-256 `bac045bee5e605ca8be10049fe08f5c1f4cf47d562a0d53e8eb2ef411d8fa93e`, 10,572,757 bytes.

Thus the reported `af7a86d43` and `3725c1b1` describe distinct stages of this
recorded export: parent and checkout snapshot, respectively. The original
interpreter pathname includes the parent revision; that name is not proof of
which package was imported. The snapshot/package contents and runtime imports
were not independently authenticated here. The `6be66bc` artifact-name suffix is
not a producer identity. Contract 38 remains a historical bindings claim, not a
freshly verified manifest/runtime-contract identity.

The completion stdout's first line records actual export-plan SHA-256
`2c63c7564fe0e631eccd2c6e6df2e7d26de4c108f526bbce1515c5b2cc6bb789`
(584 entries). This must **not** be conflated with the research lead's
`ac0957308f729887239701eb702d334d8c2eb0f3277f872c8d247c652cb79c51`
plan-assignment identity. The underlying allocation/plan transformation is still
unrecovered.

The retained request supplies the original argv and binds priced inputs to
`ff30459219e0df6740830e01bc446b2b53c21fab8cccd5ef55c86f11e724f3d0`.
It names the original Hessian references, cached-expert manifest, source digest
cache, and cached-producer package with source seal
`0833671bbddbc3fb7186bdbed0a905ef248c23ffdf1aeee0c083322680d20f6b`.
These are original invocation authorities, **not** independent verification of
all referenced bytes, scales, cached units or source slices. Do not substitute
these pathname identities for the missing per-unit provenance checks.

## Original failure and configuration disagreement

The retained rank-0 log is byte-for-byte the original failure (SHA-256
`2cd740e418053ce67b0fefbb4a4be59f07a100bb0ca51b36adfe10fd3e56f861`).
It fails during VllmConfig construction with
`model.language_model.layers.10.mlp.experts: expert_ids is required`.
This is before model construction, placement, collectives or generation.
It independently proves one missing map, not a fresh census of all 42 stacks.

The retained `original-engine.json` describes max model length 2049, 1 GiB KV per
rank, eager mode and chunked prefill disabled. The log instead records length
8448, 2 GiB KV and chunked prefill enabled. Engine JSON is therefore **not** a
reliable description of the attempted invocation. Retain both; do not edit one
to agree with the other. The retained gang/inputs describe a historical two-host
allocation and runtime pin `4c0fd6aca04f1c3e95b094ae006c2479043c3372`;
they do not prove a currently available allocation or that a repaired artifact
was supplied to either rank.

The original roster's SHA-256 was freshly verified as
`bb07140cb5c23ad574f32e138333942e95df5be31b2510afe0971aa32539e1d8`.
Its config/index/manifest/shard entries remain **historical claims**, not fresh
hashes of the artifact. The failing config itself was not recovered; neither a
repaired config nor a source config snapshot was invented from these claims.

## Recovery/reuse decision and concrete unavailable inputs

Listing `/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc` through the file
tool returned `permission_denied`. This root and `/mnt/shared/models` are outside
the supplied readable platform roots; no alternate-path or shell read was used
to circumvent the denial. A shell existence/readability probe had reported the
run notes, exported config and source config present/readable to the host UID,
which does **not** grant this agent permission to read their contents. This is
an access-policy boundary, not evidence that those files are missing.

The supplied writable roots are only the worktree and attempt temporary root.
No configured writable large-artifact destination or current two-rank serving
allocation was established. The admitted single-GB10 test arm is not by itself
proof of the required two-host serving facility. No GPU or export job was
submitted to work around these gaps.

Focused recovery found a completed *different* all-priced T8/BF16 export under
`/mnt/shared/tessera-measurements/t8-v1-allocation-20261005T022641Z/release-all-priced-T8-BF16`:
its `complete-artifact-receipt.json` and `allocation-run-contract.json` identify
`exported-progress` and writer operation
`55010e80d48c7fb348b2f65d87ee2f0566c72ed7765f0ca9c7f3ccbd8ae31bfb`.
Its recorded plan is
`58f1b8821ba07b815cc7a0df2015d67c28ffa71993ce046d638c78be4f7a5c8f`,
not the original export's recorded plan above. Its receipt explicitly excludes
serving qualification. It was not selected, copied or relabelled as a refresh.
Likewise PrismaQuant's historical prismaquant#1225 handoff-export counters concern
produced-output spool delivery, not proof of the Tessera issue's refreshed
serving artifact. Earlier Tessera export-work ownership and accepted refresh
status remain unknown. No competing encode was started.

## Continuation handoff to tessera#1204

Do not point the existing campaign at this evidence directory as a model.
First obtain platform-admitted access to the original run/source inputs and a
configured fresh artifact-storage destination, and recover the earlier Tessera
export owner's actual output/receipts. Reuse a completed output only after
qualifying its full input tuple and consumer compatibility. Otherwise use the
existing supported exporter, with original input bindings and authenticated
historical-cache intake, into a fresh configured root. The selected candidate
producer is not yet qualified against the explicitly pinned consumer; do not
substitute main or another PrismaQuant runtime.

Then validate every routed stack/population/map/class and storage-ordered
projection rung against the original plan, including source tensor/slice,
manifest, scale and shard/index reconciliation. Keep valid nonidentity maps and
fail-closed missing-metadata refusal. Retain generated config/manifest/index,
index membership and all shard sizes/SHA-256 values without committing weights.

Only after that, establish the configured two-rank facility and record its
actual image, interpreter, imports, native dependencies and runtime contract.
Before execution, timestamp a deciding-case rationale: both ranks loading this
exact inventory-bound artifact, reaching READY and answering one bounded request
(e.g. at most 16 new tokens). CPU metadata tests cannot exercise distributed GPU
loading, placement, collectives and generation. Target one approximately
10-minute wall-clock run on two GPUs (approximately 20 GPU-minutes); this is a
continuation planning estimate, **not** an executed allocation or authorization
for a sweep. Authenticate qualified prior results first. Retain both rank logs,
actual argv/environment/effective configuration reconciled with observations,
artifact root/inventory digest, READY evidence and request/response. Record a
concrete facility refusal if unavailable; do not manufacture a success receipt.

This package does not establish the campaign's TR3 correctness, fold-on/off
performance, profiler/energy, work/J acceptance or promotion. The historical
2026-09-26 smoke serve used another Tessera revision and cannot qualify acceptance
at the requested pin.

## Verification

All commands below used `run_tests`, the x86 arm, two workstealing workers,
`--durations=10`, native thread limits of one, `PYTHONPATH=src` and
`--basetemp="$TMPDIR/pytest"`. No CUDA device allocations, no skips, no GPU
export or serve in this attempt.

1. `python -m pytest tests/test_issue1358_readiness_receipt.py tests/test_expert_classes.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"`
   reported 238 passed and 5 failed. Four failures are the unchanged acceptance
   assertions at the absent readiness receipt. The fifth exposed an unqualified
   cross-repository issue reference in this report; it was corrected to the
   explicit PrismaQuant tracker. The impacted-test guard passed. The retained
   stdout was read back using `read_test_artifact`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6d8100a78cca2405bedb51f64092d91faf014c6eae6d4e715cba67fda6eb3644","size":8897}},"operation":"2c34ab14e752ced4fb5263e80165f4c660b71412755a6171c5230ced1935b161","receipt":"ed3f1f48eaf16f733284e969046caa81a3ee2ce30abfd76972561ae961945be5"}]

2. After the reference correction,
   `python -m pytest tests/test_issue_refs.py tests/test_expert_classes.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"`
   reported 30 passed, zero skips. Explicit metadata refusal and nonidentity
   storage-order regression checks remain passing. This does not qualify the
   actual full artifact or distributed runtime.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9264da402394bc1f8442cc96523ffdbcfa74c947bc49cb80609d84a320ffeb51","size":1063}},"operation":"14b443ce5dd824902f25b5dab426e53434db6956068b2e37cb5094f5beff3ddf","receipt":"50036d4b6b7cbf3d1ae04d8bf5380095e6ed8aefcc979d07f12727320f34b343"}]

Acceptance remains unmet at the absent readiness receipt. This is an incomplete
deliverable, not a defective test or an impossible objective as written. GPU
checks and actual TP2 readiness were not run; no GPU-arm pass is claimed.

### Supplied research results (not rerun or independently authenticated here)

The brief reports four intended missing-receipt assertion failures and separately
239 passing metadata/repository-guard tests, zero skips, both on x86 CPU and zero
device allocations. Its retained evidence lines are preserved verbatim below;
they are not current export/serve evidence.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a7591af5dc39a9ba484564a94c554bcdf0ace3ada16a1e21be9731ca04255d07","size":5993}},"operation":"960ffb9bd81c57ddd6d0081f0166fa0564d061acdd9ca2858e1294dfc74f0584","receipt":"04bbac8b3abb984d3adabc73c62303052a481a4afc3b9b31959d47aa1dbaf9ee"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0881c411914b4aa5f34005ffdf6ddc99d9919bdd94681dcee9e247c20c37bd94","size":1864}},"operation":"3920ae853b37682be2d0896c6477776882ca3242c976eef8834c94b0a814248e","receipt":"ed4fe93193f8182cbdae3239e0693d39142634a8a5e76c44fe678bac7fa08574"}]
