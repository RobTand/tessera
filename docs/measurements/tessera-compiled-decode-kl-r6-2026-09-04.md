# Compiled window-GEMV KL: fresh r6 population (2026-09-04)

**Outcome: #113's compiled measurement is complete. No numerical-quality winner
or lane-isolated quality claim is established.** Both lane serves executed the
required 50,176 GEMV launches; both fallback serves executed none and refused
preparation for all 112 modules. The cross-arm prefill lower bound is 0.019502,
so the decode separation cannot be attributed solely to M=1 GEMV arithmetic.
Against the compiled BF16 teacher, even the lower-bound ordering flips between
the two lane rebuilds. No promotion gate or release claim changes.

This is the fresh closure population for #113, not a revision of the earlier
Sparky receipt. It measures one compiled BF16 decode teacher, two independently
compiled lane serves (A1/A2), and two independently compiled fallback serves
(B1/B2), all on Sparklina (`gx10-6b77`). Each student's decode and prefill dumps
come from that student's same serve. No weights, serving defaults, format,
encoder profile or promotion rule move in this receipt.

All quoted KL values are **top-1024, teacher/student-intersection lumped KL lower
bounds**, not full-vocabulary KL. Decode covers 256 stride-16 scored M=1
positions; prefill covers 4,088 positions from eight 512-token chunks. The
corpus and tokenizer are identical within every comparison. These are two
fresh builds per lane state, not a confidence interval over build variability.
There is no latency, speed or work-per-joule claim.

## The measured result

The table names the reference first: A1 → B1 means the tool's teacher argument
is A1 and student argument is B1. It is directional, not a symmetric distance.

| Reference → student | Decode KL lower bound / top-1 | Prefill KL lower bound / top-1 |
|---|---|---|
| A1 → A2 (lane rebuild control) | 0.006155 / 97.27% | 0.000000 / 100.00% |
| B1 → B2 (fallback rebuild control) | 0.000000 / 100.00% | 0.000000 / 100.00% |
| A1 → B1, also A1 → B2 | 0.017367 / 90.63% | 0.019502 / 91.73% |
| A2 → B1, also A2 → B2 | 0.017437 / 90.63% | 0.019502 / 91.73% |

The reverse decode bounds are 0.006162 for A2 → A1, 0.017709 for B1/B2 → A1,
and 0.017500 for B1/B2 → A2; reverse cross-arm prefill is 0.019521. The full
28-row precision-preserving table, bounds, tail masses, source identities and
build checks are in [the machine-readable receipt](data/ts113-r6-pairwise.json).

B1/B2 have identical **scored payload fingerprints** in both regimes, despite
distinct fresh compiled-build fingerprints. A1/A2 have identical scored
prefill fingerprints and different decode fingerprints. A zero top-K bound
alone would not prove full-vocabulary equality; the fingerprint statement is
only about the observed payload. Cross-arm prefill is not an identity. This
is a compiled-path/build confound where the M=1 GEMV arithmetic cannot explain
the difference, not an identified new defect or a causal decomposition of the
decode bound. KL in two different regimes cannot simply be subtracted.

| Compiled BF16 → student, decode | KL lower bound | Conditional upper bound | Mean teacher tail mass | Top-1 |
|---|---|---|---|---|
| A1 | 0.437342 | 5.824334 | 0.054547 | 62.50% |
| A2 | 0.442105 | 5.823660 | 0.054488 | 61.72% |
| B1 | 0.438400 | 5.796978 | 0.054252 | 61.33% |
| B2 | 0.438400 | 5.796978 | 0.054252 | 61.33% |

The upper bounds assume the tool's explicitly declared student probability
floor, `exp(-100) = 3.720075976020836e-44`; they are not unconditional upper
bounds on an unobserved vocabulary. The missing mean teacher mass is about
5.4%, and the interval is large. A1's lower bound is below the fallback's,
while A2's is above it. **Neither a lower-bound ranking nor the top-1 sample
establishes a full-KL winner.** This supplies the missing teacher/rebuild
measurements while retaining their uncertainty.

## Protocol and immutable inputs

The image is pinned by the runtime contract, not by a mutable EUGR/latest tag:

```
vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
```

Every build sidecar reports vLLM 0.28.0, `compiled_forward: true`, `eager: false`,
complete cache evidence and at least one fresh compile. The resolved dispatch
agrees across teacher and students: `custom_ops: ['none']`, `rms_norm: ['native']`,
`fused_add_rms_norm: ['native']`. This removes the earlier eager-teacher versus
compiled-student dispatch confound; it does not remove compiled-build variation.

Staged inputs are under
`/mnt/shared/tessera-runs/ts113-fresh-sparklina-aa6-r1/inputs`.
Arm A and arm B weights are hardlinks to one inode, `56:341271`, with SHA-256
`ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad`.
The BF16 weight SHA-256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The corpus file SHA-256 is
`cf96c4744a58e925f62673b6fc09c3bd584b5d7a49c00b901d7f0bce0ab57002`;
its contract SHA-256 is
`cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d`.

The lane is present in A and refused by a read-only extension directory in B.
The launch gate derives 196 source-role units from the manifest, independently
of its 112 module containers: 256 scored positions require exactly 50,176
`window_gemv` launches in A. B must have zero such launches and exactly 112
preparation refusals. The compiled route trace declines by design; launch
evidence here comes from the model worker's actual torch-profiler trace.

## Source and continuation provenance

The population root is
`/mnt/shared/tessera-runs/ts113-sparklina-population-aa6-r6`.

| Actual measurement stages | Original source commit | PrismaBuild action |
|---|---|---|
| teacher, A1 | `1159a845f840e2401f62b1907d5c03144f609a54` | `783ab956f3da7465d3254b362c89f442fbb33ed8bd4f484fa39e635391ca0c00` |
| A2, B1, B2 | `03406be7395b7b8c2ed58f51a4d34e89bf5ba759` | `2f97d7ad9e7e18d15ace07c78b1cf990c40312f68891a8a222a6c4dbcba307ee` |

The initial action completed the teacher and A1's serve, both dumps, trace
parsing and teacher comparison. It then failed while recursively hashing the
extension root: vLLM's container TMPDIR had left unrelated private directories
there. The fix scopes the inventory to the module directory named by
`ext.WINDOW_GEMV_MODULE_NAME`. A1's serve was **not repeated**. Its remaining
inventory and stage seal were completed at recovery, before A2/B1/B2 ran.

The original `CAMPAIGN_IDENTITY.json` is preserved byte-for-byte, SHA-256
`7886f890e0e3b5b0f34de96ae9d2aac93097580cef0f7057b4cf216b28136f81`.
`CONTINUATION_IDENTITY.json` records the source per stage and the continuation's
exact changed paths. The preflight rejected any serving, plugin, encoder or
wire-source difference; only the controllers, their regression, architecture
prose and the verified PrismaBuild closure stamp differ. Teacher/A1 are not
mislabelled as having been measured by the later controller.

A1's completion-power sample is explicitly **recovery-time**, not serve-end
telemetry. Its existing sealed bytes are never rewritten. A follow-up guard
in merged PR #124 refuses recovery if either seal marker exists but verification
fails; the actual A1 recovery began with both markers absent. Independent
verification checks the current sealed payloads and original profiler hashes;
it is not retroactive proof of dump immutability before the recovery seal.

## Memory and profiling scope

The r5 run was withdrawn and none of its partials reused. Its observer measured
only 7,044.453 MiB available while a 16,527.648 MiB parser and 42,886.19 MiB vLLM
engine RSS overlapped. This was application-level overlap on one UMA pool.
PrismaBuild #9 remains the separate resource-enforcement issue; this receipt
does not claim to fix it.

In r6 the wrapper stops the decode profiler, takes the matched prefill dump,
removes the serving container and confirms its absence, then parses. The
complete profiler-file roster is hashed. Every excluded trace is stream-scanned
and must contain no `window_gemv`; exactly one rank0 model-worker trace is
parsed. The 64-GiB `MemAvailable` gate is a headroom check, not a hard memory cap;
the parser also has a 900-second timeout and a measured peak-RSS receipt.

The raw 5-second observer logs are
`/mnt/shared/tessera-runs/ts113-r6-live-telemetry-783ab956.log` and
`/mnt/shared/tessera-runs/ts113-r6-live-telemetry-2f97d7ad.log`.
Their minima describe the sampled intervals, not an unsampled whole-run bound.
Power is measured directly; GB10's `gpu_utilization` and fake memory-utilization
percentage are not used as evidence of progress or efficiency.

The two observer intervals contained 111 and 261 samples. Their minima were
22,426,160 KiB (initial action, 19:10:41Z) and 22,195,448 KiB (continuation,
19:29:17Z), both above 21 GiB available. Their largest sampled power readings
were 19.10 W and 21.36 W. These sparse values do not establish a peak or energy
integral. The memory improvement versus r5 is an observed protocol result,
not a claimed PrismaBuild enforcement guarantee.

| Arm | Post-teardown rank0 parser peak RSS (KiB) | Wall time | Swaps |
|---|---|---|---|
| A1 | 4,553,168 | 23.95 s | 0 |
| A2 | 4,321,292 | 23.41 s | 0 |
| B1 | 3,712,944 | 17.44 s | 0 |
| B2 | 3,710,696 | 17.50 s | 0 |

[The memory receipt](data/ts113-r6-memory.json) records the raw-log and parser
receipt hashes. It was produced by CPU-only PB
`70fd803b562da5f7bc9706dff62f855e9c30e877d5244d875057d32148841b36`;
summary SHA-256 is
`39486a3f4a8ae41de10a4fe1ea73adea54431191434148af78bc1fa1d129dd44`.

## Completion and independent audit

The continuation succeeded on its first attempt in 1,416 seconds, publishing
`CAMPAIGN_COMPLETE` at `2026-09-04T19:45:05Z`, SHA-256
`6c02519be366278ee2ae3d397e2b59d9898b58ea9fda71089ee8d61973194ae2`.
Its PB receipt SHA-256 is
`871bd43059a7dacaf3a31bdbbd201a3682bd94c57ef2e13cdb70642502beb5a6`.
GPU ownership was then released for the MoE campaign.

An independent, read-only CPU audit verified all five stages and the campaign's
exact seven-member hash roster: the two source-identity records and five stage
seals. Teacher has 8 sealed files, A1 26, and A2/B1/B2 24 each. Each student has
three accounted profiler files, the expected populations, finite valid
logprobs with only declared padding, matching corpus/tokenizer/dispatch and
the exact launch/refusal counts above, including all 24 phase-bin sums.
All five build fingerprints are distinct and all report fresh compilation.
This validates recorded profiler hashes/accounting, not an independent reparse
of every raw profiler event.

[The independent audit](data/ts113-r6-independent-audit.json) was produced by
PB `5a923b9af1c1053ed73457878ba88db071f0145284ec7a732e88b467acf25c26`,
one CPU core / 4 GiB on dl380g10, in 8 seconds. Receipt SHA-256:
`589f59979ce640983028da8477b5730c79662b4c1f8d249358d7426835dbf783`.

The final comparison action was CPU-only PB
`00506b6786ca90a09eda9f1f04e67fc7ec2ecc093b546a237c82a2eb50968b9a`,
one core / 4 GiB on gx10-6b77, 40 seconds, source `1b4b034`. It waited for the
campaign and pool completion before reading sealed data. Output directory:
`/mnt/shared/tessera-runs/ts113-r6-analysis-v1`; `receipt.json` SHA-256:
`fbdd96a136990f8e23130c5cc2985fe1d938cc52da49fdaf5a5b71cf18684ef4`.
Its PB receipt SHA-256 is
`f4a9f4b2b7da35efdf8cb9e3106c67f182c04e8550a68b90371ca5f9915f8c4c`.

## Reproduction and code verification

The GPU campaign and continuation both ran through deployed PrismaBuild v1,
runtime `aa6d3cfa2f77`, with all three logical GPU tokens on the one physical
Sparklina GPU reserved, two CPU cores and 64 GiB declared memory. No direct
serve, test or GPU probe was launched outside PrismaBuild.

`experiments/ts113_r6_compare.py` reads only sealed stages, checks complete fresh
compiled builds and matching dispatch, then computes both directions of every
student pair in both regimes (24 comparisons), plus four teacher-to-student
decode comparisons. It uses the existing metric tool and fingerprints, hashes
both `kl_tool.py` and `kl_estimator.py` before and after, and writes a separate
fresh sibling directory. It does not append anything to sealed stage directories.

The merged implementation is PR #124, commit
`8c1c2068d2287f46af6e704f5bdebbbb9e01388a`. Its final targeted PB action
`d859c1ca37671a81537ba56f06e455470877347b04b9beebfcc7b6826badfc62`
passed 69 tests, skipped 9 and left 0 modules uncollected, serially on dl380g10
with torch 2.11.0+cpu and no CUDA device. Receipt SHA-256:
`74aee87fcd7a9889b593925febf224d09f9e193abfee4e99c500e93c1d24466f`.
The selector narrowed to the lock, wrapper-cleanup, build-identity and pinned
runtime-image test files using the exact fetched base and direct-tree fallback.

The receipt branch's targeted build-identity run, PB
`7f6a5b484ce09bd03ff7e5afb266040292d382adfde1489e530fb1b4cd53537f`,
passed 43 tests, skipped 9 and left 0 modules uncollected on the same CPU-only
population; it also compiled the comparison collector's Python syntax. The
selector against exact merged `8c1c206` narrowed to that single test file.
Neither test run claims CUDA-surface coverage. Both runs' skip reasons verbatim:

- 6: `/home/rob/tessera-runs/compile-dispatch is not on this box`
- 1: `the two surviving compile caches from 2026-09-02 are not on this box`
- 1: `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box`
- 1: `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box`

The pre-fix failures are preserved in the implementation's PB receipts:
`8de81a096b5e` failed the parse-before-reap ordering and missing profiler-roster
tests; `bf961207564b` failed the all-exit reap-before-release trap test;
`47212b91ab27` reproduced the recursive extension inventory's permission error;
`11792c8298ed` showed both invalid-existing-seal cases incorrectly recovering.
Each was followed by its targeted green before the relevant fix was accepted.
The receipt branch additionally fixes one stale module-docstring sentence:
the teacher wrapper now pins and reads its cache root, rather than emitting
an incomplete log-only build stamp.
