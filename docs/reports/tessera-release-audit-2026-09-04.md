# Tessera release audit, 2026-09-04

The whole-codebase audit Rob asked for on #17 as a precondition of `v0.1.0`
("audit the entire Tessera codebase and report findings, evidence limits,
and a candid ship/no-ship assessment before publication"). Candidate: the
integration branch for #127 at `54cd1df` (code tip `b83fd17`; every commit
after it is documentation, the issue snapshot or this report). Contract v16,
lane-eligibility schema v5, raw contract SHA256
`75137c73bce8837713b427d977beb0eec280faccb39fd3225acf1b3bd00eb0b1`
(unchanged on the branch; receipts bind to it).

## 1. Method

Five read-only partitions, each an Opus worker at xhigh effort with a brief
naming the primary artifacts it had to read and a report format of
claim / receipt (`file:line`) / measured-vs-asserted / ship-blocking? /
suggested fix. No worker ran GPU work or `pbrun`; everything empirical in the
reports is a CPU or torch-free check, and each report ends with what it did
not read. The partitions:

| Partition | Scope | Docs / files read in full |
|---|---|---|
| serving | `src/tessera/serving/*`, `runtime_contract.json`, census tools | contract incl. all 16 changelog entries, scheme/config/contract/nvfp4_route/moe_route/sharding |
| claims | README, AGENTS, pyproject, ci.yml, ARCHITECTURE, every 2026-09-04 receipt, the serving/MoE contract doc, exl3-comparison | 29 documents plus spot-reads of 10 source files |
| export / MoE | `export.py`, `moe_layout.py`, `experiments/export_tessera_serving.py`, the ts5 drivers, the LFM campaign | 2 of the 7 ts5 drivers in full |
| core | wire, manifest, container, canonical, layout/slicing, encoder identity, kernel-lane rosters | `alphabet.py`, `encode.py`, `kernel*.py` in part only |
| tools / CI | `tools/*`, `tests/conftest.py`, `merge_suite`, `impacted_tests`, packaging, both hosted jobs | built a torch-less venv and ran `--collect-only`; did not build the wheel |

The coordinator (this session) then applied AGENTS.md's fix-or-file rule:
every finding that was a bounded change was fixed on the branch in its own
commit with the test shown failing first; everything else was filed as an
issue naming why it was not fixed here. No finding was dismissed without a
line to point at.

## 2. Headline

**No P0 in any partition.** Five reports, 68 findings by their headings
(serving 12, claims 18, export 10, core 12, tools 16): 27 fixed on the branch
(13 in code or CI, each its own commit; 14 release-facing prose findings in
`54cd1df`), 41 filed as 25 issues, #130 through #154 (section 5). Counted from the `### [P…]`
headings of the five reports, not estimated.
All five partitions returned the same verdict: ship `v0.1.0` **with the
served scope stated honestly**, which the README did not do at `6072e57`
and does at `54cd1df`.

## 3. Fixed on the branch

Each is one commit, red-first where a test could be written.

| Commit | Partition | Finding |
|---|---|---|
| `4d32cc2` | serving P1 | `trellis_input_global_scale` was `torch.empty`; uninitialised memory could pass the `> 0` gate. Now NaN-filled so the gate's own predicate refuses an unloaded scale. |
| `4e5ac52` | serving P2 | `_validate_group` checked the body against the global vocabulary, not the route's body. |
| `8f24738` | serving P3 | `_ROUTE_STATUSES` published a status no cell used and no gate defined. |
| `556aa59` | serving P2 | Prose in `config.py`/`bf16_route.py` denied a routed_moe cell that v16 publishes. |
| `e32cbc0` | export P1 | `PACKED_EXPERT_ND` matched only `.mlp` owners while every other MoE regex accepted `mlp\|feed_forward`; a packed stack under `feed_forward` was invisible to the exporter. |
| `6884cf8` | export P2 | Every safety check in the artifact-mutating `ts5_lfm_correct_passthrough.py` was a bare `assert` (stripped under `-O`). |
| `c02e77f` | export P3 | `unpack_moe_wires` rebuilt each blob byte-by-byte through `.tolist()` (42.7 ms vs 0.11 ms per 4.2 MB row). |
| `d04cce6` | export P3 | `W13_PROJECTIONS = 2` was a second literal beside the runtime's shard table. |
| `e2c696d` | core P1 | `nvfp4_scale_bytes` used a numpy keyword on a torch tensor; every fifteen-binade unit raised `TypeError` instead of a plane or a `GrammarError`. |
| `d4df86c` | core P1 | Three manifest presence flags accepted any non-zero byte, so one manifest had many byte strings. Now read as canonical bools. |
| `9725014` | core P2 | `DEFAULT_SCALE_REFIT`'s comment said the bytes do not move; they do. |
| `b83fd17` | tools P1 | Nothing verified what the built distribution contained. `tests/test_packaging.py` + `tools/check_wheel.py`, run by both hosted jobs; shown refusing a wheel with the contract removed. |
| `94e8289` | tools P2 | `ci.yml` said "No ignore list" above two redundant `--ignore` flags. |
| `54cd1df` | claims P1-P3 (14 of 18) | README overclaims (TP, streamed MoE, "one wire", continuous-rate, allocation), the retired 1.72x in the EXL3 banner, ARCHITECTURE's missing stamp and wrong 14.7%, pyproject metadata; see the commit message. |

Integration commits on the same branch: `ab75bee` (ts-5 MoE final),
`faff67b` (#129 selector), `2e12b7f` (cached-producer API), `1e93144` and
`6072e57` (cross-repo issue references and the snapshot they resolve
against).

## 4. Evidence

**Two-population suite.** `tools/merge_suite.py`, both arms through
PrismaBuild, one receipt per run; rows in `docs/status/suite-populations.md`.

| Receipt | Commit | GPU arm (serial, `--strict-cuda`, GB10, torch 2.11.0+cu130) | x86 arm (`-n 24`, dl380g10, torch 2.11.0+cpu) | Verdict |
|---|---|---|---|---|
| `20260904T180849` | `6072e57` | 2535 passed / 0 failed / 12 skipped | 2016 passed / 0 failed / 518 skipped | green |
| see section 4a | `b83fd17` | | | |

Both arms report the same effective source identity per run, and a branch
run records `master head? no` by the ledger's own rule; the merge receipt on
master's tip is section 4a.

**Torch-free suite** (the `pure` job's population, run locally at
`54cd1df`): 895 passed, 50 skipped, no module failed to collect on its own
names. Hosted CI on the PR head: run 33924008666 green at `6072e57`; the run
at `54cd1df` includes the new wheel check (section 4a).

**Wheel.** `tools/check_wheel.py` on a locally built
`tessera_quant-0.1.0-py3-none-any.whl`: the four run-time data files
present, the `vllm.general_plugins` entry point resolves to a callable with
torch never imported, the packaged contract loads and validates. The same
script exits 1 naming the file on a wheel with `runtime_contract.json`
removed.

**Served scope** (what the contract attests; unchanged by this audit):

- Dense, Qwen3-0.6B, sm_121, `vllm/vllm-openai@sha256:61fc8a89…` (vLLM
  0.28.0): three families at one rung each (E2M1x2 q896, E4M3 q1024, BF16
  q1792), resident and streamed, eager and compiled, 112/112 census in every
  mode, plugin KL-vs-BF16 0.6316 / 0.4660 / 0.004923, each bit-identical
  between residencies (the 2026-09-02 encoder). With the LDLQ block lever
  (2026-09-03, `tessera-dense4-gap-2026-09-03.md`) the same 4.0-bpp wire
  served 0.5100 against PrismaQuant NVFP4 GPTQ+JSO's 0.5106 at equal
  residency (0.9988x, parity).
- Routed MoE, LFM2.5-8B-A1B only, E4M3 q1024, resident and eager only,
  `eugr/spark-vllm@sha256:0afec8d4…` (vLLM 0.28.1rc1): 22/22 stacks and
  2,112 projections dispatched on the modular Triton FP8 route in decode
  (M=1) and prefill (M=64). Quality is a prefill top-1,024 intersection KL
  lower bound of 0.0832 (upper 1.179 at the probability floor) over 4,096
  positions with 85.1% top-1 agreement. No full-vocabulary KL, no decode KL,
  and a repetitive greedy smoke recorded and unexplained. No numeric quality
  cutoff exists in the cell-promotion contract, so this is a dispatch
  attestation, not a quality pass.

### 4a. Filled at merge

The `b83fd17` receipt, the `54cd1df` hosted run and the merge receipt on
master's tip are appended below when they complete; until a row is here it
has not been observed.

**Note, 2026-09-04, after this audit.** #131 and #133 landed as contract v17 /
lane-eligibility schema v6 (per-cell `runtime.vllm`/`runtime.torch`,
`versions.default_serve_image` in place of `versions.attested_on`, and a
required per-cell `evidence` grade). The raw SHA256 quoted at the top of this
report is the audited v16 file; the v17 file's raw SHA256 is
`ba3a3c69027a11f7bf9ef570867c58d608d0865c3e6a17dfeea53ba43ce055e6`.

**Note, 2026-09-04, after that note (#181).** A prose-only correction
replaced the three build-box absolute paths in the contract's changelog with
the tracked receipt that carries the same evidence
(`docs/measurements/tessera-gemv-lane-reachable-2026-09-03.md`). No field,
schema, cell, rung, route or bound moved, so `contract_version` stays **17**
-- one denotation, now under two digests, as `af9d23b` already did at v16.
The v17 raw SHA256 recorded above is the file this audit was run against;
the corrected file's raw SHA256 is
`2f2dee1a23b14101f0312cfbf4fcaf1542eb575841621000cfa05164e2275903`.

## 5. Filed, not fixed

Filed as issues so the tag can be cut against a list rather than a memory.
None of these changes bytes on the wire, a served number, or a claim the
README makes at `54cd1df`; each says in its body why it was not fixed on
this branch.

| Issue | Partition | Finding | Decision needed? |
|---|---|---|---|
| #130 | serving P1 | The NVFP4 route is the only route that does not cross-check its decoder against the reference. | no |
| #131 | serving P1 | `versions.attested_on` does double duty and is wrong for the two routed_moe cells; the `versions` block is required but never validated. | no |
| #132 | serving P1 | The census's `--runtime-image` is operator-asserted, verified only out of band. | no |
| #133 | serving P1 | The two routed_moe cells are machine-indistinguishable from the dense cells while carrying screen-grade evidence (a top-1,024 bound, a repetitive smoke). | **yes**, and measurement needed |
| #134 | serving P2/P3 | `predicates` is a required cell field nothing validates; `window_gemv.cu` ships twice and the published `source` is not the copy that builds. | no |
| #135 | export P1 | A routed-MoE stack's rung is gated against the dense route's published range. | no |
| #136 | export P2 | The packed 3-D expert write path is checked at name level only; `packed_expert_orientation` is unused by the plan path. | no |
| #137 | export P2 | The legacy merge guard's `SHARED` names three fields no exporter writes. | no |
| #138 | export P2 | The library streaming index prices `total_size` on wire regions, not bytes on disk. | no |
| #139 | export P3 | The exporter's `ignore` roster hard-codes an embedding name that is wrong for nested layouts. | no |
| #140 | core P1 | `slice_unit` writes a shard record mixing two coordinate frames; reproduced by the filer (a legal re-slice refused, an illegal record serialised). | no |
| #141 | core P2 | Enum decoding escapes the Tessera error taxonomy. | no |
| #142 | core P2 | Release placement ranks by a hardcoded E2M1 value table whatever the grid. | no |
| #143 | core P2 | The encoder-identity fixture set and the byte-proof matrix miss the same surfaces (shard, S6B, diagonal). | fixture half is Rob's call |
| #144 | core P2 | The truncation-terminal ladder is unreachable on any encoded artifact. | no |
| #145 | core P2 | The kernel lane's supported set is a roster restated three times, tied to the `.cu` by nothing. | no |
| #146 | core P2 | The bit-exactness gates are gated on CUDA and on one box's absolute paths (17 literals across 8 files, not the report's 10). | no |
| #147 | core P3 | Assorted P3s in wire, decode, container and fused (7 bullets; the `Reader.bool` one is moot after `d4df86c`). | no |
| #148 | tools P1 | `impacted_tests` can never return `narrowed` (the wildcard walk implicates 237 modules) and drops package `__init__` edges (25 and 39 consumers missed). | no |
| #149 | tools P1 | The version string and the plugin entry point are hand-copied into four places. | no |
| #150 | tools P1 | The publish trigger is a bare tag match: no ancestry check, no `environment`, a mutable action ref. | **yes** (with #17's PyPI question) |
| #151 | tools P2/P3 | The sdist ships tests without `conftest.py`; test infrastructure ships in the runtime wheel; `ninja`/`nvcc` are declared in no extra. | no |
| #152 | tools P2 | `--strict-cuda` asserts a device, not that the CUDA surface ran; AGENTS.md overstates it. | no |
| #153 | tools P2 | `test_kl_tool_decode_regime` pins a file outside every checkout. | no |
| #154 | tools P3 | conftest exec probe, third-party import failure on own names, an unbounded final `process.wait()` (the grace-period no-poll is deliberate and kept). | no |

Three claims findings are inside those issues rather than their own: `versions.attested_on` (#131), the contract field with missing spaces and the compiled-cell mode the census cannot check per cell (#134, #133). The claims P3 "ARCHITECTURE states what another repository's parser does" is corrected in the release-docs commit that records this report. Filed by an Opus worker from the five reports with every citation re-read at `b83fd17`; four report numbers were corrected in the filing (line moves, the absolute-path count, the impacted-tests counts) and are attributed as corrections in the issue bodies. Two findings the reports listed were already fixed in passing (`556aa59`, `d4df86c`) and were dropped with that stated.

## 6. Evidence limits

What no partition could verify, gathered from the five reports:

- **Nothing on a GPU.** Every empirical check in the reports ran on CPU
  torch or no torch. CUDA-graph capture paths, the fused Triton Viterbi,
  the compiled decode chain and every kernel are covered by the
  two-population suite's GPU arm, not by the audit.
- **No vLLM behaviour was observed.** Whether vLLM's loader refuses an
  unloaded parameter, whether `_load_w13` expects gate at `[0:N]`, whether
  the compile cache folds `additional_config["tessera"]` -- all claims about
  another runtime, treated as unattested. The served receipts are the
  evidence for them.
- **No KL, bpp or latency figure was re-derived.** The claims partition
  checked every number in README/ARCHITECTURE against the receipt that owns
  it; it did not re-measure one.
- **The LFM artifact's provenance.** Its wires were cut at source commit
  `5c56b7e5…` and its config corrected post hoc; a fresh export from the
  candidate will not reproduce its `code_sha256`/`export_identity`. The seal
  attests that artifact, not this tree's output. `encoder_fixture_id` is
  CPU-only by construction and the halves were encoded on two boxes, so the
  merge guard cannot see a GPU-only encoder divergence; the fused-Viterbi
  bit-exactness pin and the cross-box render-identity receipt are what cover
  it.
- **The packed 3-D expert path and `Glm5NextForConditionalGeneration`** have
  unit tests and no served artifact. The routed-MoE rung is gated at export
  against the dense reader range, not the routed_moe cells (filed); an
  off-1024 MoE rung exporting without refusal is a code trace, not a built
  checkpoint.
- **Large modules read in part:** `alphabet.py`'s fitted tables and
  `build_forest`, `encode.py`'s Viterbi bodies and LDLQ loop, `kernel.py`,
  `kernel_window.py`, `control.py`, 5 of the 7 ts5 drivers, the two 40-50 KB
  GEMV receipts (existence and framing confirmed, contents not audited).
- **The publish path end to end** was not exercised; that `publish` is
  skipped when `pure` fails is documented GitHub behaviour, not measured.

## 7. Ship / no-ship

**Ship `v0.1.0`, with the scope the README now states, once the decisions
below are taken.** The grounds:

- No P0. Every P1 that could change bytes, a gate outcome or a served claim
  is fixed on the branch; the remaining P1s are guards and reconciliations
  (a load-time cross-check, an export gate against the routed_moe cells, the
  selector's always-full defect, four copies of the version string, the
  publish trigger's shape) that harden the next release rather than change
  this one.
- The suite is green on both populations at `6072e57` and the code changes
  after it are seven small commits, each with its own red-first test where
  one applied; section 4a carries their receipt.
- The honesty gap the audit found was in prose, not bytes: the contract,
  the receipts and the ARCHITECTURE document already scoped the MoE
  evidence correctly; the README did not. It does now.

What a reader of this release should not expect: a multi-rank serve, a
GLM artifact, a compiled or streamed MoE serve, an MoE quality number
stronger than a prefill lower bound, or a route on any platform but sm_121.

## 8. Decisions that are Rob's

1. **Tag `v0.1.0`.** Pushing the tag runs the `publish` job and uploads to
   PyPI; that is the outward-facing step and it is not taken here.
2. **The PyPI Trusted Publisher "environment" question (#17)** and the
   filed publish-trigger hardening (bare tag match, no ancestry check,
   branch-pinned action). Either publish as-is -- a publisher configured with
   an environment would reject the run, which is recoverable by editing the
   workflow and re-tagging -- or harden first.
3. **Make the repository public.** PrismaQuant's CI checks out the pinned
   Tessera commit and fails with "Repository not found" while it is private
   (RobTand/prismaquant#187 through RobTand/prismaquant#191, RobTand/prismaquant#175).
4. **Price the two routed_moe cells' evidence.** They are machine-
   indistinguishable from the eight dense cells while their quality evidence
   is a prefill lower bound; a `quality`/evidence field or a promotion cutoff
   is a decision about what a cell means, filed as needs-decision.
5. **Whether the issues filed by this audit gate the tag under option B**
   ("no v0.1.0 until every Tessera issue is closed"). Read literally they
   do. The recommendation is to scope option B to the issues open when it
   was decided (#5, #126, #129, #17) plus the two needs-decision items; the
   rest are post-release hardening and say so in their bodies.
6. **PrismaQuant's pin** (`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`
   and the two constants in `tessera_serving_runtime_pin.py`) flips to the
   tagged commit and version after the tag exists; the pending-pin test flips
   with it.
