# The 16-bit route on RDNA4: `TESSERA_BF16_K1` served on gfx1201 (2026-09-13)

**Result.** `TESSERA_BF16_K1` — W16A16, the window body over the CHANNEL plane
with its 2^L table snapped to bf16 — is **served** on an AMD `gfx1201` device by
`tessera.serving` on a ROCm vLLM build, in both residency modes and under both
the eager and the compiled forward. Four route censuses record **all 112
declared modules** on the route, in the prefill and the decode shape, with zero
problems. At the R = 7 rung (`q256 = 1792`, wire 7.129 bpp) the served
KL-vs-BF16 is **≥ 0.004906** in the prefill-scored (batch) regime over 4088
positions and **≥ 0.004804** in the decode regime over 256 M = 1 positions,
against the unquantized model served on the same image. Both numbers are
**bit-identical between `resident` and `streamed`**, which is what the twin
check predicts: the two residencies decode the same tiles.

This is the receipt contract v24 cites when it publishes the first two AMD
cells and fills `platforms.gfx1201.serve_image`. Nothing wider is attested —
one rung, one platform, dense only, no routed-MoE cell, `max_world_size 1`.

**Scope, and it is narrow.** gfx1201 here is a Radeon RX 9070 XT running under
WSL2. Every measurement below proves the **HIP code path**: that the extension
builds through the loader's own path on a ROCm torch, that the decode and the
GEMV agree with their references, that the route serves and what it executes,
and that the served bound is what it is. It proves **nothing about gfx1151
(Strix Halo) numerics** — that part is not in this fleet, and
`docs/strix-halo-tester-protocol.md` is still what a receipt from it must
satisfy. It proves **nothing about performance on any part**: WSL2 exposes no
performance counters, and a desktop RDNA4 card says nothing about an APU. No
performance claim is made from this receipt. `qualification: device_qualified`
on the v24 cells is qualified under exactly this scope.

**Admission.** The GPU runs below are direct runs on the box, authorized by Rob
on 2026-09-12 ("gfx1201 runs locally on my machine, so, you'll test it here").
Everything that starts vLLM is PrismaBuild-exempt by standing ruling. The
export that produced the artifact ran through PrismaBuild on a GB10.

---

## 1. What was served

| | |
|---|---|
| wire | `/mnt/shared/tessera-runs/allocated/qwen3-0.6b-bf16-k1-r7-t460` — Qwen3-0.6B, grid `BF16`, `q256 = 1792` (R = 7), body `WINDOW`, plane `CHANNEL` |
| bpp | body/wire **7.1292**, on disk **7.1318**, resident-mode **16.0** (112 modules, 196 units, 440 401 920 quantized params, checkpoint 1 015 093 626 bytes) |
| encoder | manifest `git: af4b91a`, written `2026-09-13T00:23:47` (local), exported through PrismaBuild on a GB10 |
| twin | `…-t460-twin`, the exporter's `materialize_bf16_folded` rendering, verified unit for unit **on gfx1201**: 196 units checked, **0 mismatched**, streamed decoder re-checked on every 8th unit, no `quantization_config` on the twin |
| teacher | the unquantized `Qwen/Qwen3-0.6B` snapshot `c1899de2`, served on the **same image**, through the same plugin install |
| corpus | `corpus_qwen_n8_s512.json` — Qwen-tokenized, n = 8 × 512, `corpus_sha256 076d33ef…`, tokenizer `76f13c8e…` |
| metric | KL-vs-BF16, top-1024 support, teacher–student intersection, **lower bound** — `kl_tool.py`, the same instrument every sm_121 row was taken with |
| image | `192.168.1.107/prismaquant/vllm-rocm@sha256:0461258dfe253a3e0baca9c62804a4b41a21ab445b2624b6fca0d51711e14000` (private registry on dl380g10) |
| runtime | vLLM **0.30.0.dev0**, torch **2.11.0+rocm7.2.4.git5fbd98f3**, python 3.12.14, all read from inside the container |
| toolchain | hipcc **HIP version: 7.14.60850-0000000**, host ROCm 7.14.1, HIP runtime 7.2.53211 |
| device | `AMD Radeon RX 9070 XT`, `gcnArchName` → platform token `gfx1201`, wavefront 32, 32 CUs |
| instruments | `experiments/tessera_plugin_run_rocm.sh` (container, WSL2 device flags, non-editable plugin install) · `experiments/tessera_plugin_served_rocm.sh` (serve + greedy smoke + `kl_tool` dump + compare) · `tools/tessera_route_census.py` · `experiments/bf16_twin_check.py` · `experiments/moe_greedy_smoke.py` · `kl_tool.py`. The per-arm drivers that called them are workspace scripts on the box, kept beside the receipts; nothing that decides a number is in them |

**The toolchain identity is in this header and not in the cells.** A hipcc
version is a property of the image, and the image is pinned by digest; a
`runtime.hipcc` field beside `runtime.torch` would be a second spelling of the
same pin for a gate to fall out of step with. The cells carry the digest.

**The plugin is installed, not on a path.** vLLM discovers a plugin through
`importlib.metadata.entry_points(group="vllm.general_plugins")`, so the
launcher copies the mounted read-only tree into the container and runs
`pip install --no-deps --no-build-isolation` on the copy. The install is
**not editable**: what the serve resolves is a copy of the bytes the launcher
was pointed at, and no later edit can move under a running serve. Every serve
log carries the banner line
`[tsrun-rocm] vllm 0.30.0.dev0; torch 2.11.0+rocm7.2.4.git5fbd98f3; plugin ['lora_filesystem_resolver', 'lora_hf_hub_resolver', 'tessera']`.

**The image reference is port-less, and that is measured rather than chosen.**
`runtime_image._DIGEST_REFERENCE` accepts `repository@sha256:<64 hex>` and a
repository component may not carry a colon, so `192.168.1.107:5000/…` is
unpinnable by this grammar. Docker's own `RepoDigests` for the pulled image
carries four spellings and the port-less one is among them, so the reference in
the contract is the daemon's answer and not a re-spelling of it:

```
$ docker image inspect --format '{{.RepoDigests}}' <the digest above>
[192.168.1.107/prismaquant/vllm-rocm@sha256:0461258d…
 192.168.1.107:5000/prismaquant/vllm-rocm@sha256:0461258d…
 prismaquant/vllm-rocm@sha256:0461258d…
 localhost/prismaquant/vllm-rocm@sha256:0461258d…]
```

## 2. Receipts 1–3: the kernels, against their references

One suite, run on the device through the loader's own path:

```
python -m pytest -q tests/test_kernel_window_gemv.py tests/test_kernel_window_gemv_plan.py \
                    tests/test_kernel_roster.py tests/test_serving_dispatch.py \
                    tests/test_serving_backend.py tests/test_serving_platform_gate.py
281 passed, 8 skipped in 39.77s
```

| leg | what carries it |
|---|---|
| extension build through `torch.utils.cpp_extension` on the pinned ROCm torch, via the loader's own path | every device test in the suite; `test_the_extension_is_resolved_at_preparation_not_on_the_first_call` pins that it is the preparation path and not a first-call fallback. `backend.ensure_toolchain_on_path` takes its ROCm branch (`ROCM_HOME`), and the CUDA branch's own test skips with that reason stated |
| `window_decode` HIP vs `torch_window`, bit-exact | `test_synthetic_decode_is_the_definition`, parametrized over rates `1`, `2`, `4`, `mixed24`, `mixed124` × rows `512`, `1024`, `96`, `1000`; `test_every_reach_unit_decodes_byte_identically` on real units |
| `gemv` HIP vs decoded-weights fp32 matmul | `test_gemv_m_tiles_within_bound` for M ∈ {1, 2, 3, 4, 5, 8}; `test_gemv_plans_agree` across the MT/warp/column-per-item plans and both table dtypes; `test_gemv_every_weight_exact_synthetic` / `…_on_a_reach_unit` at M ∈ {2, 8} |

**What these do NOT cover, said plainly.** The attested rung is `q256 = 1792`,
i.e. rate 7, and the window-GEMV lane reads rates `(1, 2, 4)` only. So the
kernel legs above exercise the lane at the rates it serves, and the rung the
v24 cells attest does **not** reach that lane at all — §3 shows the refusal in
the serve's own words, and the cells' `executes` is the torch window because of
it. 8 skips, every reason recorded on the receipt: 5 box artifacts this box does
not hold, 2 needing two CUDA devices, 1 the CUDA branch of the toolchain repair.

Receipt: `wsl-gpu:/home/rob/agents/t460-cells/receipts/gfx1201-kernel-suite-20260913T040301Z.txt`.

## 3. Receipt 4: the census, in four combinations

`tools/tessera_route_census.py` compares what the serve **recorded** through
`emit_route` against what the checkpoint **declares**, module by module, in a
prefill shape and a decode shape.

| mode | forward | verdict | decode modules | prefill modules | other-route | problems | s |
|---|---|---|---|---|---|---|---|
| resident | eager | **served** | 112 | 112 | 0 | 0 | 27.7 |
| resident | compiled | **served** | 112 | 112 | 0 | 0 | 55.7 |
| streamed | eager | **served** | 112 | 112 | 0 | 0 | 27.9 |
| streamed | compiled | **served** | 112 | 112 | 0 | 0 | 76.5 |

Every arm records one route and one only: `contract bf16_unquantized`,
`symbol torch.mm`, `decoder torch_window`, `platform_token gfx1201`, and
`device.capability [12, 0]` — which is the reading §4.5 of the architecture doc
warns about: on HIP the capability tuple is NVIDIA's sm_120 spelling and the
platform token comes from `gcnArchName`, never from the tuple.

**The streamed arms carry the rate refusal, on every module.** The window-GEMV
lane is asked and declines, in its own words:

```
tessera_window_gemv: down_proj (column_rates [7] are outside the rates this
lane reads ([1, 2, 4]); the lane repacks each column at its own rate, so one
column out of range refuses the whole unit -- this lane does not read the unit
and the route's materialised path serves it)
```

That refusal is why both v24 cells name one launch, `{torch.mm, torch_window}`,
in both regimes and both residencies. It is derived, not transcribed:
`scheme.launch_pairs` over `ROUTE_LAUNCHES` returns exactly that pair for
`TESSERA_BF16`, dense, at rung 1792, for each of `decode`/`batch` ×
`resident`/`streamed`.

**Cell agreement.** The four arms above ran the **v23** document, which carries
no gfx1201 cell, so each reported `covered_by_cell 0` / `agrees: null` — the
correct answer for a document with nothing to agree with, and not a verdict on
the launches. The arms in §3.1 re-take the eager pair with the v24 contract
installed in the container.

Receipts: `wsl-gpu:/home/rob/agents/t460-cells/receipts/census-t460-{eager,compiled}-{resident,streamed}-*.{log,json}`.

### 3.1 The same census against the v24 cells

The same two eager arms, re-taken with the v24 document installed in the
container, so `cell_launch_agreement` has cells to compare against:

| mode | forward | verdict | agrees | decode covered / unattested | prefill covered / unattested | s |
|---|---|---|---|---|---|---|
| resident | eager | **served** | **true** | 112 / 0 | 112 / 0 | 27.3 |
| streamed | eager | **served** | **true** | 112 / 0 | 112 / 0 | 28.9 |

Every decode record resolves to `tessera_bf16_k1_dense_gfx1201_decode` and every
prefill record to `tessera_bf16_k1_dense_gfx1201_batch` — 112 modules each, none
unattested, in both residencies. The agreement is `tessera.cell-launch-agreement/3`
and it compares the launches the serve **recorded** against the launches the
cells **name**, so this is the serve-side leg of the cells and not a restatement
of the derivation in §3.

**The compiled arms cannot produce this verdict, and that is a property of the
instrument.** Under a compiled forward the recorded shapes are dynamic (`M*`
instead of `M1`/`M64`), so a record cannot be attributed to a token-count
regime: the v23 compiled arms reported `unsupported_records: 112` for exactly
that reason. The compiled forward is therefore attested by its census
**verdict** and by the smoke, not by cell agreement — and the cells' `kl`
entries name `execution_modes: ["eager"]` accordingly, while
`runtime.execution_modes` names both because both were served.

Receipts: `wsl-gpu:/home/rob/agents/t460-cells/receipts/census-t460-v24-eager-{resident,streamed}-*.{log,json}`.

### 3.2 An E4M3 artifact is refused

Loading a `TESSERA_E4M3_K1` checkpoint on this platform is refused before a
kernel, with the contract's own word — `gfx1201` publishes `null` for that
family's `executes`, which is `unbacked`, and `unbacked` is the one state the
plugin refuses on. Receipt: `wsl-gpu:/home/rob/agents/t460-cells/receipts/e4m3-refusal-20260913T041224Z.log`.

## 4. Receipt 5: the serve, the smoke and the KL

Each arm is one serve on the pinned image with the plugin installed inside the
container, a greedy smoke, then a logprob dump on the corpus contract. The
serves that carry a KL are `--enforce-eager`; the compiled forward is covered by
the censuses of §3 and by the smoke below, not by a KL.

### 4.1 The bounds

| regime | residency | positions | top-1 agree | KL ≥ (all) | KL ≥ (confident) |
|---|---|---|---|---|---|
| batch (prefill-scored) | resident | 4088 | 95.475% | **0.004906068394762042** | 0.0032516091811518677 |
| batch (prefill-scored) | streamed | 4088 | 95.475% | **0.004906068394762042** | 0.0032516091811518677 |
| decode (M = 1) | resident | 256 | 95.313% | **0.004803555532266324** | 0.0023827105355719784 |
| decode (M = 1) | streamed | 256 | 95.313% | **0.004803555532266324** | 0.0023827105355719784 |

The two residencies agree to every recorded digit in both regimes — the same
`kl_lower_max`, `kl_lower_p99` and confident-set size, not only the mean. That
is the expected answer and is itself the finding: `resident` materialises the
decoded tile once and `streamed` decodes per window at call time, and the twin
check of §1 asserts those are the same bytes. A difference here would have been
a defect.

**The two regimes are two metrics and do not compare.** A prefill dump scores
512-row forwards; the decode dump scores each position off an M = 1 forward,
which is the only regime a decode cell's launch is made in. The decode arms ran
with `--enable-prompt-tokens-details` so `kl_tool` could check the M = 1 claim
against the serve's own `usage.prompt_tokens_details.cached_tokens`; without it
the tool refuses rather than mislabelling the regime. Stride 16, the serve's KV
block size.

**Order within an arm.** The greedy smoke precedes the dump on the same serve.
That order matters for a launch histogram and not for a KL, and the histogram is
§3's, taken on its own serves.

### 4.2 Why not `kl_full_vocab`

The brief for #460 asked for `evidence.grade: kl_full_vocab` from the canonical
n = 8 × 512 contract. **That grade is not available from any instrument in this
repository**, and the fact is measured rather than conceded:
`docs/measurements/moe-evidence-debt-2026-09-04.md` §4 records that every served
KL here — dense and MoE alike — is a `kl_tool` top-K teacher/student-intersection
**lower bound**, and the tool's own header says so on every line it prints. The
canonical corpus contract is satisfied (n = 8 × 512 WikiText, the same
`corpus_sha256` every sm_121 row was taken on); what is not satisfied is the
support, and `top_k: 1024` with `kind: topk_intersection_lower_bound` is how the
contract says that. `derive_evidence_grade` then reads `kl_lower_bound` off the
entries, so the cells cannot assert a grade their receipts do not carry.

A `kl_full_vocab` cell would need an in-runtime full-vocabulary logits dump,
which is a new instrument and not an edit. Recorded as an acceptance criterion
this work could not meet.

### 4.3 The greedy smoke, as a record

`experiments/moe_greedy_smoke.py` scores every (prompt, form, interface) pair
under its own degeneration rule — *repetitive iff the completion ends in a
cycle* — on this route's arm and on a BF16 reference arm served on the same
image, and the pair it emits is the `evidence.smoke.record` both v24 cells
transcribe. Fourteen rows, seven prompts × two request forms, four raw
continuations and three through the checkpoint's own `chat_template.jinja`:

| prompt | form | interface | route arm | bf16_source |
|---|---|---|---|---|
| P0 | campaign | raw_completion | `recorded` | `recorded` |
| P1 | campaign | raw_completion | `recorded` | `repetitive` |
| P2 | campaign | raw_completion | `recorded` | `recorded` |
| P3 | campaign | raw_completion | `recorded` | `recorded` |
| P4 | campaign | chat_template | `recorded` | `recorded` |
| P5 | campaign | chat_template | `recorded` | `recorded` |
| P6 | campaign | chat_template | `recorded` | `recorded` |
| P0 | pure_greedy | raw_completion | `recorded` | `recorded` |
| P1 | pure_greedy | raw_completion | `recorded` | `repetitive` |
| P2 | pure_greedy | raw_completion | `recorded` | `recorded` |
| P3 | pure_greedy | raw_completion | `recorded` | `recorded` |
| P4 | pure_greedy | chat_template | `recorded` | `recorded` |
| P5 | pure_greedy | chat_template | `recorded` | `recorded` |
| P6 | pure_greedy | chat_template | `recorded` | `recorded` |

`contract.derive_smoke_status` reads **`recorded`** off that table (twelve rows
positive on both arms) and `derive_smoke_attribution` reads
**`unattributed`** (nothing on this arm cycles, so there is no symptom to
attribute). These are the first dense cells whose smoke word is derived rather
than asserted: the two sm_121 BF16 cells carry an asserted `recorded` because
their 2026-09-02 receipt predates this instrument.

**What the P1 rows are not.** The BF16 source cycles on P1 and the route does
not. That is one prompt on a 0.6B model at `max_tokens: 64`; it is recorded
because every completion is recorded, and **no quality claim is made from it**.
The KL of §4.1 is the quality reading, and it says the route is 0.0049 nats of
lower bound away from the source.

The greedy smoke that each KL serve ran before its dump is in those serves' own
receipts; the completion was, in every arm,
`' Paris. The capital of France is also the capital of the French Republic. The'`
against the teacher's
`' Paris. The capital of France is also the capital of the Republic of France.'`.

Receipts: `wsl-gpu:/home/rob/agents/t460-cells/receipts/greedy-smoke-gfx1201-20260913T045315Z.{log,json}`,
`wsl-gpu:/home/rob/agents/t460-cells/receipts/smoke/smoke_{wire,bf16}.json`, `wsl-gpu:/home/rob/agents/t460-cells/receipts/smoke/smoke_pair.{json,md}`.

## 5. What is not attested here

* **No gfx1151 cell.** No Strix Halo part is in this fleet. The `gfx1151`
  platform entry keeps `serve_image: null` and no cell, which is the honest
  state and not a gap.
* **No routed-MoE cell on gfx1201.** The BF16 MoE builder path was not
  exercised on HIP in this work, so there is no receipt to mint one from.
* **No performance claim.** See the scope paragraph at the top.
* **No second rung.** `attested_rungs_q256` for `TESSERA_BF16_K1` is `[1792]`
  and the cells attest that rung only.
* **No compiled-mode KL.** The compiled forward is attested by the censuses and
  the smoke; both cells' `kl` entries name `execution_modes: ["eager"]`, which
  is the sm_121 precedent and the honest scope of the dumps.

## 6. Receipt files

Every path below is relative to `/home/rob/agents/t460-cells/receipts/` on
`wsl-gpu`, the box that produced them, and is mirrored to
`/home/rob/tmp/agents/t460-cells/receipts/` on sparky. The inline citations
above spell the same paths in full.

| receipt | what it carries |
|---|---|
| `gfx1201-kernel-suite-20260913T040301Z.txt` | receipts 1–3: 281 passed, 8 skipped, every skip reason verbatim |
| `gfx1201-encoder-on-hip-20260913T040517Z.txt` | the encoder does **not** run on HIP — `window_viterbi.py:235` is PTX inline asm and hipcc refuses it (`couldn't allocate output register for constraint 'f'`). This is why the export ran through PrismaBuild on a GB10 and not here |
| `pb_export_submit.log`, `pb_export_run.log` | the export: 461 s on a GB10, `wire_bpp 7.129`, artifact under `/mnt/shared/tessera-runs/allocated/` |
| `twincheck-gfx1201-20260913T045042Z.{log,json}` | the wire and its twin are one encode: 196 units, 0 mismatched, on gfx1201 |
| `census-t460-{eager,compiled}-{resident,streamed}-*.{log,json}` | receipt 4, four arms on the v23 document |
| `census-t460-v24-eager-{resident,streamed}-*.{log,json}` | the same census against the v24 cells: `agrees true` |
| `e4m3-refusal-20260913T041224Z.log` | an E4M3 artifact refused with the contract's word |
| `kl-teacher-20260913T043045Z.log` | the prefill-regime BF16 teacher dump |
| `kl-qwen_teacher_bf16_gfx1201_decode-20260913T044011Z.log` | the decode-regime BF16 teacher dump |
| `kl-bf16k1-{resident,streamed}-eager-*.log` | the batch-regime bounds |
| `kl-bf16k1-{resident,streamed}-eager-decode-*.log` | the decode-regime bounds |
| `greedy-smoke-gfx1201-20260913T045315Z.{log,json}` | the fourteen-row smoke record and its derived word |

Earlier attempts are kept beside the ones above rather than deleted, because a
receipt directory that holds only the runs that worked is a filtered record:
`kl-teacher-2026091*T04{19,23,28}*.log` are the teacher serves that died on a
dangling HF snapshot symlink, a missing `requests`, and a port collision;
`kl-*-decode-20260913T04{39,41}*.log` are the decode dumps `kl_tool` refused
before `--enable-prompt-tokens-details` was on the serve;
`greedy-smoke-gfx1201-20260913T045101Z.log` is the smoke run that died on a
missing `tokenizers`; `census-old-r7plugin-*` are the first censuses, taken on
the **2026-09-02 sm_121 artifact** before the fresh export existed, and they
attest nothing about the cells.
