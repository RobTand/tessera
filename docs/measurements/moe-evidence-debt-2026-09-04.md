# What the two routed-MoE cells rest on, and what would thicken it (2026-09-04)

**Decision.** Rob promoted issue #133: the two `routed_moe` cells keep
`qualification: "device_qualified"`. This receipt records the evidence debt that
decision carries, so a reader three months from now learns the limit without
reconstructing it from a campaign log.

**Outcome in one sentence.** Both cells are qualified on evidence from **one
model, one residency mode, one execution mode, and one regime** -- an LFM2.5-8B-A1B
E4M3/q1024 artifact served resident and eager on the pinned EUGR image -- and the
batch cell's quality number is a **top-1024 teacher/student-intersection lower
bound of 0.0831613565** (upper bound 1.1787049112 at the declared floor), not a
full-vocabulary KL. The decode cell has no quality number at all: it grades
`route_only`, and its evidence is the route census.

The identities, hashes and PrismaBuild actions behind every number here are in
`tessera-lfm-campaign-2026-09-04.md` §§7-8. This file does not repeat them; it
states what the evidence does **not** cover and what each missing leg costs.

---

## 1. The debt is thin in three specific ways, and not in a fourth

The comparison that matters is against the eight dense cells in the same table,
because `device_qualified` means the same word on all ten. Read off
`src/tessera/serving/runtime_contract.json` at contract v17:

| cell | regime | grade | KL entries | smoke |
|---|---|---|---|---|
| `tessera_e2m1_k2_dense_sm121_decode` | decode | `route_only` | -- | `not_recorded` |
| `tessera_e2m1_k2_dense_sm121_batch` | batch | `kl_lower_bound` | top-1024, eager+compiled | `not_recorded` |
| `tessera_e4m3_k1_dense_sm121_decode_resident` | decode | `route_only` | -- | `not_recorded` |
| `tessera_e4m3_k1_dense_sm121_decode_streamed` | decode | `kl_lower_bound` | top-1024, eager; top-1024, compiled | `not_recorded` |
| `tessera_e4m3_k1_dense_sm121_batch_resident` | batch | `kl_lower_bound` | top-1024, eager+compiled | `not_recorded` |
| `tessera_e4m3_k1_dense_sm121_batch_streamed` | batch | `kl_lower_bound` | top-1024, eager+compiled | `not_recorded` |
| `tessera_bf16_k1_dense_sm121_decode` | decode | `route_only` | -- | `recorded` |
| `tessera_bf16_k1_dense_sm121_batch` | batch | `kl_lower_bound` | top-1024, eager | `recorded` |
| `tessera_e4m3_k1_routed_moe_sm121_decode_resident` | decode | `route_only` | -- | `repetitive` |
| `tessera_e4m3_k1_routed_moe_sm121_batch_resident` | batch | `kl_lower_bound` | top-1024, eager | `repetitive` |

**Where the MoE cells are not behind.** On the axes the schema records, each MoE
cell sits exactly where its regime's dense cells sit. Every batch cell in the
table, dense and MoE, is `kl_lower_bound` on a top-1024 bound. Four of the five
decode cells are `route_only`; `tessera_e4m3_k1_dense_sm121_decode_streamed` is
the exception, not the rule. The MoE `repetitive` smoke is *more* information
than the `not_recorded` that six dense cells carry: somebody ran a greedy prompt
and wrote down what came back.

The premise in the #133 issue body -- that the dense cells carry full-vocab KL
while the MoE cells carry a screen -- is wrong, and Rob corrected it on the issue
before contract v17 landed. Section 4 explains why no cell can carry a
full-vocabulary KL with the instruments this repository holds.

**Where the MoE cells are behind.** Three axes, in the order a gate can see them:

1. **Execution modes.** The MoE cells declare `execution_modes: ["eager"]`; the
   eight dense cells declare `["eager", "compiled"]`. vLLM serves compiled by
   default, so the mode the MoE cells attest is the one an operator has to ask
   for. This is the only gap that is visible in a required field.
2. **Residency.** The MoE cells require `TESSERA_SERVE_MODE=resident`. Streamed
   MoE has no cell because it has no route: the routed-MoE path materialises
   through `torch_materialize_stock`, and the streamed decode lane is a dense
   lane. This is a build, not a missing measurement.
3. **Population.** One artifact, from one model family. No field expresses
   population, so this gap is invisible to a consumer and lives only in prose --
   here and in the campaign receipt. Every dense family is also N=1 (one
   Qwen3-0.6B artifact), so this is a table-wide property rather than an MoE one,
   but the MoE cells are the ones whose architecture class the population is
   meant to represent.

**The observation that was unexplained.** The greedy smoke returned repetitive
`France is` text, and the campaign refused to attribute that to quantization
without a matched BF16 prompt. Section 7 runs that prompt: the BF16 source
returns the identical completion, so the repetition is the model and the prompt,
not the quantization. The word `repetitive` in the contract is still a true
description of what came back; it is no longer a concern about the artifact.

---

## 2. What each missing leg would prove, ranked

Ranked by what it changes in the contract per GPU-hour spent. Costs are grounded
in the campaign's own timings: a prefill dump of the 8x512 corpus took 67.5 s
against the BF16 teacher and 63.8 s against the student, and the whole
census and student PrismaBuild actions ran 201.9 s and 285.5 s including model
load and cleanup.

### Leg 0: a matched BF16 greedy smoke -- RUN, see section 7

- **Cost, as run:** one serve, 110 s to load plus one request.
- **Proved:** the repetition is the model and the prompt. BF16 returns the
  identical completion.
- **Changed:** nothing in the contract. `repetitive` still describes the
  completion accurately, so the status word stands; the control reaches a reader
  through the campaign receipt the cells already name, which now points here.

### Leg A: a decode-regime top-1024 bound on the MoE artifact

- **Cost:** two serves on the box holding the pinned image, roughly 30-45 GPU-minutes.
  Both arms must be re-dumped: `kl_tool compare` refuses a cross-regime pair
  unconditionally, so the prefill teacher on disk cannot be reused.
- **Proves:** quality under M=1 forwards, which the census attests dispatch for and
  nothing attests quality for.
- **Changes:** `tessera_e4m3_k1_routed_moe_sm121_decode_resident.evidence.grade`
  from `route_only` to `kl_lower_bound` -- the same grade as the best-evidenced
  decode cell in the table.
- **Rank rationale:** the cheapest leg that moves a `grade`.
- **BLOCKED, measured.** The attempt and its cause are in section 6. The decode
  regime cannot reach M=1 on this hybrid conv/SSM architecture with the
  instrument as written; the fix is a small, opt-in change to `kl_tool`'s decode
  sweep, and the cost above holds once it exists.

### Leg B: a compiled-mode census and prefill KL

- **Cost:** one serve, roughly 15-20 GPU-minutes. The existing prefill BF16
  teacher dump is reusable, because the regime is unchanged.
- **Proves:** that the routed-MoE route survives vLLM's compiled forward, which is
  what vLLM serves by default. The lane has been bitten here before
  (`vllm-compiled-forward-breaks-lane-hot-paths`).
- **Changes:** `runtime.execution_modes` gains `compiled` on both cells, and the
  batch cell gains a second `kl` entry. Note that the route trace cannot attest
  shapes under compile, so the census evidence is weaker in that mode.
- **Rank rationale:** cheapest leg per required field changed, but it widens a
  claim rather than deepening one. Rank it as the cheapest *unattempted* leg, not
  the cheapest unblocked one: its own gate -- that a routed-MoE model loads and
  censuses at all under `TESSERA_LANE_EAGER=0` in this image -- is unchecked, and
  leg A is a reminder that an unchecked feasibility gate is where the GPU minutes
  go.

### Leg C: a second routed-MoE population

- **Cost:** 4-6 GPU-hours. Encode (the LFM encode ran 2,358 s for half of 22
  stacks on one box), merge, census, teacher dump, student dump.
- **Proves:** that the routed-MoE claim is about routed MoE and not about
  LFM2.5-8B-A1B.
- **Changes:** nothing machine-readable. No field expresses population.
- **Rank rationale:** the most expensive leg and the only one that addresses the
  N=1 objection, but it buys no gate a new fact to read. Adding a population field
  to the cell schema would be a separate contract bump and should precede it.

### Leg D: a full-vocabulary KL

- **Cost:** an instrument build, not GPU-hours. See section 4.
- **Proves:** the exact divergence rather than a bound on it.
- **Changes:** `grade` to `kl_full_vocab` -- for whichever cell it is run against.
- **Rank rationale:** repository-wide, not MoE-specific. No cell in the table can
  reach this grade today, so running it for MoE first would invert the table.

### Leg E: streamed routed MoE

- **Cost:** a lane build.
- **Proves:** nothing until the route exists.
- **Rank rationale:** not a measurement. Listed so the absence is not read as an
  untaken measurement.

---

## 3. What a tighter K would and would not do

Raising the dump's K from 1,024 shrinks the teacher mass the compared support
cannot see (mean 0.0113, max 0.6443 on this pair) and therefore narrows the
0.0832-to-1.1787 interval. It does **not** change the grade:
`contract.EVIDENCE_KL_KINDS` distinguishes `topk_intersection_lower_bound` from
`full_vocab`, and a K of 32,768 is still the former. Treat a bigger K as a way to
make an existing number more useful, never as a way to promote a cell.

---

## 4. Why no cell carries a full-vocabulary KL, and why that is not an MoE gap

`kl_tool.py` has a **read** path for a full-vocabulary teacher
(`read_full_vocab_payload`, `compare --teacher-full-vocab`), which expects a
`<stem>.logprobs.f32.npy` memmap of shape `(positions, vocab_size)`. Nothing
produces that file. `kl_tool dump` writes top-K payloads only, and a search of
`/home/rob/dq-runs` and this tree finds no other writer.

Two further facts close the door for tonight and for any near-term run:

- **The serve cannot supply it.** LFM2.5-8B-A1B has `vocab_size: 128000` and the
  contract scores 4,088 prefill positions. A full-vocabulary dump over the OpenAI
  completions surface means `prompt_logprobs: 128000`, which is 523 million JSON
  entries that vLLM materialises as Python dictionaries. That is not a serve
  anyone runs.
- **Even the read path is still a bound.** With a full-vocabulary teacher and a
  top-K student, `cmd_compare` sets `partition = "student-support"` and passes
  `full_vocab=False` to `metric_identity`, which stamps
  `bound = "lower bound (data-processing inequality)"`. The schema's `full_vocab`
  kind requires `top_k: null` and means an exact number, so the existing read path
  cannot produce it either.

A `kl_full_vocab` grade therefore requires a new instrument -- a logits dump taken
inside the runtime, not over the HTTP surface -- and that instrument is missing
for the dense cells exactly as much as for the MoE ones.
`tests/test_cell_evidence.py::test_no_cell_claims_full_vocabulary_kl` pins that
state of affairs today.

---

## 5. Where these serves ran, and why there

On **sparky**, deliberately. The two cells name
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`
as their `runtime.image`, and sparky is the only box holding that manifest
digest -- sparklina holds `sha256:58862b38...`, three weeks older. A measurement
taken on another digest attests a different runtime and cannot enter these
cells' evidence blocks, so "run it on the idle box" is not available here.
Sparky's GPU was idle for the duration (`nvidia-smi`: 0% utilisation, 4 W, no
compute processes); its load average was CPU. One serve at a time, under
`experiments/serve_lock.sh`.

## 6. Leg A is blocked: the decode regime cannot reach M=1 on this architecture

The first attempt refused at the second scored position:

```
position 1: the serve forwarded 17 rows, not 1 (17 prompt tokens, 0 from the prefix cache).
Refusing: a dump taken at M>1 is a prefill-regime dump wearing a decode-regime label.
```

That refusal is correct and the tool is right to make it. Two explanations fit --
the stride does not match the serve's KV block size, or the hybrid model does not
reuse blocks -- and `experiments/hybrid_prefix_cache_probe.py` separates them in
one serve. Against the BF16 source on the pinned image, issuing the decode
regime's own request shapes strictly sequentially:

| request | prompt tokens | cached tokens | rows forwarded |
|---|---:|---:|---:|
| warm-up, whole chunk | 513 | 0 | 513 |
| L=17 | 17 | 0 | 17 |
| L=33 | 33 | 16 | 17 |
| L=65 | 65 | 32 | 33 |
| L=129 | 129 | 64 | 65 |
| L=257 | 257 | 128 | 129 |
| L=385 | 385 | 256 | 129 |
| L=512 | 512 | 384 | 128 |
| **L=129, repeated** | 129 | **128** | **1** |
| **L=129, repeated again** | 129 | **128** | **1** |
| whole chunk, repeated | 513 | 512 | 1 |

Neither explanation is right, and the third one the table shows is the answer.
The serve reports `block_size="16"`, `mamba_block_size="16"`,
`enable_prefix_caching="True"`, `mamba_cache_mode="align"`, and 1,648 prefix-cache
hits over 2,682 queries, so caching is on and the stride is correct. What a
request can resume from is **the end state of a request the serve has already
answered**, aligned down to a 16-token block -- not any interior block boundary of
a longer prefill. Read the middle rows in order and each one resumes from its
predecessor's end: L=33 from 17 aligned to 16, L=65 from 33 aligned to 32, and so
on. The decode sweep visits L = 1, 17, 33, ... exactly once each, so every scored
request resumes from a state 17 tokens behind it and forwards 17 rows. The
warm-up's 513-token prefill leaves attention blocks behind but no resumable SSM
state at an interior position, which is why the first scored position sees zero.

**The fix, and it is small.** The last three rows show M=1 is reachable: a request
whose prefix the serve has *answered before* resumes at `L-1`. Two variants
reach that, and they are not equivalent. Priming with **`full[:L-1]`** leaves an
end state at `L-1`, a multiple of 16 for every L in the stride-16 set, so the
scored request that follows forwards exactly one row. Issuing the scored request
twice and keeping the second reaches the same M=1, but its first call is a
second *scored-shape* forward at M=17, which a served-request histogram records
and which the dense decode receipts would then have to explain. Prefer the
`full[:L-1]` prime. Either way the cost is twice the requests, and the dump is
HTTP-bound, so the wall-clock cost is minutes.

This is not changed here. `kl_tool.py` lives outside this repository, at
`/home/rob/dq-runs/kl_tool.py`, and several agents were running against it; a
change to a shared instrument belongs in its own reviewed step, behind an opt-in
flag so the #102 receipt still reproduces byte for byte. Filed as its own issue.

**Scope.** The failure is a property of hybrid conv/SSM models, so it does not
touch the dense decode-regime receipts (`tessera-decode-regime-kl-2026-09-03.md`,
`tessera-compiled-decode-kl-r6-2026-09-04.md`), which are attention-only Qwen
artifacts where a single request already resumes at `L-1`.

> **Correction, 2026-09-05 (tessera#192).** The recommendation above -- "Prefer
> the `full[:L-1]` prime" -- was reasoned, not measured, and it is wrong.
> Measured on the same digest, three fresh chunks at L ∈ {17, 129, 257}:
> priming `full[:L-1]` leaves the scored request `cached_tokens = 0` and
> `rows = L`, 3/3; priming `full[:L]` leaves `cached_tokens = L-1` and
> `rows = 1`, 3/3. A request that ends exactly on a block boundary leaves
> nothing resumable at that boundary. The histogram objection to re-issuing
> the scored prefix is answered by shape instead: `kl_tool --decode-prime`
> sends the prime **warm-up shaped**, with no `logprobs`, so it is not a
> scored-shape forward and the three request populations stay separable.
> Two rows of the table above are also from a probe form that repeated at a
> block multiple; see the receipt for what each repeat length does.
> `docs/measurements/hybrid-decode-prime-2026-09-05.md` supersedes this
> section's fix recommendation. The blocker itself, and everything above the
> "**The fix, and it is small**" paragraph, reproduced exactly.

## 7. Leg 0 is run: the repetitive smoke is the model

The campaign recorded the Tessera student answering `The capital of France is`
with repetitive text and refused to attribute it to quantization without a
matched BF16 prompt. That prompt, byte for byte from
`experiments/tessera_plugin_served.sh:108`
(`{"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}`),
against `/mnt/shared/models/LFM2.5-8B-A1B-BF16` on the same pinned image, eager,
`--enforce-eager`, no speculative decoding:

| arm | completion |
|---|---|
| Tessera E4M3/q1024 student (campaign, 2026-09-04) | `' France is France is France is France is France is France is France is France is'` |
| **BF16 source (this receipt)** | `' France is France is France is France is France is France is France is France is'` |

The two completions are identical, character for character. **The repetition is
the model and the prompt, not the quantization.** A greedy 16-token continuation
of a five-word prompt on an 8B A1B base model is not a quality signal in either
direction, and this pair says so rather than leaving the word `repetitive` in the
contract to be read as a defect.

Serve log: `/home/rob/tessera-runs/ts133/serve_smoke.log`. Probe output:
`/home/rob/tessera-runs/ts133/smoke.log`.

## 8. What this receipt does not do

It changes no cell's `qualification`, `grade`, `route_status`, or `evidence`
block. The two measurements it adds are a control (section 7) and a blocker
diagnosis (section 6); neither is a KL number, and neither promotes anything.
Every other number is read from `tessera-lfm-campaign-2026-09-04.md`, from
`src/tessera/serving/runtime_contract.json`, or from the LFM serve logs under
`/mnt/shared/tessera-runs/ts5/lfm25/`.

---

**Note, 2026-09-04, after this receipt (#195).** Leg 0's "Changed: nothing in
the contract" and section 8's "changes no cell's `evidence` block" were true of
this receipt and are no longer true of the tree. The control section 7 ran now
travels **in** the contract: lane-eligibility schema **v7** (contract v18)
gives every `evidence.smoke` a `control` -- `{reference: "bf16_source", outcome:
"identical_completion", receipt: "docs/measurements/moe-evidence-debt-2026-09-04.md"}`
on both `routed_moe` cells -- and a derived `smoke.attribution`, which reads
`shared_with_reference` there. The status word `repetitive` still stands, as
this receipt said it should; what changed is that a consumer no longer has to
read this file to know the reference does the same thing. PrismaQuant's pin
(RobTand/prismaquant#192) refused the lane on `status` alone while that fact was
prose; it moves to `attribution`.
