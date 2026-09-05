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

**The unexplained observation.** The greedy smoke returned repetitive `France is`
text. The campaign refused to attribute that to quantization without a matched
BF16 prompt, and no matched prompt has been run. Until one is, the word
`repetitive` in the contract names an observation with no control.

---

## 2. What each missing leg would prove, ranked

Ranked by what it changes in the contract per GPU-hour spent. Costs are grounded
in the campaign's own timings: a prefill dump of the 8x512 corpus took 67.5 s
against the BF16 teacher and 63.8 s against the student, and the whole
census and student PrismaBuild actions ran 201.9 s and 285.5 s including model
load and cleanup.

### Leg 0: a matched BF16 greedy smoke

- **Cost:** none beyond a serve that is already up for another leg. One request.
- **Proves:** whether the repetitive output is the quantization or the model.
- **Changes:** `evidence.smoke.status` on both MoE cells, from `repetitive` to
  `recorded`, if BF16 repeats the same way.
- **Rank rationale:** the only leg that can *remove* a flagged concern rather than
  add a number, and it is free when piggybacked.

### Leg A: a decode-regime top-1024 bound on the MoE artifact

- **Cost:** two serves on the box holding the pinned image, roughly 30-45 GPU-minutes.
  Both arms must be re-dumped: `kl_tool compare` refuses a cross-regime pair
  unconditionally, so the prefill teacher on disk cannot be reused.
- **Proves:** quality under M=1 forwards, which the census attests dispatch for and
  nothing attests quality for.
- **Changes:** `tessera_e4m3_k1_routed_moe_sm121_decode_resident.evidence.grade`
  from `route_only` to `kl_lower_bound` -- the same grade as the best-evidenced
  decode cell in the table.
- **Feasibility gate, checked:** the decode regime proves M=1 from
  `usage.prompt_tokens_details.cached_tokens` and refuses otherwise, so it needs
  vLLM's prefix cache. Both LFM serves in the EUGR image report
  `enable_prefix_caching=True` with `Mamba cache mode is set to 'align' for
  Lfm2MoeForCausalLM`, so the mechanism is available on this architecture.
- **Rank rationale:** the cheapest leg that moves a `grade`.

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
  claim rather than deepening one.

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

## 5. What this receipt does not do

It records no new measurement. Every number above is read from
`tessera-lfm-campaign-2026-09-04.md`, from `src/tessera/serving/runtime_contract.json`,
or from the two LFM serve logs under `/mnt/shared/tessera-runs/ts5/lfm25/`. It
does not change any cell's `qualification`, `grade`, or `route_status`.
