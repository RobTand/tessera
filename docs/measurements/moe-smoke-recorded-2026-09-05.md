# Routed-MoE greedy smoke, recorded: the source and the student both answer through the checkpoint's own interface (2026-09-05)

**Status: closed for what it claims.** This is the receipt behind `evidence.smoke.status: recorded` on
the two `routed_moe` cells of the serving contract (`tessera_e4m3_k1_routed_moe_sm121_decode_resident`,
`tessera_e4m3_k1_routed_moe_sm121_batch_resident`; contract v21). It records a greedy smoke on prompts
the BF16 **source** does not degenerate on, with the Tessera **student** served beside it under the same
pinned image, eager, resident, TP=1, no speculative decoding. It is a smoke, not a KL: the claim is
*greedy coherence on both arms*, nothing about closeness.

The finding, in three lines:

1. **Raw continuation degenerates on the source.** Every raw-completion prompt (the v17 campaign prompt
   `The capital of France is` and three content-bearing passages ending mid-sentence) ends in a cycle on the
   BF16 source, under the campaign's request shape and under pure greedy alike. The student ends in a cycle
   on all but one of them too (its P2 wanders without cycling). The checkpoint is the *instruct* model and a
   raw continuation is not the interface it was trained on.
2. **Through the checkpoint's own chat template, both arms answer.** Three questions carrying the same
   content, sent to `/v1/chat/completions` so the serve applies the checkpoint's `chat_template.jinja`
   (byte-identical on both arms), read `recorded` on **both** arms in **both** request forms: no cycle, a
   `<think>` trace the model opens on its own, then an on-topic answer, then the model's own EOS. The
   student finished all six with `stop`; the source hit the 512-token cap once, mid-sentence, without a cycle.
3. **The completions are not identical between arms**, and this receipt claims nothing about that: the
   greedy paths diverge inside the trace, both stay coherent, and closeness is the KL's business, not a
   smoke's. Both arms' completions are below, verbatim, whichever way they read.

## 1. Why this measurement exists (RobTand/prismaquant#198)

PrismaQuant's pin refuses the routed-MoE lane on `smoke.status` alone, and that refusal is right under
the rule it was given: principle 5's "generates correctly" leg had no positive record for routed MoE.
Contract v18 (#195) added a *control* -- the BF16 source, given the campaign prompt, returns the same
`' France is France is ...'` character for character -- which removes the evidence *against* the route
without adding evidence *for* it: a control that degenerates too says the symptom is shared, not that the
route generates. The honest fix is a positive record, and a positive record is a Tessera measurement, so
it is recorded here, the status word moves to `recorded`, and PrismaQuant's existing status-only rule
admits the cells with no predicate change on its side.

## 2. Setup

| | |
|---|---|
| Box | sparky (GB10, 121.6 GiB unified) |
| Image | `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c` (local id identical; `runtime_image_require`) |
| vLLM | `0.28.1rc1.dev397+gfd4a15126.d20260904`, V1 engine, `enforce_eager=True`, `speculative_config=None` (both arms; `serve_require_no_spec_decode` passed on both) |
| Serve flags (both arms) | `--max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.35 --max-logprobs 1024 --enforce-eager --trust-remote-code`, TP=1, `--served-model-name kl-target` |
| Source arm | `/mnt/shared/models/LFM2.5-8B-A1B-BF16`, `quantization=None`, config sha256 `9c0255c2…d2d68`; 15.8 GiB weights, KV 21.92 GiB |
| Student arm | `/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2/full-model-r3`, `quantization=tessera` (plugin registered: `vllm.general_plugins: ['tessera', …]`), config sha256 `cbc35c81…c0795`; served **resident** (`TESSERA_SERVE_MODE=resident`), 8.59 GiB weights, KV 33.36 GiB |
| Student seal | `…/passthrough-correction-r3/artifact-seal.json`, sha256 `3208ca1e85ebcf2a512924789f1bacff3a22319cdbb67e81f4c3f36348dff634`; `checkpoint` and `checkpoint_identity` bound to the served directory before the serve (`seal_binding.json`, `bound: true`) |
| Artifact identity | `source_identity` taken before and after each serve, byte-identical (`identity_<arm>_{before,after}.json`); both directories mounted read-only |
| Tokenizer | `tokenizer.json` sha256 `695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41`, identical on both arms; the rule tokenises with the student's copy |
| Chat template | `chat_template.jinja` sha256 `c5b67247c9f736f6c40eaa91ae1ef27dc67f5b041eedaadb97a1bea5030fb0ff`, identical on both arms; it has no switch for the `<think>` trace |
| generation_config (both arms) | `{do_sample: true, temperature: 0.2, top_k: 80, repetition_penalty: 1.05}`; vLLM applies these as request defaults (logged: "Default vLLM sampling parameters have been overridden by the model's generation_config.json") |
| Tree served | Tessera `ec42d8d22148792b3764cbfa896c7d447c9cefd9` (`origin/master` at dispatch; `-e /work` inside the container, as `experiments/tessera_plugin_served.sh` does) |
| Instrument | `experiments/moe_greedy_smoke_pair.sh` (serve one arm at a time under the serve lock, probe, stamp, reap, compare) driving `experiments/moe_greedy_smoke.py`; prompts `experiments/moe_greedy_smoke_prompts.json` sha256 `1d280c010c23d156493e89d015afaad04cf94f434b3f4f60e6e797a0adb88375` |
| Build stamps | `tessera.serve_build_identity/2`, `complete: true`, `fresh_compiles: 0`, eager, no compiled build: source fingerprint `a9742a32bee8db4f…`, student `1473ade26775c434…` (`smoke_<arm>.build.json`) |
| Memory | gate 40 GiB `MemAvailable` before each serve (107 GiB seen before the source, 114 before the student); high-water: `MemAvailable` fell to 55,415,660 kB (52.8 GiB; ~69 GiB in use, box-wide) during the source serve and to 70,196,380 kB (66.9 GiB; ~55 GiB in use) during the student serve; 114 GiB after each reap; caches under `/home/rob/ts-audit-caches/ts-moe-smoke/` |
| Receipts | `/home/rob/tessera-runs/ts198/pair-r3/` on sparky: `driver.log`, `serve_<arm>.log`, `smoke_<arm>.json`, `smoke_<arm>.build.json`, `metrics_<arm>.txt`, `identity_<arm>_{before,after}.json`, `telemetry_<arm>.log`, `runtime_image.{txt,json}`, `seal_binding.json`, `pair.json`, `pair.md`; driver stdout `run-r3.log` |

PrismaBuild was not used; the serve is the repository's own `docker run` lines (source: the BF16 line of
`experiments/serve_and_dump_kl.sh`; student: the line of `experiments/tessera_plugin_served.sh`), the
`runtime_image_require` / `serve_lock` / `build_identity_stamp` helpers, and the two scripts above.

## 3. Requests

Two request forms per prompt, both `temperature: 0`:

- **campaign** -- the v17 smoke's shape, byte for byte in what it sets: `{model, prompt | messages,
  max_tokens, temperature: 0}`. With `temperature: 0` the serve decodes greedily, and the checkpoint's
  `generation_config.json` defaults still apply to what the request does not set, so `repetition_penalty
  1.05` (and a `top_k 80` that changes nothing under argmax) is in force. This is the form the contract's
  status word is read from, because it is what the campaign measured.
- **pure_greedy** -- the same plus `repetition_penalty: 1.0, top_k: -1`: argmax with nothing else.
  Recorded so a reader can see the finding does not depend on the penalty.

Raw prompts (P0-P3) go to `/v1/completions` as `prompt`, exactly like the campaign smoke. Chat prompts
(P4-P6) go to `/v1/chat/completions` as one `user` message, so the serve applies the checkpoint's own
`chat_template.jinja`; the client sends no system prompt, no template arguments and no chat flags.

| id | form | max_tokens | text |
|---|---|---:|---|
| P0 | raw `prompt` | 16 | The capital of France is |
| P1 | raw `prompt` | 64 | The Rhine rises in the Swiss Alps and flows north through Germany and the Netherlands before reaching the North Sea. For centuries it has carried grain, timber and coal between the interior and the coast, and the cities along its banks grew rich on that trade, including |
| P2 | raw `prompt` | 64 | Photosynthesis takes place in the chloroplasts of plant cells, where chlorophyll absorbs light and uses its energy to split water molecules. The oxygen released is a by-product; the hydrogen and the captured energy are used to reduce carbon dioxide into sugars, and that chemical energy is stored in |
| P3 | raw `prompt` | 64 | The first transatlantic telegraph cable was completed in 1858 between Ireland and Newfoundland. It carried only a few messages before its insulation failed, and the line went silent within weeks. Engineers spent the next several years studying the failure, and a more durable cable laid in 1866 |
| P4 | chat `messages` (one `user` turn) | 512 | In two or three sentences, explain why the Rhine mattered to the trade of the cities along its banks, and name two of those cities. |
| P5 | chat `messages` (one `user` turn) | 512 | In a short paragraph, explain where the energy that photosynthesis captures from light ends up, and what the oxygen it releases comes from. |
| P6 | chat `messages` (one `user` turn) | 512 | Briefly describe what happened to the first transatlantic telegraph cable of 1858, and how the 1866 cable differed from it. |

## 4. The rule

The instrument decides the status word, and `tests/test_moe_greedy_smoke_rule.py` pins it:

> repetitive iff the completion ends in a cycle: some period p with 2p <= L has a p-periodic suffix holding >= 2 full periods (s >= 2p), whatever its share of the completion; tokens are the artifact tokenizer's canonical encoding of the returned text

In symbols: tokens are the student tokenizer's canonical encoding of the returned text (L of them); for
each period p with 2p <= L, s(p) is the length of the longest suffix in which token i equals token i+p; the
completion is `repetitive` iff some p has s(p) >= 2p, and the report names the shortest such p. Greedy
decoding never leaves a cycle, so the *tail* is what matters: a repeat in the middle that the completion
moved past is not degeneration, and a cycle that started late is still a cycle -- where `max_tokens` cut
it is a fact about the budget, not the model. The v17 observation (`' France is'` x 8, p=2, s=L=16) is the
rule's positive control and must read `repetitive` on both arms; it does, in every run.

**What the rule does not do.** It detects cycles. It does not grade coherence or facts: a completion can
read `recorded` and be incoherent (the student's raw P2 below is exactly that -- non-periodic wandering),
and a coherent answer can carry a wrong fact. So the status word is the rule's verdict on the campaign-form
completions, and the contract flips to `recorded` on the machine condition *at least one prompt reads
`recorded` on both arms in the campaign form* (`positive_record` in `pair.json`, here `P4`, `P5`, `P6`),
while the reading of the text -- section 7 -- is reported beside it for a person, not fed into the word.

The rule was tightened once, before any chat-form result existed, and every completion of the first run
is re-scored under the final rule in `pair.json` (`rule_at_run` records what the arms ran under). See §5.

## 5. Runs: what was measured, in order

Three runs, all kept under `/home/rob/tessera-runs/ts198/`. The prompt set grew twice, each time
because the interface answered, not because a verdict did; nothing was dropped.

- **pair-r1** (P0-P3, both forms, both arms; rule at the time: *majority* -- repetitive iff the periodic
  suffix outweighs the rest). The source ended in a cycle on all eight (prompt, form) pairs. The student read
  `recorded` on three of them under that rule while visibly cycling (its P3 spends 33 tokens on a sentence
  and then repeats `' trans'` to the cap: a 31-token cycle a majority rule waved through). That is a defect in
  the rule, fixed before r2: the share of the completion a cycle occupies is set by `max_tokens`. Under the
  final rule r1 reads: source repetitive 8/8, student repetitive 7/8 (`positive_record: []`).
- **pair-r2** (P0-P6, chat prompts at 64 tokens). Source arm complete: P0-P3 reproduce r1 byte for byte;
  P4-P6 open `<think>` and plan the answer, no cycle, budget exhausted before any answer. **Student arm
  void**: vLLM's memory profiler asserted `Initial free memory 49.45 GiB, current free memory 102.8 GiB` --
  a co-tenant container on the box released ~50 GiB of unified memory during the ~5 s profile (the telemetry
  shows `MemAvailable` 40 -> 92 GiB across one 10 s sample). Not a Tessera fault, not a measurement;
  `pair-r2/serve_tessera.log` keeps the trace. The driver reaped the container and released the lock.
- **pair-r3** (P0-P6, chat prompts at 512 tokens so the answer is reached; this receipt). Both arms
  complete. P0-P3 reproduce r1 byte for byte on **both** arms; the source's 64-token r2 chat completions are
  exact prefixes of its r3 ones. Nothing else on the box held the serve lock; the foreign container present
  during r2 was idle (97 MiB) at launch.

## 6. Results (pair-r3)

Verdicts under the rule of §4; L tokens, cycle period p, periodic suffix s, and the serve's `finish_reason`
(`stop` = the model emitted EOS; `length` = `max_tokens`). "identical" is byte equality of the two arms'
completions.

| prompt | form | max_tokens | BF16 source | Tessera student | identical |
|---|---|---:|---|---|---|
| P0 | campaign | 16 | `repetitive` L=16 p=2 s=16 finish=`length` | `repetitive` L=16 p=2 s=16 finish=`length` | yes |
| P1 | campaign | 64 | `repetitive` L=64 p=10 s=64 finish=`length` | `repetitive` L=64 p=3 s=12 finish=`length` | no |
| P2 | campaign | 64 | `repetitive` L=64 p=1 s=57 finish=`length` | `recorded` L=64 p=0 s=0 finish=`length` | no |
| P3 | campaign | 64 | `repetitive` L=64 p=7 s=64 finish=`length` | `repetitive` L=64 p=1 s=31 finish=`length` | no |
| P4 | campaign | 512 | `recorded` L=292 p=0 s=0 finish=`stop` | `recorded` L=372 p=0 s=0 finish=`stop` | no |
| P5 | campaign | 512 | `recorded` L=335 p=0 s=0 finish=`stop` | `recorded` L=360 p=0 s=0 finish=`stop` | no |
| P6 | campaign | 512 | `recorded` L=512 p=0 s=0 finish=`length` | `recorded` L=304 p=0 s=0 finish=`stop` | no |
| P0 | pure_greedy | 16 | `repetitive` L=16 p=2 s=16 finish=`length` | `repetitive` L=16 p=2 s=16 finish=`length` | yes |
| P1 | pure_greedy | 64 | `repetitive` L=64 p=6 s=63 finish=`length` | `repetitive` L=64 p=3 s=12 finish=`length` | no |
| P2 | pure_greedy | 64 | `repetitive` L=64 p=24 s=64 finish=`length` | `repetitive` L=64 p=24 s=61 finish=`length` | no |
| P3 | pure_greedy | 64 | `repetitive` L=64 p=7 s=64 finish=`length` | `repetitive` L=64 p=7 s=64 finish=`length` | yes |
| P4 | pure_greedy | 512 | `recorded` L=402 p=0 s=0 finish=`stop` | `recorded` L=431 p=0 s=0 finish=`stop` | no |
| P5 | pure_greedy | 512 | `recorded` L=400 p=0 s=0 finish=`stop` | `recorded` L=428 p=0 s=0 finish=`stop` | no |
| P6 | pure_greedy | 512 | `recorded` L=468 p=0 s=0 finish=`stop` | `recorded` L=401 p=0 s=0 finish=`stop` | no |

`positive_record` (campaign form, both arms `recorded`): P4, P5, P6.

## 7. Reading the completions

- **P0-P3, both forms.** The source cycles on all eight: a whole-completion cycle on P0, P1 and P3, and on
  P2 a run of `' **'` after one sentence (campaign) or a 24-token sentence pair repeated (pure greedy). The
  student cycles on seven; its campaign P2 closes the sentence, then produces a non-periodic "final answer"
  scaffold that is incoherent but never repeats. Its cycles are shorter-period and, on P1, mix in Spanish
  (`… de la interior y el interior y el …`). None of this is a positive record and none of it is claimed as one.
- **P4-P6, both forms.** Both arms open `<think>`, restate the task, draft, close the trace and answer.
  Source: Cologne and Basel; energy into ATP/NADPH driving the Calvin cycle, oxygen from water split at
  photosystem II; the 1858 cable failed within weeks, the 1866 one was laid by the Great Eastern with better
  insulation and armouring. Student: Cologne and Basel (campaign) / Cologne and Mainz (pure greedy); the same
  energy and oxygen account; the same cable story. Every one of the student's six ends with its own EOS.
  The source's campaign P6 reached the 512-token cap two clauses into its answer, without a cycle.
- **A shared factual slip.** Three of the four P6 answers credit the 1866 cable with "repeaters": the
  source's campaign-form trace says "repeaters came later" and its answer leaves them out, its pure-greedy
  answer puts them in, and the student puts them in both times. A smoke does not grade facts; the slip is
  recorded because it was noticed, and because the source makes it too.
- **Not claimed.** That the arms agree (they do not, byte for byte: the traces diverge within the first
  sentence and the answers differ in wording), that either is more correct, or anything about closeness.
  The batch cell's KL entry is the closeness record; this receipt is the "generates" record.

## 8. The completions, verbatim (pair-r3)

Each block is the text field the serve returned (`choices[0].text` for raw prompts, `choices[0].message.content`
for chat), unmodified; token ids are in `smoke_<arm>.json`.

### 8.1 Form `campaign`

#### P0 / campaign / BF16 source -- `repetitive` L=16 p=2 s=16 finish=`length`

```text
 France is France is France is France is France is France is France is France is
```

#### P0 / campaign / Tessera student -- `repetitive` L=16 p=2 s=16 finish=`length`

```text
 France is France is France is France is France is France is France is France is
```

#### P1 / campaign / BF16 source -- `repetitive` L=64 p=10 s=64 finish=`length`

```text
 the cities of the interior and the coast, and the cities of the interior and the coast, and the cities of the interior and the coast, and the cities of the interior and the coast, and the cities of the interior and the coast, and the cities of the interior and the coast, and the cities of the
```

#### P1 / campaign / Tessera student -- `repetitive` L=64 p=3 s=12 finish=`length`

```text
 the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities de la interior y el interior y el interior y el interior y el
```

#### P2 / campaign / BF16 source -- `repetitive` L=64 p=1 s=57 finish=`length`

```text
 the cell's membrane."

** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
```

#### P2 / campaign / Tessera student -- `recorded` L=64 p=0 s=0 finish=`length`

```text
 the cell's membrane."

The final answer: "Splitting water molecules" (or "reducing carbon dioxide into sugars") (or "storing hydrogen in the cell's membrane").

Given the question: "Splitting water molecules", I'll answer: "Splitting water molecules".

Thus:


```

#### P3 / campaign / BF16 source -- `repetitive` L=64 p=7 s=64 finish=`length`

```text
. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks.
```

#### P3 / campaign / Tessera student -- `repetitive` L=64 p=1 s=31 finish=`length`

```text
. The line went silent within weeks. So we can say that the line went silent, and the line went silent.

Thus we have many messages: "the first trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans trans
```

#### P4 / campaign / BF16 source -- `recorded` L=292 p=0 s=0 finish=`stop`

```text
<think>
The user asks: "In two or three sentences, explain why the Rhine mattered to the trade of the cities along its banks, and name two of those cities."

We need to produce 2-3 sentences total? It says "In two or three sentences". So we can do either 2 sentences or 3 sentences. Must include explanation why the Rhine mattered to trade of the cities along its banks, and name two of those cities.

Thus something like: "The Rhine provided a navigable waterway that linked inland producers with major European markets, allowing cities along its banks to export goods such as grain, wine, and manufactured products efficiently. Notable trading cities on the river include Cologne and Basel." That's one sentence? Actually it's two clauses but could be considered one sentence. We need two or three sentences. Could do:

"The Rhine served as a vital commercial artery, enabling cities along its banks to transport bulk commodities and connect to broader European trade networks, which spurred economic growth and urban prosperity. Prominent examples of such cities are Cologne and Basel." That's two sentences. That includes explanation and names two cities. That satisfies requirement.

Thus answer: Two sentences. Provide that.
</think>
The Rhine served as a vital commercial artery, allowing cities along its banks to move bulk goods and link inland producers with major European markets, thereby boosting trade and urban prosperity. Prominent examples of such cities are Cologne and Basel.
```

#### P4 / campaign / Tessera student -- `recorded` L=372 p=0 s=0 finish=`stop`

```text
<think>
The user asks: "In two or three sentences, explain why the Rhine mattered to the trade of the cities along its banks, and name two of those cities."

We need to produce a response that is 2-3 sentences total, containing an explanation why the Rhine mattered to trade, and naming two cities along its banks.

We should mention something like: The Rhine was a major European trade artery linking inland Europe to the North Sea, facilitating movement of goods such as grain, wine, timber, and minerals; it allowed cities to access maritime trade routes and connect with broader markets. Two cities could be Cologne (Köln) and Basel (or Strasbourg, Mainz). So we can say: "The Rhine provided a navigable waterway that linked interior producers with the North Sea, enabling efficient transport of bulk goods and fostering market integration for the cities on its banks." Then name two cities: "Cologne and Basel were key trading hubs along this river." That would be one sentence? Actually they want two or three sentences. Could combine both in two sentences:

Sentence 1: The Rhine mattered because it offered a navigable trade route that connected inland producers to the North Sea, allowing cities along its banks to export agricultural products, raw materials, and manufactured goods efficiently. Sentence 2: Notable examples include Cologne and Basel, which thrived as major commercial centers due to their strategic positions on the river.

That's two sentences, includes explanation and names two cities. That satisfies requirement.

Thus answer.
</think>
The Rhine mattered because it offered a navigable trade route that linked inland producers to the North Sea, allowing cities along its banks to export agricultural products, raw materials, and manufactured goods efficiently. Notable examples include Cologne and Basel, which thrived as major commercial centers due to their strategic positions on the river.
```

#### P5 / campaign / BF16 source -- `recorded` L=335 p=0 s=0 finish=`stop`

```text
<think>
The user asks: "In a short paragraph, explain where the energy that photosynthesis captures from light ends up, and what the oxygen it releases comes from."

We need to produce a short paragraph (maybe 2-3 sentences) explaining where the captured light energy goes (e.g., into chemical energy stored in ATP and NADPH, which then drive carbon fixation via Calvin cycle to produce glucose). And also state that the oxygen released comes from water molecules split during photolysis.

Thus answer: The captured photon energy is converted into chemical energy in the form of ATP and NADPH, which power the synthesis of carbohydrates; the O2 released originates from the splitting of water (photolysis) in the thylakoid lumen. Provide concise paragraph.

Make sure no disallowed content. It's straightforward.

We should not add extra fluff. Provide one paragraph.

Let's craft:

"The light‑energy absorbed by chlorophyll is first transformed into the high‑energy carriers ATP and NADPH, which then supply the energy needed for the Calvin‑Benson cycle to fix CO₂ into sugars; the molecular oxygen expelled as a by‑product comes exclusively from the water molecules that are split during the photochemical reactions of photosystem II." That works.

That's a single paragraph. Done.
</think>
The light‑energy captured by chlorophyll is first converted into the high‑energy carriers ATP and NADPH, which then provide the energy required for the Calvin‑Benson cycle to fix CO₂ into sugars; the molecular oxygen released as a by‑product comes exclusively from the water molecules that are split during the photochemical reactions of photosystem II.
```

#### P5 / campaign / Tessera student -- `recorded` L=360 p=0 s=0 finish=`stop`

```text
<think>
We need to produce a short paragraph explaining where the energy captured from light in photosynthesis ends up, and what the oxygen released comes from. The answer: Light energy is converted into chemical energy stored in ATP and NADPH (the "energy carriers") which then drive the Calvin cycle to fix CO2 into glucose. The oxygen released comes from water molecules split during photolysis of chlorophyll; the O atoms are derived from H2O, not from CO2. So we need to mention that the energy captured is used to generate ATP and NADPH, which power carbon fixation, while O2 originates from splitting water (photolysis). Provide concise paragraph.

Make sure it's one paragraph, short. Let's craft:

"In photosynthesis, the photons absorbed by chlorophyll convert their energy primarily into the high‑energy molecules ATP and NADPH, which then supply the reducing power and energy needed for the Calvin‑Benson cycle to fix atmospheric CO₂ into carbohydrate. Simultaneously, the light‑driven photolysis of water in the thylakoid lumen releases molecular oxygen; the oxygen atoms come exclusively from the water molecules that are split to provide electrons, protons, and the electrons required for the photosystems." That works.

That's a single paragraph. Provide.
</think>
In photosynthesis, the photons absorbed by chlorophyll convert their energy mainly into the high‑energy molecules ATP and NADPH, which then supply the reducing power and energy needed for the Calvin‑Benson cycle to fix atmospheric CO₂ into carbohydrates; at the same time, the light‑driven photolysis of water in the thylakoid lumen releases molecular oxygen, and the oxygen atoms originate solely from the water molecules that are split to provide electrons, protons, and the electrons required for the photosystems.
```

#### P6 / campaign / BF16 source -- `recorded` L=512 p=0 s=0 finish=`length`

```text
<think>
We need to answer: Briefly describe what happened to the first transatlantic telegraph cable of 1858, and how the 1866 cable differed from it.

We should provide a concise description. The first transatlantic telegraph cable was laid in 1858, but it failed after a few weeks due to technical problems, corrosion, and perhaps damage during laying. It was abandoned after about three months. Then the second attempt in 1865-66 succeeded, with improvements: better insulation (gutta-percha), stronger cable design, better ship handling, longer length, etc. Also the 1866 cable used improved materials, more protective layers, and had repeaters? Actually repeaters came later; the 1866 cable used gutta-percha insulation and a copper core, and the cable was thicker and had better protection against water and mechanical damage. Also the 1866 cable was laid by the SS Great Eastern, and it lasted longer.

Thus answer: The 1858 cable was laid in August 1858, transmitted messages for a few weeks before failing due to insulation breakdown and mechanical stress; it was left on the ocean floor. The 1866 cable, after many attempts, was successfully laid and operated, using a more robust construction, better insulation, and a stronger steel wire core, and was able to transmit messages reliably for decades.

We need to be brief. Possibly 2-3 sentences each? Or a short paragraph. Provide both events and differences.

Let's craft:

"The first transatlantic telegraph cable, completed in August 1858, worked for only a few weeks before its insulation deteriorated and the line succumbed to mechanical stress, after which it was abandoned on the seabed. The 1866 cable, after numerous failures, was finally laid successfully by the SS Great Eastern; it employed a thicker, better‑insulated gutta‑percha sheath, stronger steel wires, and improved splicing techniques, allowing continuous service."

That is concise. That covers what happened and differences. That should satisfy.
</think>
The first transatlantic telegraph cable, finished in August 1858, managed only a few weeks of service before its insulation failed and the line broke under mechanical stress, after which it was left on the ocean floor. The 1866 cable, after many earlier attempts, was finally laid successfully by the SS Great Eastern; it used a thicker, better‑insulated gutta‑percha sheath, stronger steel
```

#### P6 / campaign / Tessera student -- `recorded` L=304 p=0 s=0 finish=`stop`

```text
<think>
We need to answer: "Briefly describe what happened to the first transatlantic telegraph cable of 1858, and how the 1866 cable differed from it." Provide a brief description. No extra formatting unless asked. So we should give concise paragraph(s). Probably mention that the 1858 cable was laid in 1858, worked for a few weeks then failed due to technical problems, corrosion, insulation failure, and was abandoned. The 1866 cable was more robust, used better materials, had repeaters, and was successfully transmitted messages. Also note differences: length, number of wires, insulation, durability, use of gutta-percha, copper sheathing, etc. Provide brief description.

We must not add disclaimer. No markdown. Just plain text. Let's produce a concise answer.
</think>
The first transatlantic telegraph cable was laid in August 1858 and initially carried messages for a few weeks before it stopped working. Corrosion, inadequate insulation, and mechanical stress caused it to fail, and the line was abandoned after only a few weeks of service.  

When the cable was successfully re‑laid in 1866, it incorporated several improvements: a thicker, more durable core; better insulating material (gutta‑percha and later rubber); additional protective steel armor; and a system of repeaters that amplified the signal. These changes made the 1866 cable far more reliable and enabled continuous, long‑distance communication across the Atlantic.
```

### 8.2 Form `pure_greedy`

#### P0 / pure_greedy / BF16 source -- `repetitive` L=16 p=2 s=16 finish=`length`

```text
 France is France is France is France is France is France is France is France is
```

#### P0 / pure_greedy / Tessera student -- `repetitive` L=16 p=2 s=16 finish=`length`

```text
 France is France is France is France is France is France is France is France is
```

#### P1 / pure_greedy / BF16 source -- `repetitive` L=64 p=6 s=63 finish=`length`

```text
 the cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its banks. The cities along its
```

#### P1 / pure_greedy / Tessera student -- `repetitive` L=64 p=3 s=12 finish=`length`

```text
 the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities of the interior and the cities de la interior y el interior y el interior y el interior y el
```

#### P2 / pure_greedy / BF16 source -- `repetitive` L=64 p=24 s=64 finish=`length`

```text
 the chloroplasts. The process is a fundamental part of the chloroplasts, and that chemical energy is stored in the chloroplasts. The process is a fundamental part of the chloroplasts, and that chemical energy is stored in the chloroplasts. The process is a fundamental part of the chloroplasts
```

#### P2 / pure_greedy / Tessera student -- `repetitive` L=64 p=24 s=61 finish=`length`

```text
 the plant cells. The plant cells are the source of the energy for the plant cells, and that energy is used to split water molecules. The plant cells are the source of the energy for the plant cells, and that energy is used to split water molecules. The plant cells are the source of the energy for the plant
```

#### P3 / pure_greedy / BF16 source -- `repetitive` L=64 p=7 s=64 finish=`length`

```text
. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks.
```

#### P3 / pure_greedy / Tessera student -- `repetitive` L=64 p=7 s=64 finish=`length`

```text
. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks. The line went silent within weeks.
```

#### P4 / pure_greedy / BF16 source -- `recorded` L=402 p=0 s=0 finish=`stop`

```text
<think>
The user asks: "In two or three sentences, explain why the Rhine mattered to the trade of the cities along its banks, and name two of those cities."

We need to produce 2-3 sentences. Must explain why the Rhine mattered to trade of the cities along its banks, and name two of those cities. We need to pick two cities along the Rhine, e.g., Cologne (Köln) and Basel, or Strasbourg and Mainz, or Düsseldorf and Cologne, or Cologne and Mainz. Provide explanation: the Rhine was a major transportation route, linking interior Europe to the North Sea, facilitating movement of goods, linking major commercial centers, enabling exchange of goods like wine, grain, timber, etc. So we can say: The Rhine provided a navigable waterway that linked inland producers with maritime trade, making cities along its banks wealthy and strategic. Two cities: Cologne and Basel. That would be one sentence? Actually we need two or three sentences. Could be: "The Rhine served as a vital trade artery, allowing inland producers to ship bulk goods to the North Sea and connecting major commercial hubs across Europe." That's one sentence. Then: "Cities such as Cologne and Basel thrived as key ports and market centers because of this access." That's second sentence. That's two sentences, includes both explanation and naming two cities. That satisfies. Provide that.

Thus answer: "The Rhine served as a vital trade artery, allowing inland producers to ship bulk goods to the North Sea and linking major commercial hubs across Europe. Cities such as Cologne and Basel thrived as key ports and market centers because of this access." That's two sentences. That works.

We must not add extra commentary. Provide that.
</think>
The Rhine served as a vital trade artery, allowing inland producers to ship bulk goods to the North Sea and linking major commercial hubs across Europe. Cities such as Cologne and Basel thrived as key ports and market centers because of this access.
```

#### P4 / pure_greedy / Tessera student -- `recorded` L=431 p=0 s=0 finish=`stop`

```text
<think>
The user asks: "In two or three sentences, explain why the Rhine mattered to the trade of the cities along its banks, and name two of those cities."

We need to produce a response: 2 or 3 sentences. It should explain why the Rhine mattered to trade of the cities along its banks, and name two of those cities. We need to pick two cities along the Rhine. For example, Cologne (Köln) and Basel, or Strasbourg and Mainz, or Düsseldorf and Cologne. The Rhine flows through many cities: Cologne, Düsseldorf, Mainz, Koblenz, Basel, etc. We need to mention why it mattered: it provided a navigable waterway for transport of goods, linking interior Europe to the North Sea, enabling trade in commodities like wine, grain, timber, etc., and fostering economic growth. So we can say: The Rhine's navigable channel allowed bulk goods to move efficiently, connecting inland producers with coastal markets, which spurred the prosperity of riverine cities. Then name two cities: Cologne and Mainz. That would be two sentences: "The Rhine's navigable channel turned the surrounding cities into vital trade hubs, allowing them to export agricultural products, raw materials, and manufactured goods to the North Sea and beyond, which fueled their economic growth. Notable cities that thrived on this trade include Cologne and Mainz." That's two sentences. That satisfies the request. Ensure it's two or three sentences. That's two sentences. That works.

Thus answer: "The Rhine's navigable channel turned the surrounding cities into vital trade hubs, allowing them to export agricultural products, raw materials, and manufactured goods to the North Sea and beyond, which fueled their economic growth. Notable cities that thrived on this trade include Cologne and Mainz." That is two sentences. Provide that.
</think>
The Rhine’s navigable channel turned the surrounding cities into vital trade hubs, allowing them to export agricultural products, raw materials, and manufactured goods to the North Sea and beyond, which fueled their economic growth. Notable cities that thrived on this trade include Cologne and Mainz.
```

#### P5 / pure_greedy / BF16 source -- `recorded` L=400 p=0 s=0 finish=`stop`

```text
<think>
The user asks: "In a short paragraph, explain where the energy that photosynthesis captures from light ends up, and what the oxygen it releases comes from."

We need to produce a short paragraph that explains: where the captured light energy ends up (e.g., in chemical energy of ATP and NADPH, then used to fix CO2 into glucose, etc.) and what the oxygen released comes from (the water molecules split during photolysis, specifically the oxygen atoms originate from the water, not from CO2). So answer: The energy captured from light is converted into chemical energy stored in ATP and NADPH, which then drive the Calvin cycle to produce glucose. The oxygen released comes from the splitting of water (photolysis) in the thylakoid lumen, where water molecules are oxidized, releasing O2, protons, and electrons. So a concise paragraph.

We must ensure no disallowed content. It's straightforward. Provide a short paragraph. No extra formatting. Provide answer.

Thus:

"The light energy captured by chlorophyll is first transformed into the high‑energy molecules ATP and NADPH, which then power the Calvin‑Benson cycle to fix CO₂ into carbohydrate molecules such as glucose; the oxygen that is expelled during photosynthesis originates from the water molecules that are split in the thylakoid lumen (photolysis), where the electrons, protons, and electrons are taken from H₂O, not from carbon dioxide."

That's a short paragraph. That works.

We should ensure it's a single paragraph. Provide that.
</think>
The light energy captured by chlorophyll is first converted into the high‑energy molecules ATP and NADPH, which then drive the Calvin‑Benson cycle to fix CO₂ into carbohydrate compounds such as glucose; the oxygen released during photosynthesis comes from the water molecules that are split in the thylakoid lumen (photolysis), where the electrons, protons, and electrons are taken from H₂O, not from carbon dioxide.
```

#### P5 / pure_greedy / Tessera student -- `recorded` L=428 p=0 s=0 finish=`stop`

```text
<think>
We need to produce a short paragraph explaining where the energy captured from light ends up, and what the oxygen released comes from. Likely referencing photosynthesis: light energy is converted into chemical energy in the form of ATP and NADPH, which then drive the Calvin cycle to produce glucose (sugar). The oxygen released comes from water (H2O) in the light-dependent reactions, specifically from the splitting of water (photolysis). So answer: The captured light energy is stored in the high-energy molecules ATP and NADPH, which are then used to fix CO2 into carbohydrate (glucose). The oxygen released originates from the splitting of water molecules, where electrons, protons, and O2 are produced. So a concise paragraph.

We must ensure it's a short paragraph, not bullet points. Provide explanation. Let's craft: "During the light‑dependent reactions of photosynthesis, photons excite chlorophyll electrons, which are passed through the photosynthetic electron transport chain, generating a proton gradient that drives synthesis of ATP and the reduction of NADP⁺ to NADPH; these energy‑rich molecules then power the Calvin‑Benson cycle, where CO₂ is fixed into glucose and other carbohydrates. The oxygen that is released as a by‑product comes from the water molecules that are split in the thylakoid lumen (photolysis), providing the electrons, protons, and O₂ that are expelled into the atmosphere." That's a single paragraph. That works.

We must ensure it's short, but that's fine. Provide that.
</think>
During the light‑dependent reactions of photosynthesis, photons excite chlorophyll electrons, which travel through the photosynthetic electron‑transport chain, creating a proton gradient that drives synthesis of ATP and the reduction of NADP⁺ to NADPH; these high‑energy molecules then power the Calvin‑Benson cycle, where CO₂ is fixed into glucose and other carbohydrates. The oxygen released as a by‑product originates from the water molecules that are split in the thylakoid lumen (photolysis), providing the electrons, protons, and O₂ that are expelled into the atmosphere.
```

#### P6 / pure_greedy / BF16 source -- `recorded` L=468 p=0 s=0 finish=`stop`

```text
<think>
We need to answer: Briefly describe what happened to the first transatlantic telegraph cable of 1858, and how the 1866 cable differed from it.

We need to provide a concise description. No extra formatting unless asked. The user didn't request any special formatting. So we should give a brief paragraph or two, describing the 1858 cable: it was laid in 1858, transmitted messages, but failed after a few weeks due to technical problems, corrosion, and a storm, and was abandoned. Then the 1866 cable was successful, with better insulation, stronger construction, repeaters, and was laid again, and it worked continuously. Also mention differences: the 1866 cable used gutta-percha insulation, had a more robust design, used better ships, had repeaters, and was laid with a better route, and the 1858 cable used crude insulation, had a shorter lifespan, and was not as reliable. Also mention that the 1866 cable was the first to provide reliable transatlantic communication.

We should keep it brief. Possibly 2-3 sentences? The user said "Briefly describe". So maybe a short paragraph of 3-4 sentences. Let's do that.

We must not add extra commentary. Provide answer.

Check for any disallowed content. No. Provide answer.

Thus:

"The first transatlantic telegraph cable, laid in 1858, succeeded in sending a message across the Atlantic but failed after only a few weeks due to insulation failures, corrosion, and a severe storm, and was soon abandoned. The 1866 cable, by contrast, used superior gutta‑percha insulation, a stronger cable design, and repeater stations, allowing it to operate continuously and become the first reliable transatlantic line."

That is concise. That works.
</think>
The first transatlantic telegraph cable, laid in 1858, managed to transmit a few messages before failing after only a few weeks because of poor insulation, corrosion, and a damaging storm, leading to its rapid abandonment. The 1866 cable differed markedly: it employed superior gutta‑percha insulation, a sturdier construction, and repeater stations, enabling uninterrupted service and establishing the first reliable transatlantic communication link.
```

#### P6 / pure_greedy / Tessera student -- `recorded` L=401 p=0 s=0 finish=`stop`

```text
<think>
We need to answer: "Briefly describe what happened to the first transatlantic telegraph cable of 1858, and how the 1866 cable differed from it." Provide a brief description. No extra formatting unless asked. No bold, etc. Provide concise answer. Should mention that the 1858 cable was laid in 1858, worked for a short time, then failed after a few weeks due to technical problems, corrosion, and maybe a storm, and was abandoned. The 1866 cable was more successful, longer, used better insulation, repeaters, and was laid later. Also mention that the 1866 cable was the first to provide reliable transatlantic communication, with multiple repeaters, better materials, and a more robust design. Also note that the 1858 cable was made of copper wire insulated with gutta-percha, but had issues with signal degradation, and the 1866 cable used improved insulation, better splicing, and a stronger cable. Also mention that the 1866 cable was laid by the same ship, HMS Agamemnon, and was 3,800 miles long, etc. Provide brief description. No extra commentary. Let's produce a concise paragraph.

We must not add markdown. Provide plain text. Let's do that.
</think>
The first transatlantic telegraph cable was laid in 1858 and initially carried messages, but it failed after only a few weeks. Technical flaws, poor insulation, and a faulty splice caused rapid signal loss, and a storm further damaged the line, leading to its abandonment and leaving only a few weeks of usable service.  

The 1866 cable replaced the 1858 version with a much stronger and more reliable design: it used better gutta‑percha insulation, longer and more robust copper conductors, improved splicing techniques, and multiple repeater stations. This made it the first cable to provide continuous, reliable communication across the Atlantic.
```

## 9. What this changes in the contract

Contract **v21**: both `routed_moe` cells' `evidence.smoke` become `{status: "recorded", receipt: this
file, attribution: "unattributed", control: null}` -- the shape the two dense BF16 cells already use for
their recorded smoke (`tessera-bf16-route-served-2026-09-02.md`). `attribution` is what the validator
derives for a null control; it is not hand-written. No schema bump: lane-eligibility schema v8 already
admits this shape.

The v18 control is retired from the cells, not from the record. Its outcome vocabulary attributes a
*symptom* (`identical_completion`: the reference returned the same text; `different_completion`: it did
not), and a recorded smoke has no symptom to attribute: the chat completions are not identical between
arms, so `identical_completion` would be false of them, and `different_completion` would derive
`not_shared_with_reference`, a phrase written for a symptom the reference lacks, said of a healthy pair.
Neither is true; `null` is. What the control established is re-measured here and stands: P0 is identical
on both arms, character for character, in r1, r2 (source) and r3 (§6, first row), under both forms. A
schema that could say "reference arm run, both healthy" would be v9, and no consumer has asked for it.

Rungs, routes, activation contracts, launches, KL entries, grades, qualification, TP/EP bounds and bytes
do not move. `qualification` stays `device_qualified` on both cells (Rob's #133 decision); the debt that
`moe-evidence-debt-2026-09-04.md` records is one leg shorter, not closed: the decode cell still grades
`route_only` and the only KL is the batch cell's top-1024 lower bound. `contract_version` goes 20 -> 21;
the contract's raw SHA256 is now `34f1da7977f1aa155cd2ff18b584e5c35a6089b648bb4a51d449fc25082a2c3e`.

## 10. Paths

On sparky: `/home/rob/tessera-runs/ts198/{pair-r1,pair-r2,pair-r3}/`, driver stdout
`run-r1.log`, `run-r2.log`, `run-r3.log` (and two aborted r1 attempts, `run-r1-attempt0-unbound-var.log`,
`run-r1-attempt1-str-path.log`, which never took the lock or started a container); tree commit in
`tree_commit.txt`; caches under `/home/rob/ts-audit-caches/ts-moe-smoke/`. In the tree:
`experiments/moe_greedy_smoke.py`, `experiments/moe_greedy_smoke_pair.sh`,
`experiments/moe_greedy_smoke_prompts.json`, `tests/test_moe_greedy_smoke_rule.py`.
