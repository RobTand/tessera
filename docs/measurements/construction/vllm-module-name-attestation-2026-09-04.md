# `vllm_module_name` against vLLM's own `WeightsMapper`

**What was measured.** Every name the producer's replay of
`WeightsMapper._map_name_with_shard` can be asked about, put through the real
`WeightsMapper` inside the pinned serving image and through
`tessera.serving.contract.vllm_module_name`, and compared.

**Why it needed measuring (issue #108).** `vllm_module_name` translates a
checkpoint module name into the namespace vLLM builds it under, so the
construction census — which observes vLLM-namespace prefixes — can be joined
to a plan written in checkpoint names. It is the one place in this repo that
computes what vLLM *would* do rather than reading what it *did*, and principle
14 has no exemption for "the algorithm is code, not a table". It cannot be
derived: vLLM publishes the rename table (the census reads it off the model
class) and keeps the loop that consumes it. So the gap is closed by
attestation — the same names through both, with any disagreement a failure.

**Runtime:** vLLM **0.28.0**, torch 2.13.0+cu130, NVIDIA GB10 sm_121, in
`vllm/vllm-openai:latest`
(`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`) —
the same image `docs/measurements/construction/qwen3-0.6b.json` was censused
in. Submitted through the PrismaBuild pool (`pbrun --gpu --cpus 4`), executed
on sparky.

## The three divergences, and what they cost

Diffed against vLLM 0.28 `model_executor/models/utils.py`:

| | vLLM | before #108 | after |
|---|---|---|---|
| substring replacement | `key.replace(substr, new, 1)` — once | every occurrence | once |
| prefix loop | falls through | `break`s on first match | falls through |
| suffix loop | falls through | `break`s on first match | falls through |
| `orig_to_new_renaming` | applied | silently ignored | **refused by name** |
| `orig_to_new_regex` | applied | silently ignored | **refused by name** |
| `orig_to_new_stacked` (populated) | applied | silently ignored | **refused by name** |
| a field vLLM adds next | applied | silently ignored | **refused by name** |

Neither committed census exercises any of them: GLM-5.3-Flash uses
`orig_to_new_prefix` with three non-overlapping rules and Qwen3 declares no
mapper at all. That is why it went unseen, and it is why the attestation ships
synthetic tables alongside the receipt tables — a shape no artifact currently
has cannot be attested by an artifact.

## Result

**In the pinned image:** `tests/test_serving_name_mapping.py` — **15 passed**,
vLLM 0.28.0, sparky, pbrun action `4e6a943d0559`, 16 s. The replay and the
real `WeightsMapper` agree on every probe name of every table: **232 names
across 9 tables** (7 synthetic, 2 committed receipts).

That run predates commit `e27d56a`, which moved `SYNTHETIC_TABLES`, `LEAVES`
and the name generator out of the test file into `tests/mapper_probes.py` so
the distance harness could share them. `src/tessera/serving/contract.py` is
byte-identical between the run and `HEAD`, and the refactor generates the same
232 names for the same 9 tables — checked by execing the pre-refactor
generator out of `git show 056bb5f` and comparing set-wise. So the receipt
covers the shipped logic; what has not been re-run in-image is the file
layout.

**How far master was from the runtime.** The in-image arm establishes
`vllm_module_name(this branch) == WeightsMapper`, so master's distance from
the runtime is master's distance from this branch, which needs no image to
measure. Over the same probe set:

| table | names where master ≠ the attested replay |
|---|---|
| `substr_twice` | 2 |
| `prefix_chain` | **22** |
| `suffix_chain` | 4 |
| `substr_then_prefix_then_suffix` | 2 |
| the three dropping tables | 0 |
| `glm53-flash-4layer.json` (committed) | **0** |
| `qwen3-0.6b.json` (committed) | **0** |

Reproduce with `experiments/vllm_module_name_distance.py master` (no image,
no GPU; it execs the named revision's function out of `git show`).

e.g. `model.` under `{"model.": "language_model.", "language_model.": "lm."}`
→ master `language_model.`, runtime `lm.`; and
`model.layers.0..block.mlp..block.down_proj` under a single substring rule →
master rewrites both occurrences, the runtime rewrites one. The two zeros on
the committed receipts are the issue's "it fails closed today", measured
rather than argued.

The pure-suite half, which needs no image, is
`tests/test_serving_construction.py`. On master `766033c` it fails 7 of its 9
new cases; the three semantic ones fail like this:

```
assert 'model.layer.0.layer.1.proj' == 'model.layer.0.block.1.proj'   # substr twice
assert 'language_model.layers.0.mlp.down_proj' == 'lm.layers.0.mlp.down_proj'  # prefix chain
assert 'model.layers.0.b_proj' == 'model.layers.0.c_proj'             # suffix chain
```

and the four refusal cases fail as `DID NOT RAISE ValueError` on
`orig_to_new_renaming`, `orig_to_new_regex`, a populated `orig_to_new_stacked`
and an invented field. After the fix: **22 passed**.

## What is still queued, and why it is not load-bearing

An in-image **master arm** — master's `contract.py` under the shipped
attestation, expected to fail — is queued in the PrismaBuild pool as action
`282a16d71082` and had not been picked up when this was written: sparky's
queue stood at 37 ready / 6 claimed, a fresh `--gpu` submit was refused with
`no live worker can run this action`, and sparklina's GPU was held by an
out-of-pool encode. That is a contended box, not a blocked measurement.

It is not load-bearing because AGENTS.md rule 8 ("show the test failing before
the fix") is already met twice over, without an image: the pure suite fails 7
of its 9 new cases on master with the lines recorded above, and the
`experiments/vllm_module_name_distance.py` table counts master's disagreement
with the attested replay name by name. The queued arm would restate that
inside the container.

## The census was the deeper hole

`tools/tessera_construction_census.py::_weights_mapper_table` iterated a
hardcoded four-name roster (`substr`, `prefix`, `suffix`, `stacked`). Fed a
mapper declaring `orig_to_new_regex`, `orig_to_new_renaming` and an invented
field, master's census returns:

```
MASTER census table: ['orig_to_new_prefix']
```

— the rules are dropped on the way into the receipt, so a producer reading
that receipt maps the name as though they were not there and **no refusal on
the producer side could ever fire**. Narrowing `vllm_module_name` alone would
not have closed #108; it would have moved the silence one file upstream. The
census now reads `dataclasses.fields` off the runtime's own mapper (AGENTS.md
rule 3), so every non-empty field lands in the receipt where the producer's
refusal can see it. The refused fields are serialised lossily on purpose —
`orig_to_new_regex` keys as `{regex, flags}`, a renaming by `repr` — because
presence, not fidelity, is what a refusal reads.

The receipt *shape* is unchanged for a mapper using only the replayed fields,
which is both committed censuses: `test_the_census_still_produces_the_tables_
the_committed_receipts_carry` reconstructs GLM-5.3-Flash's table from the
committed receipt and asserts byte equality. Neither census needs re-running.

## What this does and does not cover

* It covers `orig_to_new_{substr,prefix,suffix}` — the fields the producer
  replays — over the committed receipt tables and seven synthetic tables,
  including three that drop a name.
* It does **not** implement `orig_to_new_renaming` or `orig_to_new_regex`. A
  checkpoint whose model class declares one gets a refusal at export naming
  the field, not a mapped name. Implementing them is a follow-up with its own
  attestation, and it needs a third architecture to exist before it is worth
  anything.
* It is a **producer-side** gate. The serve side never had this bug:
  `TesseraConfig.apply_vllm_mapper` calls the real
  `hf_to_vllm_mapper.apply_list`, which is vLLM's own code running in vLLM's
  own process.
