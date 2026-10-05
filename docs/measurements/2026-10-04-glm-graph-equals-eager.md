# GLM-5.3 under CUDA graphs computes what eager computes (tessera#702)

Status: measured 2026-10-04 on sparky and sparklina (GB10, sm_121), TP 1, u1
stub B (8 layers) with and without its MTP layer, release image `5be13705`,
release `max_model_len` 8448. Refs tessera#702, prismaquant#1586, tessera#508,
tessera#695.

## Result

With Tessera at `ba663a6b`, every graph arm is a member of the eager pool on
all 48 choices of both passes of the tessera#508 equality set, and replayed
every graph it captured. The same configuration served from master
(`13e41726`) is 0 of 48.

| Arm | `compilation_config` (as given) | Captured sizes (target; draft) | Pass 1 | Pass 2 | Replays, class ≤ 2048 / class 8448 |
|---|---|---|---|---|---|
| rE1, rE2 | `--enforce-eager` | none | 48/48 | 48/48 | - |
| rG1 | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | 1..8 (pinned) | 48/48 | 48/48 | 1128 / 201 |
| rG2 | FULL_DECODE_ONLY, sizes 1..8 | 1..8 | 48/48 | 48/48 | 1128 / 201 |
| rP1 | `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` | 1..8 (pinned) | 48/48 | 48/48 | 1128 / 201 |
| rG3 | `{"mode":"NONE"}`, FULL_DECODE_ONLY, sizes 1..8 | 1..8 | 48/48 | 48/48 | 1128 / 201 |
| rG4 | `custom_ops ['all']`, eager IR priority, FULL_DECODE_ONLY, sizes 1..8 | 1..8 | 48/48 | 48/48 | 1128 / 201 |
| rGR | `{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}` (the release serve's flags) | 1..8 (pinned) | 48/48 | 48/48 | 1128 / 201 |
| mtE1, mtE2 | `--enforce-eager`, MTP k = 1 | none | 48/48 | 48/48 | - |
| mtG1 | FULL_DECODE_ONLY, MTP k = 1 | 2..16 even; 2..16 even (pinned) | 48/48 | 48/48 | 1118 / 197 |
| mtG3 | `{"mode":"NONE"}`, FULL_DECODE_ONLY, sizes 1..16, MTP k = 1 | 2..16 even; same | 48/48 | 48/48 | 1118 / 196 |
| mtGR | release flags, MTP k = 1 | 2..16 even; same (pinned) | 48/48 | 48/48 | 1118 / 195 |
| **rX3 (master 13e41726)** | as rG3 | 1..8 (stock capture) | **0/48** | **0/48** | stock: one class |

Each stub-B graph arm is judged against rE1 and rE2 (both passes each); each
MTP graph arm against mtE1 and mtE2. The eager serves reproduce each other
exactly (48/48 across serves). A choice is a member when its token ids and
every top-20 logprob list are bit-identical to the same choice of some eager
run of the same batch (`experiments/glm53_508_graph_qual/eq-member-508.py`).
Replays are the plugin's own counts (`glm53_graphs.REPLAYS`, read by the
dispatch hook); the MTP rows count the target and the draft manager, which
replayed equally.

Receipts (schema `tessera.graph_equals_eager.v1`, verdict `equal`):
`/mnt/shared/tessera-measurements/graph-attest-702/receipt-stubB-ba663a6b.json`
and `receipt-stubB-mtp1-ba663a6b.json`; raw arms under `final-ba663a6b/`.
PrismaBuild actions (one GPU action per arm, `--exclusive`): rE1 `0faadd06c573`,
rE2 `4cf2a398eff5`, rG1 `62c0d228d709`, rG2 `12267f247d3b`, rP1 `4d53ad8d31fe`,
rG3 `fe049997ad0f`, rG4 `b8a510ef479a`, rGR `acb6d93fb134`, mtE1 `19c2594fed12`,
mtE2 `b890ff64ab4f`, mtG1 `84a6dcbf76e8`, mtG3 `28b7ccbd941a`, mtGR
`d40aa5350f29`; rX3 `fe5cc70e347f` (under `final-33c7b727/`).

## Three causes, three remedies

The 2026-09-30 page measured causes 1 and 2. Cause 3 was hidden behind them
until both were removed.

1. **Operators.** With `enforce_eager=False` vLLM defaults to `VLLM_COMPILE`,
   resolving `custom_ops ['none']` and `native` norms. The plugin fills the
   unset fields with eager's (`custom_ops all`; the IR priority vLLM's own
   `KernelConfig.set_platform_defaults` resolves at mode NONE) from the
   Glm5Next `MODELS_CONFIG_MAP` hook, before vLLM resolves either default.
   rG1 and rP1 (default compile mode) are now members from step 0, where
   sbG1 and sbP1 departed at step 0.
2. **The indexer's frozen branch.** A FULL capture builds metadata at
   `max_seq_len = max_model_len`, freezing GLM's `max_seq_len <= index_topk`
   host branch at its top-k side. The plugin captures one class of FULL
   graphs per side of `index_topk` (bounds 2048 and 8448), each at its own
   upper end, and replays the class holding the step's own `max_seq_len`.
   rX3 (stock) is 0/48 with the first departure at decode step 1; rG3 (the
   same configuration with the plugin) is 48/48.
3. **Padded replay (new).** vLLM's default capture sizes `[1, 2, 4, 8]`
   replay a batch of 5 or 7 in the size-8 graph. With causes 1 and 2 removed
   (Tessera `33c7b727`), rG1, rP1 and rGR were 46/48 in both passes: exactly
   b5 choice 4 (token ids from step 7) and b7 choice 6 (from step 6), the
   same in all three arms, while rG3 and rG4, which capture every size, were
   48/48. The plugin now captures every decode count `n * (1 + k)` where the
   serve named no sizes, and refuses explicit sizes that leave a padded
   decode count. That run's arms are under `final-33c7b727/` (built from
   Tessera `33c7b727`, so no verdict of the final tree reads them).

**Fail closed, measured.** rZ1 (PrismaBuild `3bd42f9610a3`) served explicit
sizes `[1, 2, 4, 8]`. The serve refused at startup with: `tessera.glm53_graphs:
this Tessera GLM-5.3 CUDA-graph serve would not compute what its eager serve
computes, and Tessera refuses it: decode batches of [3, 5, 6, 7] tokens replay
a larger captured graph, which is another computation ...`.

## The long-context screen

Every arm also ran seven cases whose longest row passes `index_topk` (2100 to
8000 tokens), so the 8448 class replayed (195 to 201 times per arm). These
are a screen, not part of any verdict: above `index_topk` eager does not
reproduce itself. rE1 answered three identical 4000-token requests with three
different digests, differing from the first sampled token, which comes from
prefill (stock top-k writes its selection in arrival order). Graph arms land
on the same token trajectories eager does, and are members on 3 of 11 long
choices, as each eager serve is against the other.

## Scope: what this attests and what it does not

Attested, per receipt and exactly: image `5be13705`, stub B (config sha256 in
the receipt), Tessera source of `ba663a6b`, each arm's `compilation_config`
as given, MTP k (0 or 1), `max_model_len` 8448, `max_num_seqs` 8, TP 1.
`tessera.graph_receipt.verify` matches every one of those fields and
extrapolates none.

Not measured:

- TP 2, the full GLM-5.3 artifact, and the release `max_num_seqs` 4. A ship
  card verifies only against a receipt produced from its own serve.
- Served KL against BF16 under graphs; graph-vs-eager speed; capture time and
  graph pool memory with two classes (each class adds one set of FULL graphs;
  the plugin scales vLLM's profiler estimate by the class count).
- `max_model_len` above 32768, more than 64 decode rows per graph, k >= 2,
  PP/DP/context parallelism, LoRA, breakable piecewise graphs: refused by
  name, not measured.

The arm harness gained its runtime-image gate (`runtime_image_require`)
after these arms ran; they declared the image to PrismaBuild by digest and
record its local id in `engine-args-<arm>.txt`.

## Clarification, 2026-10-05: the v1 receipts and `verify` (tessera#973)

The two receipts above carry schema `tessera.graph_equals_eager.v1`, which names
no fabric. Their recorded evidence stands as written: the receipts and the raw
arms were not regenerated, and nothing above was re-measured. What changed is
the checker. Since tessera#942 (receipt schema v2, the fabric is scope),
`tessera.graph_receipt.verify` **refuses a v1 receipt for a card**, with the
reason "a `tessera.graph_equals_eager.v1` receipt names no fabric, so it cannot
attest a card's serve; produce a `tessera.graph_equals_eager.v2` receipt". So
the sentence in "Scope" that `verify` matches every one of those fields
describes the v1-era `verify`, not the current one. A v1 receipt stays readable:
`finish` still re-derives it. `tests/test_graph_receipt.py::
test_a_v1_receipt_stays_readable_but_verifies_no_card` pins both halves.
