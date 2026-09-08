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
