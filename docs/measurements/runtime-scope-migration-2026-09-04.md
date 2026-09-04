# Runtime scope migration — Tessera #126

Lane eligibility schema v5 adds required `runtime.image` and
`runtime.execution_modes` to each cell. A reader must receive the same exact
manifest image and a measured execution mode to resolve an attested cell.
Missing context is unattested. The global dense image is not a fallback.

The eight existing dense cells keep the same IDs, family, rungs, launch pairs,
activation contracts, residency modes and device qualification. They now
explicitly name the existing global digest
`vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`
and both execution modes, `eager` and `compiled`. No routed-MoE cell, TP/EP
claim, wire change or quality threshold is introduced.

## Evidence retained from the dense attestation

The runtime modes are existing measurements, not new serves:

- E2M1x2: `tessera-serving-plugin-2026-09-02.md`, section 3, records resident
  and streamed eager and compiled serves and their route census/quality arms.
- E4M3: `tessera-window-gemv-served-2026-09-03.md`, section 3, records the
  same four combinations. Its compiled census is dispatch evidence; section 4
  separately records profiler launch evidence for the streamed GEMV arm.
- BF16: `tessera-bf16-route-served-2026-09-02.md`, section 2, records four
  resident/streamed by eager/compiled censuses, all 112 modules in both
  token-count regimes, and section 3 records the served quality comparison.

Those historical census headers carry vLLM/torch and execution/residency
fields, but not individual image-digest fields. Their image association is the
existing global attestation plus the contemporaneous RepoDigests observations
in `tessera-serving-plugin-2026-09-02.md` (the two-GB10 image note) and
`tessera-ldlq-window-served-2026-09-02.md` (the served image manifest). This
migration does not rewrite those files to imply each was digest-stamped.
New census receipts must record explicit invocation image and execution mode.

## Refusals and test evidence

The validator requires a manifest pull reference, using the existing runtime
image parser's grammar; a floating tag, local image ID, noncanonical digest
or trailing newline is refused. Execution modes are a nonempty, distinct list
drawn from `eager` and `compiled`. The runtime object is closed to unknown keys.
Overlap includes image and execution mode; disjoint runtime scopes may coexist
but IDs remain unique. A scope-derived optional ID suffix hashes the canonical
image and complete execution-mode set. Explicit fields decide eligibility.

PrismaBuild pre-fix action
`465f4a6427f2f8a673c4bb1e35028b152446d4894cc13780e84ee42cd6c8cebc`
ran all 26 new cases against the old contract: all 26 failed. The distinct
failure lines were:

```text
test_every_published_cell_names_its_measured_runtime:
AssertionError: 'tessera.lane-eligibility.v4' != 'tessera.lane-eligibility.v5'
test_runtime_image_requires_an_exact_manifest_digest:
AttributeError: module 'tessera.serving.contract' has no attribute 'require_runtime_image'
test_cell_runtime_scope_uses_explicit_image_and_canonical_mode_order:
AttributeError: module 'tessera.serving.contract' has no attribute 'cell_runtime_scope'
test_cell_runtime_execution_modes_are_explicit_nonempty_and_unique:
AttributeError: module 'tessera.serving.contract' has no attribute 'cell_runtime_scope'
test_cell_runtime_scope_is_a_closed_object:
AttributeError: module 'tessera.serving.contract' has no attribute 'cell_runtime_scope'
test_a_cell_may_not_borrow_the_global_dense_image_pin:
Failed: DID NOT RAISE ValueError
test_runtime_variants_have_disjoint_scopes_and_unique_ids:
AttributeError: module 'tessera.serving.contract' has no attribute 'cell_runtime_id_suffix'
test_different_runtime_scopes_cannot_reuse_one_cell_id:
AssertionError: Regex pattern did not match. Expected regex: 'repeats.*id'
test_runtime_id_suffix_is_derived_from_the_entire_scope:
AttributeError: module 'tessera.serving.contract' has no attribute 'cell_runtime_id_suffix'
test_validator_reads_runtime_scope_instead_of_accepting_prose:
AssertionError: Regex pattern did not match. Expected regex: 'digest reference'
```

Initial green action
`a49904dca0a1cd7b0705fc487e366d88e58ffc1861d63f19791a8135fe1fd6c0`
ran `test_serving_runtime_scope.py`, `test_serving_contract.py`, and
`test_runtime_image_pin.py`: 87 passed in 1.40 seconds; receipt
`33376545b7e5409a3b969439da34c262b65cd97c7a82eb7648e078151d90f456`.
Both populations used dl380g10, serial CPU, with:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

This is producer contract coverage. The coordinated census and PrismaQuant
reader changes carry their own evidence; none of these tests is a GPU serve.

The final targeted action added explicit preservation assertions for the
dense image and execution-mode roster:
`ce35d0d79117a1b5997ee74beab4a080e33b9b56fe82adb9e7b75e7d58d54508`.
It again reported 87 passed, 0 skipped and 0 uncollected modules in
1.39 seconds on the same serial CPU population. Receipt:
`ecd50ecc66d8a09dc75110a18a96d9feb42eb84c11fbebaa7ace51ea4e7b54a9`.

## Selected reverse-reachable coverage and fixture repair

Selector action
`10b239becb8616477ae8ec4e87103ba0120788f8a2dd66caeec6d4ad2cd2dff4`
compared the immutable snapshot against fetched
`6d0b5e7f5c0acf159d775c3efd6903d899a77a83` and selected 48 files with verdict
`narrowed`. The three files above and the two census files covered by the
coordinated census change were excluded from the remaining 43-file action
`d184116c23634215d5ada74b5d74e0e739f0d6df39f9868baf20dd5278fd6988`.
Its completed first attempt was **red**: 590 passed, 2 failed, 301 skipped,
0 uncollected modules, 14 warnings, in 360.82 seconds. The producer snapshot
was `de243d4c2ce9030713fcb25ac579f86b8b1ecbb5`; the population is
`/mnt/shared/tessera-runs/ts5/lfm25/astra-v5-contract-impact-r1.surface.json`.
The subsequent retry was withdrawn through PrismaBuild after reading that
completed attempt, not represented as a successful action.

Both failures were old `test_lane_reachability.py` fixtures reaching the new
unique-ID refusal before the gate each test intended to exercise:

```text
test_a_cell_may_not_claim_the_lane_in_the_residency_that_has_none:
tests/test_lane_reachability.py:438: AssertionError: Regex pattern did not match.
Actual message: "runtime_contract.lane_eligibility.cells[3] repeats cell id 'tessera_e4m3_k1_dense_sm121_decode_resident'; every cell has one identity"
test_two_cells_may_not_cover_one_residency_of_one_regime:
tests/test_lane_reachability.py:456: AssertionError: Regex pattern did not match.
Actual message: "runtime_contract.lane_eligibility.cells[8] repeats cell id 'tessera_e4m3_k1_dense_sm121_decode_resident'; every cell has one identity"
```

The launch test now isolates its mutated cell, and the overlap test assigns
its twin a distinct, valid runtime-derived ID. Thus both still demand the
specific launch/semantic-overlap refusal rather than weakening their assertion
to any earlier error. No producer validation was relaxed.

Only the failing file was run against the pristine prior commit:
`08d4966fc9081c1abd1038c3c16e4f48d5bb279c2f2441eabb687afa4c54b122`,
42 passed, 0 skipped, 0 uncollected modules in 1.21 seconds; receipt
`dd10c4aa4f149a44f6f37517b90d19a423409456e590c064c4dbee34fc072edf`.
An earlier baseline invocation changed its sealed checkout and was correctly
refused by PrismaBuild's closure verification; it is not counted as a green
action. The quoted baseline instead used a separate pristine worktree.

The repaired file passed on the assembled v5/census/image-gate tree:
`3bf4bbda4a1c1d46327a7ed18791aefcb6d00a79368bba482a7fe246e5025181`,
42 passed, 0 skipped, 0 uncollected modules in 1.25 seconds; receipt
`ba890a1e58e4626aba47736cf18fe40b4b3c4983d260e904a8618ab5bda3573d`.
This targeted repair does not turn the earlier red population into a green
whole-run result. Merge-wide verification remains the coordinator's run.

All these populations were serial dl380g10 CPU, torch 2.11.0+cpu, with no CUDA
device. The 43-file run's skip reasons, verbatim, were:

```text
81  the encoder is a CUDA path
79  needs a CUDA device
29  the Viterbi is CUDA
24  the kernel lane is a CUDA path
23  the lane is a CUDA kernel
15  the encoder is a GPU job
14  the Tessera encoder is a CUDA path
6   /home/rob/tessera-runs/compile-dispatch is not on this box
6   needs CUDA
5   the kernel lane runs on CUDA
5   the reach checkpoint is not on this box
4   Qwen3-0.6B is not on this box
2   no stock twin
1   /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box
1   /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box
1   E2M1 publishes no reader range
1   could not import 'vllm': No module named 'vllm'
1   the PrismaQuant tree or the allocation outputs are not on this box
1   the PrismaQuant tree with tessera_formats is not on this box
1   the shipped checkpoint is not here
1   the two surviving compile caches from 2026-09-02 are not on this box
```

## Exact-image wrapper boundary after assembly

The assembled tree combines the explicit-image resolver fix with the census
wrapper that injects the selected image. Action
`20cf2a19e60c7b8b52b9acef355cff827f3d90bb93a6d0ee1f25a1a733978c88`
ran `test_census_runtime_wiring.py` and `test_runtime_image_pin.py`: 27 passed,
0 skipped, 0 uncollected modules in 1.48 seconds; receipt
`e4b2d82c41f7c91fde7dc497b1853d5fd3da23824704f7d0beae4539349e7f5c`.
It was the same serial, device-less dl380g10 CPU population, not GPU or served
quality evidence. No routed-MoE cell has been promoted by this migration.

The fixture repair's own selector action
`9ecc7ea39da7ddf9bff7d37f773e7230d999f0b124b5ce40dbc65e552b7b7669`
compared against fetched `7f7ebd2f12637cedb7fb9df4d93b57625809d39a` and
returned `narrowed` with only `tests/test_lane_reachability.py`, the 42-pass
file reported above. Its temporary pristine baseline worktree was removed
after the completed baseline receipt; no source branch was deleted.
