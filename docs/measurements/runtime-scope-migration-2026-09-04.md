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
