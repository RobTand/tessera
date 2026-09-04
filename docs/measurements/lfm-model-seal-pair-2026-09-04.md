# Explicit LFM served-artifact/seal pair — 2026-09-04

Source: `6cf952f61a52d3afb33d8499728cfa5a80ca12e6`, based on
`4fa03667310b752cf6324938683cc1ac2e152e9b`.

The served campaign driver accepts `--model PATH --seal PATH` together while
preserving its original defaults. A seal whose `checkpoint` does not exactly
name the selected model is refused before hashing model files or launching a
container. The original pre/post full checkpoint-identity checks remain.
Receipts retain the selected paths and the seal hash. This change selects a
separately prepared artifact; it does not modify either model or seal.

## Red before green

All execution used deployed v1 PrismaBuild on `dl380g10`: one CPU, 4 GB memory,
OMP/MKL/OpenBLAS threads fixed to one, 600-second action timeout. No GPU action
was submitted. Every population below reports torch `2.11.0+cpu`, no CUDA,
zero skipped tests, zero uncollected modules, and therefore no skip reasons.
These runs do not cover the CUDA-gated surface.

The first red submission exposed a test-fixture stub shadowing error. That
fixture was corrected, then the same final tests were submitted against the
unchanged base in a separate worktree. Only the corrected run is the red proof:

- Action `cf4abc8e993de61428b16fc2939abe47784422cac5131df0da3f69a8575f5e3c`:
  **5 failed, 1 passed** in 1.30 seconds. The complete override and both partial
  override cases failed with `SystemExit: 2` / `unrecognized arguments`.
  The absent and wrong checkpoint cases failed at test line 77 with
  `Failed: DID NOT RAISE ValueError`. The preservation control already passed:
  different checkpoint bytes were refused by the existing identity check.
- Action `e70dfcd054f4b258f993d3f2b1b3ef905a7354f0172f4da733c04a1ce27bb125`:
  **22 passed** in 1.24 seconds across `test_ts5_bound_model_seal.py`,
  `test_ts5_stage_attempt.py`, `test_ts5_bound_collision.py`, and
  `test_ts5_stage_cleanup.py`.
  Receipt: `8a9a1ef3c72dd621a2359a0b16997305599c533bc44215a2b7f41096d3ab7860`.

Population files under `/mnt/shared/tessera-runs/ts5/lfm25/`:

- `ts5-bound-model-seal-red-r2.surface.json`
- `ts5-bound-model-seal-green.surface.json`

The tests execute the driver's real CLI setup and prelaunch statements via AST
with source-identity and subprocess boundaries replaced; they never start a
container. This is argument/binding and cleanup proof, not served evidence.

## Required selector

Action `d8004c729cce1763ac224a76b7eae57209979d59d0b917a12101d3788e72f2d4`
fetched the exact base and ran `tools/impacted_tests.py --ref FETCH_HEAD...HEAD
--json` in this branch's sealed snapshot. It reported `narrowed`, using direct
tree comparison because the snapshot has no merge base, and selected the new
model/seal test file. Receipt:
`24b2b4107938df1b84b5d2864582d19268c01a8c55af132325e342a9e354f6fa`.

It omitted the existing AST-executed attempt/collision script consumers. This
is recorded on existing Tessera #129 and sent to its assigned selector owner;
the explicit 22-test green run above includes both omitted files. No selector
changes were folded into this bounded campaign patch.
