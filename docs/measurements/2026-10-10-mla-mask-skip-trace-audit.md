# Mask-skip per-rank trace audit tool, 2026-10-10 (Refs #1177, part of #812)

No served numbers yet. This page records the audit tool that the TP2
measurement will run, and the evidence that the tool itself is correct.

## What landed

- `src/tessera/serving/mla_sparse_sm120.py` names its dispatch words as
  module constants (`MASK_SKIP_POLICY`, `MASK_SKIP_KIND`,
  `MASK_SKIP_CONTRACT`, `NATIVE_DECODER`, `STOCK_DECODER`,
  `NATIVE_SYMBOL`, `STOCK_SYMBOL`, `NATIVE_SCHEDULE`, `STOCK_SCHEDULE`).
  `_emit` and its call sites name the constants. No behavior changed.
- `tools/tessera_attest.py` gains `audit_mla_mask_skip_traces(paths,
  world_size=...)` plus `--mask-skip-traces/--expect-world-size/--audit-out`.
  The record `tessera.mask-skip-audit/1` passes only when every rank shows a
  `native_mg_mask_skip` launch and a `stock` fallback launch under
  `mla_mask_skip_eager_dispatch`, with exact rank/world coverage and a trace
  that counts no graph replays and no compile tracing. Any other input is a
  named problem and fails the record. No kernel code changed.
- `tests/test_attest_mask_skip_traces.py` covers the pass case, five
  fail-closed cases and a source-level pin of every audited word to the
  override that emits it (no vLLM import; the CPU pool has none).

## Tool evidence (PrismaBuild, `--tag x86`)

- Red pre-fix: `170c20a8afaaf33bbbb14cc7b177593bbabccb2406ac26a09046b33c5e090899`,
  `7 failed`, `AttributeError: ... has no attribute 'audit_mla_mask_skip_traces'`.
- Green post-fix: `a3bd47914c383aabccaf8b445881508895f16b7c83f44f5a30e234f60908ce90`
  (`7 passed`), `5ab7f3d17137424373deee60f80c499d128bfe25bf2edcc8a6a44fac243ecd48`
  (`test_attest_receipt.py`, `23 passed`),
  `47020b6e8140313f76ac4d20a0d34fa9b98c3072e4ca1547182b515dda4023fd`
  (`test_route_trace.py`, `25 passed`).
  `test_serving_mla_mask_registration.py` skips: no vLLM on the CPU worker.

## Pending serve (not run)

A CPU-tagged probe action for the GB10 prerequisites never placed
(`81ec0992...`, still `waiting` after 90 s, withdrawn). A TP2 serve is two
hosts with the release configuration plus the flag; it is not a single pool
action from the control seat. The serve recipe for the follow-up:

- Baseline VB1770 configuration (noSpec, 1 GiB KV per rank, mode
  NONE with FULL_DECODE_ONLY, TP2, MNBT2048), plus
  `TESSERA_RESEARCH_MLA_MASK_SKIP=1` and per-rank `TESSERA_ROUTE_TRACE`.
  Separate reservation and separate serve from k2; never share one GPU.
- Then: `python3 tools/tessera_attest.py --mask-skip-traces
  rank0.json,rank1.json --expect-world-size 2 --audit-out audit.json`.
  The record must read `passed=true` with `native_mg_mask_skip` launches on
  both ranks and `stock` fallback launches beside them.
