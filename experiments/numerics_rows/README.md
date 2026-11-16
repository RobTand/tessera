# Numerics rows

Build against draft 10. The specification files stay outside this checkout.
Do not submit the prepared jobs until the lead gives Fleet OK.
Do not run a row before the lead supplies the frozen rule record.

Run `python3 -m experiments.numerics_rows.prepare_job` to prepare a job directory.
The command copies a Git repository and writes a job script. It submits nothing.
Each job checks the pinned kernel hash before it runs the Python command.

Use this order:

1. P-G and P-N build the libraries and probes. They save SASS, ELF sections, flags, and binary hashes.
2. Q1-G and Q1-N read the P receipts. They save runtime quantizer bytes and probe tables.
3. H1-G and H1-N read the same P receipts. They check the preconditions before the freeze.
4. C1-G and C1-N read a population JSON and the corresponding Q1 receipt.
5. CPU runs syntax checks, selected pytest files, and Bazel metadata generation.
6. Rows read all the receipts and the frozen rule record.

A population JSON has `sms` and `modules`. Each module has these fields:

- `root`: checkpoint directory.
- `name`: exact target in its `config.json`.
- `library`: `e4m3mma`, `e4m3`, `e2m1`, `value`, or `triton`.
- `contract_family`: the format family from v68.
- `pair`: the exact symbol and decoder.
- `selector_env`: the arm's selector values, for the receipt.

A canonical-cache entry also has `wire_root`, `wire_family`, `layer`, and optional `scales`.
Dense cache entries have `wire_path` and a checkpoint `scheme_template` of the same family and shape.
`wire_family` uses `routed_pair_oracle.load_wires`' existing payload and rung fields.
C1 writes separate reader weights and recipes for TP1 and both TP2 cuts.
The row recomputes separation and joint-read coverage before the native scale tests.
The population stamp job fills `wire_stamps_read`. The lead supplies the observed SM count as `sms`
and sets `reviewed` to true after review. C1 and rows refuse drafts.

The frozen rule record has `sha256`, `version`, `frozen_utc`, `reviewer`, `decisions`, and `preconditions`.
The row checks its hash against the supplied rule file.

P-G and P-N built the libraries. Q1 passed on both images. The lead holds those job results.
H1, the P3 matcher and the review-fix CPU checks still need their recorded verdicts.
The SASS matcher refuses branch joins with unknown reaching definitions.
Such a refusal leaves routed units UNATTESTED. It does not prove a slice mismatch.
The CPU job writes `checks.json`, logs, population JSON, and a metadata patch.
Read each exit status and log before using any result.
Open questions are in [questions.md](questions.md).
