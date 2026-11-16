# Numerics rows

Build against the lead's current draft. The specification files stay outside this checkout.
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

P3 can run in a CPU job after H1 completes. Combine the receipts with
`python3 -m experiments.numerics_rows.h1 --gpu-receipt H1_JSON --p3-receipt P3_JSON --out OUTPUT_DIR`.
This command checks the library and probe hashes, keeps both input receipts, and runs no CUDA work.
Use its `h1.json` for the freeze and rows. A missing or incomplete CPU proof cannot confirm H1.

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
and signs the population after review. C1 and rows require the review block, verify its source hash,
refuse changes to the reviewed content, and record the signed file's SHA-256.
`population inspect` reads framing and manifests with bounded file or safetensors slices.
It records manifest hashes and declared payload hashes. It does not verify payload bytes.
C1 and the rows still verify and hash the complete wires.
Each module reports its path, wire size, metadata bytes, time, process I/O and peak memory.
`population profile --arm ARM` measures one wire through both paths and saves CPU profiles.
Use `--arm ARM` with `inspect` to split the population into separate jobs.

The frozen rule record has `sha256`, `version`, `frozen_utc`, `reviewer`, `decisions`, and `preconditions`.
The row checks its hash against the supplied rule file.

The active target is A8S E4M3 on IMG-N. Its unsigned populations have 14 dense and 42 routed modules.
Use `population draft --a8s-tsv PATH` to read the artifact root from the lead's real-gap table.
Inspect and sign the dense population first. Routed rows wait for the contract pair fix.
Keep the population tools for later extensions. E2M1, BF16 and IMG-G E4M3 rows are on hold.

Builds add `--objdir-as-tempdir` to the cell flags and retain `-lineinfo`.
CPU job `f2rST9Aa4Dd` found identical e4m3mma, e4m3 and value libraries at both commits and on repeat.
Measured and served builds must use the same source and build paths. Read the
[tested recipe](/mnt/shared/flux-jobs/tessera-1717-SO-REPRO-d5976c1d-20261116T154120Z/results/comparison.json).
Q1 and H1 may use the retained 8bf496ab build before comparison. A P3 tool limit remains UNATTESTED.

C1 profile f2qwcb5HuDy completed in 765.44 s. Scale cases took 745.30 s; 2,829 `argwhere` calls took
514.88 s. C1 now selects the first nonzero coordinate once per group. The exact joint evaluator stays.
The A8S before/after CPU profiles must show equal case data before any speed claim or time estimate.
Read each exit status, profile, Netdata series and log. Open questions are in [questions.md](questions.md).
