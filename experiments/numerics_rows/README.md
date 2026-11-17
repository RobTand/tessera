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
- `contract_family`: the format family from the bound source contract.
- `cell_ids`: the sorted full IDs of that contract's admitted decode and batch cells.
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

The frozen rule record has `sha256`, `version`, `frozen_utc`, `reviewer`, `decisions`,
`preconditions`, and `launch_levers_sha256`. Decisions include D1 through D9.
The row checks its hash against the supplied rule file.

The active target is A8S E4M3 on IMG-N. Its unsigned populations have 14 dense and 42 routed modules.
Use `population draft --a8s-tsv PATH` to read the artifact root from the lead's real-gap table.
Inspect and sign the dense population first. Routed scope includes the IMG-N resident decode and batch cells at R1024.
Keep the population tools for later extensions. E2M1, BF16 and IMG-G E4M3 rows are on hold.

P uses the serving flags without additions. Its retained source tree stays at its original path and mtime.
Pass one full Tessera commit to `prepare_job --source-revision`; every job checks it before its payload.
P takes `--revision`; C1 and CPU commands take `--source-revision`. Q1 and H1 check P's recorded commit.
The clean A8S campaign reruns P, Q1, H1 and C1 at the decode-once default fix commit F.
The lead then supplies a rule v2 freeze record for the new H1 and the frozen lever file.
From that freeze through the last counted row, every job uses F even if master moves.
Use `build --existing-build BUILD_JSON` with the usual source, image and output arguments to write `p.json`.
It adds code identities without rebuilding or changing the input. The `code_identity` docstring defines the exact encoding.
P3 records these identities too. Confirm H1 with that CPU proof, then give the enriched H1 copy to the rows.
The row uses the serving loader. It hashes the worker's executable-mapped library and records its PID, path and mappings.
Before any numerics case, it requires that library's code digest and cubin hashes to equal the H1/P identities.
A difference is INVALID. Symbol resolution uses SASS from that mapped library. Cubin hashes remain separate from the ELF section digest.
A P3 tool limit remains UNATTESTED.

C1 job preparation copies the signed population and its bound draft into the shared job directory.
It mounts the draft read-only at the source path in the signature. The signed JSON bytes stay unchanged.

C1 profile f2qwcb5HuDy completed in 765.44 s. Scale cases took 745.30 s; 2,829 `argwhere` calls took
514.88 s. C1 now selects the first nonzero coordinate once per group. The exact joint evaluator stays.
The A8S before/after CPU profiles must show equal case data before any speed claim or time estimate.
Read each exit status, profile, Netdata series and log. Open questions are in [questions.md](questions.md).

Launch state uses the whole worker start environment under rule v2 draft 4.
Only exact frozen exclusions do not count. Unknown names count.
Captured names cannot change or disappear. Later additions are diagnostic.
Receipts store sorted names and one digest. Values remain in worker memory.
`launch-levers.README.md` describes the public-value list and the environment proof.
`start_env.py` defines the common C1, H1, row and specified-serve constants.
No source-read completeness gate remains. Math and compiled-code checks remain.
