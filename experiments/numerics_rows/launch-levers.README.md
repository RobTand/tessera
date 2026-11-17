# Launch input policy

Rule v2 draft 4 freezes `launch-levers.tsv`. Every worker records its digest.
The worker captures its whole start environment before any dependency import.
Only exact excluded names do not count. Prefixes do not control inclusion.
The source-read audit is diagnostic. Its incomplete scopes cannot block PASS.

`start_env.py` defines the container constants and the specification digest.
The source is the Tessera lane launcher and its build-identity environment helper.
C1, H1 and rows use this recipe. Cache and scratch mounts have fixed worker paths.
The serving launcher must supply the same OCI constants through its existing extra-arguments input.
The deterministic build-identity switch stays unset. Optional Tessera levers stay unset.

No A8S release worker environment was observed or saved. The specified serve environment replaces that missing observation.
`prepare_env.py` prepares a CPU-only comparison of row, C1 and independent serve-spec starts.
The proof also imports torch, Triton, vLLM and Tessera in the row worker.
It compares captured names after those imports. Added names are diagnostic.
A changed or removed captured input is INVALID before another native call.
The CPU proof cannot establish Docker GPU injection versus Podman CDI equivalence.
The GPU canary must compare its start-state digest. A difference refuses attestation.
NVIDIA_VISIBLE_DEVICES remains included. No OCI exclusion forces a match.

Environment values stay in worker memory. Receipts contain sorted names and one digest.
The exact digest encoding stays in `launch_state.digest`.
The only public values are frozen Tessera lever names and PRISMAQUANT_DEV_MODE.
Excluded-name and added-name diagnostics contain names only. Latched flags use the same public-value policy.
The environment proof writes no values or per-name hashes, including in its differences.
Its temporary comparison digests use a Unix socket and never enter files or logs.
Raw image inspection stays in private /var/tmp scratch. Results list image environment names only.
The TSV excludes VLLM_API_KEY by exact name, citing the pinned image's authentication-only source.
Unlisted names count. A new credential exclusion requires an exact entry and source evidence.

The loaded-library audit conditions every toolchain exclusion. It checks the actual mapped bytes.
Each Triton specialization binds its cubin, entry, target, constants and launch parameters.
Each profiled kernel must have one audited native, unchanged image-file or compiled-code origin.
Unknown or ambiguous origins remain UNATTESTED. No source-read completeness gate remains.
Effective torch math settings remain fixed and bound in C1, H1, rows and the serve.
The tool does not add the serve validator to Tessera's plugin.

`environment-writes.tsv` lists environment-reading sites for the Opus review.
It separates in-memory reads, child-process inputs and receipt output.
