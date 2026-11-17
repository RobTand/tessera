# Launch input policy

Frozen rule v2 freezes `launch-levers.tsv`. Every worker records its digest.
The worker captures its whole start environment before any dependency import.
Only exact excluded names do not count. Prefixes do not control inclusion.
The source-read audit is diagnostic. Its incomplete scopes cannot block PASS.

`start_env.py` defines the container constants and the specification digest.
The source is the Tessera lane launcher and its build-identity environment helper.
C1, H1 and rows use this recipe. Cache and scratch mounts have fixed worker paths.
The serving launcher must supply the same OCI constants through its existing extra-arguments input.
The deterministic build-identity switch stays unset. Optional Tessera levers stay unset.

No A8S release worker environment was observed or saved. The specified serve environment replaces that missing observation.
`prepare_env.py` prepares a CPU-only comparison of row, C1 and serve-spec starts using the shared recipe.
The proof also imports torch, Triton, vLLM and Tessera in the row worker.
It then imports vllm.config and vllm.distributed, as H1 does, and the OpenCV
loader that the lazy multimodal path imports. No device call is required.
It records environment writer names and source locations, without values.
It compares captured names after those imports. Added names are diagnostic.
A changed or removed captured input is INVALID before another native call.
The CPU proof cannot establish Docker GPU injection versus Podman CDI equivalence.
The GPU canary must compare its start-state digest. A difference refuses attestation.
NVIDIA_VISIBLE_DEVICES remains included. No OCI exclusion forces a match.

Environment values stay in worker memory. Receipts contain sorted names and one digest.
The exact digest encoding stays in `launch_state.digest`.
Only boolean or enumerated Tessera selectors and levers, and PRISMAQUANT_DEV_MODE, can have public values.
The TSV public_values column names each permitted value. Paths and commands are never public.
Excluded-name and added-name diagnostics contain names only. Latched flags use the same public-value policy.
The environment proof writes no values or per-name hashes, including in its differences.
Its temporary comparison digests use a Unix socket and never enter files or logs.
Raw image inspection stays in private /var/tmp scratch. Results list image environment names only.
The TSV excludes VLLM_API_KEY, HF_TOKEN and HUGGING_FACE_HUB_TOKEN by exact name.
The cited paths use local signed wires, not Hub authentication.
Unlisted names count. A new credential exclusion requires an exact entry and source evidence.

The loaded-library audit conditions every toolchain exclusion. It checks the actual mapped bytes.
LD_LIBRARY_PATH is an exact library-search-path exclusion. IMG-N OpenCV
cv2/__init__.py:147 unconditionally prepends the directories from config.py:3-5.
Setting the final path in the launcher cannot prevent another prepend.
The row and serve must still audit the loaded Tessera bytes and resolve every
kernel to an unchanged image file or a bound compiled identity.
Each Triton specialization binds its cubin, entry, target, constants and launch parameters.
Each profiled kernel must have one audited native, unchanged image-file or compiled-code origin.
Unknown or ambiguous origins remain UNATTESTED. No source-read completeness gate remains.
IMG-N has no vllm/_C.abi3.so. Its inventoried owner is _C_stable_libtorch.abi3.so.
The CPU code-roster check records quantizer and fill functions from that file and libtorch_cuda.so.
Effective torch math settings remain fixed and bound in C1, H1, rows and the serve.
The tool does not add the serve validator to Tessera's plugin.

The serving launcher must set TESSERA_LANE_DOCKER_EXTRA to start_env.lane_extra().
It includes --workdir=/ext and --hostname=tessera-worker with the fixed thread and OCI inputs.
The serve-spec proof arm uses this shared recipe and checks the lane source. It is not an independent runtime implementation.
IMG-N has no /work. [Podman 4.9.3 resolveWorkDir](https://github.com/containers/podman/blob/v4.9.3/libpod/container_internal_common.go#L740-L779)
refuses that missing directory before startup. The existing /ext mount avoids this refusal.
This changes the serve-spec digest. C1, H1, rows and the serve must use the new recipe.


The output sites for the Opus security review are LaunchState.fields,
LaunchState.diagnostics and LaunchState.latched_diagnostic. Row, C1 and H1 call
those methods. env_proof.worker sends temporary comparison digests on a private
socket. env_proof.run_arms writes names and one launch-state hash.
dependency_audit.image_inputs writes image names only. common.check_runtime
reads private image inspection and records the image identity, without a second
hash over Env. Flag-parser exceptions use error_text and error_trace, which
record only the exception type. Other exception records retain code errors.
No producer writes the LaunchState.mapping object. The generator keeps raw
inspection in private /var/tmp scratch and removes it on exit.
