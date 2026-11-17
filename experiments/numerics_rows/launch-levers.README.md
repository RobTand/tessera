# Launch input policy

The rule freezes `launch-levers.tsv`. Each worker records its digest.
The source inventory uses Tessera c3a3ca52. `launch-image-env.json` records
IMG-N acc684c06ec7 Config.Env. `dependency-audit-review.json` classifies all
53 inputs in the audit worker's launch state and records source file hashes.
The TSV binds that review by SHA-256. Its complete field is still false.

The A8S release worker environment was not observed. There is no saved worker
environment and no live release worker. Rows use D9: TESSERA_SERVE_MODE=resident,
no optional Tessera lever. The serve validator must capture its own worker
inputs and refuse a state without a matching PASS receipt. This tool does
not add a serving validator to Tessera's plugin.

Every present dependency-prefix name counts unless its exact policy entry
excludes it. NVIDIA_VISIBLE_DEVICES counts. The image declares `all`, but a
container runtime can override it before the worker starts. Record the worker's
value. No release-worker value is known. TRITON_PTXAS_PATH remains included.
PRISMAQUANT_DEV_MODE remains included. The harness uses NUMERICS_IMAGE_INSPECT.

Build and installer inputs do not count on these paths. The 53 decisions are
in the review's observed_name_decisions. TORCH_CUDA_ARCH_LIST and
FLASHINFER_CUDA_ARCH_LIST are toolchain diagnostics. LD_LIBRARY_PATH,
LIBRARY_PATH and native/Triton cache locations are excluded only with the
loaded-code checks. Each exclusion cites its source and condition in the TSV.

The audit measured float32 matmul precision `highest`, matmul TF32 false,
cuDNN TF32 true, FP16/BF16 reduced-precision reduction true, FP16 accumulation
false, deterministic algorithms false, warn-only false, cuDNN deterministic
false and cuDNN benchmark false. Workers retain these values. They also record
available split-K reduction flags, the precision hierarchy and the preferred
BLAS backend. C1, H1, row and serve settings must match.

`dependency_code.CodeAudit` observes the pinned Triton CompiledKernel load/run
API before preparation. It records actual cubin bytes and SHA-256, entry,
target, constants and launch metadata/grid under each unit and case. It saves
the cubin, preserves existing hooks, and refuses a changed loaded definition.
A serve supplies the expected complete binding list; a mismatch refuses before
launch. Missing load observation is UNATTESTED. A used specialization without
its compiled H1 P2 check is UNATTESTED. C1 records that its CPU path has no
device launches. H1 records its observed specializations, including P2 cases.

The row profiles preparation and each served launch configuration. It uses
CUDA events, not CPU operator span names. Every event must resolve uniquely
to the audited Tessera SASS, a function in an unchanged image library at its
image path, or an observed compiled identity. The image inventory records
measured hashes of torch and vLLM libraries. The worker verifies executable
mappings against the hashed inode and resolves CUDA symbols from that file.
A changed image library is INVALID; an unknown or ambiguous owner is UNATTESTED.
The join refuses PASS evidence without complete code-origin bindings.

The source audit is incomplete in two scopes, listed in the review: transitive
optional-package imports under vllm.utils.flashinfer, and vendor-native runtime
inputs outside the conservative dependency prefixes. The reviewed torch eager
BLAS/fill owners and Triton descriptors have explicit traces. Code identity
does not replace the remaining state audit. The row records UNATTESTED while
this review is incomplete. A CPU pass cannot close the GPU acceptance checks.
