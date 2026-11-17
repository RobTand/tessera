# Launch input policy

The rule freezes `launch-levers.tsv`. Each worker records its digest.
The source inventory uses Tessera c3a3ca52. `launch-image-env.json` records
the observed Config.Env of IMG-N acc684c06ec7 on sparklina. It is image
metadata, not an observation of a release worker.

The A8S release serve environment was not observed. There is no saved worker
environment and no live release worker. Do not infer one from a launcher.
Rows use D9: TESSERA_SERVE_MODE=resident and no optional Tessera lever.
The serve validator must capture inputs in the serve worker and refuse a
launch state that has no matching PASS receipt. Matching a release serve is
follow-up validator work and does not block the freeze by itself.

All TESSERA names and dependency-prefix names count unless an exact policy
entry excludes them. Unknown names under these prefixes count. The seven
toolchain names are diagnostics. PRISMAQUANT_DEV_MODE counts as an admission
input. A warning in dev mode cannot satisfy any required refusal.
The harness uses NUMERICS_IMAGE_INSPECT, outside the Tessera namespace.

Each name in the observed image environment has an explicit policy entry.
Names without a finished source trace are included conservatively. This is
not a claim that each affects arithmetic. MAX_JOBS controls ninja worker
count only in pinned torch/utils/cpp_extension.py:2832-2844. It does not enter
the launch hash. The existing canonical loaded-library audit still applies.

The pending dependency audit must trace preparation, quantization,
composition and native launch paths in the pinned image. It records package
versions, source hashes, dynamic reads and names outside the conservative
prefixes. Its collector reports complete=false until that review finishes.

The runtime code checks are not complete at this checkpoint. The next step
must bind every used Triton specialization and resolve each profiled device
kernel to audited native code, an unchanged image library at its image path,
or bound compiled code. A missing proof is UNATTESTED. A known identity or
math-setting mismatch is INVALID. Existing FAIL precedence is retained.
This checkpoint cannot establish a complete rule v2 attestation.
