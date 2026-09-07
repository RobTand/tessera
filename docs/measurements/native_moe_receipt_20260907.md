# Whole-MoE receipt boundary: CPU and factory evidence — 2026-09-07

This records initial implementation evidence for #395. No original 96-member
wire stack was loaded or applied in these runs. There is no whole-MoE latency,
quality result, complete engine resource price, runtime-cell promotion or
release qualification here. The next gates use the canonical regenerated
capture, original PWC wires and independently frozen PrismaQuant panel.

## CPU boundary and lifecycle checks

PB `98a457255259d924acb5c8ea97dfbe13e159551b460537a9887404a0c3b6be83`
finished on dl380g10 with exit 0: **184 passed, 0 skipped, 0 uncollected**.
Torch was `2.11.0+cpu`; no GPU surface was exercised. Compilation checks also
passed. The admitted action reserved four CPUs and 8 GiB with OMP/MKL/OpenBLAS
threads bounded to one and pytest `-n 4 --dist worksteal --durations=8` on:

- `tests/test_native_moe_operator_receipt.py`
- `tests/test_native_operator_receipt.py`
- `tests/test_native_operator_resources.py`

The tests cover the complete ordered member join, exact routing transport and
bias, observed factory configuration, phase/output gates before all timings,
mutations during timing, workspace pointer drift and timing after collector
shutdown. The panel preserves `probe_scope` when the joint probe is an explicit
first-sequence integration screen; this does not change the full native capture.

CAS payload SHA256:
`7494063f8b33c7c7f2204ebefb9e4af67e86a44d96178489fb05dcc39bc62b25`.
Actual logs, terminal coordinates and independently recomputed payload hashes
are retained in `/mnt/shared/tessera-native376-resource/native-moe-395-validation/`.
An earlier CPU tooling attempt (`501de022…`) lacked Torch and skipped the two
Torch-dependent modules; its 29 passes are not validation of this producer.

## Stock-runtime factory construction

PB `e59018cd0f10bc1f42f7bbdf8134f4e113bdbbb49d2e772605604e7f582107e7`
finished on Sparky with exit 0, using four CPUs, 12 GiB total memory and a 4 GiB
GPU budget. The image was the actual official stock base
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`,
upstream commit `1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27`. Canonical Tessera
source `382a1a97dc89618173a2c0799ac569d473b9dc2a` was installed separately with
no dependencies; all 4,967 stock vLLM files matched before and after installation.
The launcher bound the actual Docker image ID and the official RepoDigest,
then verified the owned container's image and exit status.

The stock LFM factory, supplied its grouped sigmoid routing and an explicit
synthetic FP32 selection bias, constructed E=32, hidden=2048, intermediate=1024,
top-k=4, BF16, TP1/EP1 with maximum scheduled tokens 2048. Its observed backend
was **TRITON**, and its checkpoint role mapping was `w1,w2,w3` (gate/down/up).
The source wire container role order remains explicitly `w1,w3,w2` (gate/up/down).
The factory's recorded routing-method enum value was 2. These are configuration
observations; this smoke did not execute an expert kernel or establish speed.

The selected serving config is
`experiments/configs/lfm25_first_model_clean_20260907.json`, SHA256
`1b3bd7af48a0c859a89c9c42f18aa6131a246911fdf194943369bc41b7aad134`.
Factory CAS payload SHA256:
`538cba1001794b104c3e7eb49dfebd0a384ed0e235268744c3dbd70103ad9e8f`.
`native-moe-395-validation/factory-smoke.json` extracts the actual JSON record;
`native-moe-clean-context-r4/` retains installation, image and container evidence.

Two earlier findings were corrected before this pass. Action `aee98709…`
refused the config's native `set` type; `_plain` now represents sets
explicitly and sorts their JSON values deterministically, with no repr fallback.
Action `2892c6f6…` reached the native MoE import but FlashInfer could not write
its cache after dropping UID. The launcher now sets the library's supported
`FLASHINFER_WORKSPACE_BASE` to an owned temporary path. No runtime core file
was patched to fix either failure. The first transport attempt `02d1c4d…`
failed to import the harness because the launcher omitted `/work` from its
Python path; it did not reach the factory.

## Clean collector build

PB `3df90b5dad1bbc8393d9f173976fc23b0ea8390f4cbdefcef075b5a7183e02be`
built the existing C++ observer against that stock image with exit 0. The
image's packaged CUPTI headers/libraries live under
`/usr/local/lib/python3.12/dist-packages/nvidia/cu13/`; the earlier build
`3ba096a2…` found no `cupti.h` in the old image's CUDA include location.
The source was unchanged, SHA256
`530cac2f30ccfa3d8581755b84457f6283df96f1dd6e22092200507a85533357`.
The resulting shared library is
`/mnt/shared/tessera-native376-resource/clean1970-native_operator_resources.so`,
SHA256 `337bbbca32a9ec0dbc453f05edff309a089875eff00c159b7d7e45596ea6a156`.
This is build evidence only; allocation capture on the clean runtime remains
a separate gate. The build command, actual image ID and artifact hash are in
its `.build.json` sidecar and the verified PB result.

## Clean collector allocation qualification

The first allocation run, PB `77f24643ccee47e909218eb80688898b97268883192d102df679d54cb3e21aa9`,
failed before its apply interval because the stock image ships `libcupti.so.13`
without a `libcupti.so` development symlink. Context lookup now resolves
`cuptiGetContextId` from the collector's already loaded, linked dependency.
A CPU regression first failed on that extra library lookup (`9cd52894…`), then
the three receipt/resource modules passed **185 tests**, with no skips or
missing collection, under PB `3075f972efd582a36394de9426bf8b7d3e025e1896234296d23c250c36cbc7d1`.
Its independently rehashed CAS payload is
`94a66ea4ed7c48cc2ccf22d77868d96f3d54a679ffc33ce7ee266d69efca14cb`.

PB `58d9bcb75600116e259209e14f489a24ad99d11c0935d135cbb460be3fea84e8`
then finished on Sparky with exit 0 in the same official image, reserving two
CPUs, 8 GiB total and 3 GiB GPU memory, with native threads bounded to one.
The qualification observed the deliberately transient **123,456 external CUDA
bytes** and **1,048,576 Torch bytes**. Their conservative sum was 1,172,032
bytes. CUPTI 130001 reported successful configuration and flushes, no errors,
and zero dropped records. Torch was `2.13.0+cu130`. This exercises the collector,
not model weights, kernels, latency, or full-engine fixed resources.

The checked raw trace is at
`/mnt/shared/tessera-native376-resource/clean1970-resource-qualification-r2/trace.json`,
SHA256 `b2513f4e0d6e225ed55888ba8f71abad8407f8d56dc9afd47e7235f32f65c39c`.
Its qualification sidecar SHA256 is
`d359059367d4349afe55d915216f197ef5e3b26bf9beed86250a5a0ecf4af065`;
the independently rehashed PB payload is
`8a9666bfba6340b3405dbf953d6e80e4a49b854bf662c9e5f6dff2dd3f88fdbd`.
The directory retains actual image/container inspections and the exact owned
container's completed status. No collector C++ or model artifact changed.

## Original 96-wire preparation and source execution identity

The canonical request at
`/mnt/shared/tessera-measurements/first-model-20260907/native-moe-panel-r1024/prepare-01/request.json`
has SHA256 `94096681f0573620ad59c05b12eab4bfa9abba6cb4acfc7b658f6c1729246fb6`.
Its actual shape is E=32, hidden=2048, intermediate=1792, top-k=4. The earlier
factory-only smoke's intermediate=1024 was synthetic and is not evidence for
this captured shape.

The independent panel can now carry the paired `source_execution` and
`source_execution_qualification_sha256` fields. The former has closed schema
`prismaquant.joint_aura.source_execution.v1`, an explicit root configuration,
and named module attention/expert selectors. The latter is a SHA256 or
explicit null. Both enter the panel hash; PrismaQuant owns the independent
source-backend/proof join. CPU regression PB `d1caa1f10e9d…` first refused the
two valid new panels. PB
`4ee5e00060c2ec25544f5c5b79f891a11fc244f9c1aba3e9985c762bee3ee9e7`
then passed **196 tests**, with no skips or missing collection, across the
three receipt/resource modules. Verified CPU CAS payload:
`5094c0dc045b29e1144ebb0138ffb1c45f40b74e06a7b93b1b876b50b402b522`.

Actual native preparation PB
`3f7e7ae7f37e45b188e4fba85a4cfbb1c6e0448d7d20d866357026c4b74371a4`
finished on Sparklina with exit 0. It loaded all original wires, checked their
PWC renders, and warmed both captured phases. The TRITON FP8 owner retained
four native expert weight/scale tensors and one locked workspace allocation
of **23,068,672 bytes**. All 4,967 stock vLLM files remained unchanged after
execution. The exact owned container exited 0 without OOM and was removed.
The job reserved four CPUs, 16 GiB total and 8 GiB GPU memory, with native
threads bounded to one. Sparklina is an explicit subsequent-run dependency:
the frozen runtime binds the observed GPU UUID and local image declaration.

The authoritative preflight is
`/mnt/shared/tessera-native376-resource/native-moe-original-r1024/prepare-03/preflight.json`,
SHA256 `794829fc60af194514571ff90512e4e0ebe07f1f64d3adc3c273805d3d809188`.
Its independently rehashed PB payload is
`e2dfb612885737fd29097bfd47d06c4dcce032d4055d4a334dd928bff2fdb3d3`;
raw CUPTI trace SHA256 is
`2aac6d5cb2bb0ec399264d5c5e4a4da973cebe7063c5d5e8e54de4f88c22d4b6`,
with successful configuration/flushes and no errors or dropped records.
This is untimed preparation, not a numerical verdict, operator resource
bound, full-engine resource price or release qualification.

Two prior attempts remain bounded evidence. `prepare-01` (`b4aa6b240e74…`)
reached native warmup but refused the missing launcher image-declaration pair.
The launcher now reuses the existing resolver and `container_env` against the
explicit research configuration plus actual Docker inspection, preserving
the packaged default reference separately. No packaged pin changed.
`prepare-02` (`1d207ab84d5d…`) passed, but `prepare-03` supersedes its harness
identity after adding the source-execution panel fields. No request, tensor
or wire regeneration was required for that refresh.

## Stable generated runtime and original activation refusal

The first measured and profiled attempts (`73374e99491a…` and
`dc99662066de…`) correctly refused runtime identity before numerical evaluation.
An untimed comparison (`39b928f768cf…`) isolated the only differing runtime
field to the generated Triton `cuda_utils` shared library: fresh container
compilation produced different bytes. The launcher now reuses the existing
`TRITON_CACHE_DIR`, seals every generated file after untimed preparation, and
checks the complete file/hash map before and after every subsequent action.
No native hash check was weakened and no parallel weight cache was introduced.

Preparation `5a97746e3d8949957c5ae9b611709e259fca8734c32004bfe1dd2d58afd76296`
passed on Sparklina. `prepare-04/preflight.json` beneath the preceding evidence
root supersedes preparation 03; SHA256:
`0e588e8d5217bbc6235cc2e349cae302100b498a4e04bde626f790f251f5fe28`.
Its runtime SHA256 is
`eabb129c797d76e03c110d070e45b99d69db91d67b003b21f82a6b3c45a0d1df`.
The 33-file cache seal SHA256 is
`7bc5f26063c602a51cc1b8d843ffdecb6234c81b97ccd8ce799b0a99a4752399`.
PrismaQuant independently froze panel 02, SHA256
`479206a4e228249a1a5f22e3229cb5b3b79d0ae0b4e1f31459d4d0c5c4b143e0`,
against that runtime and the unchanged original request/source proof.

Measurement `448959d34a5b975af5188a00c2a2dd40a45f79772af9cb26a374cfb5292c896e`
then reached the numerical gate and exited 2 (`numerical_refused`; PB wrapper
exit 1). Whole-owner output passed both phases, with maximum absolute errors
0.00146484375 prefill and 0.000244140625 decode. Decode input QDQ was exact.
Prefill input QDQ failed the fixed atol=rtol=0.015625 gate, maximum absolute
error 0.0703125 and maximum normalized error 2.6605080831408774. Consequently
there are **no timings and no admitted operator resource bounds** from this
attempt. The retained native weights and workspace are observations only.
The receipt is `measure-02/receipt.json`, SHA256
`dffa1514d65612a936876da8398f1553ddaa2e7c7dc36d30dae4877582fc1215`;
its raw memory trace SHA256 is
`7464e4364ed7eaf2df4369b3904898f986e31615d28314593fa82adca90d87ac`.
All 4,967 stock files and all 33 cache artifacts were unchanged; the exact
owned container exited 2 and was removed. Both-host Netdata evidence contains
ten series with none missing (`measure-02/netdata/index.json`, SHA256
`bf82b1a8948f060863bd8c2045d91cff1e0ced44d56375574cfebc63e65c06fd`).

A small admitted diagnostic,
`2c01d0c7b6aa4d2a0ad2ea22787432c267de21441da30183cd1b6574868b639d`,
passed on Sparklina and captured native FP8 code bits, FP32 scales, inputs and
reference QDQ without changing any original bytes. Of 512 prefill row scales,
256 differed from the PrismaQuant formula. Division by the captured native
scale reproduced every native code bit; division by the PrismaQuant scale did
not. The one-ULP scale differences can flip FP8 midpoint rounding: row 53,
column 232 has input -0.0908203125, native scale 0.0021623882930725813 and
PrismaQuant scale 0.002162388525903225, giving code -44 versus -40. There were
4,050 differing BF16 QDQ values; decode was exact. Compressed-tensors reproduced
the frozen reference QDQ exactly. This localizes the incompatibility to scale
and quantizer arithmetic; it does not yet establish a replacement formula.
No tolerance was relaxed. The diagnostic JSON is
`activation-diagnostic-01/diagnostic.json`, SHA256
`851f140f29abed96922014ae00d7f90ab252511480b9af60ef10bd0189d004c2`;
its independently rehashed safetensors capture SHA256 is
`8452ec28f9b325e52fcc6745ee4aa16d6828374e45650380b3e56ade770fffb2`.
Stock runtime and sealed cache bytes remained unchanged after this diagnostic.

## Explicit first-model KV capacity and coherent native context

The coordinator selected a new integration configuration,
`experiments/configs/lfm25_first_model_fixed_kv_20260907.json`, SHA256
`f5064609d62a3e61ef1d9bb87b2b62ea10b31b71759db3dce666543d7585233e`.
It supplies `kv_cache_memory_bytes=412286976` for eight 4096-token requests,
with unchanged 2048 scheduled-token limit and eager TP1. This new configuration
supersedes neither the bytes nor the claims of earlier auto-sized captures.
The stock 0.35 utilization setting still controls startup free-memory admission;
it is bypassed only for KV sizing when explicit bytes are supplied.

An admitted CPU projection through the exact installed stock1970 helpers
reproduced all four original full-engine KV outer placement descriptors.
The shared physical pool has six 32768-byte pages per block, one full-attention
group needing 256 pages per request and three recurrent align-mode groups
needing two each. Eight requests plus one shared null block need 2097 blocks,
or 412286976 bytes. Recurrent padding is included; the four aliased group
placement descriptors must not be summed. The projection explicitly does not
claim that r3 retained complete resolved recurrent specs or that the new pool
has been observed in an engine. The new config carries mandatory expectations
for the next actual capture rather than admitting the projection as fixed cost.

The derivation at
`/mnt/shared/tessera-native376-resource/kv-capacity-1970/diagnostic-01/derivation.json`
has SHA256 `23e506bf87e36e3fb4e05caf15e29450aad887d0d0e89fab453d9a3b67787268`.
PB `d400e8ba22f66aebae0f82e6fe672733cf6e31811f643f1d7574167920e2e748`
ran CPU-only in the official ARM container with two CPUs and 8 GiB, exit 0,
and all 4967 stock files unchanged. Verified CAS payload:
`5ec24cf68bee92d0b8c66a7d4c2da6560fb7a5d7f0b4fefe4299964badf47b3b`.

The native resolver previously refused the extra explicit-capacity argument.
Regression PB `7d295e5d84ba48f2d22e4da1675bae094fd83294a61d8a8750a3ff2d75f95a18`
recorded one failure and five passing rejection cases. The resolver now binds
positive explicit bytes directly into the actual `CacheConfig`; omission keeps
the preceding automatic-capacity context. Invalid values still refuse.
PB `e836133da60099ff4c8eaae8ef6dca91aecdb34f908da396744807c1671cb7ee`
then passed **202 tests**, with no skips or missing collection, using four CPU
workers on DL380 and Torch2.11.0+cpu. Verified CAS payload:
`c385697c68331888833a502757a7ea2b1aee0a836a6ffc97f01317826d772972`.
Initial attempt `6afcb4429835…` used an interpreter without Torch, skipped the
module and exited5; it is missing-tool evidence, not the behavioral regression.
New native request, runtime preflight and independent panel identities are
required after this resolver/config change. No previous resource observation
is relabeled as belonging to the new configuration.

## Source checkout and installed package identity

Canonical producer382's source-tree encoder hash is
`57809bff862b880dc397e6d271a80c04d6c87d1af3bba12c076648fc5443355c`.
The native runtime's installed-package hash is
`8239d56568b0b2298a05c61b7a2dc0c85b7fc76a25c4bb6dae0fe2d7ecdf428b`.
Both use the same `encoder_source_sha256` algorithm. Packaging excludes
`tessera._dev*` in `pyproject.toml`: five development Python files exist in
the archive and are absent from the installation. Every one of the 67 installed
files matches the archive's bytes and size, with no extra installed file.
Hashing the installed subset of the archive reproduces the runtime hash exactly.
The source-code hash includes 71 archive files or 66 installed files; those
counts exclude the separately sealed runtime contract JSON.

This is a verified packaging distinction, not evidence of JIT source mutation.
The producer source-tree and installed serving package identities remain
distinct scopes and cannot replace one another. Full-engine serving observations
must join the actual installed package identity and file roster of the native
capture. The read-only artifact comparison is retained at
`/mnt/shared/tessera-native376-resource/native-moe-original-r1024/package-identity-diagnostic.json`,
SHA256 `aa2560d05b1a666cca404f01c2bd7649c3fdb7f38b2f18f9cf126bd5308bde3d`.

## Corrected fixed-KV original-wire native observation

The final original-96 panel uses request `prepare-03/request.json`, independently
frozen `panel-03.json` (file SHA256
`2090eb469f5e46f001ab27da49068937ba56cd7ea22e849ad3453a646522c9c3`),
and native runtime identity
`a0a72558667a2836856d952e10d64014f6fb7b60d86bc0fabd464ca0435692ed`.
It retains the fixed-KV configuration `f5064609...`, the corrected shared FP8
activation arithmetic and the actual native TRITON whole routed owner boundary.
Router computation is outside this measured boundary; selected IDs and weights
are checked against the original source capture. No sum of expert leaf timings
is substituted for the whole invocation.

PB timing action `1cbfe802699317ec7443bf68d41360ec5493b143a60f38146e4818e8d7155bed`
and separate profiler action
`b46560b08862f233e5fb29461855081aff5fd5bbf5e2cb307c02a298ecd40b78`
both ran on Sparklina, with actual exit zero and complete scope cleanup. Each
reserved four CPUs, 16 GiB aggregate and an 8 GiB GPU subset, with one native
thread per library. The exact runtime and canonical panel identities match
between the two processes. The sealed Triton cache is unchanged in both; loaded
package origins match the canonical installed source files and all 4,967 stock
vLLM files remain unchanged.

For 32 CUDA-event samples after eight warmup iterations, median whole-apply
latency was **1.906431973 ms for prefill M=512** and **0.263167992 ms for decode
M=1**. Both numerical gates passed: maximum absolute output error was
0.0009765625 and 0.000244140625 respectively, and activation QDQ matched exactly.
The independently profiled invocation recorded two fused MoE kernels totaling
1692.047 us for prefill and 235.883 us for decode. These are a baseline boundary
price and an instrumented replay, not an optimization delta.

The measured operator residency is 353,042,432 bytes. Persistent vLLM workspace
is separately 23,068,672 bytes. Complete incremental operator scratch bounds,
including output, are 7,367,680 bytes for prefill and 14,848 bytes for decode.
These bounds do not establish model-fixed resources or cross-operator workspace
composition and remain insufficient for an admissible full-model runtime table.

Evidence under `/mnt/shared/tessera-native376-resource/native-moe-original-r1024/`:

- `measure-03/receipt.json`: SHA256 `e2c5c85158b405fa419c095788328b2895268df19761581ec72b05be8f0d04a4`.
- `measure-03/receipt.json.memory.json`: SHA256 `b72417b4f243f3c296dc0fdd0153503eb4fba8496db85b6cb37a414e95198a83`.
- `measure-03/pb-audit.json`: SHA256 `57c80ded5a0bf1e50723e93e663580501906868910965056e275928c7b357c29`; rehashes every stdout-bound artifact and CAS payload `a0963620f429d2785a43adca9370d90cabf7b721b94ca32a335812b0196a9d39`.
- `profile-02/profile.json`: SHA256 `f9b834c75bd48faa07ef713ea3a870bb83ce6b6d40517e3bd2f4568907f90b0e`; includes both CPU and CUDA profiles, numerical gates and separately hashed Chrome traces.
- `profile-02/pb-audit.json`: SHA256 `1771b260117204c8c330b4b8bd311bd7634a9a10f452d7aed73ad8fcf0ea5b18`; rehashes profiler artifacts and CAS payload `119290d887d43f1c342002b87c69ebc9e60a0882ebd479ef7478f3e3a65915a1`.
- `measure-profile-netdata-03/index.json`: SHA256 `46f6c4d1f108ff51b1463adbf96af75a3d1d68b73b78cc25a2d485ab24762948`; all ten selected raw power/clock/memory/CPU series from both GB10 hosts were captured with no missing series. The finite panels have low duty cycle; these host-level series do not resolve per-operator energy or establish saturation.

The first PQ consumption of these unchanged measured bytes failed with
`KeyError: 'workspace'`: its fixture had invented top-level workspace fields
absent from the producer's actual v1 receipt. The actual frozen panel carries
the identity and resources carry the observed digest/bytes. PrismaQuant issue
RobTand/prismaquant#328 tracks the reader repair and receipt-shaped regression;
no measured receipt is rewritten to accommodate it.

PQ reader repair RobTand/prismaquant#329 subsequently passed its receipt-shaped regression and
84 CPU tests, then consumed the unchanged native receipt through PB
`5c5626d6e391486c1d5ebb174becb53651b808d429161814b47338c27776fe5f`.
The accepted observation at
`/mnt/shared/tessera-measurements/first-model-20260907/native-moe-panel-r1024/consumed-03.json`
has SHA256 `4ec5fe3f02048549ab3abe8a4926572aac91ab89d3333a45d74d8b89617b6ddb`.
It preserves all timings and resource values above, with full-model runtime
admission still false. The independent CPU validation/CAS audit is
`workspace-reader-328-audit.json` beside that output, SHA256
`061b971bc210256174b29e14b1ce60cf75a29ce226c80ba0c7b4ea556f5bcda9`.

Final delivery checks after merging current master (including cached dense
export PR #403) ran through PB
`33033d7f7429c98b36f048b29b423aba399b7a0da8debd5429d9e3096953a65e`.
Native measurement modules compiled and 204 tests passed, including all 202
native receipt/resource tests; the sole failure was the offline issue index
missing the newly filed PrismaQuant workspace-reader issue/PR. Refreshing the
existing index fixed that reference failure: PB
`ddc8bc3679f34ed8a03c2fcfe3f5cb8ba3bacb4aca15b4531c90833257962640`
passed all three reference checks. Both runs reported zero skips or missing
collection. The final reference action exited zero with complete scope cleanup;
`delivery-cpu-audit.json` beside the native evidence rehashes its CAS and
canonical receipt and records the initial terminal failure without relabeling it.
The audit SHA256 is
`56a8e58aa014aceb28f648d94eed086074638f7009657af7c76a56c2743c91c7`.
