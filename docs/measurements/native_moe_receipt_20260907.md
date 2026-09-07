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
