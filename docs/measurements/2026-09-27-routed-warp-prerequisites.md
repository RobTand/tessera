# Routed warp-specialization prerequisites — #640 round 2

## Status

**Not a performance result or an admission receipt.** The additive research
prototype is unadmitted and not imported by serving. Existing v39 identities,
cells, defaults and numerical bounds remain unchanged. No new cell, PR, merge,
TP1 census or TP2 census exists from this round. Full E2M1 routed fusion is not
implemented. There is no fused-oracle pass or after-performance claim.

Branch rebased onto master `52bc86ea1228dbf2ee5c8e0484b36d632e3b5262`.
Working evidence directory:
`/home/rob/tmp/claude-campaign-20260926/pi/kernel-640/`.
Image X throughout:
`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.

## Verified prerequisites

- SM121a compilation and native BF16, E4M3 and block-scaled E2M1 MMA worked with
  nvcc **13.0.88** and `cuda::barrier` in torch **2.13.0+cu130** on NVIDIA GB10.
  Three targeted tests passed, all three device-allocating, zero skips and zero
  uncollected modules, `-n 2 --dist=worksteal --durations=20 --strict-cuda`.
  The E2M1 case also checks nonuniform row/K-group scales and codes, not merely
  all ones. This one-tile gate does not establish multi-frame overlap.
- Pre-fix: those three cases failed with
  `ModuleNotFoundError: No module named 'tessera.routed_warp'` at
  `tests/test_routed_warp_mma.py:10`; zero device allocations/skips/uncollected.
- Hardware attributes: 48 SMs, 32 lanes/warp, 1,024 threads/block,
  1,536 threads/SM, 65,536 registers/SM, 49,152 shared bytes/block,
  101,376 opt-in shared bytes/block, 102,400 shared bytes/SM, 24 MiB L2.
- The existing E4M3 grouped kernel's emitted PTX actually uses
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` after conversion.
  Its source's FP8 `tl.dot` is not proof of native FP8 MMA. BF16 emits the BF16
  k16 atom; A4 emits the block-scaled E2M1 k64 atom. All three directly chain
  accumulator operands, with no `add.f32` in the inspected kernels.
  The BF16 dump includes two cached earlier FP8 entries; only `bf16-2/3.ptx.txt`
  are BF16. This is an inspection of old kernels, not evidence for new ones.
- Image Python activation source was captured. C++ activation source was absent
  at the inspected package, `/opt/vllm` and `/workspace/vllm` candidates; no
  broader source-availability claim follows.
- 69 selected CPU integration tests passed on dl380g10, torch CPU population,
  two work-stealing workers, zero skips/uncollected and zero CUDA allocations:
  `tests/test_runtime_image_pin.py`, `tests/test_serving_native_extensions.py`,
  `tests/test_routed_pair_profile_exit.py`. This verifies image guards and that
  the new JIT loader is not serving-reachable; it is not kernel numerics.

## Prototype and failure

The uncommitted window prototype has paired gate/up plus activation and a
separate down phase. Producers fill a two-slot shared-memory ring while
consumers issue MMA; full/empty barriers guard publication/reuse. Each branch
keeps its own gathered A tile/permutation. CSR is built once. Each CTA owns an
expert/output tile and loops only over live route tiles, with no routing-dependent
host synchronization. The research predicate is uniform R4, L14, 64-aligned
geometry and SM121; BF16 requires folded arithmetic. No old predicate narrows.

Initial M16/N64/K64 follows one M atom, one 32-byte R4 sector and the existing
K frame. Eight producer warps own packed words' eight output rows; eight
consumers own N8 atoms. Two slots are the minimum for overlap. This is an initial
mapping, not a measured optimum. Cross-warp cache reuse is not a coalescing claim.

The first full window smoke **failed** on sparklina with
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
at the synchronize following the new phases. It produced no valid numerical
result. Source review found an unmasked last-word lookahead; a bound was added.
That omission is real, but it has **not been proved the only cause**.
The current source compiles and loads in image X with **no GPU exposed**; that
is not a GPU verification of the guard.

A memcheck retry remained READY behind G2. While it waited, inspection of the
installed Compute Sanitizer help found that its filter needs `kns=...`, not
Nsight Compute's `regex:...` spelling. The local wrapper was corrected. The
invalid request was withdrawn through the supported PB client from READY,
with **zero attempts and zero reservation tokens released**. No running process
was stopped. No corrected GPU rerun has been submitted while the requested
independent design review is pending. Bounded `pbwait --wait-s 600` windows and
complete admission records were consumed; this was not called a scheduler bug.

The new FP8 primitive uses native FP8 rather than the old F16 lowering. It still
must pass the unchanged independent oracle. The eight-expert old/new smoke's
screen is explicitly NOT that oracle. The prototype also changes route reduction
to `index_add_`; its numerical effect remains unverified. Full E2M1 fusion must
preserve its exact accumulation/activation boundaries or stop for Claude's
specified deviation review. No tolerance was widened.

## PB ledger

All eight published actions below are terminal and consumed. Successful stdout
CAS payloads were hash-verified; failures/withdrawal are not successes.

| Action key | Outcome | Purpose |
|---|---|---|
| `e69908aa0637ae091a003092e996f672bdca792e7bc7498c78f443cd5ee0bd65` | failed rc1 | Expected pre-fix 3-case import failure |
| `0cc3b8ee1c530f8bb6a9b759e843ee1dcb8a74db9ed510efa1d901337f7d98c0` | done rc0 | 3 CUDA MMA/barrier tests |
| `a32976fe6b5ca1df61594cb998dcb2f171d92f09547726b14e4c753fc9a32065` | done rc0 | Old-kernel PTX and activation inspection |
| `b4304688c1cc01bb55ed027033b4ba699d42864e783512d4e48efd85bf042e2c` | done rc0 | Initial full-source CPU-only compile, 29.30 s |
| `6423b844c2aa4d4f0b28399994e1b9dfcb5af22ea56b1cd7840cb3c3a8a42284` | failed rc1 | First window smoke, illegal memory access |
| `d7e93324b94064b30c43803b44f108245cd74247949c30802225c706918a6b29` | withdrawn, no attempt | Invalid memcheck filter; withdrawn from READY |
| `cc22063c565fda860fc6ead8825c05b93ad4e22585e02597a4acb72a6c2e1529` | done rc0 | 69 CPU integration guards |
| `e6bc7791412b46634e4c887368dd273165d570cc1afccea975442854bb9acf76` | done rc0 | Current relocated/guarded CUDA source compile, no GPU, 36.97 s |

An earlier `--host-class` submission was refused before queueing because it
requires measurement mode. The non-timing tests then omitted that flag.

Latest compile receipt SHA256:
`c56a090f7b39f0f032cf0e8fc9a7c8d227c0790d83aaa966d8415d577177d2b6`;
stdout CAS:
`773586377ad06aa57f2f47ebadddcaae43fe6a7ab3bc617b23ee565e4a54ba03`.
MMA receipt: `acfcbf6be043131456a703af2effad4caccf267860a9f1c183a2ab1be20fcec5`.
Its stdout CAS: `42c2dd0f7139bdf043d3e8c10305b7537fa269d393cfcb9b4cb699c6c3b71d56`.
CPU guards receipt: `b4f9360d3c8b480972b44921f9670488ccf48fef271892cc4ba7539c4e538af1`.
Their stdout CAS: `88a133fcedff4954df5da5cdc7572ea2a171e02f8f8677df54423a227dd04908`.

Commands are retained in `round2-*.pb.log` and wrappers:
`experiments/routed_warp_tests.sh` (including CPU-only compile mode),
`experiments/routed_pair_action.sh --window-smoke` through the contract-probe
wrapper. All used published pbrun/pbtest, two admitted CPUs, threads/MAX_JOBS=1,
12 GiB aggregate/4 GiB GPU for GPU probes, 8 GiB/no GPU for CUDA compilation,
and portable GB10 image-based placement. CPU tests used pb-cpu, 4 GiB aggregate.
Reported box power during CPU-only compilation is **external load**, not work/J
or power attributable to the compiler. No round-2 performance measurement exists.

## Consultation and pending decision

One read-only `zai/glm-5.3` consultation, run
`cae68fc4-de6b-41ee-81d2-3888b8704faf`, with supervisor dialogue and no fallback.
Report: `round2-design-review.md`. Its P1 asm-tail objection was rejected after
checking the pinned CUTLASS header and the exact nonuniform-scale GPU result.
Its unmeasured cycle counts and fixed stage/warp caps were not adopted.

Claude was asked to arrange an independent **Fable design review** of the actual
pipeline, decode addressing and scheduling derivation, using `FABLE-REQUEST.md`:
https://github.com/RobTand/tessera/issues/640#issuecomment-5852642082
This is not an E2M1 loss-of-exactness escalation: E2M1 routed fusion does not
exist yet. Implementation is paused for that requested coordinator consultation;
no source should be promoted from these prerequisites. No PB work is pending.

Side prose fixes, separate commits: `6ba8ae10a` clarifies the existing BF16
activation boundary; `7cc5cefd3` removes the retired span-2 decoder from package
commentary. Production code for existing identities was not changed. The WIP
source and packaging updates remain uncommitted for review. Host LSP reports
image-only vLLM imports and pre-existing optional annotations in the doc-only
adapter; these were recorded, not described as a wholly clean workspace.
