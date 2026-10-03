# Paired-K32 continuation acceptance packet

Refs #857 / PR #862. Supersedes pending-GPU and execution-policy statements in the historical 2026-10-02 compile screen. Historical protocols and banks are unchanged. No GPU work or new tests were launched for this continuation; no kernel, native bank, resident layout, wire, opt-in guard, serving default or pin changed.

## Completed bounded numerics and native evidence

Evidence root: `/mnt/shared/astra-resume-20261002/t8_performance/`.
`paired-k32-direct-numeric-v3/RESULT.json` records exact flag0/flag1 word equality for nine real outputs (gate/up, down routes, reduced output at M1/512/2048) and six synthetic outputs (down K128 fallback and K192 paired boundary). Accepted result SHA256: `a5a8e479a8ab6e9ad26944d554aa7c88e2dda41423b4882e7eaeac9ae129c9d9`. Independent fp64/derived-bound proof remains restricted to the declared token0 prefix, not full batch/model quality.

Production CUDA source SHA256: `41414b331db7224cb79ad0ac9b9c87c1e06af80d6ea4fa6306a9384f6b10ec85`; input manifest: `95e4a9f2116d402cab87dee6630765ad81cd3435abedd478589fa1be345e5b9b`.
Baseline flag0 ELF: `b89ba2d62705abf0152fbc8bb6b338e7bd60f88199211b5f6d61effddfa0c705`; candidate flag1 ELF: `bbbbb4d30815b64dd1bd65534d7a853bda0d77c948204fd1ff9aee917ab36f13`.
These are the retained `paired-k32-10f3d16d-{flag0-native,native}` banks. `MATCHED-NATIVE-FLAGS.json` records the sole compile-flag difference, `-DTESSERA_ROUTED_FUSED_PAIRED_K32=1`. Existing CPU PB native build/resource evidence remains: GU original121/paired120 registers, down122/124; all four specifications STACK/LOCAL/LDL/STL zero; static shared1024B distinct from dynamic76240/59664B. No-spill evidence is not a speed result.

## Historical timing: complete but not performance acceptance

`paired-k32-direct-abba-v1` completed A1/B1/B2/A2, 10 warmups and30 CUDA-event samples per arm/case, separate three-call native profiles and3s power windows. It reuses accepted raw output guards. Numerical and timing driver hashes differ; equivalence is scoped to identical production source/native/readset, not the whole harness.

| M | A1 ms | B1 ms | B2 ms | A2 ms | Reported median-of-arm-medians A/B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 |0.501184|0.490112|0.495440|0.505088|1.021024|
| 512 |11.863568|11.667680|11.619392|12.914576|1.064030|
| 2048 |13.363376|13.511248|13.533616|14.290672|1.022525|

These are observations, **not accepted causal speedups**. M1 is unpaired fallback: its apparent improvement cannot establish paired scheduling benefit. M2048 candidate blocks are slower than A1: A1 IQR13.3224–13.3956ms, B1 13.4945–13.5342ms, B2 13.5012–13.5735ms, A2 14.2437–14.3264ms. M512 IQRs are A1 11.7294–11.9908ms, B1 11.6519–11.6874ms, B2 11.6013–11.6421ms, A2 12.8430–12.9688ms. Tight within-block events do not repair between-block drift; these are not120 independent replicate measurements per cell.

Retained `paired-k32-sol/ABBA-STAGE-CLOCK-CPU-ANALYSIS.json` reports M512 GU/down profile times7.6861/3.5112ms A1 versus7.7445/3.5336ms B1 and7.7534/3.5237ms B2. M2048 GU is7.9587ms A1 versus8.1642/8.1053ms B1/B2. Separate profiles therefore do not support a robust barrier-reduction gain. A2 M512/M2048 clocks average2430.1/2300.8MHz versus A1 2501.1/2453.4 and B1 2522.0/2475.0. A2 power66.86/69.52W versus A1 75.28/83.73W; A2 host user CPU35.15/32.634% versus B1 7.765/6.793%. These are observed confounds, not proof of a named process causing the change.

Both-Spark `netdata-both-hosts.json` covers epoch1790977557–1790977676. Sparky user/system means11.218/2.126%, load1 mean2.82; Sparklina0.604/0.809%, load1 mean0.817, GPU power7W and reported utilization0 throughout. This is compatible with a quiet second Spark, not proof of first-Spark isolation. GPU collection cadence10s despite1s returned buckets cannot qualify subsecond event windows or per-arm energy. GB10 utilization is not saturation evidence. Energy remains HOLD.

## Corrected policy and remaining decision

Historical custom-op benchmarking was incorrectly classified as covered by the vLLM service exemption. Timing reports have empty PB action fields; they are **not PB-exclusive measurement receipts**. Only actual service operation is exempt. Preserve the accepted bounded numerical observations; do not restamp old evidence or repeat completed numerics solely to relabel provenance.

The remaining performance decision is whether paired scheduling wins at admitted M512/M2048 under matched isolated clock/CPU conditions without harming fallback/decode. Current observations cannot answer it. Any root-authorized PB-exclusive timing-only comparison must reuse exact accepted output guards, ELFs, image and readset, collect matched native profiles and actual launch shared-memory/carveout, retain per-block raw samples and both-Spark CPU/power/residency, and predeclare clock/CPU confound handling. Do not discard A2 post hoc or normalize timings by clock. Served TP2 throughput/quality/decode remains necessary for any full-model or work-per-joule claim. No default/pin/ship gate promotion follows.

### Existing launch-owner compatibility

Reuse existing containment, not a second dispatcher. Historical paired main rejects PB action context. `bench_t8r.sh` rejects simultaneous `BENCH_STRICT_STAGED` and direct input transport; this separates input transports, not PB ownership. Simply prefixing the historical main with pbrun is not a qualified launch.

The PM controller `experiments/t8r_speed/piece_major_pb_action.py` at commit `88ab364a708507072524d151050c98c9a2f58067` imports only `run_direct_arm(deadline_s=600)`/`owned_cleanup` from the external owner:
`/mnt/shared/astra-resume-20261002/t8_performance/piece-major-common-c236b7aa/execution-owner-timing600/experiments/t8r_speed/paired_k32_action.py`, SHA256 `adcc84dba715a3fca0ed712601948f768f76777af11e77e1504ef91431578828` (read live and hash checked). It retains held-original-FD transport with ordinary PB Docker shim/scope, not invented residency leases. Peer reports its exercised PB-exclusive PM result `f10b6eaa`; that is PM-vs-legacy, **not paired qualification or paired profiles**.

Paired-specific controller/source/protocol/bit/native qualification is still required before a new action. Bind the owner hash, current published helper generation, immutable image, exact accepted numeric result/input/source/ELFs, fresh output namespace, and explicit root GO. Proposed resources: published pbrun, `--exclusive`, GPU demand1, CPU2/native1, total UMA memory16GiB with GPU subset8GiB, bounded600s owner deadline, preserved PB-assigned affinity and container scope. Native thread variables OMP/MKL/OPENBLAS/NUMEXPR/MAX_JOBS all1. This is a compatibility contract, not a ready-to-run or authorized command; paired main cannot inherit PM proof without its own bound wrapper.

Independent performance reviewer currently owns PM #868 only; no independent paired approval is implied. Root retains final acceptance/GO. Existing TS857 remains open for this outstanding performance proof; PR862 remains draft and opt-in. No additional paired measurement was submitted.

## Source-only summary integrity repair (#889), 2026-10-03

Review found that the existing reducer checked the count of raw events but
trusted a separately advertised median. Its CPU fixture even supplied30 raw
ones for every arm while advertising baseline2/candidate1. The new control
uses consistent even-length samples and rejects invented medians, NaN,
infinity, zero, negative or boolean samples, and boolean advertised medians.
The existing `serving.timing_panel.timing_summary` owns finite-positive
validation and the conventional median; no second statistical rule is added.

Causal PB `a61f802516772ca79521191c2e8ed837e6bd595dfa6524efef770fdcf8c81c4f`
on the old reducer recorded7 failed/7 passed, with all seven new corruptions
accepted when they should refuse. This is a synthetic CPU-control population,
not replay of the accepted15 numeric outputs or the historical ABBA data.
Frozen source/input manifests and native banks retain their original hashes;
changed controller code needs its own bound source qualification before any
future GPU action. Historical results above are neither recomputed nor promoted.

Corrected PB `4d6418a789c0df1c0e00a06ab2e810f800985d91ec389dd01c0e887a0eb21aec`
completed28 CPU controls, zero skips/uncollected/CUDA allocations onDL380G10:
the full paired-action file and the existing torch-free canonical timing owner
controls. No GPU smoke, numeric replay, real-data reduction or new native build
ran. This repair closes only the raw-event integrity defect, not #857 or #855.

The child PRs actual hosted pure run then exposed11 inherited missing-Torch
failures inside mixed pure/Torch test files. Only those four tensor/helper
test functions now report the absent optional dependency with importorskip;
the remaining pure C++/admission controls stay collected. PB
`13967d2d5fe15b739cfb8a3174f7b0bfd1ce4f3515acc05547c578bfb9adddbb`
executed all11 affected cases with real CPU Torch:11 passes, zero skips,
zero uncollected modules/CUDA allocations. Canonical receipt
`2478d7ca8960f8702def8a56ccee6f15b8b88e05a058bb46cb4400cf5afa6941`.
This includes the source-extracted host C++ resource probe, not a new CUDA
native build or native numerical qualification.
