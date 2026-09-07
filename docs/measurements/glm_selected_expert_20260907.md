# GLM selected-expert functional control — 2026-09-07

The eager research path passed on the actual GLM E288/top8 owner: selected
packed-window tiles and scales were exact, and complete operator outputs were
bit-exact against the independent full stock FP8 owner. Incorrect expert maps
changed outputs in every case. This is an operator correctness result, not
a throughput, whole-model fit, generation or quality result.

Source: `be0934bbfb`, using `PreparedWindow.stack` and
`PreparedTesseraFp8Module.stack`. The packed owner holds an expert axis and
decodes device IDs into fresh tensors with an explicit eight-expert chunk
bound. The existing window reader and stock vLLM MoE kernels do the work;
there is no CPU weight stream, persistent decoded pool or production route
change. The current batch requires matching window layouts across experts.

The three original expert-0 projection wires from the
[q512 control](glm_q512_control_20260907.md) are repeated across E=288. To
exercise expert identity, the diagnostic down-projection row scale is multiplied
by `1 + bit(global_expert_id, row modulo 9)`. All 288 signatures are distinct.
These are deliberately distinct effective diagnostic weights, not 288 trained
experts. Routing uses `(37 * slot + 19) modulo 288`, with the selected expert
axis reversed before building stock's global-to-compact `expert_map`.

The oracle first compares compact stock tensors to the full independent stock
owner. It then checks every selected packed-decoded tile and scale against
the independent producer materialization and compares the resulting output.
Finally, it swaps two entries in the expert map and requires a changed output.
Both stock legs own their tensors; the result is not a workspace-alias check.

| Case | Tokens | Selected experts | Stock compact map max error | Packed selected max error | Deliberately wrong map max error |
|---|---:|---:|---:|---:|---:|
| Decode | 1 | 8 | 0 | 0 | 0.1875 |
| Every expert | 36 | 288 | 0 | 0 | 0.875 |
| Input multiplied by 64 | 4 | 32 | 0 | 0 | 32 |

The native owner has H=4096, I=2048, top8, clamp=10, TP=1 and EP=1. It is
constructed from the actual GLM model factory configuration and then invokes
the actual `Glm5NextMoE`/`RoutedExperts` class. Trained routing, shared-expert
execution and full-engine generation are outside this control. The original
three repeated-source resident controls also passed in this process.

## Allocator observations

Packed owners occupy 1,236,484,096 bytes for w13 and 617,119,744 for w2,
including their row scales: 1,853,603,840 bytes total. Temporary expansion
remains substantial. Each case synchronizes, preserves the prior global
high-water mark, resets peak counters, and captures decode peak **before**
the independent verification tensors are allocated.

| Case | Fresh decoded weight bytes | Decode peak above existing allocations | Peak including existing allocations and verification |
|---|---:|---:|---:|
| Decode | 201,326,592 | 2,365,816,832 | 18,963,419,648 |
| Every expert | 7,247,757,312 | 11,358,404,608 | 33,519,086,592 |
| Input multiplied by 64 | 805,306,368 | 3,775,102,976 | 20,373,567,488 |

The whole-control peak was 33,519,086,592 allocated bytes and 39,246,102,528
reserved bytes. It includes the resident consumer, full independent stock
oracle, packed research owners and verification temporaries. It is not an
isolated serving footprint. Arithmetic joining the per-case and retained
global counters was checked in the final audit.

No profiler-based speed or work-per-joule comparison was performed. Eager
`torch.unique` synchronizes dynamic cardinality, and the runtime reported
default MoE configurations for E=288, E=8 and E=32. These need profiling and
representative performance measurement before any serving recommendation.

## Execution and validation

The direct vLLM control ran on Sparklina, CPUs 5–8, with a 48 GiB container
cap from 19:28:19 to 19:30:18 UTC. The immutable official image was
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`;
all 4,967 inventoried core files stayed unchanged. Launcher and container
exited zero, no OOM occurred, and harness files stayed unchanged. Exact
container `81f696ffa2464385eff1aa15858a717d91ce1701bf5ec5436923ffd5d7b77629`
and PID 3311074 were independently absent after cleanup; the GPU process roster
was empty. No startup timing is used as a performance claim.

PB CPU validation at `922ea756` selected 66 files across 20 actions, each with
two xdist workers, worksteal distribution, one native thread per worker and
a 2 GiB reservation: **1,532 passed, 156 skipped, zero uncollected modules**.
All used CPU Torch 2.11.0. The skips were 137 GPU-dependent cases and 19 absent
box artifacts; each verbatim reason is retained in the audit. All action
exits, cleanup records, CAS payloads and source bundle hashes were checked.

Pre-change proofs comprise four missing-window-stack failures and two
missing-FP8-stack failures. The first FP8 attempt on Lina lacked Torch and
collected no tests; after installing the official CPU wheel in both Sparks'
scoped test environments, the exact original source snapshot reproduced both
failures. The final targeted window/FP8/reference suite passed 27 tests with
12 CUDA-only skips. Two additional empty-case refusal tests passed after
their demonstrated failures. The final harness compile passed through PB.

One separate nearby fix: prepared windows now clone initial-state provenance
so caller mutation cannot invalidate their private owner. Its regression
failed before the one-line fix and passed afterward. No wire bytes changed.

Evidence root: `/mnt/shared/tessera-glm-native-20260907`. Exact commands and
source/configuration bindings are in the request, launch and audit files.

| Artifact | SHA-256 |
|---|---|
| `selected-functional-audit.json` | `91420270e2d7e34c457bd5183cbdeadde8b1291d59d09ab194ff942f0d0fd2b2` |
| `selected-control-lina-01/selected-expert-receipt.json` | `8a5ef859646c27d463843eaac545aa3f8aa0694d3c734c2614e1eb944f9dd8de` |
| `selected-request-04.json` | `d77d8a27c8e7fc3c3ff4d4eaf5668847281b144815cfb85d74a042779afe4d75` |
| `selected-source-be0934bb.tar` | `22230eec4f6a9d28c017da21deb7a1953070ad318a7b638fc109c801de58dfcd` |
| `selected-impacted-audit.json` | `a047116f6ce1dd5b043b925ae529b8fb74670a20d70286b34aaffd6ab4309d35` |
| `selected-regressions-audit.json` | `cafc1530a43253553bdbf9b021b18d633860af91924af526f62eccc0bdef045e` |

Earlier unexecuted requests 01–03 and their source archives are retained as
superseded, with explicit predecessor bindings. Production remains resident;
packaged runtime cells, release pins and promotion gates are unchanged.

## Master integration — 2026-09-07

After prerequisite #414 merged, integration commit `0c4156e595` merged master
`9afbcfe752` without conflict and retained its full-engine resource changes.
All five serving/harness Python files remain byte-identical to the measured
`be0934bbfb` source. No GPU rerun was needed or performed.

The dependency selector covered 72 files, including newly merged full-engine
readers. PB fanout used 20 actions with two xdist workers, one native thread
per worker and 2 GiB per action: **1,680 passed, 156 skipped, zero uncollected
modules**. All tests ran on dl380g10 with CPU Torch 2.11.0; the skips remain
137 GPU-dependent cases and 19 absent box artifacts. All terminal exits,
cleanup records, CAS payloads and source bundle hashes were checked. A separate
portable PB compile action `60716b724aae` passed on Sparklina for the five
serving/harness modules and three affected test modules.

`selected-integration-source-identity.json` records paired hashes for the five
measured files, SHA-256
`99fbc331a7fc7660c63ad7d2ef94ba755334633e9df18f76d0f638a3601700f0`.
`selected-integration-audit.json` binds the selector, command, all 20 results,
compile, source identity and prior functional audit, SHA-256
`c2f9afa5b82522d3800e8d81e1f5a85970f13e61dd09b476d75892900b0242fe`.

## Final reader integration

Root merged the newly reviewed device-unpack reader from master
`bc4946285a1f8c66d0be159d641477a00892776d` at `33b2a813`.
The only conflict was the offline issue snapshot, regenerated through the
existing refresh tool. All five measured serving/harness files remain
byte-identical to the selected-expert measurement.

PB `618f3a0a4afc2c3803e33b6b9cab159ba8fe7cca8066d3a0a70081aae8ef1e8d`
passed 27 focused window, FP8 and issue-reference tests; 34 CUDA cases were
skipped explicitly (22 BODY-unpack and 12 serving cases), with zero uncollected
modules. Portable DL380 CPU4/memory4 GiB, Torch 2.11 CPU, worksteal/native
threads one. GPU coverage remains the separately recorded measurements;
no GPU run was repeated for this source-disjoint merge.
Root verified actual exits, canonical CAS receipt/payload, source bundle and
resource cleanup in `/home/rob/tmp/selected-reader-root-interaction-audit.json`.
The preceding action `172640ce0c0a8cff92535733c3f9aa31a21f5a9a0308221b95b1a97aa0489c5c`
was invalid collection: root named a nonexistent architecture test file;
pytest exited 5 and PB exited 1 with no success receipt. The corrected
selection above replaced it.
