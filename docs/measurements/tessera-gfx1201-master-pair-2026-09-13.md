# One commit, two instruction sets: the weights-only Tessera-16 wire is the same object

**Date:** 2026-09-13 · **Issue:** #472 · **Boxes:** wsl-gpu / DESKTOP-P5UOGNJ
(AMD RX 9070 XT, gfx1201, torch 2.11.0+rocm7.2.4, Triton 3.7.0+rocm) and
sparklina (GB10, sm121, torch 2.11.0+cu130, Triton 3.6.0) · **Runner:**
PrismaBuild only.

#472 established that a weights-only `TESSERA_BF16_K1_R1792` encode produces
the same wire on gfx1201 and on a GB10. Every run behind that statement was
taken on a different Tessera tree, though: the GB10 bytes came from the
serving-runtime pin `1c827abc`, the gfx1201 default-path bytes from #476's
branch head. "Only the HIP branch changed, so the CUDA bytes cannot have
moved" is an argument, not a measurement. This is the measurement: one commit,
both boxes, each taking the path it chooses for itself.

## What ran

`tessera.export.encode_linear` on Qwen3-0.6B
`model.layers.0.mlp.down_proj.weight` (1024x3072 bf16, sha256 `01cc5c35...`),
rung `TESSERA_BF16_K1_R1792`, weights-only: `grid`/`q256`/`name`/`verify` and
nothing else, the keyword surface `prismaquant.tessera_render` forwards. Two
timed repeats per box. The harness refuses to start if
`TESSERA_WINDOW_FUSED_MAX_RATE` is set, so the path each box takes is
`fused_available()`'s decision and not an environment variable's.

Tessera master `33d3a050d912a9669af10e7c7526496ffe5c06ab`, vendored into the
submitted snapshot and stamped into each result as `tessera_vendored_sha`
(`tessera.__version__` is `0.1.0` on every commit and identifies nothing).

## Result

| | wsl-gpu (gfx1201) | sparklina (GB10) |
|---|---|---|
| `fused_available()` | **false** | **true** |
| Viterbi path | reference | fused |
| blob sha256 | `873af26a17ebcec669295a2d...` | `873af26a17ebcec669295a2d...` |
| blob bytes | 2 788 066 | 2 788 066 |
| `encoder_fixture_id()` | `03bbc5b1c56d55e1...` | `03bbc5b1c56d55e1...` |
| render sha256 | `b5a441d79f98f7ef...` | `b5a441d79f98f7ef...` |
| render MSE | 7.603092910812848e-08 | 7.603092910812848e-08 |
| cold / warm | 30.81 s / 30.13 s | 3.74 s / 3.14 s |
| params per second (warm) | 104 k | 1.00 M |
| device memory | 2.46 GB framebuffer peak | 1.34 GB torch reserved |
| PrismaBuild cgroup peak | 1.52 GB | 1.11 GB |

`cmp` on the two blob files reports no difference. The four earlier
weights-only blobs (#472's `sparky-a`, `sparky-ref`, `wslgpu-ref`,
`wslgpu-default`) digest to the same `873af26a...`, so six encodes across two
instruction sets, two Viterbi paths and three Tessera trees have now produced
one wire.

A `mem_gb=4` demand was sized from this: 1.52 GB host peak on the larger side,
and on a GB10 the device bytes are charged to no memcg, so 1.11 + 1.34 GB is
the figure that box must actually hold.

## Throughput, and the one number that does not exist

The unit is 3 145 728 parameters. Warm, gfx1201 encodes it at 104 k
parameters per second and the GB10 at 1.00 M -- **9.6x**. The gap is the
missing kernel, not the board: on the *reference* path a GB10 is slower than
gfx1201 (#472, 51.9 s against 30.9 s warm). #481 tracks the HIP `_mul` that
would close it.

#472's first figure was 3.5x, and it was taken on sparky rather than on
sparklina. That box was not idle: its PrismaBuild box windows read 79.9 W and
82.0 W mean GPU power (86.0 W peak, 61% of the 140 W reference) across the two
encode actions `eaed04d3bca7` and `4923ea1946ab`, against 37.3 W mean here for
the same unit -- more box-level power for the same work, and slower
(`sparky-a` 7.16 s cold / 8.75 s warm, `sparky-b` 10.12 s / 8.48 s, against
3.74 s / 3.14 s on idle sparklina). What else the box was running is not
recorded; that it was running something is. Plan with the idle figure.

**There is no power reading on wsl-gpu, and no work-per-joule comparison is
possible.** PrismaBuild's box window for the gfx1201 action carries a GPU group
whose `telemetry_class` is `memory_only`: device name, UUID and framebuffer
bytes, sourced from the admission broker's HIP probe. WSL2 runs no amdgpu
driver, so `rocm-smi` answers `Driver not initialized`, `/sys/class/hwmon` is
empty and `/sys/class/drm` holds only `version` -- there is no sensor to read,
at any sampling rate. The GB10 side does have one (`pqteld`: 37.3 W mean,
73.2 W peak against the 140 W SoC reference, 52% of envelope at peak), and it
is recorded here for the GB10 arm alone. `gpu_utilization` is not substituted
for the missing half: on the GB10 action it read 96% at peak while power sat
at half the envelope, which is the reading CLAUDE.md principle 15 exists to
refuse -- and on the gfx1201 action there is no utilization series either.

## Receipts

| label | tag | key | state |
|---|---|---|---|
| `wslgpu-master` | wsl-gpu | `1b83796224a492880282758b602e348d3892d407b817e98b0289b20b6f414a11` | done, rc 0, 66.7 s |
| `lina-master` | sparklina | `03754b58a001b7f4fec642c71a3ca533e4ca600e24bdf65109b8ec9f2df6acaf` | done, rc 0, 10.9 s |

Both: `pbrun.py --priority -10 --cpus 4 --demand gpu=1,mem_gb=4 --timeout-s
1500 --detach`. Records under
`/mnt/shared/prismabuild-fleet/pb-queue/done/<key>.json`; result JSON and wire
blobs under `/mnt/shared/agents-w472-rocm/<label>/`.
