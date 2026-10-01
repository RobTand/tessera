# Deep word staging on single-rate launches (opt-in)

Refs #807.

## Change

`routed_fused_window.cu` can stage a single-rate launch's words more than two
chunks ahead of their decode. The switch is the environment variable
`TESSERA_ROUTED_FUSED_WORD_STAGES`:

- Unset or 3: today's kernel. The loop's wait and issue distance are now
  written as `AHEAD = WS - 1`, which is the same constant at two and three
  stages, and the three-stage prologue is unchanged.
- n > 3: `routed_fused` builds the E4M3-instruction library with
  `-DTESSERA_ROUTED_FUSED_WORD_STAGES_ONE_RUN=n` into its own `_ws<n>` build
  directory. Each single-rate launch then cycles through
  `deep_word_stages(mode, slot)` word stages: n, or as many as sm_121's
  101,376 B block holds at the widest superblock if that is fewer. Each stage
  carries its chunk's staged stream history at its tail (`deep_stage_ints`),
  so the 768 B history region is laid out but unused.
- Two-run launches, the value library and the 16-bit E4M3 library keep three
  (or two) stages.

The copies, the decode, the MMAs and their order are the three-stage
kernel's, so the output is bitwise the three-stage launch's by construction.
No contract, rung, route or `executes` entry changes, and the default does
not change.

The kernel's `constexpr word_stages(mode, slot_words, two)` is the one
authority for the stage count. `routed_fused.word_stages` mirrors it and
takes the pair's two-run flag as a required keyword. The library publishes
`WORD_STAGES_ONE_RUN`, which the loader checks. `launch_pair` sizes the block
from the pair's compile-time slot and refuses a `Params.slot_words` that is
not that slot.

| Requested stages | Gate/up slot 4 / 8 / 12 / 16 | Down and dense slot 4 / 8 / 12 / 16 |
|---|---|---|
| 5 | 5 / 5 / 5 / 5 | 5 / 5 / 5 / 5 |
| 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 |
| 64 (clamped) | 39 / 21 / 15 / 11 | 51 / 28 / 19 / 15 |

## Why [E]

At three stages, a chunk's words are issued two chunks ahead of their decode.
If the producers' chunk period is pinned by loaded memory latency, more
stages in flight shorten it.

For this reading:
- The chunk period is about 620-750 ns at every M (R1024 gate/up: M = 1 640,
  512 617, 2048 653 ns).
- 14% more SM clock bought 0.5% (R1024 gate/up, M = 512: 7.624 ms under NCU
  at 2.15 GHz, 7.583 ms in the bench at 2.42-2.48 GHz).
- At M = 512, DRAM reads run at 70% of 232 GB/s.

Against it (NCU, M = 512, R1024 gate/up,
`l512-20260930/ab-staged-20260930T191324Z/port-ncu/src-k2.csv`):
- The producers' own word wait (`DEPBAR`) is 0.90% of warp samples.
- The producer barrier (`BAR_PROD`) is 10.0%, and at most 7 x 0.90 = 6.3% of
  that can be word latency seen through the barrier. At least 37% of it is
  skew in the producers' own decode.
- The rest of producer time is issue on the shared/MIO pipe: wait 14.3%,
  short_sb 8.0%, mio 6.7%.

That brackets the lever at M = 512 at about 2-15% of the routed time.

## Measurement

### Compile gate [M]

PrismaBuild row `312f56a0`: CPU only, dl380g10, 16 CPUs, 184 s, exit 1 (one
pair's expectation was too strict; see the last item). Root:
`/mnt/shared/tessera-measurements/t8r-speed-20260929/l512-20260930/compile-gate-20261001T172619Z`,
with `out/gate.txt` and `out/gate.json`. The harness is
`experiments/t8r_speed/compile_gate.sh` and `sass_cmp.py` on
`claude/routed-d2-harness` `2c834dc4a7`.

It compiles `routed_fused_window.cu` from source snapshots of master
`dbce87af05`, D1 `07ebdf8284` and this change `d1bb964426`, with each
library's production flags. The toolchain is nvcc and ptxas 13.0.88, g++-13,
and the image's torch 2.13.0+cu130 headers. `sass_cmp.py` compares SASS
instruction text and resource usage per kernel.

- **Toolchain parity.** The D1 E4M3-instruction library object built on sparky
  by PrismaBuild (`abl-lut-20261001T111516Z/ext-fix`) and the same source
  built here match on 137 of 137 kernels.
- **Flag off against D1.** Identical on all four libraries: value 101 of 101
  kernels, 16-bit E4M3 107 of 107, E4M3-instruction 137 of 137, E2M1 77 of 77.
- **Flag on, at 5, 8 and 64 requested stages.** The 72 single-rate kernels
  change, and every two-run and non-routed kernel is identical (65 of 137). No
  kernel uses local memory or stack. At 5 and 8 stages no kernel's register
  count changes (at most 122). At 64 (clamped), seven kernels move between
  115-122 and 120-124 registers.
- **D1 against master.** The value, 16-bit E4M3 and E2M1 libraries are
  identical. In the E4M3-instruction library, 40 of the 72 single-rate kernels
  are identical. Each of the other 32 differs in one instruction: an integer
  subtract in the descriptor search with its operands swapped
  (`IADD3 R5, PT, PT, R13, -R4, RZ` against `IADD3 R5, PT, PT, -R4, R13, RZ`).
  The instruction count and registers are unchanged. `d1-master/single_rate_diff.py`
  classifies every changed line, and none is left unclassified. Integer
  addition is exact, so this cannot change an output. The pair expected
  bit-identical SASS on single-rate kernels, which these 32 kernels fail; that
  is the row's exit 1. The two-run kernels differ as D1 intends: they return
  to the register path, at +1 to +9 registers on the 64-row variants.

Go/no-go: PrismaBuild row `0fc0bb3b`, which times the prototype of this change
(`analysis/make_deep.py`, the same staging) at 5 and 8 stages against D1. It
runs on the A8S release artifact, R1024 L10, at M = 1, 512 and 2048 plus
recorded routing, forward then reverse. Go needs about 2% or more at M = 2048
in both passes. Under that, this change is a measured negative and stays in
draft.

Before default-on:
- Bitwise on every cell: the A8S routed L3/L10/L24/L44/L45 groups and the 14
  dense groups, at M = 1-8, 512, 2048 and 2049, on balanced and recorded
  routing, forward and reverse.
- The routed and dense window test suites pass with the flag on and off.
- Faster on every timed cell, decode M = 1-8 included, with an NCU profile
  after the change at M = 2048 and Netdata power read against the envelope.
- The served gate (TR3 KL and decode greedy identity) before merge.
