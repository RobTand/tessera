# The window Viterbi's R = 8 cliff was a register spill, and the flat unroll was never right

**Date:** 2026-09-02 · **Box:** sparky (GB10, sm_121) · **Issue:** Tessera #11
**Files:** `src/tessera/window_viterbi.py`, `src/tessera/encode.py`
**Reproducers:** `experiments/window_viterbi_r8_diagnosis.py`,
`experiments/window_viterbi_scan_unroll_sweep.py`

## 1. What was recorded, and what it left open

`docs/measurements/tessera-bf16-route-2026-09-02.md` §10 measured the shape of
the cliff and attributed it correctly to the Triton step kernel: the reference
torch chain is flat in the rate (per step the trellis evaluates
`low * FAN = 2^L` transitions whatever `R` is, so flatness is what a correct
implementation must show) while the fused path went 0.75 s / 1.47 s / 65.0 s at
R = 6 / 7 / 8 on 1024x1024 at L = 14 — **9.8x slower than the path it exists to
replace**, having been 8.7x faster two bits down.

It left two things open, and named them as open:

- the *mechanism* — the unroll doubling per bit accounts for a 2x-per-bit
  trend and "does **not** account for the extra 20-40x at `FAN = 256`";
- the *fix* — a `_tile` change to stop the `[BC, BL]` tile collapsing, offered
  explicitly as "a hypothesis, not a measurement", with the warning that it may
  spill worse than the unroll costs.

`0739f33` then shipped the dispatch rule (`WINDOW_FUSED_MAX_RATE = 7`), which
made R = 8 encodes ~9x faster **by not using the kernel**. That is what issue
#11 calls a workaround, and it is.

## 2. The mechanism, read off the compiled kernel

The class minimum is `FAN - 1` **independent** masked loads feeding a
**dependent** select chain. Flat-unrolled, the scheduler hoists the loads to
cover their latency and holds one live value per hoisted load per element a
thread owns. `_tile` shrinks the tile as `2048 / FAN` while the iteration count
grows as `FAN`, so the live set grows like `FAN` however the tile is sized.

`experiments/window_viterbi_r8_diagnosis.py` reads `n_regs` and `n_spills`
straight off the `CompiledKernel` the launch returns — not a timing, so it is
valid on a contended box. L = 14, and the launch geometry is otherwise
identical across every rate (`grid = (16, 16)`, `num_warps = 4`, `BC = 2`,
`front_out = 2048` elements a program):

| R | FAN | BL | lanes/thread | n_regs | n_spills | PTX lines |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 64 | 1.00 | 40 | 0 | 705 |
| 5 | 32 | 32 | 0.50 | 64 | 0 | 1913 |
| 6 | 64 | 16 | 0.25 | 96 | 0 | 3376 |
| 7 | 128 | 8 | 0.125 | **128** | 0 | 6327 |
| 8 | 256 | 4 | 0.0625 | 40 | **690** | 8363 |

`n_regs` tracks the hoisted live set right up to R = 7 and then **falls** at
R = 8 while 690 bytes of spill appear: ptxas stopped trying to keep the chain
in registers and put it in local memory. A spill inside a serial dependent
chain is a local-memory round trip per predecessor, 255 of them per class per
step. That is the extra 20-40x, and it is a **step function**, not a trend —
R = 7's live set fits in 128 registers and R = 8's does not fit in 255.

**This refutes the recorded candidate fix.** Widening `BL` puts *more* elements
under each hoisted load, so it moves the spill down the rate ladder rather than
away. The warning attached to that hypothesis in the BF16 doc was right, and
the hypothesis is now dead rather than untested.

## 3. The fix, and the part nobody had measured

Spell the scan as a runtime `tl.range` with an IR unroll factor instead of a
flat `tl.static_range`. The loop body is the same six lines in the same order,
so the selects run in the same order and the answer is the same answer; what
changes is how many loads may be in flight.

At R = 8 that is the rescue the mechanism predicts. What was **not** predicted,
and is the more useful half of the result, is that the flat unroll was costing
2-3x at rates that never spilled at all — wherever the live set was large
enough to crowd the register file and not yet large enough to spill.
`experiments/window_viterbi_scan_unroll_sweep.py`, 1024x1024 at L = 14, one
process, an idle box (`nvidia-smi --query-compute-apps` empty, 6.9 W of a
~140 W envelope before each run), seconds:

| R | reference | flat unroll | loop ×32 | flat→loop | loop vs reference |
|---|---:|---:|---:|---:|---:|
| 4 | 4.079 | 0.173 | 0.161 | 1.07x | 25.3x |
| 6 | 4.003 | 0.498 | 0.193 | 2.58x | 20.8x |
| 7 | 4.131 | 0.927 | 0.290 | 3.20x | 14.2x |
| 8 | 3.974 | **38.190** | **0.490** | **77.9x** | **8.1x** |

`states` are `torch.equal` and `sse` is the same float against the reference in
every cell. `n_regs` is 40 and `n_spills` 0 at every rate under the loop.

There is **no rate at which the flat unroll wins**, so the policy is uniform
rather than a threshold: `_scan_unroll` returns the same factor everywhere. It
holds off L = 14 as well — L = 12 (R = 4/6/8: 2.7x / 2.2x / 74x), L = 16
(R = 4/8: 1.04x / 81x) and arity 2 at L = 14 (R = 7/8: 1.54x / 2.4x, where the
flat form was already spilling 44 bytes at 255 registers).

**The factor is measured, not derived.** At R = 8, factors 1 / 2 / 4 / 8 / 16 /
32 / 64 / 128 ran 1.491 / 0.836 / 0.552 / 0.568 / 0.516 / 0.499 / 0.480 /
0.477 s, all with `n_spills = 0`; 8 / 16 / 32 were then swept across R = 4..8
at L = 12 and L = 14. **32** is at or within 3% of the best in every cell and
spills nowhere. `TESSERA_WINDOW_SCAN_UNROLL` moves it, and `=0` restores the
flat unroll, which is how the table above is reproduced.

## 4. The crossover moves, because the cliff was what put it where it was

`WINDOW_FUSED_MAX_RATE` was **7**, and 7 was the cliff's edge, not the
algorithm's. With the scan fixed the fused path degrades smoothly and the
crossover is where the work actually runs out. Same harness, L = 14:

| R | 4 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference (s) | 4.08 | 4.00 | 4.13 | 3.97 | 4.14 | 4.19 | 4.47 | 4.30 |
| fused (s) | 0.16 | 0.19 | 0.29 | 0.49 | 0.89 | 1.67 | 3.16 | 6.01 |
| fused vs ref | 25.3x | 20.8x | 14.2x | 8.1x | 4.6x | 2.5x | 1.4x | **0.72x** |

so `WINDOW_FUSED_MAX_RATE = 11`. **The wire's own rate domain is 1..8** (code
rate 1..8 over an 8-bit-native alphabet, which is what
`runtime_contract.json`'s `reader_rate_bound` names), so every rate an artifact
can carry now runs on the kernel. 11 is where a caller reaching *past* the wire
stops being served by it.

Why this matters beyond a benchmark: R = 6-8 is Tessera-8 and the whole BF16
route, where the alphabet keeps paying ~1.6-1.8x per bit. Every one of those
encodes has been running the reference since `0739f33`.

## 5. What did not change

**Not one byte.** `experiments/audit_byte_baseline.py` before and after, run
**with the GPU visible** so the fused path is the one under test (the default
`CUDA_VISIBLE_DEVICES=""` would have exercised only the torch reference and
proved nothing about this change):

```
0 changed of 36
```

14 encodes and 22 decodes, identical. The encode cases reach `viterbi_window`
at R = 4 (`e4m3-1024`, window body over the CHANNEL plane) and at R = 7 arity 2
(`e2m1x2-sub-512c`), which are rates whose scan spelling this change moves.
`tests/test_window_viterbi_fast.py` holds the identity line directly: 52
passed, including the new `test_both_scan_spellings_are_one_answer`, which runs
both spellings and the reference at R = 4, 7, 8 and requires `torch.equal` on
the states and `==` on the `sse` float.

## 6. The guards

Three tests, and what each would have caught:

- `test_the_step_kernel_spills_at_no_rate_the_wire_can_carry` — the mechanism,
  as a property of the compiled kernel rather than of the clock, so it does not
  flake on a loaded box. On `master` it fails with
  `the fused window step spills 690 bytes at R=8 (n_regs=40)`.
- `test_auto_takes_the_fused_path_at_every_rate_the_wire_can_carry` — the
  crossover must sit above the wire, not inside it. On `master`:
  `WINDOW_FUSED_MAX_RATE=7 excludes rate 8, which the wire carries`.
- `test_both_scan_spellings_are_one_answer` — a regression guard, not a
  fail-before: it pins that the knob stays a machine choice. It cannot run on
  `master`, which has no knob.

The two pre-existing crossover tests hardcoded `L = 10` and asked for
`WINDOW_FUSED_MAX_RATE + 1`, which is now 12 and violates `rate <= window_bits`.
They derive `L` from the constant now, so the next crossover move is one edit
and not three.

## 7. Scope

Measured on GB10 (sm_121) with Triton 3.6.0 and torch 2.11.0+cu130, at
L = 12, 14 and 16, arity 1 and 2, rates 1-12, on random targets. **Not**
measured: another GPU architecture, another Triton version, or a real export's
end-to-end wall clock (the sweep times `viterbi_window` on a synthetic tensor,
which is the kernel's own cost and not an export's). A box whose scheduler
behaves differently can restore the old spelling with
`TESSERA_WINDOW_SCAN_UNROLL=0`, and should re-run
`window_viterbi_r8_diagnosis.py` first — the register table is the evidence,
and it is cheap.
