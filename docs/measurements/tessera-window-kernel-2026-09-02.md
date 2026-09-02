# The fused window kernel: what it decodes, what it costs, and what it does not beat

2026-09-02. `src/tessera/kernel_window.py`, `tests/test_kernel_window.py`,
`experiments/bench_kernel_window.py`. Commits `1675a1f`, `05558b3`, `793c09d`,
`3a7d026`, `b1576f0`, `79f9c72` on `worktree-agent-ad5a8f4b23cadfa9f`.

**The headline, stated first because it is the answer to the question that was
asked.** The fused decoder replaces the pure-torch window reader at **40-202x**
(eager) and **6-32x** (torch.compile) per unit, and the whole 196-unit reach
checkpoint decodes DRAM-cold in **9.72 ms** (49.57 us/unit). The torch reader
was not run on a cold rotation, so there is no measured whole-checkpoint
baseline to divide that by, and this receipt does not extrapolate one. The
fused GEMV **does not beat the
resident FP8 GEMV per token at any measured shape or model**: summed over a
model's Linears at M=1 it is **1.065x behind on Qwen3-0.6B**, **2.343x behind on
Qwen3-4B** and **1.271x behind on the 27B shape list**, against a resident FP8
tile plus a fused per-token activation quantiser. The reason is **occupancy, not
bytes**: on both kernels *no* memory unit ncu exposes on this device is above
37% of its peak, and L2 -- the unit that stands in front of memory -- is at
15-22%, so reading 4 bpp instead of 8 buys nothing while the kernel cannot keep
the memory pipeline fed. (**ncu's DRAM counters are unavailable on GB10** --
`dram__*` returns n/a on this integrated part -- so nothing below is a measured
DRAM number; see section 5.) The clearest single statement is `17408x5120`: both
kernels are slow there, the fused GEMV reads **half the bytes** `_scaled_mm`
does, and the two are **level (1.02x)**. A kernel whose time does not move when
its bytes halve is not limited by bytes.

---

## 0. Conditions. Read these before any number below.

Every number here was taken on **sparklina** (GB10, sm_121, 48 SM, 24 MB L2,
273 GB/s spec, ~140 W envelope) with the venv
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python` (torch 2.11+cu130).

**The box was shared for the whole session.** Four to seven other workers' CUDA
processes were resident throughout (exports, pytest runs, campaign drivers).
Every bench row in the JSONs carries `mean_w` / `max_w` and
`cuda_procs_max` / `cuda_procs_min` sampled across that row's own timing window,
because a timing taken beside five other processes is a different number from
the same timing taken alone and a receipt that cannot say which it is cannot be
read later. Idle draw before each arm was **30-40 W** of a ~140 W envelope,
which is the honest reading of how loaded the card was: not idle, not saturated.

Arms and their conditions:

| arm | local time | other CUDA procs | serve container up |
|---|---|---|---|
| decode | 12:58 | 6-7 | no |
| gemv | 13:02 | 5 | no |
| ablate | 13:03 | 5 | no |
| ncu gemv 2560x9728, 1024x3072 | 13:04-13:11 | 4-6 | from ~13:06 |
| ncu decode + the M=4/M=8 sweep | 13:12-13:20 | 4-6 | **yes, two containers** |
| profile | 13:07 | 5 | **yes, starting** |
| bandwidth | 13:08 | 4 | **yes** |

**The serve lock, and a mistake I made with it.** I took sparklina's serve lock
at 16:16:22Z by writing `/home/rob/tessera-runs/serve.lock/owner` by hand rather
than through `experiments/serve_lock.sh` (whose owner line is
`$$ <utc> $SERVE_LOCK_OWNER`; mine had no pid). By **17:06:20Z** that lock was
gone and pid 2762875 (`experiments/serve_and_dump_kl.sh`, a BF16 teacher dump)
held it. **It was not the stale rule that removed mine**: `serve_lock.sh` breaks
a lock only when it is older than 3600 s *and* no container is running, and mine
was 50 minutes old at 17:06:20Z. The mechanism is unknown; the likeliest is
another worker's unconditional `serve_lock_release`. Taken with my own mistake
below, **two different workers deleted a lock they did not hold in one
afternoon** — that is a systemic hole in the protocol, not one person's slip, and
it is flagged as such to the coordinator. — it had been *waiting* since ~12:58 and started its vLLM container at
~13:06. So the decode, gemv and ablate arms ran with no serve on the box; the
**profile and bandwidth arms did not**, and the later ncu runs (the decoder, and
the M=4/M=8 comparison) ran with **two** containers up. ncu serialises the
kernel it profiles and its register/occupancy/percent-of-peak counters are
architectural, so those are unaffected; its **`Duration` on those later runs is
the figure to treat as an upper bound**.

At **~17:08Z I then deleted that lock believing it was still mine.** It was not.
Pid 2726607 acquired it 2.5 minutes later and started a second vLLM, so two
serves overlapped for roughly a minute and two more `serve_and_dump_kl.sh` runs
queued behind. **Harm check: none found.** Both dumps completed at full size —
`/mnt/shared/tessera-kl/qwen_teacher_bf16_lina.json.npz` (33,532,602 B, 13:09)
and `qwen_rot_teacher_lina.json.npz` (33,532,598 B, 13:12), each with its
`.meta.json`. Two vLLM containers were up when I finished. This is recorded
because the next worker to read a KL number taken on sparklina this afternoon
needs to know a second serve was briefly beside it.

**Bandwidth denominators — three probes agree, one arm disagrees, unresolved.**
On sparklina at 13:08 with 4 other processes: SM read-only **103.0 GB/s**, SM
read+write **102.8**, copy engine **101.4** (256 MiB each,
`bench-bandwidth-20260902-130832.json`). But `torch._scaled_mm` moved a cold
12.5 MB FP8 tile at **205.6 GB/s** in the same session. The three probes and the
one real kernel disagree by 2x and I did not resolve it. **The authority used
below is ncu's own percent-of-peak figures**, which are measured against the
device's own counters rather than against any probe of mine -- and which are
therefore unaffected by this discrepancy. What ncu cannot give on GB10 is a DRAM
counter at all (`dram__*`: n/a), so its `Max Bandwidth` is the maximum over the
memory pipelines it *can* see (L1/TEX, L2), not a DRAM utilisation. The
unresolved 103-vs-205 gap is listed again in section 6.

The earlier read-only probe was `torch.sum`, which is a two-stage reduction
whose second stage is a separate launch: it read **11.5 GB/s** on a box where a
copy ran at 61. It is replaced by a one-pass Triton reader that retires one
value per program — the shape every kernel here has. Do not use a reduction as
a bandwidth probe.

---

## 1. What was built

`src/tessera/kernel_window.py` (~1030 lines) — the fused lane for the WINDOW
body (schema minor 2), grid `E4M3`, `window_bits` 14, span 1, over the CHANNEL
scale plane. Nothing here re-derives a table the reference decoder builds: the
packing is `lane_planes.pack_window_planes`, the code table is the unit's own
ALPHABET plane through the grid's `native` map, the row scale is
`scale_channel.channel_scale_field`'s expression.

Two kernels and two families over them:

- `_decode_kernel` -> `decode_fp8_tile(...) -> uint8 [rows, cols]` (+ the fp32
  row scale beside it) for the FP8 route, and `decode_value_tile(...) ->
  <table dtype> [rows, cols]` **with the row scale already applied** for the
  value/BF16 route. One kernel, `SCALED` a `tl.constexpr`, output allocated at
  `table.dtype`.
- `_gemv_kernel` -> `window_gemv` / `window_value_linear`: `x @ W.T` at
  M <= `GEMV_MAX_M` = 8 that never materialises the tile, fp32 accumulation,
  split-K over atomics, row scale applied in the epilogue.
- `window_linear` / `PreparedWindowUnit.linear` dispatch on M *inside* the op
  (where `x.shape[0]` is concrete), so a compiled forward never guards on the
  token dimension.

All ops are `torch.library.custom_op(..., mutates_args=())` with
`register_fake`, static-shaped, no `int()` on the token dim, no `data_ptr`
fingerprinting, no in-place mutation of an aliased pool.

**TP (amendment 1, and the coordinator's later concrete form).** A TP shard is
an ordinary unit whose columns begin from the last `window_bits` bits of the
parent column's stream, and this lane supports **both spellings of that start,
which are provably the same object**:

- **In the wire.** `pack_window_planes` puts `L` zero bits in front of every
  column, and code `t`'s window begins at `offset + (t + 1) * R`, so for
  `(t + 1) * R < L` that window *reaches into the pad*. The kernels read the
  window out of the stream — they do not assume the pad is zero — so a shard
  that fills its pad with its parent's last `L` bits decodes correctly with no
  extra plane and no flag. Attested by
  `test_the_initial_window_may_travel_in_the_stream_pad`, byte-identical to the
  definition (section 2).
- **Beside the wire.** Every entry point also takes an `initial` window as an
  input tensor, one int64 per column, default zeros. The kernel adds it in the
  `(t + 1) * R < L` positions, which is exactly the arithmetic the pad performs,
  and the test asserts the two agree byte for byte.

Nothing bakes the zero start in, and the lane needs no change when
`tessera.layout.slice_unit` lands: whichever representation the slicer emits,
this decodes it.

**The BF16 family (amendment 2).** The table dtype and element width are
parameters: `uint8` E4M3 byte codes (16 KB) for the FP8 route, `bfloat16` values
(32 KB) for the value route, and the GEMV accumulates from either. `float16` is
the FP8 route's internal GEMV table and is *exact* for this grid (asserted on
all 254 legal bytes, not argued: three mantissa bits over 2^-9..448 sits inside
fp16's normals).

**The seam W2 calls.** `window_module_decode(units)`,
`window_module_row_scale(units)`, `window_module_linear(x, units)` — pure python
over prepared units, one launch per unit, concatenating on the output axis.
`window_module_row_scale` **refuses the value family**: that scale is already
inside the decoded tile and handing it back to a caller that would pass it to
`_scaled_mm` beside the tile is a silent second multiplication.

The named call sites in the lane (**not edited** — W2 owns that tree):

- `src/tessera/serving/fp8_route.py:115` `prepare_tessera_fp8_module` — builds
  `prepare_window(...)` at `:153` and byte-compares against `materialize_fp8` at
  `:157`. This is where `kernel_window.prepare_from_parsed` / `prepare_window_unit`
  replaces the torch reader's prepared object; the two expose the same
  `steps` / `decode()` surface deliberately.
- `src/tessera/serving/fp8_route.py:108` `PreparedTesseraFp8Module.decode()` —
  becomes `window_module_decode`.
- `src/tessera/serving/fp8_route.py:250` `TesseraFp8LinearMethod.apply` — the
  forward. At M <= 8 it can call `window_module_linear` instead of decode-then-mm.

**Activation contract (principle 14) — W2's call, not mine.** The FP8 route's
GEMV multiplies **bf16 activations** directly against the reconstructed weight,
i.e. W4A16-on-the-wire, where the decode-then-`_scaled_mm` path is W8A8. A lane
that turns the GEMV branch on is changing what its `emit_route` record must say.
The docstrings on `window_linear` and `window_module_linear` say so; the gate
belongs in the lane, and this receipt does not assert what the runtime executes.

---

## 2. Acceptance item 1 — bit exactness

`tests/test_kernel_window.py`, **31 passed** on sparklina
(`experiments/results/kernel-window/pytest_blockc16.log`, 128 s, at the tuned
`BLOCK_C=16`; `pytest3.log` is the same suite at 64, 27 passed before the two
column-mask shapes and the stream-pad case were added). Skips cleanly without
CUDA (`pytest.mark.skipif(not torch.cuda.is_available())`) and without the reach
checkpoint (a second mark), so a box with neither runs zero and fails none.

```
ssh sparklina 'cd /home/rob/tmp/wt-kernel && PYTHONPATH=src TMPDIR=/home/rob/tmp \
  TRITON_CACHE_DIR=/home/rob/.triton-cache \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest tests/test_kernel_window.py -q'
```

- **All 196 units** of the reach checkpoint: `torch.equal(prepared.decode(),
  materialize_fp8(...))` and `torch.equal(prepared.row_scale, scale.float())`.
- **The stock twin's tensors** (12 units): the fused tile equals the *written
  checkpoint's* `weight` bytes and `weight_scale`, not merely the decoder's
  output — an agreement with the decoder alone would not catch a checkpoint
  written from something else.
- **A fused module's stacking**: `window_module_decode` over a `gate_up_proj`'s
  roles equals the twin's single stacked tensor. This is the only test that
  would catch a helper which decodes every unit correctly and concatenates them
  backwards.
- **Synthetic units** at `(256,128)`, `(200,176)`, `(128,1024)`, `(96,80)`,
  `(64,64)`, `(96,37)`, `(37,1)`, `(72,45)` and more — deliberately including
  non-multiples of the 256-row block, of the 32-lane x 8-vector tile, and of
  **every plausible `BLOCK_C`** — against a step-at-a-time walk of the body's
  definition, not against another closed form that could share a mistake. The
  `37`-column shapes exist because `BLOCK_C` is a tuned constant: `176` and
  `112` are off 64 but *multiples of 16*, so at the new default they would have
  stopped exercising the column mask, silently.
- **Nonzero initial window, both representations.** A random per-column start
  passed as a side tensor decodes correctly and differs from the zero start, in
  both families. And — the TP branch's representation —
  `test_the_initial_window_may_travel_in_the_stream_pad` writes the start into
  the **wire's own `L` pad bits** (which is what `pack_window_planes` leaves in
  front of every column) and decodes with `initial = 0`: byte-identical to the
  side-tensor decode *and* to the step-at-a-time definition, with the GEMV on
  the same stream inside tolerance of its own tile. The kernel reads the window
  out of the stream rather than assuming the pad is zero, so a shard needs no
  extra plane; the side tensor stays as the equivalent second spelling.
- **Compiled forward**: `torch.compile(fullgraph=True, dynamic=False)` over both
  families' ops, held to bf16's own step (split-K sums in a launch-dependent
  order).
- **The families refuse each other's tables** (`decode_fp8_tile` on a float
  table, `prepare_window_values` on a uint8 table), and a module refuses mixed
  families.

## 3. Acceptance item 2 — the GEMV's tolerance

`test_gemv_matches_the_dequantised_reference` holds `window_gemv(x, ...)` to
**2e-6 relative** against `(decoded_tile.float() * row_scale) @ x.float()` on
real units with random bf16 `x`. The bar is derived, not chosen: both sides
accumulate in fp32 over `cols` <= 9728 terms, so the worst-case relative error
is `~K * 2^-24` in the pathological cancelling case and `~sqrt(K) * 2^-24` in
practice: at K=3072 that is 1.8e-4 worst case and 1.0e-5 typical, so 2e-6 of
the row's own magnitude sits an order under the practical bound and three under
the worst case. Checked at M = 1, 2, 4, 8 on six real units. A **one-hot
probe** is checked separately at *exact* equality, where nothing is summed and
no tolerance is owed — that is the test that would catch a scale or an index
error hiding inside a tolerance.

Two derived bars appear elsewhere and are stated because they are *not* the same
thing:

- The **module** seam returns bf16, so it is held to `2^-8` at M<=8 and `8e-2`
  past the cap (where activations are quantised to E4M3).
- The **value family's** GEMV is compared to the **unrounded**
  `table[state] * scale` product, not to the rounded tile: the kernel never
  materialises the tile, and holding a more accurate kernel to a less accurate
  reference is a bug in the test, which is what it was until it was fixed.

## 4. Acceptance item 3 — the timing table

### 4a. Decode, per unit — before and after the `BLOCK_C` flip

The decoder's column-block width was 64 when the table below was first taken and
is **16** now; the flip is a tiling constant, it changes no byte, and section 6
lever 1 has the ncu evidence that chose it. Both tables are kept, because the
delta is the claim.

**After (`BLOCK_C=16`, the default), `bench-decode-20260902-134054.json`**, taken
while holding sparklina's serve lock (`decode_blockc16_lina.log`; 4 other CUDA
processes, **no container**, 62-67 W against a ~140 W envelope):

| shape | fused | GB/s moved | bf16-tile fused | GB/s moved | torch eager | torch compiled | vs eager | vs compiled |
|---|---|---|---|---|---|---|---|---|
| 1024x3072 | **21.19 us** | 223.0 | 24.90 us | 316.0 | 5213.6 us | 729.8 us | 246.1x | 34.4x |
| 3072x1024 | 19.84 | 237.9 | 21.23 | 370.6 | 5500.4 | 531.6 | 277.3x | 26.8x |
| 1024x2048 | 14.02 | 224.7 | 15.59 | 336.6 | 3604.8 | 433.8 | 257.2x | 30.9x |
| 2048x1024 | 14.36 | 219.1 | 14.50 | 361.6 | 2495.8 | 114.7 | 173.8x | 8.0x |
| 1024x1024 | 8.69 | 181.3 | 8.56 | 306.6 | 438.0 | 74.5 | 50.4x | 8.6x |

**DRAM-cold rotation over all 196 units** (221 MB of plane, nothing L2-resident):
**30.94 us/unit, 109.0 GB/s moved**, 67.1 W mean / 68.7 W max. **Whole-checkpoint
decode 6.06 ms** (was 9.72). Per-unit hot the flip is **1.10-1.33x**; on the cold
rotation **1.60x** — the cold path is where the occupancy mattered most, which is
the same story lever 1 tells.

**Before (`BLOCK_C=64`)**, `bench-decode-20260902-125834.json`. L2-hot (one unit repeated; at 1024x3072 the
wire+tile is 7.9 MB inside a 24 MB L2, which is why the "GB/s moved" column
exceeds the 273 GB/s spec — **these are not DRAM numbers**). Interleaved
round-robin arms, min of rounds. Torch baseline is the pure-torch window reader
snapshotted by path so a concurrent edit to the serving tree cannot move it.

| shape | fused | GB/s moved | bf16-tile fused | GB/s moved | torch eager | torch compiled | vs eager | vs compiled |
|---|---|---|---|---|---|---|---|---|
| 1024x3072 | **23.28 us** | 203.0 | 24.35 us | 323.2 | 4708.5 us | 735.2 us | 202.3x | 31.6x |
| 3072x1024 | 23.79 | 198.5 | 24.56 | 320.3 | 4649.4 | 425.3 | 195.5x | 17.9x |
| 1024x2048 | 18.68 | 168.6 | 20.12 | 260.8 | 2605.1 | 334.0 | 139.4x | 17.9x |
| 2048x1024 | 18.89 | 166.6 | 19.61 | 267.4 | 2616.9 | 113.1 | 138.5x | 6.0x |
| 1024x1024 | 11.00 | 143.2 | 11.50 | 228.1 | 441.6 | 72.3 | 40.1x | 6.6x |

Power 35-45 W, 6-7 concurrent CUDA processes on every row, no container.

**DRAM-cold**, rotating through all 196 units: **49.57 us/unit, 68.1 GB/s
moved**, 35.8 W mean / 40.3 W max. Whole-checkpoint decode **9.72 ms**.

The two runs are an hour apart on a shared box, so read the *ratio within each
run* (fused vs torch) as the solid number and the between-run delta as
corroborated by ncu rather than established by wall clock: ncu, serialised and
cache-flushed, puts the same flip at **1.37-1.71x** on three shapes (section 6).

**The BF16 family's write side is nearly free**: the value tile is *twice* the
bytes and costs 2-5% more time at every shape at `BLOCK_C=64`, and 0-18% more at
16 (at 1024x1024 it is 1% *faster*, which is inside the run's spread). Its decode
moves 306-371 GB/s where the FP8 one moves 181-238, which is the same statement:
the tile write is not what this kernel is spending its time on.

**Is the decoder bandwidth-bound? No, and this is the acceptance line it misses
— though the `BLOCK_C` flip halves the gap.** At the default the same shape now
reads **L1/TEX 50.5% and L2 50.7% of peak** (section 6 lever 1) with **zero**
spills, against the figures below. The counters below are the **`BLOCK_C=64`**
kernel, kept because they are what named the lever;
ncu on `_decode_kernel` at 1024x3072 (`ncu_decode_1024x3072.txt`,
cache-flushed, so DRAM-cold: 56.16 us):

```
Max Bandwidth          21.61 %      Memory Throughput      21.61 %
L1/TEX Cache Thpt      22.94 %      L2 Cache Throughput    21.61 %
Compute (SM)           18.00 %      Mem Busy               18.30 %
Registers Per Thread   255          Local Memory Spilling Requests  64,512
Theoretical Occupancy  16.67 %      Achieved Occupancy     16.31 %
Waves Per SM           2            L1/TEX Hit Rate        88.86 %
```

255 registers per thread, **64,512 spill requests** (336 a block), 16.7%
occupancy, and no memory unit ncu can see above **23%** of its peak (there is no
DRAM counter on this part). That is what the lever was named against. At
`BLOCK_C=16` it is 92 registers, **0 spills**, 37.0% occupancy and ~50% of both
cache units — better, and still **not** at the write-out bound. That is an honest miss against "target: bandwidth-bound", and
the register pressure is the named lever (section 6).

### 4b. GEMV at M=1, per Linear shape

`bench-gemv-20260902-130213.json`, `gemv_lina.log`. All arms DRAM-cold: each
rotates through enough replicas to make the working set 96-134 MB (4-5x L2).
Five arms interleaved, min of rounds:

- `fused_gemv` — mine, reads the ~4.07 bpp wire, never materialises.
- `bf16_linear` — `F.linear` (cuBLAS) on a resident bf16 tile, 16 bpp.
- `fp8_lane_quant_plus_mm` — the A side **as the lane runs it**: `torch.compile`d
  per-token E4M3 quantisation (one or two fused kernels) + `_scaled_mm` on a
  resident 8 bpp tile.
- `fp8_quant_plus_mm` — the same with the quantiser eager (5-6 launches).
- `fp8_mm_only` — `_scaled_mm` alone on a pre-quantised activation.

The last two exist because timing against the eager quantiser credits the fused
GEMV with ~11 us it never saves; both ratios are reported, and the gap between
the lane arm and the eager arm *is* the launch overhead.

| shape (rows x cols) | fused | GB/s of wire | bf16 | lane quant+mm | mm only | GB/s of tile | fused/lane | fused/mm |
|---|---|---|---|---|---|---|---|---|
| 1024x1024 | 11.27 | 46.7 | 9.91 | 11.61 | 10.39 | 101.0 | 0.97x | 1.08x |
| 1024x2048 | 17.63 | 59.7 | 25.68 | 16.45 | 15.03 | 139.5 | 1.07x | 1.17x |
| 1024x2560 | 20.06 | 65.6 | 73.05 | 19.88 | 18.43 | 142.2 | 1.01x | 1.09x |
| 1024x3072 | 23.70 | 66.6 | 82.63 | 22.45 | 20.95 | 150.1 | 1.06x | 1.13x |
| 1024x5120 | 34.55 | 76.2 | 67.34 | 35.60 | 33.70 | 155.6 | 0.97x | 1.02x |
| 2048x1024 | 16.01 | 65.6 | 22.53 | 15.91 | 14.80 | 141.7 | 1.01x | 1.08x |
| 2560x4096 | 76.50 | 68.6 | 126.94 | 58.12 | 56.64 | 185.1 | 1.32x | 1.35x |
| 2560x9728 | 337.71 | 36.9 | 380.03 | 126.39 | 122.86 | 202.7 | 2.67x | 2.75x |
| 3072x1024 | 22.83 | 69.0 | 32.20 | 19.94 | 18.48 | 170.3 | 1.15x | 1.24x |
| 4096x2560 | 63.45 | **82.7** | 121.62 | 55.17 | 53.61 | 195.6 | 1.15x | 1.18x |
| 5120x5120 | 353.21 | 37.1 | 395.79 | 129.89 | 128.16 | 204.5 | 2.72x | 2.76x |
| 5120x17408 | 1315.61 | 33.9 | 1836.04 | 948.87 | 938.54 | 95.0 | 1.39x | 1.40x |
| 9728x2560 | 356.15 | 35.0 | 634.50 | 122.76 | 121.21 | 205.5 | 2.90x | 2.94x |
| 17408x5120 | 971.35 | 45.9 | 1608.18 | 947.30 | **949.67** | 93.9 | **1.02x** | **1.02x** |

Power 29-37 W, 4-6 concurrent CUDA processes on every row.

As a fraction of the 273 GB/s spec the fused GEMV reads the wire at **12-30%**;
as a fraction of the 103 GB/s this box's probes measured, **33-80%**. Neither is
a ceiling: ncu says no memory unit it can see is above 37% of peak while it runs
(section 5).

**Versus BF16.** The fused GEMV beats cuBLAS bf16 on every shape at or above
1024x2048 for M <= 4, by 1.1x-2.6x. It **loses** at 1024x1024 (11.27 vs 9.91)
and it loses to bf16 at M=8 on six of the 0.6B shapes. "Beats bf16 everywhere"
would be wrong.

### 4c. us/token summed over a model's Linears, M=1

Per-model totals = sum over every Linear of the model, one launch each, eager
(no CUDA graph). A graph-captured lane pays roughly one launch (~4 us) less per
Linear on **all** arms, so these totals are inflated in absolute terms; the
ratios are the claim.

| model | Linears | fused | bf16 | fp8 resident (lane quant+mm) | mm only | fused/fp8 | fused/mm |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 196 | 3515.5 us | 6021.8 | 3301.6 | 3038.3 | **1.065x** | 1.157x |
| Qwen3-4B | 252 | 44282.4 | 73573.3 | 18898.3 | 18446.3 | **2.343x** | 2.401x |
| 27B (assumed shapes) | 448 | 258165.2 | 382634.0 | 203165.1 | 202342.9 | **1.271x** | 1.276x |

At M=2/4/8 the gap widens (0.6B: 1.12x / 1.28x / 2.08x; 4B: 2.40x / 2.54x /
2.81x; 27B: 1.35x / 1.53x / 1.69x).

**The verdict the brief asked for, in one line: the fused GEMV does not beat the
resident FP8 GEMV per token — at any shape, at any M, on any of the three
models.** It beats resident BF16 per token on all three: **1.71x** (0.6B), **1.66x** (4B),
**1.48x** (27B).

### 4d. The M=8 cells that looked anomalous were contention, not the kernel

The wall-clock table has `1024x2560 M=8` at 78.80 us and `1024x3072 M=8` at
84.87 us — non-monotone in `cols` (1024x5120 M=8 is 60.49) on the same launch
shape. Under ncu, which serialises and times the kernel itself:

| shape | M=4 | M=8 | ratio | regs M=4 -> M=8 | warps active M=4 -> M=8 |
|---|---|---|---|---|---|
| 1024x2560 | 34.53 us | 51.52 us | 1.49x | 166 -> 209 | 21.54% -> 15.54% |
| 1024x3072 | 36.29 | 53.66 | 1.48x | 166 -> 209 | 21.60% -> 14.47% |
| 1024x5120 | 53.92 | 78.30 | 1.45x | 166 -> 209 | 21.13% -> 14.43% |

Uniform and monotone. The two wall-clock cells are contention artifacts and the
0.6B M=8 total inherits them. The real M=8 cost mechanism is visible here:
**M=8 costs 43 more registers per thread and a third of the occupancy.**

**`GEMV_MAX_M = 8` is not derived.** The coordinator asked for kernel support to
M=8 and it is there and correct, but the two instruments disagree on where
decode-then-mm overtakes the fused path: on the wall-clock bench at 1024x3072
the fused M=8 (84.9) loses to decode+mm (23.3 + 20.9 = 44.2), putting the
crossover near M=4; under ncu's cache-flushed timing the fused M=8 (53.7) still
beats decode+mm (56.2 + ~21 = 77). Neither instrument is the lane. **The cap
should be set by the lane's own graph-captured measurement, and until then it is
a constant, not a result.**

### 4e. torch.profiler

`experiments/results/kernel-window/trace_kernel_window.json` and
`profile_kernel_window.txt`, 50 decodes + 50 GEMVs on
`model.layers.0.mlp.down_proj` (1024x3072):

```
tessera_window::window_gemv       5.715 ms CUDA (82.29%)   50 calls
  _gemv_kernel                    5.715 ms
tessera_window::decode_fp8_tile   1.184 ms CUDA (17.05%)   50 calls
  _decode_kernel                  1.184 ms
aten::zeros/zero_/fill_           45.95 us CUDA            50 calls   0.919 us each
cuLaunchKernelEx                  4.063 us CPU each
```

Two kernels, no hidden work, and the GEMV's `torch.zeros` output allocation is
a real third launch worth **0.92 us** per call — a named lever below.

**One number in this table is not trustworthy and I am flagging it rather than
quoting it.** The profiler reports the GEMV at **114.3 us** per call where the
bench says 23.70 and ncu says 29.63 on the same shape, while `decode` agrees
across all three (23.7 / 23.3 / —). The profile arm ran at 13:07:47, about a
minute after another worker's vLLM container started on the same GPU (section 0).
I did not re-run it: by the time I finished, **two** vLLM serves were up, so a
re-run would be worse, not better. The timing table above therefore rests on the
bench (interleaved, min-of-rounds) and on ncu (serialised, cache-flushed), and
the trace is here for the launch structure, not for the absolute GEMV time.

## 5. Where the time goes

Three instruments, one answer.

**ncu, `_gemv_kernel`** (`ncu_2560x9728.txt`, `ncu_1024x3072.txt`):

| | 2560x9728 (4B down_proj) | 1024x3072 (0.6B down_proj) |
|---|---|---|
| Duration (cache-flushed) | 203.17 us | 29.63 us |
| **Max Bandwidth** (max over the memory pipelines ncu can see) | **20.22%** | **20.64%** |
| L1/TEX Cache Throughput | 37.40% | 44.16% |
| **L2 Cache Throughput** | **14.77%** | **22.35%** |
| Memory Throughput (SOL) | 35.91% | 34.74% |
| DRAM | *counter unavailable on GB10* | *unavailable* |
| Compute (SM) Throughput | 34.84% | 31.00% |
| Registers Per Thread | 160 | 160 |
| **Local Memory Spilling** | **0** | **0** |
| Theoretical Occupancy | 25.0% (register-limited) | 25.0% |
| Achieved Occupancy | **13.67%** (6.56 warps/SM) | 21.18% |
| Waves Per SM | **0.6** | 0.89 |
| L1/TEX Hit Rate | 96.10% | 93.32% |
| Warp cycles per issued instruction | 7.24 | — |
| ...of which L1TEX scoreboard stall | **43.0%** | — |

**No memory unit ncu exposes on this device is above 37% of its peak while this
kernel runs, and L2 -- the unit standing in front of memory -- is at 15-22%.**
ncu has **no DRAM counter on GB10** (`dram__bytes_read.sum` and its siblings
return n/a on this integrated part), so "the DRAM is idle" is not a thing this
receipt can measure; what it can say is that a kernel saturating memory would
show a busy L2, and this one does not. It is not bandwidth-bound; it is
latency-bound at an occupancy of three warps per scheduler out of twelve, and
43% of its issue cycles are spent waiting on the wire load's L1TEX scoreboard.
Three warps cannot hide that latency. Halving the weight bytes cannot help until
the occupancy rises — which is exactly what `17408x5120` shows: both kernels take
~950 us there, and the fused one gets that time while reading **half** the bytes
`_scaled_mm` reads. A kernel whose time does not move when its bytes halve is not
limited by bytes. (I do **not** claim `_scaled_mm` is at its own bandwidth bound
there: it moved 93.9 GB/s on that shape but 205.6 GB/s on a cold 5120x5120 in the
same session, so 93.9 is not a ceiling I have established -- see section 6.)

**The ablation agrees** (`bench-ablate-20260902-130333.json`; the same kernel
with one term compiled out, `tl.constexpr` so the branch is free):

| removed | 1024x3072 | share | 2560x9728 | share |
|---|---|---|---|---|
| — (full) | 21.17 us | | 324.99 us | |
| wire read | 17.64 | -17% | **109.21** | **-66%** |
| value gather | 18.96 | -10% | 320.36 | -1.4% |
| both | 23.58 | *+11%* | 380.24 | *+17%* |
| split-K atomic | 19.46 | -8% | 331.87 | 0 |
| activation load + FMA | 21.91 | *+3%* | 330.70 | 0 |

On the collapse shape **two thirds of the time is the wire read** and the table
gather is free (L1 hit 96.1%). On the small shape nothing dominates because the
kernel is short enough that launch and tail costs are comparable to the work —
removing *both* memory terms makes it slower, which is the signature of a
latency-bound kernel whose scheduler was hiding something. Two shapes was the
difference between "no term dominates" and a diagnosis.

**The split policy is a lever worth ~1.16x, not the cause.** `_gemv_split` rounds
the wanted split **down** to a power of two, so `rows=2560` gets 16 where 25 was
wanted: 160 blocks, 0.6 waves. ncu at 2560x9728:

| split | duration | warps active |
|---|---|---|
| 16 (default) | 202.21 us | 13.69% |
| 32 | 215.62 | 20.55% |
| 64 | **174.53** | 22.05% |

Not monotone, and 1.16x at best. An earlier wall-clock split sweep said "nothing
helps here"; that sweep was contention-masked, and ncu is the instrument that
could see through it.

## 6. Levers, each with its number

Not attempted. Named with the evidence that would justify them.

1. **Registers.** The GEMV is register-limited to 25% theoretical occupancy at
   160/thread (209 at M=8); the decoder spills, 255/thread and 64,512 spill
   requests at 16.7%. The state is carried as int64 where 14 bits are live —
   int32 state and smaller live temporaries are the first thing to try, on both
   kernels. ncu's own estimate for the occupancy gap alone is 45-75%.
2. **The split policy.** Round to nearest rather than down: 1.16x at 2560x9728
   (ncu), free.
3. **The `torch.zeros` output.** 0.92 us of a 23.7 us call is a third launch per
   GEMV — 4% at the 0.6B shapes. A kernel-side initialisation or a
   caller-provided buffer removes it, but the buffer must stay functional (no
   mutation of an aliased pool) for the compiled forward.
4. **The prepared-buffer layout.** The wire's per-column stride puts each
   program's 132-byte reads 1.3-4.9 KB apart. A row-block-major prepared buffer
   (preparation-time only — **the wire on disk does not change**) would make
   them contiguous. Unquantified; it is a hypothesis, and the ablation says the
   wire read is where the time is on the shapes that matter.
5. **The bandwidth-probe discrepancy** (103 GB/s from three probes vs 205 GB/s
   from `_scaled_mm`, same box, same hour). Unresolved, and it matters because
   it is the denominator every "% of peak" claim divides by.

## 7. What the BF16 variant still needs

The parametrisation is built and tested; the *route* is not measured.

- **Real units.** Every BF16 test here uses a synthetic bf16 table over a random
  stream, per the brief. When that worker's grid merges, the family needs the
  same 196-unit byte-identity sweep against its own reference decoder.
- **Rates other than 4.** Both families are tested at R=4 only, because the
  reach checkpoint is uniform R=4 on all 286,720 columns. The kernel's span
  arithmetic covers R up to the point where `window_bits + (VEC-1)*R + 7 <= 64`
  (so R <= 6 at VEC=8, R <= 7 at VEC=7); nothing has exercised R != 4 on real
  bytes.
- **A GEMM comparator.** The value route is W16A16 into a stock BF16 GEMM, so
  its serving comparison is `decode_value_tile` + cuBLAS versus a resident bf16
  tile. Decode costs 2-5% more than the FP8 route's; the rest is unmeasured.
- **No served KL.** Nothing here touches quality.

## 8. What is not done

- **M > 8.** The GEMV's accumulator is `[MBLK, LANES, VEC]` fp32 in registers,
  so M grows the block linearly and the register wall is already the binding
  constraint at M=8 (209/thread, 14.5% occupancy). Past 8 the lane should
  decode-then-GEMM; where exactly is unsettled (section 4d).
- **MoE.** Packed experts are one unit per expert per role; nothing here batches
  them, and a fused-MoE lane would want a grouped launch rather than the
  per-unit loop in `window_module_linear`.
- **A single stacked-module GEMV.** A three-role `qkv_proj` issues three
  launches. One launch is possible (the units would have to share one packed
  plane) and is not built; at the 0.6B shapes a launch is ~4 us against a ~20 us
  kernel, so this is worth roughly 15% of the small-shape total.
- **CUDA-graph capture of the lane's forward.** Every number here is eager.
- **The activation-contract gate.** Section 1: W2's call, per principle 14.

## 9. Four corrections to earlier claims in this work

- **The initial-window hoist was wrong and is reverted** (`793c09d`). Moving the
  initial-window term behind `if pid_p == 0` — a program-uniform branch, so it
  looked free — cost **10.6x** on decode: 1024x3072 went 23.9 -> 253.1 us,
  1024x1024 11.4 -> 25.5, 4096x2560 261.5 -> 740.3 (`hoist_ab.py`, min of 9
  rounds). A branch whose result is a `[BLOCK_C, LANES, VEC]` tensor makes
  Triton carry that tensor across an `scf.if`, and it leaves the register file
  to do it. The suggestion was mine to take and the measurement overruled it;
  the lesson is in the code where the next person will read it.
- **`793c09d`'s message says "idle GB10". It was not.** The process list from
  that afternoon puts four other CUDA processes on the box, started about a
  minute after I took the lock, and their `etime` arithmetic says they were
  running throughout. The 10.6x survives that (it is far larger than any
  contention effect); the "w2 36.1 vs w8 39.1 us" comparison in the same commit
  does **not**, and should be read as a preference confirmed by the direction of
  a 5-shape x 3-M x 36-config sweep, not as a measured 8%. History is not
  amended; this is the correction.
- **"The DRAM is 80% idle" was an over-read of ncu, and this receipt no longer
  says it.** ncu's `Max Bandwidth` is the maximum over the memory pipelines it
  can see; on GB10 the `dram__*` counters are **not among them** (n/a on this
  integrated part), so no number here is a measured DRAM utilisation. What is
  measured is L1/TEX at 22-44% of peak and **L2 at 15-22%** — and since every
  byte from memory crosses L2, a kernel saturating memory would show a busy L2.
  The conclusion ("not bandwidth-bound; occupancy-bound") is unchanged; the word
  "DRAM" was doing work the instrument had not done. Related: I called
  `_scaled_mm` "genuinely DRAM-bound at 93.9 GB/s" on 17408x5120 while my own
  table shows it at 205.6 GB/s on a cold 5120x5120 in the same session. 93.9 is
  not a bound I established, and the 17408x5120 argument does not need it: both
  kernels take ~950 us there and the fused one reads half the bytes.
- **The first `BLOCK_C` sweep measured the wrong kernel, and the discrepancy it
  produced is what caught it.** It called `_decode_impl` with the unit's
  `value_table` rather than its `code_table` — a float table, hence a different
  element width, a different output dtype and a different register budget. It
  read 56 spill requests a block where the FP8 path reads 336, at the same
  shape, grid and duration; that 6x is what said "these are two binaries, not
  one". The sweep in section 6 is the re-run on `decode_fp8_tile`'s own path
  (`ncu_blockc.py` takes `code|value` explicitly now so the mistake cannot be
  made silently again). The direction was the same on both paths and 16 won on
  both, but the numbers that chose the default are the FP8 ones.

## 10. Reproducing

The `BLOCK_C` sweep of section 6 lever 1 (`ncu_blockc_sweep.log`), which needs
`/home/rob/tmp/kernel-window/ncu_blockc.py` on the box:

```bash
ncu --kernel-name regex:decode --launch-skip 5 --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
  python ncu_blockc.py <rows> <cols> <block_c> code   # or `value` for the BF16 family
```

```bash
rsync -a --delete --exclude .git --exclude __pycache__ <worktree>/ sparklina:/home/rob/tmp/wt-kernel/
ssh sparklina 'cd /home/rob/tmp/wt-kernel && PYTHONPATH=src TMPDIR=/home/rob/tmp \
  TRITON_CACHE_DIR=/home/rob/.triton-cache \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python experiments/bench_kernel_window.py \
  --arm {bandwidth|decode|gemv|ablate|profile} --out /mnt/shared/tessera-runs/kernel-window'
```

`--arm` takes **one** value per invocation. ncu (no sudo needed;
`RmProfilingAdminOnly: 0` on this box) — send its Triton cache somewhere
disposable so it never root-owns `~/.triton-cache`:

```bash
ssh sparklina 'cd /home/rob/tmp/wt-kernel && PYTHONPATH=src TMPDIR=/home/rob/tmp \
  TRITON_CACHE_DIR=/home/rob/tmp/kernel-window/triton-ncu \
  /usr/local/cuda/bin/ncu --kernel-name regex:{gemv|decode} --launch-skip 5 --launch-count 1 \
  --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats \
  --section SpeedOfLight --section LaunchStats \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  experiments/ncu_window_target.py {decode|gemv} <rows> <cols> [M] [split]'
```

Raw outputs: `experiments/results/kernel-window/`.
