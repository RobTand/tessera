# Tessera window-body GEMV at M=1 on GB10 (2026-09-02)

**Outcome.** A fused CUDA GEMV that reads the ~4.07 bpp E4M3 window wire
(schema minor 2, L=14, CHANNEL plane) directly and beats the resident-FP8
lane (`per-token quant + torch._scaled_mm`, 8 bpw) per decode token on the
Qwen3-4B Linear list by **<<OP_4B>>x (op: zero + kernel + bf16 cast) /
<<KERNEL_4B>>x (kernel alone)** at M=1, against the brief's target of >= 1.3x
and the byte-ratio bound of ~2.0x (the lane reads 8 bpw at ~203 GB/s, the
wire is 4.07 bpw). The kernel runs at <<GBS_BIG>> GB/s on the big 4B shapes,
<<FRAC_READ_BIG>> of what a plain CUDA streaming read achieves on the same
bytes (232-253 GB/s; the 273 GB/s spec is not reachable by any reader we
measured, see section 4) and <<FRAC_273_BIG>> of spec. The debug decode and
the GEMV path are both bit-exact against `materialize_fp8` on all 196
reach-checkpoint units. Everything below was measured on **sparklina** while
four foreign CUDA jobs were running on it; the box state is recorded next to
every row.

- Module: `src/tessera/kernel_window_gemv.py` + `src/tessera/csrc/window_gemv.cu`
  (standalone; `kernel.py`, `kernel_window.py` and `serving/` untouched).
- Tests: `tests/test_kernel_window_gemv.py` (51; skip cleanly without CUDA /
  the reach checkpoint).
- Bench: `experiments/bench_kernel_window_gemv.py`, `experiments/gemv_chain.sh`
  (serve-locked), `experiments/bw_probe_cuda.py` (bandwidth ceiling),
  `experiments/ncu_window_gemv_target.py`, `experiments/gemv_receipt_tables.py`
  (the tables below are generated from the bench JSON).
- Raw results: `/mnt/shared/tessera-runs/gemv/` (JSON per arm, logs, ncu text).
- Commits (branch `worktree-agent-a1644313db5da8a4e` on master `fb84e41`):
  `85bdb4b` v1, `dfa5d37` v2, `60fccd0` every-weight tests + power arm,
  <<RECEIPT_COMMIT>> receipt.

## 1. Conditions

- Box: sparklina, NVIDIA GB10 (sm_121, 48 SMs, 100 KB smem/SM, 24 MB L2,
  273 GB/s spec unified memory), torch 2.11.0+cu130, nvcc 13.0, ncu present.
  No `dram__*` counters exist on GB10 (ncu's "Memory Throughput %" is against
  the busiest memory unit it can see, not DRAM); the DRAM fraction in every
  table is **wall-clock bytes over the measured streaming-read rate** and over
  273.
- Foreign load: `bf16_route_weight_space.py`, `ldlq_window_sweep.py`,
  `tessera_window_wire.py` (+ one more python) from other sessions were on the
  GPU for every v2 row (`procs` column = other CUDA processes min-max during
  the row, from `nvidia-smi --query-compute-apps`). The v1 chain ran with
  only my own two processes. v1 and v2 therefore are **not** compared row to
  row across box states; the v1 -> v2 delta quoted in section 6 is from the
  plans arm, where each plan is measured in the same run.
- Serve lock held through each chain (`experiments/serve_lock.sh`, owner
  `gemv-bench-<pid>`); `docker ps` empty; no docker serve on the box.
- Power: the interleaved bench rows read 30-40 W because seven arms and a
  sync per round leave the GPU idle most of the wall-clock; section 9 runs the
  kernel alone back to back (2000 launches, no sync) and reads power through
  it -- that is the number to read against the envelope. Clocks were pinned
  by the box at 2411 MHz throughout.
- Timing: CUDA events, min over interleaved rounds (arms rotate every round,
  so drift hits all arms alike); cold by rotating through >= 96 MB of
  replicas per shape so no arm reads from L2. The "fused kernel" arm
  accumulates into a scratch buffer that is never read (timing only); the
  "fused op" arm is what a serving path would run (`zeros` + kernel + bf16
  cast).

## 2. The wire, and what the kernel does with it

The E4M3 window body: per column a rate-R bit stream (R in 1..4, uniform
R=4 in the shipped reach checkpoint), state
`s_t = ((s_{t-1} << R) | bits_t) mod 2^14` from 0 down the output rows, a
stored 2^14-entry `window_codes` table mapping state -> E4M3 code (through
`grid.native`, which folds 0x7F/0xFF to 0x7E/0xFE), and one fp16 row scale x
one fp32 global scale (the CHANNEL plane). ~4.07 bpp on the reach wire.

**Load-time repack** (`repack_window_body`, a bijection of the body plane
bits -- the wire is not changed, only reordered): rows padded to 512-row
tiles with zero codes; columns stably sorted by rate; tile g holds each
column's 512 codes as one contiguous MSB-first chunk of `16 R` little-endian
u32 words (the wire's bytes reversed within each word), so a lane's rows
are one aligned load.

**Kernel** (`window_gemv_kernel<L=14, RPL, MT, TBL, ...>`): one warp per
column chunk; each lane owns RPL consecutive rows (16 at M <= 2: one `uint2`
per lane at R=4; 8 at M >= 4); the 14 bits of history a lane needs come
from the previous lane by `__shfl_up_sync` (lane 0 loads the previous word,
or the previous tile's last word, or 0); states are cut with
`__funnelshift_r` at compile-time immediates; the table lives in shared
memory (bf16, 32 KB: `value = __uint_as_float(tbl[s] << 16)`, exact for
E4M3; or fp32, 64 KB); one FMA per (row, batch row); per-item reduction
through a padded shared array, then one coalesced `atomicAdd` per row with
the row scale folded in (`y_i = s_i * sum_k t_ik x_k` -- the scale is
applied on the output, never folded into the tile). Blocks are persistent
and loop over work items (the table is staged once per block), `x` is
staged in shared memory double-buffered with the next item's `x` prefetched
into registers, and the next item's descriptor is prefetched one item
ahead. A balanced planner cuts each rate run of each tile into segments
minimising `ceil(n_tiles * S / grid) * (ceil(n / S) + item_cost)` (the
grid's tail is the thing being balanced), segments capped at 1024 columns
at M <= 2 and 256 above. Register budget `__launch_bounds__(512, 2)` for
M <= 4 (64 regs, 2 blocks/SM) and `(512, 1)` at M=8 (128 regs; the v1 M=8
build spilled its 64 accumulators and ran 7x behind the lane).

Refused rather than approximated (a `GrammarError` naming the fallback):
rate-3 columns, rate-1 columns at M >= 4 (no 8-row lane), L != 14,
non-WINDOW bodies, non-CHANNEL planes, RELEASE / diagonal / rotation
transforms, M > 8.

## 3. Bit-exactness

All on sparklina, `PYTHONPATH=src python -m pytest tests/test_kernel_window_gemv.py`:
51 passed (`/mnt/shared/tessera-runs/gemv/tests_v2.log` for the 47 of v2,
plus the 4 every-weight tests, 41.8 s).

- **Decode kernel vs `materialize_fp8`:** byte-identical E4M3 tiles and
  exact scales on all 196 units of the reach checkpoint
  (`/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook`),
  and on the stock twin's members (`...-reach-stock-twin`, fused q/k/v and
  gate/up looked up per member).
- **The GEMV path itself:** identity slices `x = [e_j .. e_{j+M-1}]` over
  every column, at M=2 (16 rows per lane) and M=8 (8 rows per lane), on a
  mixed-rate synthetic unit (1000 rows: two partial tiles; rates 2 and 4)
  and on a real reach unit: `y[m] == w[:, j+m] * scale` exactly for every
  weight (`test_gemv_every_weight_exact_*`). This proves every weight
  through `run_item`'s own lookback (lane 0, tile boundaries, the RPL=8
  half-tiles), not only through the debug decoder.
- **GEMV vs the reference `(tile.float() * scale) @ x`:** within
  `2 K 2^-23 sum_j |w_ij x_j|` (fp32 accumulation, the order of summation is
  the only difference) on every reach unit at M=1, and a one-hot column
  exact on every unit; M tiles 1,2,3,4,5,8 checked on synthetic units.
- The definition is also tested from scratch: a step-at-a-time Python
  reference of the state recursion (`reference_states`) against the decoder
  on synthetic streams at rates 4/2/1/mixed, rows on and off the tile, and
  the repack is checked to be a bit-count-preserving bijection.

## 4. What the memory system gives a reader (the denominator)

`experiments/bw_probe_cuda.py` (`/mnt/shared/tessera-runs/gemv/bw_probe_sweep.json`),
plain CUDA kernels via `load_inline`, cold by rotation, 2 CUDA procs on the
box (mine):

| reader | 12 MB | 26 MB | 64 MB | 256 MB |
|---|---|---|---|---|
| streaming read, best launch shape (GB/s) | 244 | 253 | 246 | 234 |
| copy, GB/s moved (read + write) | 119 | 218 | 109 | 111 |
| `torch.clone` 256 MB, GB/s moved | | | | 115 |
| `torch._scaled_mm` fp8 5120x5120 M=1 cold, GB/s read | | | 203 | |

The best any reader gets is **~250 GB/s = 0.92 of the 273 GB/s spec**; the
sustained read power was 84-91 W. So "bandwidth-bound" on this box means
~240 GB/s, and the fp8 comparator itself reads at 203 GB/s.

## 5. The number (v2, chain `v2`)

<<GEMV_TABLES>>

## 6. Launch-shape sweep (v2, chain `v2plans`) and the v1 -> v2 delta

<<PLANS_TABLES>>

<<V1V2_DELTA>>

## 7. Ablations: where the time goes

Each ablation deletes one part of the kernel under the same launch (results
wrong, timing only): `no gather` keeps the wire read and the FMAs but skips
the table lookup; `no wire read` fabricates the words in registers;
`no FMA` drops the multiply-accumulate; `neither read` drops both the wire
read and the gather.

v1 (chain `chain`, box quiet, default plan `rpl16/w16/b96/c128`):

| shape | kernel | no gather | no wire read | no FMA | box |
|---|---|---|---|---|---|
| 4096x2560 | 39.1 | 33.9 (+13%) | 26.9 (+31%) | 38.1 (+2%) | 2-2 / 34.7 W / 2411 MHz |
| 1024x2560 | 17.0 | 15.2 (+10%) | 14.6 (+14%) | 16.0 (+6%) | 2-2 / 37.0 W / 2411 MHz |
| 2560x4096 | 38.4 | 33.3 (+13%) | 26.7 (+30%) | 37.3 (+3%) | 2-2 / 36.8 W / 2411 MHz |
| 9728x2560 | 66.0 | 60.8 (+8%) | 40.3 (+39%) | 65.1 (+1%) | 2-2 / 39.0 W / 2411 MHz |
| 2560x9728 | 65.2 | 60.2 (+8%) | 40.2 (+38%) | 64.5 (+1%) | 2-2 / 40.1 W / 2411 MHz |

v2 (chain `v2`, box shared):

<<ABLATE_TABLES>>

<<ABLATE_TEXT>>

## 8. ncu (M=1, default plan, 1 launch after 5 warm-ups)

Sections Occupancy / MemoryWorkloadAnalysis / WarpStateStats / SpeedOfLight /
LaunchStats / ComputeWorkloadAnalysis / SchedulerStats; files
`/mnt/shared/tessera-runs/gemv/ncu_{chain,v2}_<shape>_M1.txt`. "Memory
Throughput" is ncu's busiest-memory-unit percentage (L1/TEX on this kernel),
not a DRAM fraction -- GB10 exposes no `dram__` counters; the DRAM fraction
is in section 5.

v1 (box quiet):

| metric | 1024x2560 | 2560x4096 | 2560x9728 | 4096x2560 | 9728x2560 |
|---|---|---|---|---|---|
| Duration | 25.79 us | 48.54 us | 85.60 us | 48.38 us | 86.40 us |
| Registers Per Thread | 64 | 64 | 64 | 64 | 64 |
| Theoretical / Achieved Occupancy | 66.67 / 62.66 % | 66.67 / 64.59 % | 66.67 / 65.52 % | 66.67 / 64.89 % | 66.67 / 65.02 % |
| Waves Per SM | 1 | 1 | 1 | 1 | 1 |
| Memory Throughput (busiest unit) | 28.52 % | 41.41 % | 43.75 % | 41.43 % | 43.27 % |
| Compute (SM) Throughput | 27.89 % | 36.60 % | 34.39 % | 36.54 % | 33.97 % |
| L2 Cache Throughput | 8.53 % | 11.26 % | 14.94 % | 11.29 % | 14.83 % |
| No Eligible (of scheduler cycles) | 71.61 % | 67.99 % | 68.53 % | 67.01 % | 67.73 % |
| Block limit: registers / smem | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |

(ncu's own duration is longer than the bench's because of its clock control
and serialisation; the ratios are what to read.)

v2 (box shared):

<<NCU_TABLES>>

<<NCU_TEXT>>

## 9. Sustained power (the kernel alone, 2000 launches back to back)

<<POWER_TABLE>>

<<POWER_TEXT>>

## 10. The bf16 value family (no fold)

The same kernel is instantiated for a bf16 table (`table_dtype="bf16"`,
`prepare_value_unit(body_bits, rates, window_bits, values, scale=...)`,
values = raw table entries): the tile the lanes multiply is the raw bf16
table value and the per-row fp32 scale is applied on the output
(`y_i = s_i * sum_k t_ik x_k`), so nothing is ever rounded into a folded
bf16 tile. `decode_values` returns the raw values tile and the scale
separately, which is the contract for a prefill (M > 8) path: the decoder
emits `(tile_bf16, scale_fp32)` and the caller runs a cuBLAS GEMM and scales
the output. Tests: `test_value_family_bf16_table` (GEMV against the fp32
reference on a synthetic bf16 table), `test_value_family_scale_on_output`
(`decode_values` exact; a folded `bf16(t*s)` tile is shown to differ).
Measured M=1 speed of the bf16-table build equals the E4M3 build's (the
E4M3 table is stored as bf16 in shared memory in the default plan; the fp32
table build is the slower one, section 6).

## 11. M > 1

<<M_TEXT>>

## 12. What remains

- **Not a serving integration.** Nothing in `src/tessera/serving/` calls this;
  the Gridbook lane still materialises FP8. Wiring it in is a separate task
  (the unit is prepared once from the parsed artifact; the per-call surface
  is `window_gemv(unit, x, out=)` / `window_linear`).
- **Rate-3 columns** (no aligned lane load at R=3 with RPL=16 -> 48 bits;
  needs a 3-word path) and **rate-1 columns at M >= 4** (RPL=8 -> 8 bits
  per lane) are refused; the shipped reach wire is uniform R=4 so no
  checkpoint unit hits either. `L != 14`, RELEASE bodies, diagonal /
  rotation transforms: refused (the reach wire uses none).
- **Prefill (M > 8):** refused; the decode-to-bf16 + cuBLAS path
  (`decode_values`) is the intended route and is not benchmarked here.
- **Small shapes** (0.6B list, <= 1.5 MB of wire) are launch/setup-bound
  (section 5); a CUDA graph over a whole layer's GEMVs is the fix, not
  measured here.
- **Box state:** every v2 row was measured beside four foreign jobs; a quiet
  re-run of the `gemv` arm is one command
  (`experiments/gemv_chain.sh` with `DO_PLANS=0 DO_ABLATE=0 DO_NCU=0`).

## 13. Commands

```
# sync the worktree and run everything under the serve lock (sparklina)
rsync -a --delete --exclude .git --exclude __pycache__ <worktree>/ sparklina:/home/rob/tmp/wt-gemv/
ssh sparklina 'cd /home/rob/tmp/wt-gemv && OUT=/mnt/shared/tessera-runs/gemv TAG=v2 experiments/gemv_chain.sh'
#   (env inside: PATH=<venv>/bin:/usr/local/cuda/bin CUDA_HOME=/usr/local/cuda PYTHONPATH=src
#    TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache TORCH_EXTENSIONS_DIR=/home/rob/tmp/torch-ext-gemv)
# pieces
python experiments/bench_kernel_window_gemv.py --arm gemv   --tag v2 --out $OUT            # all models, M=1,2,4,8
python experiments/bench_kernel_window_gemv.py --arm plans  --tag v2plans --out $OUT --shapes 4096x2560,1024x2560,2560x4096,2560x9728,9728x2560,1024x1024
python experiments/bench_kernel_window_gemv.py --arm ablate --tag v2 --out $OUT
python experiments/bench_kernel_window_gemv.py --arm power  --tag v2 --out $OUT
ncu --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats --section SpeedOfLight \
    --section LaunchStats --section ComputeWorkloadAnalysis --section SchedulerStats \
    --kernel-name regex:window_gemv_kernel --launch-skip 5 --launch-count 1 \
    python experiments/ncu_window_gemv_target.py --rows 2560 --cols 9728 --M 1
python experiments/bw_probe_cuda.py
python -m pytest tests/test_kernel_window_gemv.py -q
python experiments/gemv_receipt_tables.py --dir $OUT --gemv 'bench_gemv_v2_*.json' --plans 'bench_plans_v2plans_*.json' \
    --ablate 'bench_ablate_v2_*.json' --power 'bench_power_v2_*.json' --ncu 'ncu_v2_{shape}_M1.txt'
```

## 14. Consultations

No fable-* or opus-* workers were spawned. The `advisor` reviewer was
consulted twice (design/result review before the receipt; final check); its
asks -- prove every weight through the GEMV path, add a sustained-power
measurement, state contention per row, do not present ncu's memory
percentage as a DRAM fraction -- are all in this receipt.
