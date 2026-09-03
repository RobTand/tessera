# Tessera mathematics audit handover — 2026-09-02

Source: exhaustive read-only audits of `src/tessera/`, `src/tessera/serving/`
decode paths, `src/tessera/csrc/*.cu`, `docs/` rate-distortion, and
`experiments/` harnesses. Findings below go to evaluation as stated. No code
was changed in this audit. No GPU runs were performed.

Conventions: `[C]` code fact with `file:line`, `[M]` measured via `python3`
on CPU, `[I]` proof or inference. Severity: P0 silent-wrong-weight or
fail-open, P1 wrong number or irreproducible claim, P2 hardening or docs.

---

## 1. Exact arithmetic — `fp8.py`, `scale_codec.py`, `exact.py`

Verified correct: E4M3FN tables (bias 7, `m/512` subnormals, max 448, min
1/512, NaN `0x7F/0xFF`), E8M0 `2**(E-127)` with `0xFF` banned,
`classify_half` round-trip (252 legal / 3844 illegal of 4096 at clip 0),
480 NaN-pattern exactly `(134,15),(135,7)`, canonical
`(E,1,1)->(E+1,0,0)` with `base+1>254`, census `2826/966/61744=65536`,
pack round-trip bijections, clip-shift equivalence interior.

### P0

* `src/tessera/scale_codec.py:135,143` — float `clip_exponent` leaks to
  float. `[M]` `compose_half(127,HalfWord(0,0),0.5)==1.4142135623730951`.
  Fix: require `int` (`operator.index`), raise `ScaleCodecError` otherwise.

### P1

* `src/tessera/scale_codec.py:135-144` — `compose_half(0xFF,…)` returns
  `2**128` instead of raising. NaN base has no scale. Fix: raise on
  `base==0xFF`.
* `src/tessera/fp8.py:120-136` — `e8m0_encode_exact(0.5)` raises bare
  `AttributeError`; `encode(1)==127`. Fix: return `None` or raise
  `TypeError` consistently for non-`Fraction`.
* `src/tessera/scale_codec.py:147-158` — `value==0` would classify
  `NORMAL` (`encode(0)==0x00`, not subnormal). Contradicts `:16-18`
  zero-illegal. Unreachable today, fragile. Fix: explicit rejection.

### P2

* `src/tessera/fp8.py:100-102` — `e4m3fn_is_subnormal(0x101)==True`. Add
  `0<=byte<=255` guard.
* `src/tessera/fp8.py:91-97` — `encode(1)==0x38` via hash equality. Pin
  contract to `Fraction` or document numeric-tower acceptance.
* `src/tessera/fp8.py:59-68` — negative zero `0x80` never round-trips
  (`decode 0x80=0`, `encode 0=0x00`). State 254 entries / 253 values.
* `src/tessera/exact.py:20-36` — `bits_to_bytes(16.0)->2.0 float`;
  `as_ratio(0.5)` crashes while `as_ratio(1)` works. Enforce
  `int` / `Fraction`.
* `src/tessera/scale_codec.py:166` — `ILLEGAL_INEXACT_UNDERFLOW` names
  mid-range mantissa gaps. Rename `ILLEGAL_INEXACT` or document.
* Document clip-shift legality interior-only (16 mismatches all at
  `0xFF`) and canonical clip-independence.

---

## 2. Grammar / canonical / bitio / planes / manifest / layout

Verified: `|A_R|=2^(R+1)`, partition `16`, Bresenham exactness and total
quota (`acc_i=(iK) mod N`), capacity `cap-R`, `min(limit,c)` and depth
inversion, `RELEASE_BITS=4`, LEB128 minimality + `2**64` + zigzag +
`finish` + ratio lowest-terms, MSB-first order, normative widths, prefix
sums, `WHOLE_PLANE`, Geometry `>0` + `group%half==0`.

### P0

* `src/tessera/bitio.py:56-81` — `check_padding_zero` reads past declared
  extent. `[M]` 3-bit `a0` with `bit_length=3` raises `past extent`
  instead of passing. Fix: read padding from physical bytes, not `limit`.
* `src/tessera/layout.py:166,182` — `positions//group`, `//half` truncate.
  `[M]` `positions=10,group=32->0`, 10 weights scaleless. Fix: require
  `positions%group==0`, `%half==0` (as done for `rows%arity`,
  `steps%span`). Same for `_slice_block_plane` (`layout.py:925`).
* `src/tessera/manifest.py:666-710`, `src/tessera/footprint.py:90-106` —
  partial superblock boundary unenforced. BODY `(25,25,25,25)` with
  terminal 30 passes though not at quota boundary. Fix: enforce
  `count in {prefix sums}` for `PER_SUPERBLOCK` / `PER_BLOCK` partials.
* `src/tessera/layout.py:282` — `superblocks=max(1,len(rates)//sb)` floor.
  `[M]` `257//256=1`, should be 2. Trailing partial gets no granule.
  Fix: ceiling division.
* `src/tessera/layout.py:314-319` — `divmod(total,superblocks)` spread vs
  true per-block sums. Rates `[1,1,1,1]` sb 2 with completion
  `[2,2,0,0]`: true `(4,0)`, code `(2,2)`. Restart table misdescribes
  stream. Fix: sum per superblock from `rates` / `spec`.

### P1

* `src/tessera/grammar.py:340-341,367,376` —
  `validate_descendant_map` hardcodes `cap=3`, `GRID_CODES=16`.
  `[M]` rate 5 rejected though legal at `cap=7`. Fix: add
  `cap` / `grid_codes` params.
* `src/tessera/canonical.py:183-190` — reader blob bound missing. Writer
  rejects `len>=1<<32` (`:127-128`); reader accepts if bytes present.
  Fix: `if length>=_MAX_BLOB: raise` on read.

### P2

* `src/tessera/bitio.py:26` — `write(1,0)` accepted. Require `value==0`
  at `width==0`.
* `src/tessera/manifest.py:146-155` — `quantizable_params` only `>0`.
  Wire `qp=1e12` understates bpp. Validate `1<=qp<=positions` or document
  `lm_head` convention at the check site.
* `src/tessera/grammar.py:284` — `superblock_quota_ok` vacuous-True when
  `columns<sb`. Document or require `columns>=sb`.
* `src/tessera/grammar.py:167-179` — `q256_from_root(0)==0` but reverse
  rejects. Close loop.
* Sub-byte truncation ambiguity (1-bit planes, non-byte-multiple counts).
  Require byte-multiple quota boundaries or define last-byte
  canonicalisation. `build_terminal` should pre-check spec
  prefix-consistency instead of failing only at `Manifest`.

---

## 3. Container / footprint / calculator / wire / alphabet / codebook

Verified: header 24B, `24+manifest+plane`, truncation uniqueness,
three-way byte agreement, prefix digest, per-plane padded digests,
four-quantity separation, `bpp=8*bytes/qp`, MSB-first column-major,
AnchorForest partition enforcement, tree-VQ hoist contiguity. Closed
forms at 4096²: 2.0 / 2.25 / 2.50 / 1.25, `scale_plane_full=0.50`,
diagonals `0.0078125`.

### P0

* None beyond §2 shared items (§2 B5-B8 apply here as C2-C3).

### P1

* `src/tessera/container.py:92-94` vs `:246-249` — serialiser passes
  `side_bytes=0`; parser passes `24+manifest`. Same `wire_bpp` field
  changes meaning. Fix: pass `HEADER+len(manifest)` at write.
* `src/tessera/layout.py:582-589` — `_steps_of` tries `(1,2,4,8)` only.
  k=3 over E2M1 (4096 codes, legal per `alphabet.py:243-301`) raises
  spuriously. Fix: search divisors.
* `src/tessera/container.py:163-172` — padding check assumes MSB-first,
  ignores `descriptor.bit_order`. Branch on `bit_order`.
* `src/tessera/planes.py:280-281`, `src/tessera/footprint.py:185-190` —
  REFERENCE returns 0 and bundle sums zeros. Referenced content never
  counted. Fix counting or forbid REFERENCE until charged.
* `src/tessera/calculator.py:126-128` — `plane_rate=count*bits/qp`
  ignores padding. 1-element 1-bit BODY: `1/qp` vs stored `8/qp`. Fix:
  `8*byte_length/qp`.
* `src/tessera/calculator.py:205,229,239-295` — TCQ derived figures pass
  `alphabet=b""` (0B forest); real artifacts charge `len(blob)`
  (`layout.py:155-158`). Cited `2.5008` vs derived `2.5` = 1677.7B at
  4096², remainder manifest side bytes not derivable here. Do not quote
  2.5008 as derived. Publish forest+header+manifest breakdown.
  `~1.4KB` forest figure needs per-artifact evidence from
  `unit_artifact.py`.

### P2 / gates

* Tree-VQ unserialisable by design (`alphabet.py:513-520,615-633`).
  Correct fail-closed; any shipped-tree claim false until VALUES-plane
  lands. `_hoist:93-94` tie arbitrary-left — document.
* Explicit `bitorder="big"` (`wire.py:81,85,139,150`); trailing-pad
  policy in `unpack_uniform`; `layout.py:416` zero-digest placeholder to
  sentinel/raise; `layout.py:184` `max_released` ignored when spec
  present — separate full-extent from slice.

---

## 4. Trellis / Viterbi / window — `trellis.py`, `encode.py`, `window_viterbi.py`

Verified: 64-state rate-1/2 with 4 subsets, `MSB(nxt)=bit`,
anticipated-completion, window `((state<<R)|bits) mod 2^L` with start 0
and exact inverse traceback, chunk invariance, int64 traceback offsets,
crossover dispatch, fused contract direction.

### P0

* `src/tessera/encode.py:324-330,356-360` vs `trellis.py:303-314` — span
  tie divergence at `L>=3`. Worked `L=3,s=0`: scalar `(0,1)`, fold
  `(1,0)`, same SSE 0, different bits. `L=2` safe. Fix: document
  cost-only equivalence or lexicographically faithful fold.
* `src/tessera/encode.py:314,1256` — `**2` vs fused `_mul(d,d)`.
  `pow` not guaranteed bit-identical to multiply. Fix: `diff*diff`.
* `src/tessera/encode.py:1253-1256` — completion ignores weights at
  `arity>1`. `w=[100,1]`, A errs `(0,10)`, B `(6,6)`: unweighted picks B,
  weighted picks A. Fix: thread `sub_w` into completion.

### P1

* `src/tessera/encode.py:1349-1418,1435-1442` — `sse` meaning flips
  (weighted sum vs unweighted normalised vs release-best). Fix: report
  one definition plus optional second.

### P2

* Torch `min`/`argmin` first-minimal assumed
  (`encode.py:319,335,338,601,611`). Replace branch min with strict-`<`
  select; pin tie test incl `inf` CPU+CUDA.
* Chunk last-ulp `sse` (`:590-612,369-371`). Deterministic accumulation
  or documented tolerance.
* `memory=0` traceback `state>>-1` fault. Guard.
* `shift=bit_length-1` (`:344`). Use `point_bits`.

---

## 5. LDLQ / refit / compensate / diagonals / fused / CHANNEL

Verified: Cholesky fail-closed with `L=Lc B^{-1}`, `D=BB^T`, snap to `I`;
scheduling-only slice; residual on original `W` with whole-matrix
CHANNEL scale; final SSE on `work`; S6b/LUT exact-cost keep-winner; 1D/2D
CHANNEL quadratics; floor-after-guard order; diagonal-weighting no-op
proof; binade resnap exactness.

### P0

* `src/tessera/encode.py:647-662` — pack `floor` clips despite no-clip
  claim. `peak=6,headroom=1,amax=6.6` clips 10%; `ceil` avoids. Fix:
  `ceil` or fix doc.
* `src/tessera/encode.py:658-660` — pack never `d=1` (ratio in `(0,2)`,
  comment `[1,4)` false). Halves `10,1` dominated by `E-1` shift. Fix:
  try both bases at pack.
* `src/tessera/scale_channel.py:245` — CHANNEL missing `B>0` hold (LUT
  has it at `encode.py:943-944`). `W=[1,-1],U=[-1,1]` collapses to
  `6.1e-5` and passes guard, next trellis clips `16384×`. Fix:
  `valid=(A>0)&(B>0)`.
* `src/tessera/scale_channel.py:186-188` — `land_at_least` bump after
  clamp can emit `inf` (`65504*1.000976->inf`), then `work/inf=0`,
  `0*inf=nan`. Fix: clamp/fail-closed pre-bump.
* `src/tessera/fused.py:136` — `shared_lut_global` accepts subnormals.
  `own=32,shared=1024`, entry `0x08` moves to `2^-11`, passes
  finite+equality but kernel field arithmetic wrong at `exp 0`. Fix:
  enforce `0x08-0x7E`.

### P1

* `src/tessera/encode.py:688-694` vs `:1349-1355` — monotone alternation
  false under default (refit true SSE vs trellis normalised). Document
  coupling or force matched metric.
* `src/tessera/encode.py:721-723` — 3-candidate S6b search misses optimum
  (`g=[1,16]` misses `E=2`, cost 10 vs tried 49). Exhaustive `E` sweep
  (~10 values).

### P2

* `scale_channel.py:230-238` metric shape unchecked — `GrammarError` on
  `len`/`shape`.
* `scale_channel.py:186` overshoot 1–2 ulps, not minimal — `nextafter` /
  bit-increment loop.
* `diagonals.py:97-105` clamp asymmetry; `compensate.py:147-151`
  alignment fail-open (assert divisibility or require receipt);
  `encode.py:1418` normalised SSE (report true alongside); fused CHANNEL
  global gap (add sharing or refuse); nits (ratio comment, short-block
  comment, clamp 128 vs 126, `inv` vs triangular solve, `2^-14` doc).

---

## 6. Kernel / decode / stock / BF16

Verified: replay-table composition, history-reversal fold, fused-tuple /
span-2 / subset algebra, ±0 nibble preservation (`0.0==-0.0` collapses by
value, built from codes), S6b relabel exact-iff-normal with triple-layer
exclusion (`0x01 kernel 0.00878 vs true 0.00195`), E4M3-in-fp16/bf16
exactness, one-hot `equal` vs dense tolerance discipline (atomics),
scale-once-on-fp32, halo tiling, M-pad 1/2/4/8, repack bijection +
padding invariance, stock bit-exactness, BF16 plain-tile decode.

### P0

* `src/tessera/kernel.py:409-435,592-629` — missing `rows%8` guard.
  Packer (`lane_planes.py:84-85`) and GEMM (`:1389-1394`) require it;
  sliced/wide assume it (`:542-543`). `rows=1002,pad=8` breaks shifts and
  alignment silently. Fix: `if rows%8: raise` both wrappers.
* `src/tessera/kernel.py:409,592,898,1183,1356` — missing `cols%half`
  guard (only `:1113-1114` checks). `cols=1000,half=16`: GEMVs drop 8
  columns; GEMM OOB reads scale idx 62 of 62. CUDA native already checks
  (`serving/csrc/tessera_nvfp4.cu:241`). Fix: guard all wrappers + GEMM.
* `src/tessera/kernel_window_gemv.py:373-422,179-238`,
  `src/tessera/bf16_route.py:178-258` — CUDA GEMV + BF16 drop TP-shard
  start state. Reference, serving, and Triton thread `initial_state`;
  these paths never read it. Shard decodes from zero. Fix: thread init
  or refuse with `GrammarError`.

### P2

* Int64 scale indices (`kernel.py:383,556-558,745-747`); "normals"
  wording (`kernel_window.py:117-120` — "exactly representable");
  odd-`P` refusal doc; S6b dummy table arg (`:1151`).

---

## 7. Rate-distortion / measurement

Size arithmetic passes: `4.011723`, `4.37245` (whole-checkpoint by
definition), NVFP4 `163.2656/4.0809/170.8556/+7.2956/4.461%`, MXFP4,
`4.298915` (assumes `lm_head` held — conservative). Frontier
recomputed: TCQ 2.5 `1.4012×`, CHANNEL 4.0 L12 `0.98518/1.0467`, 3.0
`0.9568/0.9585/0.973×`, TCQ-CHANNEL `1.4140×`, FP8 floor `1.318×`, L14
`0.93969` / `0.9465`. `σ=3>σ=1` every leg, holdout disjoint, status-doc
`1.7226/1.2576/42%/1.369/1.142/1.590/+64%` pass.

### P1

* `experiments/tessera_frontier.py:70-73,127-139` — `COPY_FROM` reads
  `.get("experts")` from files keyed `arms`; copies nothing. Frontier
  JSON has zero NVFP4/Gridbook arms. `tessera-one-format.md:76` false.
  Fix key/shape or drop.
* `docs/tessera-one-format.md:108` — NVFP4 row: `0.1188` is RTN amax
  `a4`, not GPTQ+JSO; priced 6.0 not 4.5; `1.53×` matches nothing (1.76 /
  1.09 / 2.47). Delete or re-derive from named arm.

### P2

* Stamp source hash (both frontier JSONs `git: unknown`).
* Emit `served_vs_exl3_w4a16` geomeans in summary
  (`tessera_frontier.py:231-244`).
* Name mean types before cross-harness compare (geomean vs arithmetic).
* Imatrix "≤0.5%" exceeded once (`K32 L42.up -0.525%`) — "rung-mean
  ≤0.5% (max 0.53%)".
* "+53% weight error" is out-MSE (`1.229²-1`); weight-rel `+36.4%`.
  Name metric/leg.
* "23×" pre-reach-fix (post `7.38×`) — qualifier at `:262`.
* Unify `0.01171875/0.01172256/0.011723`.
* Header: served-leg rule (E2M1->a4, E4M3->a8), nominal vs measured bpp.

---

## 8. Optimality opinion

Subproblem-optimal, system-approximate. Exact layers (tables, census,
partition, Bresenham-uniform, parabola refits, linear-nearest LUT,
CHANNEL quadratics, LDLQ direction, decode composition) are optimal for
their stated costs. End-to-end alternation is not jointly optimal:
refit true-SSE vs trellis normalised-SSE vs unweighted completion vs
third-definition reporting. Uniform quota ignores sensitivity and the
1.4–1.5× `W4A4` leg. Bodies/planes do not compose uniformly (window +
CHANNEL wins; TCQ + CHANNEL worst; `E2M1x2` cap stays on trellis; `E4M3`
floor near 6 bpp; `L=16` residency bound). Kernels exact, not
throughput-optimal (`LSU`-bound lane retained). Highest-value fixes for
rate-distortion per work: single-cost coupling, exhaustive scale search,
`B>0` hold, sensitivity-weighted quota.

---

## 9. Suggested fix order

1. P0 silent-wrong-weight / fail-open: §6 F1-F3, §2 B4-B8, §5 E4/E6/E7,
   §4 D1-D4, §1 A1-A2.
2. P1 wrong numbers / irreproducible: §3 C1/C7/G1/G2, §5 E1-E3, §2
   B1-B2, §4 D5.
3. P2 hardening and docs: remainder. No exponent / bias / NaN /
   subnormal / census / pack / Bresenham / capacity / LEB128 / MSB-first
   / prefix-sum / replay / reversal / tuple / span-2 / nibble / stock /
   BF16 / size-arithmetic error beyond listed items.
