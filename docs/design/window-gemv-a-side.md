# The window GEMV's A side: what it is, what it costs, and how it is declared

**Decision (#42), 2026-09-03.** The window GEMV's activation side is
`bf16_unquantized` -- W?A16 -- and it must be declared as its **own contract
family over the E4M3 grid**, not folded into `TESSERA_FP8`. That is option
**(a)**. Option (b), a per-regime activation contract, is refused for a reason
that is not about surface: the gold KL instrument is prefill-shaped, so under
(b) every served KL receipt would measure the prefill arm while every
generated token ran the other one.

The cheapest correct spelling of (a) is a **second grid on the existing
`TESSERA_BF16` route** plus its own `formats[]` row, following the
`TESSERA_NVFP4`-holds-two-grids precedent, not a fourth `ROUTES` entry. A
fourth entry is the same decision with a more legible family name and a larger
surface; §4 states both and says which fields decide between them.

Nothing here changes what ships. The E4M3/W8A8 family stays exactly as it is,
and the A16 family is not a default until a served A/B says so (principle 3).

---

## 1. The A sides this codebase can express over the E4M3 window wire, today

A *route* is a family and a family publishes exactly one activation contract:
`contract.py:436-440` builds `contracts_by_family` from
`ROUTES[family]["activation_contract"]` and `:469-473` refuses any
`lane_eligibility` cell that differs -- for every regime, since the check
ignores the `regime` axis it iterates past. `tools/tessera_route_census.py:100`
holds the same single expectation and `:181` compares it against every module's
record in both the prefill and the decode phase.

Three contracts exist (`scheme.py:86-93`). Over the E4M3 window wire --
scalar E4M3 grid, WINDOW body, CHANNEL plane -- exactly **two** of them are
reachable:

| A side | contract string | how it is expressed today | reachable on the E4M3 wire |
|---|---|---|---|
| **A8**, per-token dynamic E4M3 | `fp8_per_token_dynamic` (`scheme.py:87`) | `fp8_route.py:272` `native_ops.native_fp8_quant(x2)` then `:277` `torch._scaled_mm(a_q, b.t(), scale_a, scale_b)` | **yes** -- this is what ships, attested at `q256 = 1024` in `runtime_contract.json` v7 |
| **A16**, unquantised bf16 | `bf16_unquantized` (`scheme.py:93`) | two implementations, one contract -- see below | **yes**, and it needs no new decoder |
| **A4**, static NVFP4 | `e2m1_group16_ue4m3_static` (`scheme.py:86`) | `nvfp4_route`, E2M1 grids over the LUT plane | **no** -- different grid and plane; `kernel_window_gemv` refuses non-CHANNEL planes (`kernel_window_gemv.py:42`) |

The A16 contract has two implementations over these bytes and **both already
exist**:

* **Decode, M <= 8 -- the GEMV.** `csrc/window_gemv.cu:67`
  `const __nv_bfloat16* __restrict__ x; // [M, K] bf16`;
  `kernel_window_gemv.py:42` "The activation contract is W4A16: `x` is bf16,
  accumulation fp32". `prepare_from_parsed` builds the 2^L table as the
  **E4M3 values in bf16** (`kernel_window_gemv.py:440-441`), so the E4M3
  family and the value family run identical arithmetic:
  `y_i = s_i * sum_k t_ik x_k` on an fp32 accumulator, the row scale applied
  once on the output.
* **Prefill, any M -- the existing pure-torch window decoder.**
  `serving/window.py:209-212` documents that a **floating** `code_map` makes
  the gather yield values instead of codes. `fp8_route.py:153` passes
  `torch.tensor(grid.native, dtype=torch.uint8)`; passing the same map viewed
  as bf16 values instead yields a bf16 tile from the same wire, which
  `torch.mm(..., out_dtype=torch.float32)` times the row scale consumes --
  literally `bf16_route.apply()` (`bf16_route.py:337-338`) over a different
  alphabet.

**Proven, not read** (`experiments/window_gemv_a_side.py`, CPU, no checkpoint):
the two `prepare_window` calls over one wire produce tiles holding the *same
numbers*, and bf16 is lossless over all 254 legal E4M3 bytes. So an A8-vs-A16
comparison over these bytes is a matched pair on the weight side -- identical
weight values, bit for bit -- and the only difference is the A side. That is
what makes it a measurement rather than two treatments in one arm.

## 2. The claim in #10 that this issue exists to retire

#10's acceptance line is "KL identical (it is bit-exact, so it must be)".
There are **two independent reasons** that cannot hold, both from the receipt
and the tests rather than from argument:

1. **The bit-exactness is on the W side only.**
   `tessera-window-gemv-2026-09-02.md` §3 proves the decode byte-identical to
   `materialize_fp8` on 196 units, and the GEMV path against the reference
   `(tile.float() * scale) @ x` -- a matmul on an **unquantised** `x`. Nothing
   was compared against `torch._scaled_mm` with a per-token-quantised A side,
   because the two compute different functions.
2. **Even at one A side, kernel and torch differ by accumulation order.** The
   same section bounds GEMV against the reference by
   `2 K 2^-23 sum_j |w_ij x_j|` -- "fp32 accumulation, the order of summation
   is the only difference". An A/B is two arms with their own KLs, never an
   identity check.

## 3. What the A side costs, from receipts that already exist

### 3a. The screen: six GLM-5.3 routed-expert projections, output space

`docs/measurements/tessera-window-body-2026-09-02.md`, six-tensor geomeans,
**pinned start, table charged** (the caveat-corrected headline, so these are
the wire as it actually serialises):

| arm | bpp | `out` = A16 | `a8` = W8A8 | A8 / A16 | vs EXL3 K4 `out` 0.06736 |
|---|---:|---:|---:|---:|---|
| **window L=14, R=4 -- the shipped wire** | 4.023 | **0.06321** | **0.06759** | **1.069x** | A16 **0.938x** (a win); A8 1.003x (a tie) |
| window L=12, R=4 | 4.012 | 0.06640 | 0.07059 | 1.063x | A16 0.986x; A8 1.048x |
| window L=14, R=5 | 5.023 | 0.03392 | 0.04151 | **1.224x** | A16 0.989x |

Read plainly: **at the rung the FP8 route serves, the A side is the difference
between beating EXL3 K4 at matched bytes and tying it.** And the penalty grows
as the encoder improves -- 1.046x on the older per-channel trellis at 4.008 bpp
(`tessera8-targets-2026-09-01.md` §2: `out` 0.0780 vs `W?A8` 0.0816), 1.069x on
the L=14 window body, 1.224x at R=5, 1.62x on FP8 RTN at 8 bpp (0.0189 vs
0.0306). Every lever that improves the weight leg raises the price of the A8
leg, because that leg is a constant: `tessera8-targets` §2 measures the
activation legs alone at **A4 0.0866 / A8 0.0241**, and the composites in these
tables are the RSS of the two (0.0780 (+) 0.0241 = 0.0816, to four digits).

The same table is why the A side, not the weight codec, is the axis that
matters at 4 bytes: **W8A8 at 4.0 bpp beats W4A4 at 4.0 bpp by 1.44x**
(0.0816 vs 0.1176) purely because the A4 leg is 3.6x the A8 leg.

### 3b. The gold metric: what is served, and what is not

`docs/measurements/tessera-stock-lane-served-2026-09-02.md` and
`tessera-dense-reach-fix-2026-09-02.md`, exact `kl_tool` KL-vs-BF16 on
`corpus_qwen_n8_s512.json` against a BF16 teacher on the same image, Qwen3-0.6B,
vanilla vLLM 0.28:

| arm | A side | bpp wire / resident | KL |
|---|---|---:|---:|
| Tessera-8 E4M3 window, reach-aware start | **A8** | 4.07 / 8.0 | **0.1512** |
| Tessera-8 E4M3 window, pre-reach | A8 | 4.07 / 8.0 | 0.4699 |
| FP8 RTN per-channel | A8 | 8.0 / 8.0 | 0.0205 |
| production NVFP4 GPTQ+JSO | A4 | 4.5 / 4.5 | 0.5106 |
| Tessera E2M1x2 q896 | A4 | 4.00 / 4.5 | 0.6404 |
| `TESSERA_BF16` route, q1792 (`tessera-bf16-route-served-2026-09-02.md`) | A16 | 7.129 / 16.0 | 0.004923 |

**There is no served A16 arm of the E4M3 window wire.** The A16 row above is a
different grid at a different rung, and it is not a control for this question.
So every number in §3a is a screen (principle 3), and the honest statement is:
the structural argument for A16 is measured on GLM Gaussian-input experts in
output space; on the one dense model where this wire has been *served*, only
the A8 arm exists.

Two served signals do bear on the A side, and neither is a matched pair:
the stock-lane receipt records that Tessera-K2's weights "served W4A4 read
0.640 against 0.192 decoded" (A4 costs 3.33x, but across images), and its
HF exact-weight check shows the Tessera-8 checkpoint's *weights* produce the
same degenerate completion the served W8A8 arm does -- i.e. on that arm the A8
path served the weights faithfully and the weights were the failure.

## 4. Why (a), and in which shape

**Against (b), the per-regime contract.** It is the configuration a performance
engineer wants: the FP8 tensor core at prefill, the GEMV at decode, one
artifact. Its price is not surface, it is the receipt. The gold instrument is
`kl_tool` at n=8 x seqlen=512 -- a **prefill-shaped** forward, 4088 scored
positions from 8 sequences. Today a `regime: decode` cell in
`runtime_contract.json` is legitimately attested by that prefill-shaped KL
*because the contract is regime-invariant*: the number measured at M=512 is the
number that runs at M=1, up to accumulation order. (b) deletes exactly that
property. The decode cell's A side would differ from the prefill cell's, no
decode-shaped KL instrument exists, and the published KL would be the A8 arm
while every generated token ran A16. A cell attested by a measurement of the
other arm is worse than an unattested cell. (b) becomes available the day a
decode-shaped KL instrument exists; it does not exist, and building one is not
in either dependent's scope.

**For (a).** One contract per family means the KL receipt, the priced A side,
the census record and the executed route are one object again, and the two
products -- W8A8 over the FP8 tensor core, W?A16 over a bf16 GEMM and the GEMV
-- are declared separately because they *are* separate, with the quality gap of
§3a between them.

**(a)'s cost, stated with the win.** A16 gives up the FP8 tensor core at
prefill: `fp8_route.py:271` records that `bf16 x fp8` is refused by
`_scaled_mm` on this hardware, so an unquantised A side means a 16-bit GEMM on
an upcast tile -- roughly half the tensor-core rate, and a 2 byte/weight
transient in streamed mode against the FP8 route's 1. **Unmeasured**: no
prefill A/B of a bf16 GEMM against `_scaled_mm` at these shapes exists, and the
GEMV receipt's bf16 column is explicitly a contended upper bound (§5 of that
doc). Also: one artifact cannot mix the two A sides per module without mixing
families, and `FUSED_MODULE_FIELDS` makes the family a module fact -- so
choosing the GEMV is a re-export, not an env-var flip.

**Which spelling.** Two, and the choice between them is a naming judgment, not
a technical fork:

* **A second grid on `TESSERA_BF16`** -- `grids: ("BF16", "E4M3")`. This is the
  established shape: `TESSERA_NVFP4` already holds two grids and the contract
  publishes one `formats[]` row per (route, grid) pair, resolved by
  `contract.reader_rate_grid(route, grid)`. `validate_tessera_scheme` (`scheme.py:438`)
  gates `grid in ROUTES[family]["grids"]`, so widening the tuple makes the
  scheme legal. New work: one `formats[]` row (family e.g.
  `TESSERA_E4M3_A16_K1`, `activation_contract: bf16_unquantized`), one
  `_FAMILY_TO_ROUTE` line (`contract.py:569-573`), a grid branch in
  `bf16_route.prepare_tessera_bf16_module` (`bf16_route.py:177-179` refuses a
  non-BF16 grid today; the E4M3 branch builds the table from `grid.native`
  values and checks against `materialize_fp8` upcast rather than
  `materialize_bf16`). **No new census expectation, no new `ROUTE_TP_AXES`
  row, no new builder** -- the route's contract, decoder and GEMM symbol are
  unchanged. Costs: `ROUTES[TESSERA_BF16]["short"]` and `grid_kind` become
  wrong for the second grid and must go per-grid or be reworded, and the
  checkpoint's `scheme["family"]` would read `TESSERA_BF16` on a 4.07-bpp
  artifact -- legible in the contract family name, misleading in the route
  name.
* **A fourth `ROUTES` entry** -- an honest route name at the cost of a builder,
  a census `contract_for`/`decoder_for` row, a `ROUTE_TP_AXES` row and a
  `_FAMILY_TO_ROUTE` line.

Either way **#51 must land first**: `route_for_grid` (`scheme.py:329-332`)
returns the first route holding a grid, and the export gate resolves through
it (`scheme.py:372`), so the moment two routes hold `E4M3` the gate's answer
depends on dict order.

## 5. What #10 and #47 each need from this

**Both:**

1. The A side they wire to is `bf16_unquantized`. Neither may dispatch the
   GEMV inside `TESSERA_FP8`'s `apply()`.
2. **#52 first.** `window_gemv` decides on the token dim in Python
   (`kernel_window_gemv.py:516` `_m_tile(M)`, `:522` the pad, `:536` `out[:M]`)
   and its 51 tests contain no `torch.compile` arm. Every route here is served
   eager **and** compiled and the census records both; this lane has been
   broken by exactly this shape before.
3. **The native-extensions table goes red.** `serving/ext.py:137-140` states
   that `kernel_window_gemv` is excluded from `NATIVE_EXTENSIONS` *because*
   nothing under `tessera/serving/` reaches it, and
   `tests/test_serving_native_extensions.py` enforces that by walking the
   import graph. The first route that imports it needs a `NATIVE_EXTENSIONS`
   entry with a `when_unavailable` block per residency -- and
   `contract._validate_native_extensions` requires `source` to start with
   `csrc/` **relative to the serving package**, while the `.cu` lives at
   `src/tessera/csrc/window_gemv.cu`. Move the source or widen the rule; do
   not paper the entry over it.
4. **A fourth `DECODERS` constant** (`telemetry.py:61-64` has three, and
   `torch_window` is deliberately shared by the FP8 and BF16 routes because it
   *is* the same object). A route that uses the kernel at M <= 8 and the torch
   decoder at prefill has two decoders per family, so the census expectation
   becomes per-`(family, regime)`. Design it once for both issues, and land
   **#53** with it (`decoder_for` is unguarded and would `KeyError` mid-run).
5. **The A/B protocol.** Two arms, two KLs, two latencies -- never an identity
   check (§2). The matched pair is a hardlinked directory pair whose
   `config.json` differs only in the declared family, the
   `retarget_checkpoint_to_plugin.py` pattern the BF16 receipt used (one inode,
   1 015 088 026 bytes on both); §1 proves the weight values are then identical
   bit for bit and the A side is the only difference. Eager **and** compiled,
   both residency modes, census clean in both phases.

**#10 specifically (the FP8 route / E4M3 wire):**

* The A/B is runnable at the **already-attested rung**: `TESSERA_E4M3_K1`
  attests `q256 = 1024`, which is rate 4, inside `SUPPORTED_RATES = (1, 2, 4)`
  and at `WINDOW_BITS_SUPPORTED = (14,)`. That is the asymmetry with #47 --
  #10 needs no new rung, only a new family row and its cells.
* Its arm-B prefill needs no new decoder: `prepare_window` with a floating
  `code_map` (§1). Check before building a second packing whether the GEMV's
  own `Repacked` can serve prefill too -- `decode_values` (`kernel_window_gemv.py:559-580`) is a full-tile decode over the same words, gated only on
  `unit.family == "value"`, and the E4M3 unit's table already holds bf16
  values.
* **M=2 -> MT=4** is per-unit, not per-M: `4 if 1 not in unit.rep.rates else 2`,
  because `:517-521` refuses `mt >= 4` on a rate-1 column and `:526` drops
  `rpl` to 8. It does not belong in `_m_tile`, and under #52 the M-tile has to
  leave the traced region anyway -- which is where a per-unit choice lives.
  Profile before and after on a quiet box (sparklina): the existing kernel
  numbers were taken with 6-8 CUDA processes resident and are contended upper
  bounds.

**#47 specifically (the BF16 route / value family):**

* #47 is right that it has **no A-side decision to make** -- the route already
  publishes `bf16_unquantized` and the GEMV already executes it. This decision
  changes nothing about #47's contract; it only settles that the *seam* #47
  opens (the fourth decoder constant, the per-`(family, regime)` expectation)
  is the seam #10 will use too, so it is built once.
* Its blocker stays rate coverage: the attested rung is `q256 = 1792` (rate 7),
  outside `SUPPORTED_RATES`. The A/B belongs at a uniform `q256 = 1024`, which
  needs its own served receipt and its own cell before anything attests it.

**Downstream, flagged not resolved:** two contract families over one wire at
identical bytes means PrismaQuant prices two A sides for the same unit. That is
a pipeline consequence of declaring the product honestly, and it belongs to
whoever owns the pricing read, not to either of these issues.

## 6. What this does not license

* It does not make A16 the default. The evidence for A16 is a **screen** --
  output-space error on six GLM routed-expert projections, whose inputs are
  Gaussian (rotation measured dead there at 1.003x). The one served number on
  this wire is the A8 arm. A default change needs the A/B of §5.
* It does not say the A8 penalty measured on GLM experts transfers to dense
  models. It might be larger or smaller: dense Qwen inputs carry the
  outlier columns the CHANNEL plane is blind to
  (`tessera-stock-lane-served-2026-09-02.md`, `layers.2.mlp.down_proj`), and
  those interact with a per-token activation quantiser in a way nothing here
  measures.
* It says nothing about prefill throughput. The bf16-GEMM-vs-`_scaled_mm` cost
  of A16 at prefill shapes is unmeasured, and the GEMV's own "vs bf16" column
  is a contended upper bound by its receipt's own §5.
