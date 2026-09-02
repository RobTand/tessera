# The 16-bit route: a BF16 alphabet under the window body

**Date** 2026-09-02 · **Branch** `worktree-agent-a20d5878f9c99effa` · **Boxes**
sparky (host tests) and sparklina (every GPU job).

W1 measured the question and answered it
(`docs/measurements/tessera16-alphabet-floor-2026-09-02.md`): the window body's
error stops falling at R ≈ 6 not because the trellis runs out of shaping but
because **the E4M3 alphabet runs out of values** — 0.0222 from R = 6 upward on
L5.gate_proj, against the same trellis over a bf16 alphabet still halving
(0.06692 / 0.03445 / 0.01789 / 0.00937 at R = 4..7, ~1.93× per bit). This
document builds that alphabet into the format: a grid, a wire recipe, a decode,
an exporter, tests, and the numbers PrismaQuant's allocator will see.

The one-line result: **`TESSERA_BF16` is a real third family** — it writes,
reads back bit-exactly, decodes three ways that agree to the bit, exports a
checkpoint plus a plain-BF16 twin that a stock loader serves, and it is priced
by the accountant at its true width. It is not yet served by a lane and not yet
selectable by the allocator; both hand-offs are §7 and §8.

---

## 1. Environment and exact commands

```
python   /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python   (torch 2.11+cu130)
env      PYTHONPATH=<worktree>/src  TMPDIR=/home/rob/tmp  TRITON_CACHE_DIR=/home/rob/.triton-cache
outputs  /mnt/shared/tessera-runs/bf16/
```

Host tests (sparky):

```
PYTHONPATH=src pytest tests/test_bf16_route.py -q -p no:randomly     # 27 passed
PYTHONPATH=src pytest tests -q -p no:randomly                        # full suite
```

GPU jobs (sparklina, one GPU, chained so they never contend):

```
TESSERA_GIT=fc2c1c1 bash experiments/bf16_export_qwen.sh 1536 1792   # export + twin + twin check, R=6 and R=7
bash experiments/bf16_weight_space_run.sh                            # the allocator's table, GLM then dense
bash experiments/bf16_tail_run.sh                                    # W1 identity, structural twin check, stock HF greedy
```

---

## 2. The grid: the whole of bf16, and the code *is* the bit pattern

`alphabet.BF16_GRID` is a `PayloadGrid` of **65 536** scalar codes whose code
`i` is the bf16 bit pattern `i`. Two consequences, and they are the reason this
shape was chosen over W1's 8192-code 32-binade window:

- **`payload_bits = 16`** — the rate cap the brief asks for falls out of the
  grid rather than being a constant, and the window body's cap is
  `payload_bits` (`export._plan_for`), so R = 4..16 are ordinary rungs.
- **The ALPHABET plane is literally the kernel's table.** A table entry is a
  bf16 word, so `plane.view(torch.bfloat16)` *is* the 2^L-entry decode table —
  no index, no value table, no transformation (§7).

The 256 non-finite patterns (exponent 255: 128 Inf/NaN per sign) cannot be grid
values, so `_bf16_legal` maps each to the largest finite magnitude of its sign
(`0x7F7F` / `0xFF7F`). They stay addressable — the grid is a closed 65 536-code
space — but nothing selects them: the window table is Gaussian quantiles, and
the test asserts zero non-finite entries at the shipping (L, σ).

**Why the code space is not a cost here.** PrismaQuant refuses a 2^16 code
space because a TCQ step scores `2^payload_bits` anchors. A window body has no
forest at all — `export._plan_for` returns the grid in their place
(`src/tessera/export.py:469`, `:486`) — and scores `2^window_bits` states per
step, 16 384 at the default L. The grid is touched exactly once per unit, to
snap the 2^L table. This is the distinction §8 asks PrismaQuant to encode.

**The snap had to be rewritten.** `torch.cdist` computes `x² + y² − 2xy`, not
`|x − y|`, and at 65 536 codes it also wants a 4 GB distance matrix.
`encode._nearest_scalar_code` is an exact float64 `searchsorted` over the
grid's distinct values, ties to the lower code, used for any scalar grid wider
than a byte; the byte-wide path is untouched. Measured, this is not cosmetic:
on W1's own 8192-code grid, cdist in float32 mis-picks **30 of 16 384** table
entries (§6), and the exact path matches bf16 round-to-nearest-even on all
16 384.

---

## 3. The wire

`export.BF16_RECIPE` — WINDOW body, span 1, CHANNEL scale plane, `L = 14`,
`window_seed = 0`, `window_sigma = None`, `channel_sigma = 1.0`. It is
`E4M3_RECIPE` with the alphabet swapped, deliberately: the brief asks for the
E4M3 default unless a measurement says otherwise, and none does. `channel_sigma`
is **stated, not searched** — `_default_sigma` would build a 4096×65 536
float64 distance matrix per candidate, and it is degenerate on a grid with 8
exponent bits, where the nearest-value error is scale-free rather than having
an optimum. Recorded in the config for replay exactly as E4M3's is.

`wire_recipe(BF16_GRID, q256)` returns it, widening the table above R = 14 so
the window is never narrower than a code (`_window_bits_for`).

**Serialisation.** `SERIALISABLE_GRIDS` is now four grids, and the fourth is
the first one whose code does not fit in a byte. The ALPHABET plane carries
`code_bytes << L` bytes, little-endian, and this needs **no schema minor** —
the grid is recovered by digest search over `SERIALISABLE_GRIDS`, so a reader
without BF16 fails at the profile id with the list of grids it does implement,
never at the plane. Written up as
`docs/schema/prismaquant.tessera.v1.md` §1d.

**Accounting.** `calculator.terminal_rate` hardcoded one byte per table entry —
the exact `format-cost-models-must-not-be-special-cases` bug class, live on the
byte gate. It now takes `code_bytes`, defaulting to 1 so every pre-existing
figure reproduces. The BF16 table at L = 14 is 32 KiB: **0.0312 bpp** on a
2048×4096 unit, **0.25 bpp** on 1024×1024. On small units that is the number a
budget must carry, not a rounding.

---

## 4. Decode: three paths, one rendering

| entry point | what it returns | who calls it |
|---|---|---|
| `decode.materialize_bf16(unit, forest, code)` | one `bfloat16` tile | the twin writer, and a resident-mode lane |
| `decode.materialize_bf16_unscaled(...)` | `(bf16 code tile, fp32 row scale)` | **what a lane should call** |
| `bf16_route.stream_bf16_tile(streamed)` | one `bfloat16` tile, from the packed wire | streamed resident mode |
| `bf16_route.stream_bf16_unscaled(streamed)` | the no-fold pair, from the packed wire | **the product mode** |

**The rounding rule, stated once.** `materialize_bf16` builds the fp32 product
`grid_value(code) × row_scale × global_scale` through `unit_scale_field` and
applies **one** round-to-nearest-even at the end. Not two, not a rounded scale:
two roundings of one product is a rendering the encoder never scored. The test
asserts `tile == (values.float() * scale[:, None]).to(bfloat16)` bitwise.

**And a lane should not do that.** W1 §B is the reason: folding the row scale
into a bf16 tile costs 0.0011–0.0022 in absolute relative output error on
*every* arm — EXL3, FP8 RTN, both window grids, every rate — because it is a
property of bf16's 7-bit mantissa and the activations, not of the weights. Its
*share* grows as the coding error shrinks: 2.2% at R = 4, **15.9% at R = 7**,
28.5% on EXL3 K8. A lane holding the wire need not pay it, and the reason is
the plane's shape rather than a trick: a CHANNEL scale is one factor per
**output row**, so it commutes with the matmul, `x (s ⊙ W)ᵀ = (x Wᵀ) ⊙ s`. The
lane runs the stock BF16 GEMM on the *code tile* — exactly bf16 already, since
every table entry is a bf16 value, so the cast rounds nothing — and applies the
row scale in fp32 as an epilogue, the same epilogue `lane_planes` already
builds for this plane on the kernel lane. Measured on random activations in the
test: **2.8e-7** relative for the epilogue against **1.7e-3** for the fold.

The twin cannot avoid it (one tensor, no scale), so **the twin's served error
is a ceiling on the route's, not its value.** W1 also measured that an *fp16*
fold costs 0.0000–0.0005 — an order of magnitude less than bf16 — so a twin
written in fp16 is a cheaper ceiling if a served gate ever needs a tighter one.
Not built here.

---

## 5. Export

`experiments/export_gridbook_tessera.py --grid BF16` writes modules declaring
family `TESSERA_BF16`, and `--stock-twin DIR` writes, alongside it, a plain
BF16 safetensors of the decoded tiles under the *source's own tensor names*,
with `quantization_config` removed and every passthrough tensor copied
verbatim. That twin is an ordinary checkpoint: vanilla vLLM (or HF) serves it
with no plugin, which is how a served KL gate will run before the lane exists.

<!-- EXPORT TABLE -->

---

## 6. W1 identity

<!-- W1 TABLE -->

---

## 7. Weight-space evidence at matched bytes

<!-- WEIGHT SPACE TABLES -->

---

## 8. Hand-off: the serving plugin (W2)

The plugin adds a **third family** alongside `TESSERA_NVFP4` (W4A4) and
`TESSERA_FP8` (W8A8): `TESSERA_BF16`, **W16A16**, materialising into an
ordinary bfloat16 weight. It needs no new flag — the existing
`GRIDBOOK_TESSERA` / `GRIDBOOK_TESSERA_MODE` pair selects it — and no new
packing: there is no stock quantized layout to build, because bf16 *is* the
stock layout.

**One import, no Triton.** `src/tessera/bf16_route.py` is pure torch, so a
runtime forbidden from importing Triton imports this module unchanged.

```python
from tessera.bf16_route import (BF16_FAMILY,          # "TESSERA_BF16"
                                prepare_bf16_unit,     # at load
                                stream_bf16_unscaled,  # product mode
                                stream_bf16_tile)      # resident/correctness
from tessera.decode import materialize_bf16, materialize_bf16_unscaled
```

**At load**, per unit (`parse_unit_artifact` gives `parsed.unit`):

```python
streamed = prepare_bf16_unit(parsed.unit, device="cuda")   # refuses TCQ / non-CHANNEL /
                                                           # release / diagonals / rotation
streamed.resident_bytes            # counted, not estimated -- what the mode holds
```

**Product mode (streamed), the call to make:**

```python
values, row_scale = stream_bf16_unscaled(streamed)   # (bf16 [out, in], fp32 [out])
y = torch.nn.functional.linear(x, values)            # stock BF16 GEMM, no fold
y = y * row_scale                                    # fp32 epilogue on the output rows
```

`values` is `nn.Linear.weight` layout — rows are output channels, which is
exactly why the scale is an epilogue (§4). **Do not** call `stream_bf16_tile`
in the product path: it folds, and the fold is a floor of ≈0.0015 relative
output error that the epilogue does not pay (2.2% of the error at R = 4, 15.9%
at R = 7).

**Resident / correctness mode** — 16 bpp, a correctness path only, and the
mode a twin comparison uses: `materialize_bf16(unit, grid, code)` returns the
one folded tile, and it is *bit-identical* to `stream_bf16_tile(streamed)`.
`stock.materialize_stock` already routes a CHANNEL-plane BF16 unit to it, so a
loader that dispatches on the stock helper needs no change.

**What the route spec should say** (mirroring `TESSERA_FP8`'s row in the
Gridbook contract): family `TESSERA_BF16`, weight dtype `bfloat16`, activation
dtype `bfloat16`, no weight scale tensor on the stock side, `activation
contract "w16a16"`, streamed residency = the artifact's own wire bpp,
resident residency = 16 bpp. There is no `--moe-backend` requirement and no
kernel gate: the GEMM is the runtime's own BF16 GEMM.

**Caveat to carry into the lane's receipt.** A streamed BF16 route decodes an
*entire* tile per forward, and at 16-bit output width the decode's own memory
traffic is the same order as the tile it produces — the FP8 route's economics
(0.55 GiB streamed vs 0.73 resident) do not transfer unchanged. Nothing here
measures the lane's throughput; that is the lane's measurement to take.

---

## 9. Hand-off: the fused kernel (B1)

The fused window kernel needs **one** change to serve this grid, and the grid
was shaped so that it is the smallest possible one.

**The table is `2^L` bf16 words, not `2^L` bytes.** On E2M1/E4M3 the ALPHABET
plane holds an *index* into the grid's value table and the kernel materialises
the value; on BF16 a code **is** a bf16 bit pattern, so:

```python
table = alphabet_plane_bytes.view(torch.bfloat16)   # 2^L entries, no gather, no LUT
```

At the default `L = 14` that is **32 KiB** of shared memory (16 384 × 2 B),
against 16 KiB for a byte-wide grid's *value* table in bf16 — the same shape
the kernel already stages, twice the entries. `PayloadGrid.code_bytes` is the
one place the width is decided (1 or 2; anything else is a `GrammarError`), so
a kernel switch on `code_bytes` covers every grid that exists.

Everything else is unchanged: the body plane is `R` bits per position packed
exactly as `lane_planes.pack_window_planes` writes it for E4M3, the state
recurrence is the same `((state << R) | bits) mod 2^L`, the scale plane is the
same CHANNEL fp16 row word times an fp32 global, and the epilogue is the one
already built for that plane. `build_window_values` still returns 65 536 fp32
values for the *shared grid table*; the kernel does not need it on this route,
because the plane already holds values.

**Not built here**, and deliberately: no BF16 GEMV/GEMM variant was written or
measured. A 16-bit route's product mode is a stock BF16 GEMM after a decode
(§8), so the fused kernel is an optimisation of the decode, not a requirement
for the family.

---

## 10. Hand-off: PrismaQuant's menu

`prismaquant/prismaquant/tessera_formats.py` needs **three** changes, one of
which is a bug that will otherwise refuse the family outright.

**(a) The base.** Add the row:

```python
_HARDWARE_BASES = {
    "E2M1": (16, "NVFP4", 120),
    "E4M3": (256, "FP8_E4M3", 89),
    "BF16": (65536, "BF16", 0),      # size, terminal format, minimum SM
}
```

Minimum SM is 0: a bf16 GEMM is not an architecture gate. `payload_bits` falls
out as 16 and the family is `TESSERA_BF16_K1_R<rate>`, arity 1 only (a tuple
over this grid is not a thing anything can encode).

**(b) `ANCHOR_BUDGET_BITS` must become body-aware — otherwise BF16 is dead on
arrival.** `TesseraFamily.__post_init__` (`tessera_formats.py:491`) refuses
`payload_bits >= ANCHOR_BUDGET_BITS = 16`, so `BF16` arity 1 raises at
construction. The guard's premise is true *of the TCQ body only*: "encoding
scores `2**payload_bits` anchors at every trellis step". Under a WINDOW body
there is no forest at all and the step scores `2**window_bits` states —
`tessera/export.py:469` ("A window body has no forests") and `:486` (the plan
returns the grid in their place). The cost the guard is protecting against is
real and should stay; it just has to be charged against the body that pays it:
refuse when the family's *only* legal body is TCQ, and price a WINDOW family's
encode at `2**window_bits` instead. E4M3 at arity 2 stays excluded (its TCQ
rungs are what the wall was written for); BF16 arity 1 becomes legal at the
default L = 14, which is 16 384 states, an order of magnitude under the wall.

**(c) The accountant's table width** (`tessera_formats.py:439`):

```python
total += Fraction((1 << wire.window_bits) * 8 * Q256_UNIT, rows * columns)
```

hardcodes one byte per table entry. On BF16 it is two, so this **under-prices**
a BF16 unit by `(1 << L) * 8 / (rows*cols)` bpp — 0.031 bpp on 2048×4096 and
**0.25 bpp on 1024×1024**, live on the DP and on the byte gate. This is the
same bug Tessera's own `calculator.terminal_rate` had; the fix there was a
`code_bytes` parameter defaulting to 1, and the same shape works here (the
width is a function of `_HARDWARE_BASES[base][0]`: 1 if `size <= 256` else 2).

**(d) The serving route.** `tessera_serving_route` gains a BF16 branch:
`contract="w16a16"`, `act_bits=16`, `act_dtype_name="bfloat16"`,
`act_group_size=0`, `activation_source_format=None`, lane `LANE_STOCK`
(`base in _HARDWARE_BASES` already yields this). The stock materialisation is
`tessera.decode.materialize_bf16`; the served route is §8's.

**(e) Two things to get right in the menu, not in the code.**

- **Price the wire, not the tile.** A `TESSERA_BF16_K1_R6` unit costs ~6.03
  bpp on the wire and 16 bpp only in the resident correctness mode. The byte
  gate must read the streamed residency, and the artifact card must say which
  residency it priced — the same discipline the FP8 route already carries.
- **This is not the BF16 passthrough row.** Principle 11 forbids *synthesising*
  BF16 from a dequantised source; `TESSERA_BF16` synthesises nothing — it is a
  4-to-8-bit wire whose alphabet happens to be bf16, and its terminal tile is
  bf16 for the same reason NVFP4's is fp4. `PASSTHROUGH_SOURCE_REQUIREMENTS`
  must not pick it up.

---

## 11. Consultations, scope, and what remains

<!-- SCOPE -->
