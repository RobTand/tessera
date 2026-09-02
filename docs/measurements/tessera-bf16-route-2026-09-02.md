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
selectable by the allocator; those hand-offs are §8 and §10, and each names the
exact change its owner has to make.

---

## 1. Environment and exact commands

```
python   /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python   (torch 2.11+cu130)
env      PYTHONPATH=<worktree>/src  TMPDIR=/home/rob/tmp  TRITON_CACHE_DIR=/home/rob/.triton-cache
         (TRITON_CACHE_DIR is not optional: the default cache is root-owned and
          every kernel test fails with PermissionError without it)
outputs  /mnt/shared/tessera-runs/bf16/
```

Host tests (sparky):

```
PYTHONPATH=src pytest tests/test_bf16_route.py -q -p no:randomly     # 29 passed
PYTHONPATH=src pytest tests -q -p no:randomly                        # 570 passed, 860s
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
  no index, no value table, no transformation (§9).

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

**The route does not fold. The twin does, and it is the only thing that does.**

| entry point | what it returns | who calls it |
|---|---|---|
| `decode.materialize_bf16(unit, forest, code)` | `(bf16 tile, fp32 row scale)` | **the route** |
| `bf16_route.stream_bf16(streamed)` | the same pair, from the packed wire | **the route, streamed — the product mode** |
| `decode.materialize_bf16_folded(...)` | one folded `bfloat16` tile | the twin writer, and `stock.materialize_stock` |
| `bf16_route.stream_bf16_folded(streamed)` | the same tile, from the packed wire | reproducing a twin from the wire |

The pair is the default-named entry point deliberately: the safe rendering
should be the one a caller gets without asking, and the fold should have to be
spelled.

**Why.** W1 §B: folding the row scale into a bf16 tile costs 0.0011–0.0022 in
absolute relative output error on *every* arm — EXL3, FP8 RTN, both window
grids, every rate — because it is a property of bf16's 7-bit mantissa and the
activations, not of the weights. Its *share* grows as the coding error shrinks:
2.2% at R = 4, **15.9% at R = 7**, 28.5% at EXL3-K8 quality. An fp16 fold costs
0.0000–0.0005; not folding costs nothing at all.

**And nothing has to fold**, because of the plane's shape rather than a trick:
a CHANNEL scale is one factor per **output row**, so it commutes with the
matmul, `x (s ⊙ W)ᵀ = (x Wᵀ) ⊙ s`. The route runs the stock BF16 GEMM on the
tile — exactly bf16 already, since every table entry is a bf16 value, so the
cast rounds nothing — and applies the scale to the GEMM's *output* in fp32,
`y_i = s_i · Σ_k t_ik x_k`, the same epilogue `lane_planes` already builds for
this plane on the kernel lane. Measured on random activations (64×256 unit,
R = 6, L = 8, 512 rows of `randn`): **4.04e-7** relative for the epilogue
against **1.65e-3** for the fold — the same 0.0015 floor W1 measured on real
GLM rows, which is the point: it is bf16's mantissa, not the weights.

**The rounding rule for the one thing that folds, stated once.**
`materialize_bf16_folded` takes the pair, multiplies in fp32, and applies
**one** round-to-nearest-even. Not two, not a rounded scale: two roundings of
one product is a rendering the encoder never scored. The test asserts
`folded == (values.float() * scale[:, None]).to(bfloat16)` bitwise.

The twin cannot avoid the fold (one tensor, no scale), so **the twin's served
error is a ceiling on the route's, not its value** — §7 prices that ceiling on
the six GLM units. A twin written in fp16 would be a tighter ceiling (W1's
0.0000–0.0005); not built here.

---

## 5. Export

`experiments/export_gridbook_tessera.py --grid BF16` writes modules declaring
family `TESSERA_BF16`, and `--stock-twin DIR` writes, alongside it, a plain
BF16 safetensors of the decoded tiles under the *source's own tensor names*,
with `quantization_config` removed and every passthrough tensor copied
verbatim. That twin is an ordinary checkpoint: vanilla vLLM (or HF) serves it
with no plugin, which is how a served KL gate will run before the lane exists.

**Qwen3-0.6B, 196 units in 112 fused modules, 440 401 920 quantizable
parameters.** Both rungs stamped `git: fc2c1c1`.

| rung | wire bpp | on-disk bpp | resident-mode bpp | encode | per unit | checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| R = 6 (`--q256 1536`) | **6.1292** | 6.1317 | 16.0 | 1832 s | 9.35 s | 960 037 778 B |
| R = 7 (`--q256 1792`) | **7.1292** | 7.1317 | 16.0 | 2132 s | 10.88 s | 1 015 088 026 B |

`bf16_twin_check.py` re-opens both checkpoints and carries no encoder state:
for every unit it parses the wire bytes, materialises them *the twin's
way* (`materialize_bf16_folded`), and asserts **bitwise** equality with the
twin tensor; every eighth unit is also decoded by
`stream_bf16_folded` and compared to the same tensor. It then checks the twin is
structurally the source — same tensor names, same shapes, every tensor
`BF16` — and that the config carries no `quantization_config`.

| rung | units checked | mismatched | streamed re-checked | mismatched | twin `quantization_config` | tensors src / twin | non-bf16 | secs |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| R = 6 | 196 | **0** | 24 | **0** | absent | 311 / 311 | 0 | 12.7 |
| R = 7 | 196 | **0** | 24 | **0** | absent | 311 / 311 | 0 | 13.0 |

No tensor is missing from the twin, none is extra, no shape differs, and every
tensor is `BF16` — the twin *is* the source checkpoint with 196 tiles replaced.

**And a stock loader agrees.** `bf16_twin_greedy.py` calls
`AutoModelForCausalLM.from_pretrained(twin, dtype=torch.bfloat16)` with nothing
else set — no plugin, no `quantization_config`, no trust_remote_code — and
generates greedily:

| arm | "The capital of France is" | top-1 |
|---|---|---|
| BF16 source | ` Paris. The capital of France is also the capital of the Republic of France…` | ` Paris` −0.465 |
| R = 6 twin | ` Paris. The capital of Italy is Rome. The capital of Spain is Madrid…` | ` Paris` −0.358 |
| R = 7 twin | ` Paris. The capital of France is also the capital of the French Republic…` | ` Paris` −0.392 |

All three emit byte-identical Python for `def fibonacci(n):`. This is a
*loadability* check, not a quality measurement — two prompts decide nothing —
but it is the only direct evidence that the served gate has a checkpoint to run
on.

**Reading the three rates.** `wire_bpp` is what a lane holds; `on_disk_bpp`
adds the container's own framing (0.0025 bpp, ~139 KB over 196 units);
`resident_mode_bpp` is 16.0 by definition — the correctness path, not the
product. The wire's excess over the nominal R is the CHANNEL rows plus the
32 KiB table, and on a 0.6B model that is not small: **+0.129 bpp at R = 6**,
because Qwen3-0.6B's Linears are 1024×3072 and smaller, where a 32 KiB table
is 0.25 bpp on its own. On a 2048×4096 GLM expert the same table is 0.031 bpp.
The accountant charges it either way; a menu that quotes one number for
"R = 6" across shapes is quoting the wrong one (§10e).



---

## 6. W1 identity

W1's BF16 arm ran **in memory**, over a grid built in its script — the finite
normal bf16 values across a 32-binade window, 8192 codes — snapped by
`torch.cdist(...).argmin` in float32. The library's grid is the whole of bf16,
65 536 codes, snapped exactly in float64, and it goes through the real wire.
Three differences, and each is measured rather than argued away
(`experiments/bf16_route_w1_identity.py`).

**1. Grid width: no difference at all.** At the shipping `L = 14`, σ = 1.0,
seed 0, the two grids snap the 16 384 table quantiles to **identical values** —
`value_mismatches: 0`. Every quantile lies deep inside W1's 32-binade window
(the table spans 7.6e-5 to 4.05), so the wider grid buys no extra reach here; it
buys the *cap* (`payload_bits = 16`) and the kernel's table view. Reach 4.00 σ,
**2116 distinct values** in a 16 384-entry table — W1's §D observation, on this
grid.

**2. The snap: 30 entries of 16 384, and they are cdist's.** On W1's own grid,
its float32 `cdist` picks a different code from the exact float64 search on
**30 of 16 384** entries; the same 30 differ from bf16 round-to-nearest-even.
The library's table matches bf16 RNE on **all 16 384**. So W1's receipt's
"snap-vs-RNE differences" were the `x² + y² − 2xy` expansion, not a tie rule —
which also means the ties W1 worried about never bind at these parameters.

Host-reproducible in a minute, no GPU:

```
python -c "…window_table(BF16_GRID, 14, sigma=1.0, seed=0, half=16)…"
grid_width {'library_codes': 65536, 'w1_codes': 8192, 'entries': 16384,
            'value_mismatches': 0, 'reach': 4.0, 'distinct': 2116}
snap       {'cdist_vs_exact': 30, 'exact_vs_rne': 0, 'cdist_vs_rne': 30}
```

**3. Wire vs memory, and what the 30 entries cost.**

On L5.gate_proj (2048×4096), the same tensor W1 carried deepest, encoded four
times each way. The library arm is the **real wire** (`encode_linear_planes` →
bytes → `read_unit_artifact`); the W1 arm is `encode_unit` → `reconstruct_unit`
over W1's grid with its `cdist` table patched in, exactly as W1 ran it. Both
priced at the same bytes, because both tables are two bytes an entry.

| R | bpp | library `wt` | W1 `wt` | library `out` | W1 `out` | library / W1 (`out`) |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 4.0352 | 0.06942 | 0.06942 | 0.06693 | 0.06692 | 1.00019 |
| 5 | 5.0352 | 0.03572 | 0.03572 | 0.03447 | 0.03445 | 1.00069 |
| 6 | 6.0352 | 0.01859 | 0.01859 | 0.01788 | 0.01789 | 0.99927 |
| 7 | 7.0352 | 0.00973 | 0.00973 | 0.00937 | 0.00937 | 1.00033 |

**The 30 entries are worth nothing measurable** — every ratio is within 0.07%,
in both directions, which is the size of the arms' own noise and not a
preference. The exact snap is still the right rule (it is what makes the table
*bf16 rounding*, statable in one line), but the correctness argument for it is
the 4 GB matrix, not the error.

**And this reproduces W1's published table.** W1's L5.gate_proj BF16 window
`out` at R = 4..7 was 0.066915 / 0.034449 / 0.017890 / 0.009371; the library
wire gives 0.06693 / 0.03447 / 0.01788 / 0.00937. The wire is W1's experiment,
to four or five digits, with the artifact written. `out_bf16` reproduces the
fold too: 0.00949 at R = 7 against `out` 0.00937 is a fold of 0.00150, against
W1's 0.001492.

`wire_equals_memory` is `true` at every rung (the reader's tensor is the
encoder's, bitwise), which is `encode_linear_planes(verify=True)` re-asserted
from the bytes on the other side.

---

## 7. Weight-space evidence at matched bytes

**Both arms are weights-only, deliberately.** Neither arm is given a Hessian:
`encode_linear_planes` is called with the weight and nothing else, so both the
E4M3 and the BF16 wire are the same encoder with the same settings and only the
alphabet different. That is the A/B the question asks for, and it needs no H.

It is **not** the shipping encoder as of the same day: on
`worktree-agent-a54b4e34fc953c2bb` a per-unit Hessian turns on LDLQ (σ = 1.0,
block 32) plus an exact full-H row-scale refit on the CHANNEL plane, worth
served KL 0.1512 → 0.1046 on Qwen3-0.6B at identical bytes. That path is the
**same WINDOW body over the same CHANNEL plane**, and this route adds a grid
and a recipe and touches neither the encode loop nor the refit
(`git diff master...HEAD -- src/tessera/encode.py src/tessera/export.py` is a
table snap, a recipe constant, a width helper and a config map), so it applies
to the BF16 alphabet exactly as it applies to E4M3 once merged — and it applies
to **both arms of every table below equally**, which is why leaving it out does
not tilt the comparison. Absolute errors here will fall when it lands; the
BF16-vs-E4M3 ratios are what this section is for.

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
from tessera.bf16_route import (BF16_FAMILY,        # "TESSERA_BF16"
                                prepare_bf16_unit,  # at load
                                stream_bf16)        # product mode
from tessera.decode import materialize_bf16        # the same pair, unstreamed
```

**At load**, per unit (`parse_unit_artifact` gives `parsed.unit`):

```python
streamed = prepare_bf16_unit(parsed.unit, device="cuda")   # refuses TCQ / non-CHANNEL /
                                                           # release / diagonals / rotation
streamed.resident_bytes            # counted, not estimated -- what the mode holds
```

**Product mode (streamed), the call to make:**

```python
values, row_scale = stream_bf16(streamed)     # (bf16 [out, in], fp32 [out])
y = torch.nn.functional.linear(x, values)     # stock BF16 GEMM, nothing folded
y = y * row_scale                             # fp32 epilogue on the output rows
```

`values` is `nn.Linear.weight` layout — rows are output channels, which is
exactly why the scale is an epilogue (§4). **Do not** call `stream_bf16_folded`
in the product path: it folds, and the fold is a floor of ≈0.0015 relative
output error that the epilogue does not pay (2.2% of the error at R = 4, 15.9%
at R = 7).

**Resident mode** — hold `values` and `row_scale` instead of the wire; the
GEMM and the epilogue are the same two lines, so the residency choice is not a
quality choice. **Folded** (`materialize_bf16_folded` /
`stream_bf16_folded`, bit-identical to each other) exists for the twin and for
`stock.materialize_stock`; a lane that calls it has silently chosen the twin's
error for no reason.

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

**(c) The accountant's table width** — `wire_overhead_q256`, `tessera_formats.py:439`, reached from `artifact_bpp` and so from `tessera_render`'s `exact_bits_per_param` / `TesseraShapeRate`, i.e. from the DP and the byte gate:

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

**Consultations.** No Fable worker was consulted: nothing here turned out to be
a hard sub-question. The two candidates named in the brief resolved by reading
code rather than by reasoning about it — the table construction is W1's own
`window_table` with a different grid, and the rate-cap algebra is already in
`export._plan_for` (`payload_bits` under WINDOW, `payload_bits − 1` under TCQ).
Two `advisor()` calls were made, one before committing to the grid shape and
one before declaring done; the second produced four of the checks in this
document (the encoder-drift diff, the structural twin check, the
`grid_from_config` hole, and the PrismaQuant accountant line).

**What is measured here, and on what.** Weight-space and (on GLM) output-space
error, on real tensors, both arms through the real wire, priced at the bytes
actually written. The GLM rows are the same six routed-expert projections and
the same held-out capture rows every other Tessera measurement uses; the dense
rows are six Qwen3-0.6B Linears including the three `down_proj` the dense
outlier work identified as the hard ones.

**What is not measured, and must not be inferred.**

- **No served KL.** No lane serves `TESSERA_BF16` yet, so there is no
  KL-vs-BF16 number and none should be quoted from this document. The twin is
  the vehicle for that gate and it exists; the gate has not been run.
- **No lane throughput.** §8's caveat: a streamed 16-bit route's decode traffic
  is the same order as the tile it produces, and the FP8 route's residency
  economics do not transfer unchanged.
- **No fused kernel.** §9 states the one change needed; nothing was built or
  benchmarked.
- **No allocator wiring.** §10 states the three changes PrismaQuant needs, one
  of which (`ANCHOR_BUDGET_BITS`) currently refuses the family outright.
- **L = 14 was inherited, not searched, for this grid.** It is E4M3's measured
  default. A wider table costs 2 bytes an entry here rather than 1, so the
  L-vs-rate frontier is *not* the same curve as E4M3's and re-measuring it is
  the obvious next experiment. W1's own reasoning about L = 16 (its §"L = 16:
  not run") applies with the table cost doubled.
- **`channel_sigma = 1.0` is stated, not searched** (§3).
- **Rates above 8 are untested by measurement.** They are legal to R = 16 and
  the wire round-trip test covers the widening path, but the product range is
  4..8 and that is where every number here lives.
