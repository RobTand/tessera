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

**The result, in one table** — six GLM routed experts, output space, geomean,
both arms through the real wire at the bytes actually written:

| R | BF16 / E4M3 at the same R | BF16 / E4M3 **one rung up** | BF16 / EXL3 K=R |
|---|---:|---:|---:|
| 4 | 1.000 | 1.94 | **0.932** |
| 5 | 0.993 | 1.53 | **0.939** |
| 6 | **0.797** | **0.828** | **0.955** |
| 7 | **0.433** | — | — |
| 8 (one tensor) | **0.231** | — | **1.002** |

Below 6 bits the alphabet is free and costs 0.016 bpp. At 6 it is worth a
whole bit. At 7 the E4M3 arm has stopped being a rung. And the BF16 arm holds
0.93-0.96x of EXL3 across the range where the E4M3 arm inverts at 6 and
collapses at 7. §7.

**And W1's prediction reproduces on a different model, harness and H.** At
8 bpp on six dense Qwen3-0.6B Linears, H-weighted geomean:
**BF16 R = 8 0.00783 · E4M3 R = 8 0.02280 · FP8 RTN 0.02341** against W1's
predicted 0.0079 / 0.0234 / 0.0238 — **within 3% on all three arms**, six
tensors, through the real wire. At 1% more bytes than a full FP8 tile,
**3.0x less error**. §7c.

`TESSERA_BF16` is a real third family: it writes, reads back bit-exactly,
decodes four ways that agree to the bit, exports a checkpoint plus a plain-BF16
twin a stock loader serves, and is priced by the accountant at its true width.
It is not yet served by a lane and not yet selectable by the allocator; those
hand-offs are §8 and §10, and each names the exact change its owner has to
make.

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
bash experiments/bf16_r8_dense_run.sh                                # twin re-check through the renamed folded path, then dense R=8
python experiments/bf16_route_w1_identity.py --out .../w1_identity_merged.json   # the same identity check, re-run from the MERGED tree (see 11)
python experiments/bf16_window_rate_cliff.py                         # fused vs reference at R=6,7,8 -- whose R=8 cliff is it (see 11)
```

Both exporter runs were made at `fc2c1c1`, when the exporter was still
`experiments/export_gridbook_tessera.py`; the merge with `master` (§11) moved
that file to `experiments/export_tessera_serving.py` and the launcher above
names the new path. The exported bytes are the old name's, and the module the
shim now forwards to is the same code plus this branch's BF16 family.

`bf16_r8_run.sh` is in the tree beside these and is **superseded**: it ran the
GLM set at R=8 first, which is 1442 s (E4M3) + 969 s (BF16) per 2048x4096
tensor -- about four hours for six -- so it was killed after `L5.gate_proj`
(whose row is in `weight_space_glm_r8.json`, written per tensor) and the dense
set, where the R=8 question is actually sharp, was run first instead.

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
E4M3 default unless a measurement says otherwise, and none does.

> **Superseded on `window_sigma`, 2026-09-03 (#48).** A measurement does say
> otherwise, and it is the spread rather than the constant. `BF16_RECIPE` now
> names `window_sigma = 1.0` — byte-identical to `None` here, because
> `channel_sigma` is 1.0 — and `wire_recipe` scales it per rung as
> `sqrt(max(R, 4)/4)`, so the body's reach in row-RMS grows with the rate
> instead of staying at the R=4 value. Bytes move at every BF16 rung above
> R=4; the rest of this section stands.
> `docs/measurements/tessera-bf16-reach-recipe-2026-09-03.md`. `channel_sigma`
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
`docs/schema/prismaquant.tessera.v1.md` §1e.

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

> **Command note, 2026-09-02 (#41, #9).** The serving exporter now refuses a
> wire this plugin build publishes no decode for. `BF16` was one for about a
> day: `--grid BF16` needed `--allow-unserveable`, which writes the arm as a
> research artifact and stamps the refusal into the manifest's `serving_gate`
> block. It no longer does. `serving.scheme.ROUTES[TESSERA_BF16]` holds the
> grid, the window body and the CHANNEL plane, and `runtime_contract.json` v5
> publishes the reader range [256, 4096], so the arm below exports with no
> flag. Nothing about the bytes or the twin ever changed; only the flag, and
> now not even that. See `docs/tessera-serving-and-moe-contract.md` §11.

`experiments/export_tessera_serving.py --grid BF16` writes modules declaring
family `TESSERA_BF16`, and `--stock-twin DIR` writes, alongside it, a plain
BF16 safetensors of the decoded tiles under the *source's own tensor names*,
with `quantization_config` removed and every passthrough tensor copied
verbatim. That twin is an ordinary checkpoint: vanilla vLLM (or HF) serves it
with no plugin, which is how the served KL gate ran: the route and its twin
are the two servings of one encode, measured against each other in
`docs/measurements/tessera-bf16-route-served-2026-09-02.md`.

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

### 7a. Six GLM routed experts, output space

Layers 5 / 20 / 42 x `gate_proj` / `up_proj`, 2048x4096, real captured
activations, the 1024 held-out rows every other Tessera measurement uses.
`out` is `|x(W - Ŵ)ᵀ| / |xWᵀ|`; geomean over the six, bpp is the arithmetic
mean of what was actually written. EXL3 reconstructions come from
`/home/rob/dq-runs/exl3-ref`, quantized by its own quantizer.

| arm | bpp | `wt` | `out` |
|---|---:|---:|---:|
| EXL3 K=4 | 4.0117 | 0.08629 | 0.06736 |
| EXL3 K=5 | 5.0117 | 0.04387 | 0.03429 |
| EXL3 K=6 | 6.0117 | 0.02246 | 0.01753 |
| EXL3 K=8 | 8.0117 | 0.00608 | 0.00475 |
| FP8 RTN per-channel, LS-refit | 8.0078 | 0.02080 | 0.01881 |
| E4M3 window R=4 | 4.0195 | 0.06940 | 0.06273 |
| **BF16 window R=4** | 4.0352 | 0.06924 | **0.06276** |
| E4M3 window R=5 | 5.0195 | 0.03580 | 0.03242 |
| **BF16 window R=5** | 5.0352 | 0.03560 | **0.03219** |
| E4M3 window R=6 | 6.0195 | 0.02323 | 0.02100 |
| **BF16 window R=6** | 6.0352 | 0.01851 | **0.01673** |
| E4M3 window R=7 | 7.0195 | 0.02233 | 0.02020 |
| **BF16 window R=7** | 7.0352 | 0.00968 | **0.00874** |

**The ratios, which are the answer.** Lower is better; `< 1` means BF16 wins.

| R | BF16 / E4M3 same R | BF16 / E4M3 **one rung up** | BF16 / EXL3 K=R | BF16 / FP8 RTN (8.008) |
|---|---:|---:|---:|---:|
| 4 | 1.0004 | 1.936 | **0.932** | 3.337 |
| 5 | 0.9929 | 1.533 | **0.939** | 1.712 |
| 6 | **0.797** | **0.828** | **0.955** | 0.890 |
| 7 | **0.433** | — | — | **0.465** |

Read it as three statements:

- **Below 6 bits the alphabet is free and costs 0.016 bpp.** BF16 and E4M3
  are the same encoder there, to 0.1-0.7%, and the wide table's second byte is
  the only difference. Nobody should pay it below R = 6.
- **At 6 bits the alphabet is worth a whole bit.** BF16 at 6.035 bpp is
  **0.828x** of E4M3 at 7.020 bpp — the 16-bit route at R buys more than the
  8-bit route at R+1, on every one of the six units (0.806-0.855). That is the
  allocator's question, and it changes the answer.
- **At 7 bits E4M3 is not a rung any more.** 0.433x, because the E4M3 arm
  stopped improving (0.02100 -> 0.02020 from R=6 to R=7, a 4% gain for a whole
  bit) while the BF16 arm halved. And BF16 R=7 at 7.035 bpp is **0.465x** of a
  full FP8 RTN tile at 8.008 — better than 8-bit, at a bit less.

**And BF16 holds against EXL3 across the whole range**, 0.932-0.955x, where
the E4M3 arm inverts at 6 (1.198x) and collapses at 7 (2.318x). This is W1's
headline, reproduced through the wire.

**R = 8, one tensor.** R=8 costs ~24 min per 2048x4096 encode on this box, so
only L5.gate_proj was carried there (the sweep was stopped after it; the dense
R=8 screen below is the cheaper form of the same question):

| arm | bpp | `out` |
|---|---:|---:|
| E4M3 window R=8 | 8.0195 | 0.02217 |
| **BF16 window R=8** | 8.0352 | **0.00512** |
| EXL3 K=8 | 8.0117 | 0.00511 |
| FP8 RTN | 8.0078 | 0.01844 |

**0.231x of E4M3 at the same rate, and 1.002x of EXL3 K8** — the 16-bit route
lands *on* EXL3's 8-bit point, which the 8-bit alphabet misses by 4.3x. The
E4M3 arm's 0.02217 is its saturation value from R=6 (0.02309 / 0.02218 /
0.02217 at R=6/7/8 in W1's deeper run): the alphabet, not the trellis.

### 7b. The twin's fold, priced

Every arm above was also scored after folding to bf16 (`out_bf16`), which is
what the stock twin ships. `fold = sqrt(out_bf16² − out²)`, geomean over the
six units:

| arm | fold | as % of `out` |
|---|---:|---:|
| BF16 window R=4 | 0.00126 | 2.0% |
| BF16 window R=5 | 0.00144 | 4.5% |
| BF16 window R=6 | 0.00135 | 8.1% |
| BF16 window R=7 | 0.00134 | **15.4%** |
| E4M3 window R=4..7 | 0.00145-0.00182 | 2.4-6.2% |
| EXL3 K=4 / K=6 / K=8 | 0.00128 / 0.00143 / 0.00136 | 2.0% / 8.2% / **28.6%** |
| FP8 RTN | 0.00143 | 7.8% |

**It is a constant, not a tax on this format.** 0.0013-0.0018 in absolute
terms on every arm at every rate, including EXL3's and RTN's, because it is
bf16's 7-bit mantissa meeting these activations. Its *share* grows only
because the coding error shrinks underneath it. **This is the twin's cost, not
the route's** — `materialize_bf16` returns the pair and pays none of it (§4),
and a served number taken on the twin is therefore a ceiling.

**How big a ceiling — corrected 2026-09-02.** An earlier reading of this table
turned the 15.4% column into "at R = 7 the route is ~15% better than its own
twin". It does not follow, and the definition three lines up is why: `fold` is
defined *in quadrature*, `out_bf16 = sqrt(out² + fold²)`, so a 15.4% share
raises the twin's error by `sqrt(1 + 0.154²) - 1 = 1.2%` and its **squared**
error — the quantity an output-space KL tracks — by 2.4%. Fifteen percent was
never in this table. Transferred to the dense Qwen units of §7c, where `out` at
R = 7 is 0.01214 and the fold constant is the same 0.00134, the share is ~11%
and the squared-error gap ~1.2%. **Served, it is smaller still**: on the R = 7
dense Qwen wire the twin's KL is 1.0011x the route's on `all` and 0.9961x on
`confident` — a signed disagreement, i.e. below what an n=8 x 512 corpus
resolves (`tessera-bf16-route-served-2026-09-02.md` §3, issue #45). The fold is
still the twin's cost and not the route's; what is not established is that it
is *visible* at this rung.

### 7c. Six dense Qwen3-0.6B Linears, H-weighted

`layers.2` {`down_proj`, `gate_proj`, `q_proj`, `o_proj`} plus
`layers.14.mlp.down_proj` and `layers.27.mlp.down_proj` — the three `down_proj`
being the units the dense-outlier work identified as the hard ones. `h` weights
each input column by the diagonal H the stock census captured; geomean over
the six.

| arm | bpp | `wt` | `h` |
|---|---:|---:|---:|
| FP8 RTN per-channel, LS-refit | 8.0182 | 0.02622 | 0.02341 |
| NVFP4 GPTQ+JSO (production export) | 4.5000 | 0.09237 | 0.08374 |
| E4M3 window R=4 | 4.0577 | 0.07211 | 0.07374 |
| **BF16 window R=4** | 4.1063 | 0.07188 | **0.07339** |
| E4M3 window R=5 | 5.0577 | 0.03924 | 0.04044 |
| **BF16 window R=5** | 5.1063 | 0.03744 | **0.03898** |
| E4M3 window R=6 | 6.0577 | 0.02803 | 0.02744 |
| **BF16 window R=6** | 6.1063 | 0.01997 | **0.02150** |
| E4M3 window R=7 | 7.0577 | 0.02665 | 0.02375 |
| **BF16 window R=7** | 7.1063 | 0.01078 | **0.01214** |
| E4M3 window R=8 | 8.0577 | 0.02652 | 0.02280 |
| **BF16 window R=8** | 8.1063 | 0.00640 | **0.00783** |

| R | BF16 / E4M3 same R | BF16 / E4M3 one rung up | BF16 / NVFP4 4.5 | BF16 / FP8 RTN 8.02 |
|---|---:|---:|---:|---:|
| 4 | 0.995 | 1.815 | **0.876** | 3.134 |
| 5 | 0.964 | 1.421 | 0.465 | 1.665 |
| 6 | **0.784** | **0.905** | 0.257 | 0.918 |
| 7 | **0.511** | 0.532 | 0.145 | **0.518** |
| 8 | **0.343** | — | 0.094 | **0.334** |

The same three statements hold on dense weights, and the table costs more here
(+0.049 bpp, not +0.016, because these Linears are 1024x3072 and smaller). Two
additions:

- **At R = 6, 6.11 bpp, the 16-bit route is already better than a full FP8
  tile at 8.02** (0.918x) and better than the 8-bit route a whole bit above it
  (0.905x). At R = 7 it is **1.93x better than FP8 RTN at 0.9 bpp less**.
- Both Tessera arms beat the *production* NVFP4 GPTQ+JSO export at 4.5 bpp
  while being smaller (4.06-4.11 bpp), which is the already-known Tessera
  result and is here only as a sanity rail.

**At 8 bpp the two alphabets separate by 3x, and the screen reproduces W1's.**
This is the point the whole family was built for, so it is stated as a
prediction that was checked rather than as a result that was found. W1's
alphabet-floor screen predicted **BF16 R = 8 0.0079 vs E4M3 R = 8 0.0234 vs
FP8 RTN 0.0238**. On six dense Qwen Linears through the real wire the
H-weighted geomean is **0.00783 / 0.02280 / 0.02341** — the same order, and
every arm **within 3%** of the predicted value (0.9% / 2.6% / 1.6%), on a
different harness, different tensors and a different H. On plain Frobenius the
order is the same and wider (0.00640 / 0.02652 / 0.02622).

Two readings of that row, and only the second is the family's claim:

- **The 8-bit route is done improving and the 16-bit one is not.** E4M3 goes
  0.02744 -> 0.02375 -> 0.02280 at R = 6, 7, 8 — it has spent its alphabet,
  and the last bit buys 4%. BF16 goes 0.02150 -> 0.01214 -> 0.00783, still
  taking ~1.6-1.8x per bit on this axis (1.7-1.9x on `wt`, W1's ~1.93x). The
  floor is the alphabet's, measured twice now.
- **A 16-bit-alphabet trellis at 8.11 bpp is 3.0x better than a full FP8 tile
  at 8.02 bpp** — 1% more bytes, three times less error — and **BF16 at R = 7
  (7.11 bpp) is 1.9x better than E4M3 at R = 8 (8.06 bpp)**, a whole bit
  cheaper. Above ~6 bpp the 8-bit route is not a rung the allocator should be
  offered when this one exists.

**Per-unit crossover — and the one unit that does not.** The crossover rate
(lowest R where BF16 beats E4M3 at the *same* R, the conservative reading
since BF16 is paying more there):

| unit | shape | `wt` | `h` | BF16 R=8 / E4M3 R=8, `h` |
|---|---|---|---|---:|
| `layers.2.mlp.down_proj` | 1024x3072 | R=4 (0.9988x) | **never in R=4..8** | **1.048** |
| `layers.2.mlp.gate_proj` | 3072x1024 | R=4 (0.9975x) | R=4 (0.9935x) | 0.197 |
| `layers.2.self_attn.q_proj` | 2048x1024 | R=4 (0.9920x) | R=4 (0.9880x) | 0.424 |
| `layers.2.self_attn.o_proj` | 1024x2048 | R=4 (0.9967x) | R=4 (0.9955x) | 0.243 |
| `layers.14.mlp.down_proj` | 1024x3072 | R=4 (0.9980x) | R=5 (0.9626x) | 0.300 |
| `layers.27.mlp.down_proj` | 1024x3072 | R=4 (0.9979x) | R=4 (0.9849x) | 0.257 |

Five of six cross at R = 4 on both axes. **`layers.2.mlp.down_proj` never
crosses on the H-weighted axis** — 1.010x / 1.019x / 1.018x / 1.044x / 1.048x
at R = 4..8 — while on plain Frobenius it is 0.72x at R = 6, 0.41x at R = 7
and **0.23x at R = 8**. The R = 8 point sharpens the diagnosis rather than
softening it: on this one unit the 16-bit alphabet is worth 4.3x on plain
error and *nothing at all* on the columns H actually weights, which is what a
reach problem looks like and is not what an alphabet problem looks like.
The residual is concentrated in its high-H columns, and this is the unit whose
rows the reach fix was written for. The likely mechanism is the σ: at the same
L and seed the BF16 table reaches **4.00 σ** where E4M3's reaches 4.08, and
`BF16_CHANNEL_SIGMA` is **stated, not searched** (§3). Do not read this row as
an alphabet result; read it as the measurement that says searching σ (and L)
for this grid is the first experiment, not an optional one. It is also why the
menu should keep both alphabets below R = 6, where they are within 1%
anyway.

---

## 8. Hand-off: the serving plugin (W2)

> **Taken up 2026-09-02 (issue #9).**  `serving/bf16_route.py` and the
> `TESSERA_BF16` `ROUTES` entry exist and are built to this section's spec: no
> new flag, no new packing, the row scale an fp32 epilogue on an
> `out_dtype=torch.float32` GEMM and never folded, and a load-time element-for-
> element check against `materialize_bf16`.  Two things below are *not* what
> shipped, and both are the contract's own vocabulary rather than a change of
> mind: the rungs go in `attested_rungs_q256` (`candidate_rungs_q256` is a
> deprecated alias that must carry the identical list), and both were **empty**
> with **no `lane_eligibility` cell** on the day the route landed, because "the
> route status is `backed`" is a claim about a runtime and needs a receipt, not
> a hand-off's say-so (principle 14, which this section already says).  The
> receipt arrived the same day: contract v5 attests `q256 = 1792` and publishes
> two `sm_121` dense cells on four route censuses plus a served KL against the
> twin — `docs/measurements/tessera-bf16-route-served-2026-09-02.md`.  One rung,
> one platform, dense only; every other rung in the reader's `[256, 4096]`
> range still resolves `unattested`, which is the same rule, not an exception
> to it.

The plugin adds a **third family** alongside `TESSERA_NVFP4` (W4A4) and
`TESSERA_FP8` (W8A8): `TESSERA_BF16`, **W16A16**, materialising into an
ordinary bfloat16 weight. It needs no new flag — the checkpoint's
`quant_method: "tessera"` selects the plugin and `TESSERA_SERVE_MODE` declares
the residency, exactly as for the other two — and no new packing: there is no
stock quantized layout to build, because bf16 *is* the stock layout.

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

**Two names, and they are not the same name.** `serving/contract.py:198-205`
says so itself: a `runtime_contract.json` `formats[].family` is a **payload**
name a producer prices (grid + arity — `TESSERA_E4M3_K1`), while a
`scheme.ROUTES` key is **what the tile is on the hardware** (`TESSERA_FP8`),
and `_FAMILY_TO_ROUTE` maps one to the other. The checkpoint's per-target
`scheme["family"]` is the *route* key — `validate_tessera_scheme` checks it
against `TESSERA_FAMILIES = tuple(ROUTES)` — so the exporter here writes
`TESSERA_BF16`, the route key. A contract row needs the payload name beside
it; `TESSERA_BF16_K1` follows the existing convention, and picking it is the
plugin's call, not this branch's.

**What the route spec should say.** The `ROUTES` entry, filled in the shape of
the `TESSERA_FP8` one at `scheme.py:95-102`:

```python
TESSERA_BF16: {
    "grids": ("BF16",), "plane": "CHANNEL",
    "short": "BF16",
    "builder": ("tessera.serving.bf16_route", "build_tessera_bf16_method"),
    "tile": "bf16 (stock bfloat16 weight, one fp32 scale per row -- an epilogue, not folded)",
    "columns_multiple": 1,          # a BF16 GEMM has no K quantum of ours
    "activation_contract": BF16_ACTIVATION_CONTRACT,   # the A side is unquantized
}
```

and the contract rows: `kind "tessera_wire"`, payload family
`TESSERA_BF16_K1`, `name_pattern "TESSERA_BF16_K1_R{k}"`, candidate rungs the
q256 the artifact was built at (1536 and 1792 here), `native_terminal_q256`
4096, `residency_modes ["resident", "streamed"]`. Streamed residency is the
artifact's own wire bpp; resident is 16 bpp. There is no `--moe-backend`
requirement, no `requires_serve_flags` beyond the mode, and no kernel gate:
the GEMM is the runtime's own BF16 GEMM, so the cell's `route_status` is
`backed` rather than `backed_with_serve_flag` — subject to the plugin's own
attestation, not to this receipt's say-so (principle 14).

**The extension point is already written down.** `scheme.py:105-111` describes
this family unprompted — *"a third family (say a WINDOW/CHANNEL body whose
alphabet is snapped to bf16 and decoded to a bf16 tile for the stock GEMM) is
one route module, one `ROUTES` entry naming its builder, and its contract rows
-- and nothing here or in `lane` has to be edited to admit it"*. That is the
shape this hand-off assumes; `bf16_route.py` is the route module it names.

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
(`base in _HARDWARE_BASES` already yields this). The two renderings are
**different functions and must not be confused** (§4):
`tessera.decode.materialize_bf16` returns the *pair* `(bf16 tile, fp32 row
scale)` and is what a lane consumes; `tessera.decode.materialize_bf16_folded`
returns the single folded tile and is the **stock/twin** materialisation — the
one a `materialize_stock`-style path or a plain-BF16 export writes, and the one
that pays the fold rounding priced in §7b. The served route is §8's.

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
Six `advisor()` calls were made, one before committing to the grid shape and
five in the endgame. They produced eleven of the checks in this document: the
encoder-drift diff, the structural twin check, the `grid_from_config` hole,
the PrismaQuant accountant line, the stale `materialize_bf16` sentence in
§10d, this section's contention note, the merged-tree signature check on
`encode_linear_planes`, the route-namespace correction in §8, the merged-tree
identity re-run above, the fused-vs-reference discriminator that turned the
R = 8 cost from a suspicion into an attribution, and — last — the two
overclaims corrected in place here: "three digits" became the exact bound
(within 3%), and the `_tile` fix was demoted from "the real fix" to a
hypothesis that may spill worse than what it replaces.

**Merged with `master` at `f3e7d0a`** (the serving plugin, TP slicing / schema
minor 4, the window kernel, LDLQ + the Hessian plumbing). Two conflicts, both
resolved by hand: this document's status entry, and the exporter -- `master`
moved `experiments/export_gridbook_tessera.py` to
`experiments/export_tessera_serving.py` and left a shim at the old name, so
**the BF16 family lives in `export_tessera_serving.py`** and the shim is
`master`'s. Everything else auto-merged. The no-touch files (`serving/`,
`kernel*.py`, `compensate.py`, `scale_channel.py`, `layout.py`) are untouched
by this branch.

**Every number in this document predates that merge — and one of them was
re-taken after it, to check that this matters less than it sounds.** They were
taken at `fc2c1c1` (stamped in the artifacts and in the sweep JSONs), before
`master`'s LDLQ + full-H refit landed. Rather than assert from `master`'s
docstring that a weights-only encode is unchanged, §1's identity harness was
re-run from the merged tree
(`w1_identity_merged.json`, `TESSERA_GIT=dcbd902`) and reproduces the
pre-merge out-space errors **exactly**: 0.06693 / 0.03447 / 0.01788 / 0.00937
at R = 4 / 5 / 6 / 7, with `value_mismatches` still 0 and
`cdist_vs_exact_mismatches` still 30. The reason is visible in the signature —
`encode_linear_planes` defaults `ldl=None` and `refit_metric=None`, so the new
machinery is opt-in and a weights-only call reaches the same code — but the
check is the evidence, not the reasoning (principle 14, applied to this
branch's own claims).

**What that verification does and does not cover.** It covers the BF16 wire at
R = 4..7 on one GLM tensor, end to end through the real bytes. It does **not**
re-take the E4M3 arm, the six-tensor GLM sweep, or any dense unit. Those are
still `fc2c1c1` numbers, and when the H-aware encoder is actually *used* it
will move both arms — so for §7 the *ratios* remain the claim. Re-running the
sweep with LDLQ on is the first thing that would sharpen it; re-running it
weights-only would, on this evidence, reprint the same table.

**Test state after the merge:** `pytest tests -q` is **792 passed, 5 skipped,
8 failed** on the host. All 8 failures (`test_serving_nvfp4_route.py` ×3,
`test_serving_sharding.py` ×5) reproduce **identically on a clean `master`
checkout** (`git archive master | …`, 8 failed / 36 passed) and none touch
this branch's files; they are `master`'s to fix.

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
- **No allocator wiring.** §10 states the changes PrismaQuant needs, one of
  which (`ANCHOR_BUDGET_BITS`) currently refuses the family outright.
- **Both arms are weights-only**, and the H-aware encoder (LDLQ + full-H
  CHANNEL refit) landing on another branch applies to both equally — see §7's
  preamble. Absolute errors here will fall when it lands; the ratios are the
  claim.
- **L = 14 and σ = 1.0 were inherited and stated, not searched, for this
  grid** (§3). This is not a formality: at the same L and seed the BF16 table
  reaches **4.00 σ** where E4M3's reaches 4.08, and on the one dense unit whose
  rows are outlier-heavy (`layers.2.mlp.down_proj`) the H-weighted error tracks
  that reach — BF16 is 1.01-1.04× *behind* E4M3 there at every rate, while
  being 0.41-0.72× ahead on plain Frobenius (§7c). Searching (L, σ) for this
  grid is the first experiment, and a wider table costs two bytes an entry
  here, so the L-vs-rate frontier is a different curve from E4M3's. W1's
  reasoning about L = 16 (its "L = 16: not run") applies with the cost doubled.
- **The R = 8 *GLM* point is one tensor, not six.** R = 8 costs ~24 min per
  2048×4096 encode on this box (see the encode-cost bullet below), so that
  sweep was stopped after L5.gate_proj and the budget was spent on the dense
  R = 8 screen instead — which *did* complete on all six units (§7c) and is
  the one the alphabet-floor prediction is checked against. What remains
  unmeasured is the six-**expert** GLM R = 8 geomean and, with it, a
  BF16-vs-EXL3-K8 number on more than one tensor.
- **R = 8 costs 40-80x R = 7, and the cost is the window kernel's, not the
  family's.** *(2026-09-02, later the same day: the attribution below stands
  and the diagnosis under it is now COMPLETE, so read the two proposals at the
  end of this bullet as history. The extra 20-40x this bullet could not account
  for is a **register spill** -- 690 bytes per thread at R = 8 and none at
  R <= 7, read off the compiled kernel -- and the fix is the class scan's
  spelling, not `_tile`: the candidate `BL` widening is REFUTED, because a
  wider tile puts more elements under each hoisted load. The dispatch rule this
  bullet proposed shipped as `WINDOW_FUSED_MAX_RATE = 7` and has now been
  withdrawn: the fused path is 8.1x faster than the reference at R = 8 and the
  crossover moved to 11. See
  `docs/measurements/tessera-window-viterbi-scan-2026-09-02.md`, issue #11.)* On one dense tensor (`model.layers.2.mlp.down_proj`, 1024x3072,
  identical code path, both arms) the encode seconds run **2, 3, 5, 10** at
  R = 4, 5, 6, 7 -- a clean 2x per bit -- and then **424 (E4M3) / 639 (BF16)**
  at R = 8; on `gate_proj` the R = 8 pair is 820 / 740 against the same 10 s at
  R = 7. Compile time is not the explanation: every R = 8 call has *identical*
  constexprs (`ARITY`, `FAN`, `SIZE`, `HAS_W`, `BACK_U8`, `BL`, `BC`), so the
  second, third and fourth encodes reuse the cached kernel, and the slowest of
  the four is the last. Nor is it launch geometry: reading `_tile`
  (`window_viterbi.py:238-249`) at L = 14, `bl * FAN` is pinned under 1024 and
  `bl * FAN * bc` under 2048, so `grid` is **(16, width/2)** and `warps` is
  **4** at every rate from 4 to 8, `BL*FAN*BC` is 2048 at every rate, and the
  launch count (`2 + steps`) does not depend on the rate at all. What *does*
  scale is the class-minimum loop `for f in tl.static_range(1, FAN)`
  (`window_viterbi.py:172`): it is **fully unrolled**, so its instruction count
  is `FAN - 1` (15 -> 255 over R = 4..8) while its per-iteration tile
  `[BC, BL]` shrinks as `2048 / FAN` (128 -> 8 lanes). The doubling is what the
  clock sees -- a serial unrolled loop costs its length -- and the shrinking
  tile is why it is never recovered: past `FAN = 64` the tile is at or under a
  warp (32 -> 16 -> 8 lanes at R = 6, 7, 8), so each iteration issues at
  roughly fixed cost however narrow it gets. That accounts for the
  2x-per-bit trend. It does **not** account for the extra
  20-40x at `FAN = 256`, and rather than leave that as a suspicion the
  discriminator was run (`experiments/bf16_window_rate_cliff.py`): **the same
  tensor, both implementations, one process**, `viterbi_window(impl="fused")`
  against `impl="reference"` (the torch chain, the definition) on a 1024x1024
  tensor at L = 14.

  | impl | R = 6 | R = 7 | R = 8 | R6->R7 | R7->R8 |
  |---|---:|---:|---:|---:|---:|
  | `reference` | 6.548 s | 6.544 s | 6.631 s | 1.00x | **1.01x** |
  | `fused` | 0.753 s | 1.474 s | **65.004 s** | 1.96x | **44.09x** |

  `sse` is identical between the two at every rate (392.791046 / 122.382999 /
  50.547596), so this is one answer computed two ways, not two answers.

  **The reference path is flat in the rate** — which is the algebra: per step
  the trellis evaluates `low * FAN = 2^L` transitions whatever the rate, so
  rate-independence is what a correct implementation *should* show, and the
  torch chain shows it to 1%. **The entire cliff is the Triton step kernel's**,
  and it is worse than "slow": at R = 8 the fused path is **9.8x slower than
  the reference it exists to replace**, having been 8.7x faster at R = 6. The
  crossover is between R = 7 and R = 8. Attributed, not suspected.

  Two consequences for whoever owns `window_viterbi.py`. **The cheap fix is a
  dispatch rule, not a kernel rewrite:** `viterbi_window`'s `impl="auto"`
  currently takes the fused path whenever the input is CUDA and Triton is
  present (`encode.py:538-543`); making `auto` prefer the reference at
  `1 << rate` above the crossover would make R = 8 encodes ~10x faster and is
  bit-exact by the table above. **The candidate fix in the kernel itself is
  `_tile`, and it is a hypothesis, not a measurement:** the cap
  `bl * FAN < 1024` shrinks `BL` as `FAN` grows, so the fully unrolled
  class-minimum loop doubles in length while its `[BC, BL]` tile falls to 8
  lanes at `FAN = 256`, and holding `BL*BC` at a warp would stop that. But the
  kernel's second half builds `state` and `cost` as `[BC, BL, FAN]`, so
  widening `BL` multiplies the branch-cost register footprint by the same
  factor (`BL = 16` at `FAN = 256` is 8192 floats a program) and may spill
  worse than the unroll does. Whoever owns the kernel should measure it, not
  take it from here. **Neither was changed here** —
  `encode.py` is shared with three other branches mid-measurement, this
  branch's diff is meant to be a grid and a recipe, and changing a hot path
  under five running jobs is the "don't change a kernel mid-A/B" landmine.
  Recorded as a hand-off with the measurement attached.

  One caveat on the *sweep's* seconds, separate from the above. The box is
  shared: six GPU processes from other branches were resident on sparklina
  while §7c's R = 8 arms ran (32 W of a ~140 W envelope, `gpu_utilization`
  96%, six host cores each at ~100% — principle 15's exact signature,
  utilisation saying "saturated" while power says one-quarter loaded), so the
  424 / 639 / 820 / 740 s figures are contended wall-clock. The discriminator
  above is immune to that: both arms saw the same box in the same process.

- **One timing column is contended, and no error number is.** The tail
  runner's `pgrep` guard listed the export and the sweep but not
  `bf16_twin_check.py`, so it started at 14:25:03 while the R=7 twin check was
  still running, and the W1-identity and structural-twin stages then overlapped
  the GLM sweep. Every error figure here is deterministic and unaffected; the
  `secs` columns in `w1_identity.json` and in the first GLM tensor are the only
  contended numbers, and neither is quoted as a claim. §5's per-unit encode
  times (9.35 / 10.88 s) were taken with the GPU to itself.
- **Rates above 8 are untested by measurement.** They are legal to R = 16 and
  the wire round-trip test covers the widening path, but the product range is
  4..8 and that is where every number here lives.
