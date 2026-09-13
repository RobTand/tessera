# Where a Hessian-aware encode diverges between gfx1201 and GB10

**Date:** 2026-09-13 · **Issue:** #472 · **Boxes:** sparklina (GB10, sm121,
torch 2.11.0+cu130) and wsl-gpu / DESKTOP-P5UOGNJ (AMD RX 9070 XT, gfx1201,
torch 2.11.0+rocm7.2.4, Triton 3.7.0+rocm) · **Runner:** PrismaBuild only.

A `TESSERA_BF16_K1_R1792` encode of one unit produces a byte-identical wire on
both boxes with weights only, and two different wires with the Hessian-aware
keywords. #472 measured the difference but did not separate its cause: whether
`ActivationSource.for_unit` **produced** different `ldl` bytes on gfx1201, or
the encoder **consumed** identical bytes differently. This separates it. Both
are true, and the larger half is the first.

## Unit and inputs

Qwen3-0.6B `model.layers.0.mlp.down_proj.weight`, 1024x3072 bf16, sha256
`01cc5c35…`. The Hessian is the same deterministic synthetic 3072x3072 fp32
XtX shipped in the snapshot on every arm (sha256 `9cac8ba1…`, and it digests
to that on both boxes after the device round trip), so the encoder's inputs
are fixed across ISAs by construction. It is not a calibration capture and
prices nothing. `ldlq_sigma` 1.0, `ldl_block` 32, `refit_gauss_seidel` false,
`refit_reach_floor` false on both boxes. `float32_matmul_precision` is
`highest` and `allow_tf32` is false on both, so TF32 is not in this.

## Stage by stage inside `for_unit`

`block_ldl` was wrapped, not reimplemented, so every tensor digested is the
one the real path builds.

| stage | GB10 sha256 (24) | gfx1201 sha256 (24) | equal |
|---|---|---|---|
| XtX after `.to(device, fp32)` | `9cac8ba17de3e60d7a8296f` | `9cac8ba17de3e60d7a8296f` | **yes** |
| `H_reg` = `regularize_hessian` | `e203ba647b13df4f1518336` | `e203ba647b13df4f1518336` | **yes** |
| `torch.linalg.cholesky(H_reg)` | `e3602c8ddb0e8293e2ba89e` | `9d8dc851beae36c1a73fd75` | **no** |
| `block_ldl` output (`ldl`) | `ce0cd5f8d9c63b496eea8bb` | `6f9f632eea4a5a049b4f5bd` | **no** |
| `refit_metric` (= H) | `9cac8ba17de3e60d7a8296f` | `9cac8ba17de3e60d7a8296f` | **yes** |

`regularize_hessian` is cross-ISA exact including its damping scalar: the
`float(diag.mean())` that is added to every diagonal element is the same
double on both boxes (`40bff9ec60000000`), and so is `float(H.sum())`
(`417804e000000000`). **The first divergent stage is
`torch.linalg.cholesky`** — cuSOLVER on one box, rocSOLVER on the other.

How far apart, on the dumped tensors:

| tensor | elements differing | rel. Frobenius | max abs | that entry's magnitude |
|---|---|---|---|---|
| `cholesky` | 4 230 175 / 9 437 184 (44.82%) | **1.490e-07** | 4.578e-05 | 90.36 (‖·‖ max 91.99) |
| `ldl` | 4 293 639 / 9 437 184 (45.50%) | **1.133e-07** | 4.470e-08 | 0.0203 (‖·‖ max 1) |

fp32 epsilon is 1.192e-07. Both relative Frobenius differences are **one
epsilon**. The large *relative* errors are confined to entries that are
numerically zero: only 519 of 9.4 M `cholesky` entries move by more than
0.1% relatively, and none of those exceeds 1.4e-03 in magnitude. This is a
backward-stable factorisation computed two ways, not a bug in either.

## Which half of the divergence is which

Four wires of the same unit. The cross-feed encodes on one box with the
*other* box's `ldl` — every other input already digests equal — so the two
causes can be held apart.

| `ldl` from | encoder ran on | wire sha256 (24) | render MSE |
|---|---|---|---|
| GB10 | GB10 | `e29b64fe6ca0a0192d42c429` | 8.189365e-08 |
| gfx1201 | gfx1201 | `5e325c0967a6e1e779b585f5` | 8.189844e-08 |
| **GB10** | **gfx1201** | `728fcdded3777ab700110049` | 8.188513e-08 |
| **gfx1201** | **GB10** | `287a4fcd5424ee43459c82da` | 8.191608e-08 |

Four inputs, four distinct wires. Feeding GB10's `ldl` to the gfx1201 encoder
does **not** reproduce GB10's wire, so the encoder does not consume identical
bytes identically either. Decoding and differencing the renders attributes
the two halves (percentages are of 3 145 728 elements, at the decoder's fp32,
which is the convention #472's 54.0% was measured in; the same pairs at the
unit's bf16 — what a served checkpoint stores — are in parentheses):

| held fixed | varied | elements differing |
|---|---|---|
| encoder box (gfx1201) | the `ldl` | **53.78%** (35.15%) |
| the `ldl` (GB10's) | encoder box | **15.28%** (7.91%) |
| — | both (the real cross-box case) | **53.96%** (35.21%) |

For scale, the Hessian-aware wire differs from the weights-only wire of the
same unit in 73.68% of elements, and the encode's own `render_max_abs_err`
is 3.857e-02. The largest render disagreement between any two of the four
wires is 9.77e-03, a quarter of that, and all four render MSEs agree to
4 parts in 10^5.

**So: the `ldl` input is the larger cause and the encoder's own consumption is
the smaller one, and neither is zero.** The weights-only path — the window
Viterbi body and the channel scale plane — is bit-exact across the two ISAs
(`873af26a17ebcec669295a2d` on both), so the divergence lives entirely in the
LDLQ compensation arithmetic.

## Reading

This is expected cross-ISA floating-point nondeterminism, amplified by a
discrete decision, and not a defect:

- The inputs are bit-identical and the first divergence is a vendor LAPACK
  call. Two backward-stable Cholesky implementations with different blocking
  and rank-k update order do not agree bit for bit, and these agree to one
  fp32 epsilon in norm, which is as close as fp32 allows.
- The encoder's residual half is the same phenomenon one level down: the LDLQ
  compensation and refit are BLAS reductions whose split and order are a
  property of cuBLAS or rocBLAS, not of the recipe.
- The amplification is structural, not numerical. The trellis takes an argmin.
  Near a tie, an epsilon moves the winner, and every later step inherits the
  new state. That is why half the elements move while the render MSE does
  not: these are equally good quantisations, not a better and a worse one.

There is therefore no small, verifiable fix. Making the wires bit-comparable
would require pinning both vendors' Cholesky blocking **and** every reduction
order inside the refit — neither is exposed, and a deterministic reduction on
one side alone would change GB10's wires without making them match.

## What the contract must record

1. **A Hessian-aware Tessera wire is reproducible within an ISA, not across
   one.** gfx1201 reproduced its own wire exactly on a rerun, and two GB10
   boxes (sparky and sparklina) produce the same wire as each other. The
   campaign's existing within-box reproducibility claim stands; a cross-ISA
   one does not.
2. **A row that compares wires by digest must record the producing ISA**, or
   it will read an expected library difference as a defect. `ldl`'s own sha256
   is the tightest available discriminator and is already computed inside
   `for_unit`.
3. **Comparability across ISAs is on the render metric, not the bytes.** The
   four wires here agree on `render_mse` to 4e-05 relative and on
   `render_max_abs_err` exactly.
4. **Weights-only Tessera-16 quanta are cross-ISA byte-comparable** and can be
   mixed freely between GB10 and gfx1201 workers. Only the H-aware class needs
   the ISA recorded.

## Receipts

PrismaBuild, `--priority -10`, all `--demand gpu=1`; records under
`/mnt/shared/prismabuild-fleet/pb-queue/done/<key>.json`, outputs under
`/mnt/shared/agents-w472/`.

| label | tag | key (12) | what |
|---|---|---|---|
| `lina-stages` | sparklina | `9af4d7507c7e` | GB10 stage digests + dumps |
| `wslgpu-stages` | wsl-gpu | `55960eca45f4` | gfx1201 stage digests + dumps + cross-feed of GB10's `ldl` |
| `lina-crossfeed` | sparklina | `b94b5dc7c3f5` | mirror cross-feed of gfx1201's `ldl` |
| `wirediff` | sparklina | `ac07361bcdae` | decode and pairwise-difference the four wires |
| `wslgpu-default` | wsl-gpu | `53861600dc18` | the default path on gfx1201 after the `fused_available` guard |

Earlier receipts this builds on are in #472: `a76ee88b5bae` (GB10 control),
`7e5561c153f9` and `69cbdc0a3d72` (the reference-path arms),
`eaed04d3bca7` / `4923ea1946ab` (GB10 fused, and its within-box control).
`fa7cb2daef07`, a duplicate GB10 control pinned to `sparky`, was withdrawn as
superseded: `a76ee88b5bae` answered it on sparklina and `9af4d7507c7e`
answered it again with stage digests.
