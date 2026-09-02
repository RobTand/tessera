# Math-audit triage (verified) — 2026-09-02

Seven Muse Spark 1.3 workers verified the audit section by section, read-only,
each required to reproduce every finding with a pasted snippet and its exact
output, and to say whether the fix changes encoded BYTES. Reports:
/home/rob/tmp/audit/report{1..7}.md

## Dismissed — NOT-REPRODUCED (do not fix)
- §1 P2 `ILLEGAL_INEXACT_UNDERFLOW` names "mid-range gaps": exhaustive sweep over
  all 256x16 halves at every clip -> every such value is <= 15/1024, i.e. always
  subnormal-band. The name is accurate.
- §4 P0 `x**2` vs `x*x`: 22,052,527 bit patterns + 1M randoms + edges -> 0
  bitwise differences on torch 2.11.0+cu130.
- §4 P2 torch `min`/`argmin` first-minimal: documented in torch's own docstring,
  not an assumption; the fused kernel's strict-`<` scan matches by construction.
- §4 P2 `shift = points.bit_length()-1`: `points` is always 2^(R-1), so this is
  identically `point_bits`. No-op.
- §6 P2 "fp16 normals": all 254 finite E4M3 bytes round-trip fp16 bit-exactly.
- §3 G2a "explicit bitorder": the calls do NOT pass `bitorder=`; the hardening
  ask stands, its premise does not.

## Tier A — real, reachable on a shipping path
1. **§5 P0-3 CHANNEL refit has no `B>0` hold** (`scale_channel.py:245`). Its LUT
   sibling has `valid=(A>0)&(B>0)` (`encode.py:943`). With `B<=0` the candidate
   is negative, lands at the smallest positive fp16 scale, and the parabola
   *prefers* it (err_new ~ 0 < err_old) -- the row's scale collapses to 6.1e-5
   and every weight in it decodes to ~0. CHANNEL is the default plane for E4M3
   and BF16. Byte-affecting. **Must be measured on real units before and after:
   how many rows fire today.**
2. **§5 P0-4 `land_at_least` emits `inf`** (floor 65505 -> stored inf), and
   `0*inf = nan` poisons the `A s^2 - 2 B s` comparison, so `nan < err` silently
   keeps or drops scales. Only already-broken rows change bytes.
3. **§5 P0-5 `shared_lut_global` admits subnormal moved bytes** (`fused.py:136`
   checks finite+equal, not the 0x08..0x7E normal range). The kernel decodes a
   scale byte by field arithmetic `2^(e-7)(1+m/8)`, which is wrong for exponent
   field 0. The audit's own instance is wrong (0x08->2^-11 is refused); seven
   neighbouring entries reproduce. Load-time only, no stored bytes.
4. **§5 P2-10d fused CHANNEL gap**: `prepare_tessera_module` feeds `scale_lut`
   into `shared_lut_global` unconditionally, so a CHANNEL+TCQ export (a
   supported recipe) hits `AttributeError: 'NoneType'` instead of a
   `GrammarError`. Serving hole, no bytes.
5. **§6 P0 TP shard start state dropped** -- and worse than the audit says.
   `kernel_window_gemv.py` and `bf16_route.py` never read `initial_state`
   (0 occurrences); `kernel_window.py:359` reads `getattr(unit,"initial_window")`
   but `SlicedUnit` carries `initial_state` (`layout.py:474`), so the Triton
   parsed path decodes shards from zero too. Proven behaviourally: a shard's
   first ceil(L/R) rows decode wrong. Nothing in `serving/` imports either
   module today, so it is reachable via the public API and the bench harnesses,
   not via a served lane -- but TP row-sharding is a shipped design promise.
6. **§6 P0 missing `rows % 8` / `cols % half` guards** in 5 of 6 kernel
   wrappers. The packer refuses these units, so no artifact encodes them; a
   hand-built call gets silently dropped columns (GEMV) or an out-of-range
   scale-group read (GEMM). Guards only, no bytes.
7. **§2 P0-4/P0-5 partial trailing superblocks, reachable from `encode_linear`.**
   The verifier filed these latent; they are not. Nothing upstream refuses
   `columns % 256 != 0` (`unit_artifact.py:204` pins `superblock = 256`
   unconditionally), and `encode_linear(randn(64, 640), grid=E2M1x2, q256=896)`
   succeeds with a BODY descriptor of **two** granules for **three**
   superblocks, counts `(76800, 76800)` where the true per-superblock split is
   `(61440, 61440, 30720)`. Whole-unit decode is unaffected (the reader does not
   seek), which is why every test passes -- but the restart table is the
   segment-local seek contract, and TP sharding is a shipped design promise.
   Byte-affecting for any unit whose column count is not a multiple of 256.
   **Weaker than the audit for BODY on conforming shapes**: q256 *is* the
   per-superblock bit total, so a real `R1142` artifact's four superblocks each
   sum to exactly 1142 and the divmod spread is exact by construction.
8. **§2 P1-6 `Manifest.schedule` raises on a real E4M3 checkpoint.** The
   verifier called this latent; it is not. `manifest.py:714` builds a
   `RateSchedule` with the default cap=3, and a real on-disk
   `TESSERA_E4M3_K1_R1142` (rates 4 and 5) raises
   `GrammarError: rate 4 outside the shaped domain (1,2,3)`. Zero callers today,
   so nothing breaks -- but a property on a valid artifact is broken. The fix
   needs `grid_codes` threaded too (at cap 7 the grid is 256, not 16).

## Tier B — real, latent (no live path), cheap
- §4 P0 completion `argmin` ignores `sub_w` (`encode.py:1407` computes it,
  `:1414` never uses it). Live only for arity>1 TCQ with completion>0; nothing
  in `src/` passes completion != 0 and the shipping recipe for sub-cap E2M1x2 is
  the window body. Arity 1 is provably harmless (a positive scalar cancels).
- §2 P0-1 `check_padding_zero` reads past the declared extent AND has zero
  callers -- so the padding-canonicality rule the wire declares is enforced
  nowhere.
- §2 P0-2 `positions // group_weights` truncates to 0 scale entries for
  non-divisible positions; `Geometry` accepts it. (The `:925` half of the audit
  does not reproduce: the `:920` guard makes that division exact.)
- §2 P0-3 partial-superblock terminal counts are not intersected with the
  descriptor's prefix sums; a boundary-violating terminal validates.
- §2 P1-7 `Reader.blob` has no `_MAX_BLOB` bound (the writer does).
- §2 P2-8 `write(1, 0)`-style zero-width writes silently drop the value.
- §2 P2-9 `quantizable_params` is only checked `> 0`; a wire value of 1e12
  validates and understates bpp by orders of magnitude.
- §2 P2-11 `q256_from_root(0) == 0` while `root_from_q256(0)` raises.
- §3 C1 `side_bytes` means two different things in `serialize` vs `parse`, so
  the reported `wire_bpp` differs (23/4 vs 543/64 on a toy). Both reports are
  discarded, so no bytes -- but the number is quotable and wrong on one side.
- §3 C3 the pad-bit check ignores `bit_order` (LSB_FIRST is never constructed).
- §3 C4 REFERENCE planes charge 0 bytes and nothing constructs one.
- §3 C5 `plane_rate` ignores padding; dead helper, zero callers.
- §3 G2b `unpack_uniform` slices trailing slack without checking it is zero.
- §3 G2c `build_terminal` with `plane_region=None` emits `sha256(zeros)` -- a
  look-valid digest of data never hashed. `calculator.terminal_rate` hits this
  every call.
- §3 G2d `max_released` is ignored whenever `spec` is present.
- §1 the whole section: `compose_half` takes a float `clip_exponent` and
  silently degrades `Fraction` to float; `compose_half(0xFF)` returns 2^128;
  `e8m0_encode_exact(1.0)` raises bare `AttributeError` while `(1)` returns 127;
  `e4m3fn_is_subnormal(0x101)` is True; `bits_to_bytes(16.0)` returns a float;
  `as_ratio(0.5)` raises `AttributeError`. All type-discipline, no bytes.
- §5 P2-8 wrong-shaped refit metrics raise raw `RuntimeError`, not GrammarError.
- §5 P2-9 `land_at_least` overshoots the minimal fp16 word by up to 2 ulps
  (263/1500 cases).
- §5 P0-1/P0-2/P1-7 (S6b pack `floor` clips the group peak by up to 6.7%; the
  `d` delta bit is provably always 0, so half the refine code space is dead;
  the 3-candidate base search misses the optimum when a group spans >2 octaves).
  **Off the shipping wire**: every recipe uses CHANNEL or LUT, never S6B.

## Tier C — documentation only
§5 P1-6 (the "monotone in weight-space error" claim is unqualified while the
`encode_unit` signature default is `trellis_weighting="none"`; `export.py` ships
`"scale"`, so the shipping path is matched), §5 nits, §4 `sse` means three
different things and the returned one is pre-release/stale, §4 chunk-size
last-ulp `sse`, §4 `memory=0` (refuse it), §3 C2/C6/G1, §2 P2-10/P2-12,
§6 odd-P docstring and the S6b double-`scale_plane` argument.


## The proof harness

`experiments/audit_byte_baseline.py` is how each fix proves its byte claim
rather than asserting it. It hashes two things: the serialised unit for a fixed
matrix of (grid, rung, shape, weighting) -- including the 640- and 320-column
shapes that carry a partial trailing superblock -- and the tensor every
`.tessera` file on this box decodes to. Run it before the change and after, and
`--diff` the two. Deterministic across processes (it seeds from `zlib.crc32`,
not `hash()`, which `PYTHONHASHSEED` randomises). Baseline at 3ea7ec3: 14
encodes, 22 decodes, 0 changed across two runs.

A fix may legitimately change future bytes. It may never change what today's
bytes mean, and the decode half of the baseline is what says so.

## One thing the audit does not touch

§4 P1 says the returned `sse` is pre-release and its meaning flips with
`trellis_weighting`, and the verifier notes `tests/test_ldlq_*.py` read it. The
LDLQ receipt merged at 3ea7ec3 does **not**: its weight-space columns are `out`
(output-space error through 8192 held-out rows) and `plain` (unweighted
weight-space error), both computed by the sweep itself. The `0.7805x` screen and
the `0.9313x` GLM cross-check are unaffected.
