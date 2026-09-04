# `_fit_lut`'s eps stop test moved no bytes, and could only ever have moved one ulp's worth (2026-09-04)

**Claim (measured).** `2f6a15a` replaced `_fit_lut`'s swap accept test
`cost < base * (1 - 1e-9)` with `cost < base * (1 - torch.finfo(cost.dtype).eps)`.
Issue #106 asked whether that moved the sixteen E4M3 entries the LUT scale
plane puts on the wire. It does not:

* **Byte identity, four arms.** `experiments/audit_byte_baseline.py
  --encode-only` gives **0 changed of 31 rows** for `2f6a15a^` against
  `2f6a15a` (the commit itself) and **0 changed of 31** for master against
  master with the literal restored (the same question asked of today's
  encoder). The same harness reports **5 changed of 31** between `2f6a15a^`
  and master, so it is a live instrument and not a null one. A whole real
  Qwen3-0.6B Linear at its shipped width, with its real Hessian, hashes
  identically across all four arms, LDLQ off and LDLQ at its default alike.
* **The decision never fires.** Mirroring every accept decision twice inside
  the real swap loop: **38,192 trials on x86 and 10,368 on GB10, 0 that the
  two tests decide differently.** The closest any trial came to the boundary
  was a relative improvement of **9.298e-06** -- about **78x** the float32
  epsilon the test compares against.

Zero differing decisions is a stronger statement than an equal hash. The
mirror follows the running code's own verdict, so it stays in lockstep with the
real encode; a trial where the two tests agree leaves both encoders in the
identical state. No disagreement anywhere in a corpus therefore means the two
take the **identical trajectory** on it, rather than happening to land on the
same bytes.

**So #106 carries no byte move.** Whether that lifts it out of the three-item
wire bundle -- this, `muse/ts-87-landfloor`'s default-path E4M3 byte move, and
#101's behaviour-derived identity -- is that decision's call; this is an input
to it. #87 and #101 stand where they were: this receipt says nothing about
either, and #101's hole is exactly why a receipt had to be taken by hand rather
than read off a field.

## What the change can decide, exactly

`base` is a float32 cost widened to a Python float, so `base * (1 - step)` is
exact in float64 and the test accepts a trial iff its improvement exceeds
`base * step`.

* `base * eps` is one ulp of `base`'s binade (`2^(e-23)` for `base` in
  `[2^e, 2^(e+1))`), so the eps test refuses the smallest representable
  improvement -- one float32 step inside a binade, and two at a binade's foot,
  where the steps below are half-width.
* `base * 1e-9` is nearly two orders below the *smallest relative step float32
  can express at all* (`eps / 2 = 5.96e-8`). On this dtype the literal was
  never a threshold: it accepted every improvement the arithmetic could
  represent.

So the whole behavioural difference is **trials that improve the running cost
by at most one ulp of its binade**. Everything larger is taken by both tests,
identically. `tests/test_lut_stop_ulp_band.py` walks five steps down from
fifteen bases -- a binade's foot, just above it, its middle and its top, at
three scales -- and pins the band from both sides rather than arguing it.

Two consequences the issue's framing implies neither of:

* On float32 the change can only ever **reject**. It cannot introduce a swap.
* On float64 costs (`eps = 2.2e-16`) it is a *tightening* and would accept
  swaps the literal refused. **No production path reaches that regime**:
  every one of the 38,296 `_lut_cost` calls in the x86 corpus accumulated in
  float32, which is the issue's premise turned from a reading of the code into
  a count.

## The E4M3 grid does not reach `_fit_lut` at all

The issue asks for a receipt "on at least one real E4M3 unit". The sixteen
entries are E4M3 *values*, but they are the **LUT plane's** table, and on the
current wire the E4M3 *grid* is not on the LUT plane: `export.wire_recipe`
gives `E4M3_RECIPE` the CHANNEL plane (`export.py:654`, and the same at
`2f6a15a^`), `BF16_RECIPE` likewise, and `encode.encode_unit` guards
`_pack_scales_lut` and `_refit_scales_lut` with
`elif scale_plane is ScalePlaneKind.LUT` (`encode.py:2347`, `:2780`). So
`_fit_lut` runs on the **E2M1 and E2M1x2** recipes -- `TCQ_RECIPE` at the cap
and `E2M1X2_SUBCAP_RECIPE` below it -- and nowhere else.

That is not an inference from reading: the trace counts swap trials per case,
and every `e4m3-*` and `bf16-*` row is **0**.

Worth writing down, because a receipt taken only on E4M3 rows would have
printed "0 changed" for a reason that has nothing to do with the threshold.
That is issue #39's lesson repeated -- a byte proof is only a proof of the
arithmetic its corpus reaches. Read literally, the issue's ask cannot be
satisfied: there is no E4M3 unit whose bytes this code path can touch. What is
answered instead is the question behind it -- whether the sixteen entries moved
on any wire that carries them.

## The measurement

`experiments/lut_stop_ulp_trace.py` wraps `encode._lut_cost` and mirrors the
swap loop's state: `_fit_lut` is the only caller, the pass-start call is the
one whose table equals the table the loop currently holds, and every other call
is a trial differing in exactly one entry. For each trial it records both
verdicts and follows the new one.

The mirror is checked against the loop's own shape rather than trusted. A pass
runs for every invocation, and a further pass only after a pass that accepted
something (`encode.py:1512-1535`: `if not improved: break`), so

    invocations  <=  pass_starts  <=  invocations + accepts

always. A trial misread as a pass start would push the count up; an invocation
that returned before the swap loop would push it down. Both effects are bounded
by this and neither run leaves the band, the two audit-matrix runs sitting on
its upper edge because every accepting pass there accepted exactly once.

| | x86 (dl380g10) | GB10 (sparky) | x86 + 2 real units |
|---|---|---|---|
| `_fit_lut` invocations | 72 | 60 | 82 |
| swap pass starts | 80 | 67 | 104 |
| accepted swaps | 8 | 7 | 79 |
| bound `[inv, inv+acc]` | [72, **80**] | [60, **67**] | [82, 161] |
| swap trials | 11,040 | 10,368 | 38,192 |
| `_lut_cost` calls in float32 | 11,120 (all) | 10,435 (all) | 38,296 (all) |
| **decisions the two tests differ on** | **0** | **0** | **0** |
| closest trial to the boundary | 8.288e-05 (695x eps) | 8.296e-05 (696x eps) | 9.298e-06 (**78x** eps) |

Two architectures matter here because `_lut_cost` is a float32 `sum()`: a
different reduction order is a different last bit, and therefore an independent
chance for a one-ulp event. Neither box produced one. (The GB10 column is the
encode matrix only; the x86 columns add the release rows, which is why their
counts are higher.)

The real units are what tighten the margin, from 695x to 78x, and they are the
reason the receipt is not taken on the audit slice alone: `_fit_lut`'s
near-ties are a property of how many halves a unit has, and the harness's value
cases cut the committed slice to 16x128.

Trials by case, x86 (the two real Linears from the `--full-unit` leg, the
rest from the harness's own cases) -- the LUT-plane rows are the whole
exposure:

| case | swap trials |
|---|---|
| `model.layers.2.mlp.down_proj` [1024, 3072], real H, E2M1x2 sub-cap | 20,640 |
| `model.layers.2.self_attn.k_proj` [1024, 1024], real H, E2M1x2 sub-cap | 6,512 |
| `e2m1x2-sub-512-128c/h1` (real weight + real H) | 5,792 |
| `e2m1-256-128c/completion` (real weight + real H) | 1,632 |
| `release` rows (E2M1 at cap, with refit) | 672 |
| `e2m1x2-sub-512c/none`, `/scale` | 432, 416 |
| `e2m1-256-512c/none`, `/scale` | 240, 432 |
| `e2m1x2-cap-512c/none`, `/scale` | 176, 400 |
| `e2m1x2-cap-640c/none`, `/scale` | 192, 320 |
| `e2m1x2-cap-384c/none`, `/scale` | 240, 96 |
| every `e4m3-*` row (6) | **0** |
| every `bf16-*` row (5) | **0** |

## The bytes

Each arm is a `git archive` of one commit, run with a **byte-identical**
`audit_byte_baseline.py`, value-slice fixture and real-unit fixture (`sha256`
checked across all four), same interpreter, same thread count, same box.

| comparison | what it isolates | changed |
|---|---|---|
| master vs master, re-run | the digests are reproducible at all | 0 of 31 |
| `2f6a15a^` vs `2f6a15a` | **the commit itself**, as the issue asks | 0 of 31 |
| master vs master + `step = 1e-9` | the same question of **today's** encoder | 0 of 31 |
| master vs `2f6a15a^` | positive control | 5 of 31 |

The positive control's five rows are all `bf16-*` -- the BF16 route landed in
the 35 `encode.py` commits between `2f6a15a` and master. Every LUT-plane row is
unchanged across all four arms, including across those 35 commits, and so is
every `e4m3-*` row -- the latter trivially, at 0 swap trials.

### One whole real Linear, at its shipped width

The audit harness's value cases cut the committed slice to 16x128 -- a few
dozen halves. `_fit_lut`'s near-ties are a property of how many halves it is
fitting sixteen entries to, so the receipt is also taken on a whole
`model.layers.2.self_attn.k_proj` [1024, 1024] of Qwen3-0.6B against that
capture's real H, encoded at the E2M1x2 sub-cap rung (`q256=512`), which
`wire_recipe` puts on the LUT plane. 299,654 bytes, hashed whole:

| arm | LDLQ off (`ldlq_sigma=None`) | LDLQ at its default (sigma 1, block 32) |
|---|---|---|
| master (`766033c`) | `56f5636a...b742641` | `8e8f8b97...517e69f` |
| master + `step = 1e-9` | `56f5636a...b742641` | `8e8f8b97...517e69f` |
| `2f6a15a^` | `56f5636a...b742641` | `8e8f8b97...517e69f` |
| `2f6a15a` | `56f5636a...b742641` | `8e8f8b97...517e69f` |

Both columns are one digest across all four arms. The two columns differ from
each other, which is the point of running both: `ActivationSource`'s default is
`DEFAULT_LDLQ_SIGMA = 1.0` (`export.py:174`), so an exporter handed an H runs
LDLQ, and LDLQ moves the residual the refit's targets are drawn from -- a
*different* set of `_fit_lut` trials, not the same one measured twice. The eps
test moves neither.

The tracer, run in the master arm on the same unit, reproduces master's
LDLQ-off digest exactly, so wrapping `_lut_cost` does not perturb the encode
and the trial counts above are counts of the encode that produced these bytes.


## Scope -- what this does not prove

* **The corpus is the claim.** These are the audit harness's shape, value and
  release rows plus two real Linears at their shipped width. A LUT unit whose
  targets put a swap within one ulp of the running cost would still change
  hands, and no corpus of this size rules that out. What it bounds is how close
  real encodes come: the nearest of 48,560 trials across two architectures was
  78x the threshold away.
* **The trial-level mirror is the LDLQ-off leg.** The 48,560 mirrored decisions
  are all `ldlq_sigma=None`. The LDLQ-default leg -- which is what an exporter
  handed an H actually runs -- is covered by byte identity across four arms, not
  by a decision-by-decision count. A one-ulp event there would be invisible to
  the trial table above and visible only as a changed digest, and no digest
  changed.
* **Nothing here is served.** This is a byte-identity receipt, not a quality
  measurement, and under principle 3 that is all it claims to be.
* **The reduction order is part of the receipt.** A last-bit claim only holds
  under the thread count and device it was taken on, which is why one arm was
  run twice on the same box before any arm was compared to another.
* **This is a receipt, not a fix.** `src/tessera/encode.py` is untouched, and
  so is `tests/test_lut_stop_dtype.py`.

## Environment

| | |
|---|---|
| x86 arm | dl380g10, `/home/rob/venvs/pb-cpu`, torch 2.11.0+cpu, 4 threads, `CUDA_VISIBLE_DEVICES=` |
| GB10 arm | sparky (aarch64, sm121), system python, torch 2.10.0+cpu, 1 thread, `CUDA_VISIBLE_DEVICES=` |
| arms | `git archive` of `2f6a15a^`, `2f6a15a`, `766033c` (master), and master with `step = 1e-9` |
| corpus | `experiments/audit_byte_baseline.py` encode + release rows; `tests/data/audit_value_slice.pt`; two whole Qwen3-0.6B Linears with the real `h_full_qwen06b.pt` H |

Code read, not changed: `src/tessera/encode.py` (`_fit_lut`, `_lut_cost`,
`_pack_scales_lut`, `_refit_scales_lut`, `_refit_scales_lut_metric`),
`src/tessera/export.py` (`wire_recipe`, `ActivationSource`). Harness:
`experiments/audit_byte_baseline.py` (unchanged; the arms run the
byte-identical file). Tracer: `experiments/lut_stop_ulp_trace.py`. Tests:
`tests/test_lut_stop_ulp_band.py` (31 pass), with
`tests/test_lut_stop_dtype.py`, `tests/test_encoder_fit_caps.py` and
`tests/test_lut_exact_fit.py` (53 pass together). Data:
`/mnt/shared/ts106-arms/` (`audit_*.json`, `realunit_*.json`, `trace_*.json`,
the four arm checkouts, and the real-unit fixture).
