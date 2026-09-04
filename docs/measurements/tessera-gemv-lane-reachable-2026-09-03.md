# The window-GEMV lane is reachable, and an artifact that cannot reach it says so

**Issue #104.** 2026-09-03. Branch `claude/ts-104-gemv-rates`, base master `42615e4`.

## What was wrong

`prepare_fp8_gemv` requires every column rate of a unit to be in
`kernel_window_gemv.SUPPORTED_RATES = (1, 2, 4)`: the lane repacks each column's
code stream at that column's own rate, and a 16-row lane exists only where R
bits per code is a whole number of bytes (R=3 would need 6-byte lanes).

A rung is a **root** rate, and `grammar.bresenham_rate_schedule` realises a root
by mixing only the two rates bracketing it. So q256 1006 (root 3.930) is columns
at rate 3 *and* columns at rate 4, and **every** unit of that checkpoint refuses
the lane at load. The streamed FP8 route catches the refusal and serves the same
bytes through the torch window decode -- a substitution the module reports as
`state: served`.

Every allocated checkpoint we hold carried a rate outside the set. All four of
#91's serve censuses therefore logged 112 of 112 modules refusing the lane while
every receipt recorded one route, `symbol torch._scaled_mm`, `decoder
torch_window`, `problems: []`. The two arms were one lane state wearing two
names.

## The decision, and what it cost

The decision (not relitigated here) was to **build a rate-constrained checkpoint,
not to widen `SUPPORTED_RATES`**. The rate axis is continuous by design -- 2-D
(rung x completion depth), q256 the realisable set -- and pinning the format to
three values for one kernel lane pays a permanent quality cost for a measurement
convenience. The kernel's constraint is the kernel's; the checkpoint comes to it.

### What that constraint admits (exact, code-derived)

`experiments/ts104_rate_constraint_cost.py`, leg 1 -- every integer rung in the
E4M3 family's own published reader range, filtered by `grammar.rate_set`:

```
=== leg 1: which rungs the tessera_window_gemv lane can read
    lane column_rates [1, 2, 4]; TESSERA_E4M3_K1 reader range [256, 2048] (cap 7)
    readable q256: 258 of 1793 integers -> ['256..512', '1024']
    i.e. root rate in [(1.0, 2.0), (4.0, 4.0)]
    the whole open interval (2, 4) and everything above 4 is UNREACHABLE
```

**258 of 1793 rungs.** Root in [1, 2] and the single point root = 4. The whole
open interval (2, 4) is gone, and so is everything above 4: an integral root 3 is
uniform rate 3, and any fractional root in between mixes a 3 or a 5.

q256 = **1024** is the one readable rung at the 4-bpp knee, and it is already the
E4M3 family's `attested_wire` rung, so `check_recipe` passes it unchanged.

### What it costs in quality (a SCREEN, then a served number)

Leg 2 is a multi-choice knapsack over the fused serving groups on the published
per-unit RD table (`rung_rd_curves_2026-09-03`), once over every measured rung and
once over the readable subset. It is a **screen** and says so in three places:
another campaign's table, seven layer-0 units with the rest BF16; its own receipt
records per-unit costs summing to 84.8% of the jointly measured damage; and R1024
is **interpolated** between the measured 1006 and 1044.

| byte budget | constrained / unconstrained oracle | constrained / uniform arm |
|---|---|---|
| uniform R512 (4.07 MB) | 1.074x | 1.000x |
| uniform R749 (5.89 MB, ~3.0 bpp) | 1.320x | 1.148x |
| uniform R1006 (7.87 MB, ~4.0 bpp) | **2.782x** | **2.594x** |
| uniform R1024* (8.00 MB, 4.07 bpp) | 1.138x | 1.000x |
| uniform R1262 (9.83 MB) | 2.451x | 2.075x |

The R1006 row is a **byte cliff**, not a slope: at a strict 7.87 MB budget the
constrained arm cannot afford uniform R1024 (8.00 MB) and has to drop a group to
R512, leaving 18% of the budget unspendable. The honest matched-bytes statement at
this knee is the R1024 row: **1.138x the unconstrained oracle, 1.000x the uniform
arm** -- i.e. the constraint costs nothing against a uniform artifact at the same
rung, and it costs the freedom to allocate.

## What changed (the defect half)

An artifact that cannot take the lane it was built to exercise now says so at
plan time, and a census in which the engaged arm engaged nothing is now a
refusal. Three read points, one predicate:

**1. Plan time -- `experiments/export_tessera_serving.py --require-lane LANE`.**
Reachability is a function of the RUNG alone (`grammar.rate_set(root, cap)`
returns `{floor(root)}` for an integral root and `{floor, floor+1}` otherwise),
so it is decidable before a shape is read -- and it is refused there, at argument
time, beside `check_recipe`, at the default rung and at every `--plan-json`
override. The refusal names the offending rates. The requirement is stamped into
the manifest as `requires_lanes`, so it travels with the bytes.

**2. Serve time -- `serving/telemetry.note_lane_refusal`.** Both streamed routes
clear a note before the lane attempt and set it on the fallback branch, so a
refusal is a value on the layer rather than a stderr line nobody aggregates. It is
deliberately **not** a 13th `ROUTE_FIELDS` entry and is never touched from
`apply()` (issue #52: the compiled forward breaks lane hot paths).
`bf16_route.gemv_refusal_for_unit` returns the reason where `gemv_eligible_for_unit`
returned only a bool -- the verdict is unchanged, what is new is that it comes with
the reason, on the non-exception path that previously left no trace at all.

**3. After the fact -- `serving/census.py::lane_engagement`, wired into
`tools/tessera_route_census.py --require-lane`.** The per-module check is a check
on AGREEMENT, and the streamed FP8 route's decode regime legitimately admits both
the GEMV pair and the materialised one -- so a serve in which the lane prepared
for nothing passes module by module. `lane_engagement` asks the question that
cannot: did the lane this arm requested take any units at all? The verdict is over
the census, not over each phase, because the lane owns a REGIME (`GEMV_MAX_M = 8`)
and the prefill forward takes the torch decode by design. `all_required_engaged` is
three-valued: `true`, `false`, or `null` when nobody said what to require -- and
`null` is the state every receipt in the #104 report was in.

**The predicate is published, not hardcoded.** `runtime_contract.json`
(contract_version 11) gains `native_extensions[].lane = {decoder, requires}`, and
`requires` carries `column_rates`, `window_bits`, `body`, `plane`. A test ties the
contract's copy to `kernel_window_gemv`'s own constants -- the same tie
`loader_axes` vs `ROUTE_TP_AXES` has -- because `ext` is read by a producer with no
torch and cannot import the kernel. A lane with no predicate **omits** the block:
an empty one would read as a claim of no constraint.

**The rate axis was not narrowed.** Nothing here changes what may be encoded or
served. What is refused is only the CLAIM that a given artifact exercises a given
lane.

## Scope: two things recorded, not fixed

1. **The BF16 family's own attested rung cannot reach its lane either.**
   `TESSERA_BF16_K1`'s attested wire rung is q256 1792 -- root 7 exactly, cap 15 --
   so `rate_set` is `(7,)` and the streamed BF16 GEMV lane is as unreachable as the
   FP8 one was. Pinned by `test_the_bf16_familys_own_attested_rung_cannot_reach_the_lane`
   so the day someone attests a reachable BF16 rung, the test says so. Not fixed
   here: choosing that family's rung is a wire decision, not this issue's.
2. **The E4M3 `lane_eligibility` cells declare `scaled_mm_w8a8` for both regimes
   and there is no cell for the GEMV lane.** The census below is the observation
   someone could attest one from; writing the cell is a contract change with its
   own gate, out of scope here.

### What the served pair is, and what it is not

The served KL below prices the **wire** at q256 1024 against the untouched
q256 1006 baseline. It is **not** a lane measurement: `kl_tool dump` echoes
512-token prompts, so every forward is M = 512, far past the GEMV's
`GEMV_MAX_M = 8`, and both arms decode through `torch_window`. The lane's own
numerics are a separate, already-measured fact (bit-exactness and throughput:
`docs/measurements/tessera-window-kernel-2026-09-02.md`); this pair answers
(g) -- what the rate constraint costs at matched-ish bytes on a served metric --
and (b) is answered by the census, which drives a one-row decode.

The two arms are **not byte-matched**: R1024 is 8.00 MB of body wire against
R1006's 7.87 MB, +1.8%. That difference is the constraint at this knee (there is
no readable rung between root 2 and root 4), and it is stated wherever the pair
is quoted rather than absorbed into a ratio.

## The fail-before (AGENTS.md rule 8)

Both new files against a pristine master checkout of `42615e4`
(`/home/rob/tmp/ts104-master`, the same two files copied in):

```
_______________ ERROR collecting tests/test_lane_reachability.py _______________
E   ImportError: cannot import name 'rate_set' from 'tessera.grammar'
_______________ ERROR collecting tests/test_census_engagement.py _______________
E   ModuleNotFoundError: No module named 'tessera.serving.census'
2 errors in 0.64s
```

On this branch: `50 passed`.

Two behaviours inside those files carry their own fail-before, recorded in the
commits that changed them rather than in a checkout that predates them:

* `test_a_decode_only_lane_is_engaged_and_not_a_failed_census` fails on the
  per-phase verdict `b412930` shipped (the prefill forward legitimately takes the
  torch decode, and the arm that works would have been refused);
* `test_a_load_time_refusal_is_not_multiplied_by_the_phase_count` fails on the
  summed refusal counter, reporting `224 of 112 module(s)`.

## (c) The plan-time refusal, and the plan-time acceptance

The accepting side, from the chain's own re-export of the shipped artifact:

```
  --require-lane tessera_window_gemv: --grid E4M3 --q256 1024 -> column rates [4], readable
```

The refusing side, live, on the rung `uniform-R1006` was actually built at
(`experiments/export_tessera_serving.py ... --q256 1006 --require-lane tessera_window_gemv`):

```
tessera lane 'tessera_window_gemv' for --grid E4M3 --q256 1006: q256=1006 is root rate
3.9297, which bresenham_rate_schedule realises as column rates [3, 4] -- and [3] is
outside the rates this lane reads ([1, 2, 4], runtime_contract.json
native_extensions[tessera_window_gemv].lane.requires.column_rates). EVERY unit of a
checkpoint at this rung would refuse the lane at load and be served by the fallback, so
an artifact built to measure the lane would measure the fallback. Re-plan on a rung whose
rate set is inside [1, 2, 4]: an integral root lands every column on one rate
(q256 = 256*R), and a fractional root mixes only the two rates bracketing it. The rung is
still legal to ENCODE and to SERVE -- the tessera_window_gemv lane is one launch inside a
route, and a unit it cannot read is served by that route's other path (for the window
GEMV: the torch window decode plus _scaled_mm, same bytes, slower). What is refused here
is only the CLAIM that this artifact exercises the lane.

--require-lane tessera_window_gemv was passed, so this plan is refused HERE -- before a
single unit is encoded. Drop the flag to write the rung anyway (it serves, through the
route's other path), or re-plan onto a rung the lane reads.
```

**It refused before any encode work**, which is the whole point of putting it at
argument time:

```
plan refusal exit: 1
encode progress lines in that log: 0
output dir after the refusal: ls: cannot access '/home/rob/tmp/ts104-refused-must-not-exist': No such file or directory
```

## (a) The checkpoint, and its full rate histogram

`/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024` -- Qwen3-0.6B,
`--grid E4M3 --q256 1024`, 846,726,118 bytes, 357 s.

`tools/tessera_lane_preflight.py` over **every** unit (`parse_fused` then
`parse_unit_artifact` on every `<module>.wire_bytes` member, not a sample of eight):

```
=== /mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024
    196 units in 6.4s; rate sets {'4': 196}; columns by rate {'4': 286720}
    lane tessera_window_gemv (rates [1, 2, 4]): READABLE -- 196/196 units readable
```

One rate set across the artifact, `{4}`; 286,720 columns, all at rate 4; 196 of 196
units readable.

The re-export **through** `--require-lane` produced byte-identical wire, so the flag
costs nothing but a stamp:

```
wire sha before=7f1193b1f014d957b7e7b78af897302223bd86a49b867a1b1b847ea9a18c68ec
           after=7f1193b1f014d957b7e7b78af897302223bd86a49b867a1b1b847ea9a18c68ec
BYTE-IDENTICAL re-export
```

and the requirement now travels with the bytes:
`tessera_serving_manifest.json` -> `"requires_lanes": ["tessera_window_gemv"]`.

### The six allocated checkpoints, read from their bytes

Read-only, same tool, every unit:

```
=== /mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R750
    196 units; rate sets {'2,3': 196}; columns by rate {'2': 20160, '3': 266560}
    lane tessera_window_gemv (rates [1, 2, 4]): UNREACHABLE -- 0/196 units readable; offending rates {'3': 196}
=== /mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1006
    196 units; rate sets {'3,4': 196}; columns by rate {'3': 20160, '4': 266560}
    lane tessera_window_gemv (rates [1, 2, 4]): UNREACHABLE -- 0/196 units readable; offending rates {'3': 196}
=== /mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1262
    196 units; rate sets {'4,5': 196}; columns by rate {'4': 20160, '5': 266560}
    lane tessera_window_gemv (rates [1, 2, 4]): UNREACHABLE -- 0/196 units readable; offending rates {'5': 196}
=== /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-3.0
    196 units; rate sets {'1,2': 28, '3,4': 168}; columns by rate {'1': 6384, '2': 79632, '3': 168896, '4': 31808}
    lane tessera_window_gemv (rates [1, 2, 4]): UNREACHABLE -- 28/196 units readable; offending rates {'3': 168}
=== /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0
    196 units; rate sets {'2,3': 28, '3,4': 28, '4,5': 140}; columns by rate {'2': 6384, '3': 99792, '4': 142128, '5': 38416}
    lane tessera_window_gemv (rates [1, 2, 4]): UNREACHABLE -- 0/196 units readable; offending rates {'3': 56, '5': 140}
=== /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-5.0
    196 units; rate sets {'3,4': 28, '4,5': 28, '5,6': 140}; columns by rate {'3': 38640, '4': 61488, '5': 134400, '6': 52192}
    lane tessera_window_gemv (rates [1, 2, 4]): UNREACHABLE -- 0/196 units readable; offending rates {'3': 28, '5': 168, '6': 140}

REFUSED: 6 (checkpoint, lane) pair(s) cannot take the lane. Any claim about that lane
measured on these bytes is a claim about the fallback.
```

The one partial is worth naming: `alloc-3.0` has **28 of 196** units readable (its
`{1,2}` group), so a lane count on that artifact would have been non-zero and still
mostly fallback. The rest are total.

## (b) The served census: the lane took 112 of 112 modules

`tools/tessera_route_census.py` on the new checkpoint, `TESSERA_SERVE_MODE=streamed`,
eager, Tessera's own vLLM plugin, `--require-lane` resolved off the artifact's own
`requires_lanes`:

```
verdict: served | elapsed 93.7 s | prompt_tokens 64
decode  : {('tessera_window_gemv::gemv', 'window_gemv', 'M1',  tile_m 1): 112}
prefill : {('torch._scaled_mm',          'window_gemv', 'M64', tile_m 0): 112}
lane_engagement.all_required_engaged: true
lane_engagement.engaged_modules_max:  {"tessera_window_gemv": 112}
lane_engagement.declared_by_artifact: ["tessera_window_gemv"]
lane_refusals: {}
problems: []
```

**Read the two fields separately, because they answer different questions.**
`decoder` is `window_gemv` exactly when the lane PREPARED for that module (the
`tessera_gemv` holder exists) and `torch_window` when it did not, so the engagement
count is the #104 question -- did the lane prepare for anything -- answered directly.
`symbol` is which kernel actually launched: the decode phase ran
**`tessera_window_gemv::gemv` at `tile_m 1` on all 112 modules**, and the prefill
phase (M = 64, past `GEMV_MAX_M = 8`) ran `torch._scaled_mm` off the same repack,
which is the lane working as specified.

This is the first non-zero `window_gemv` count we have. Every prior census of this
lane read `torch_window` on 112 of 112.

## (d) The same command on the untouched baseline is now a REFUSAL

`/mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1006`, unmodified bytes,
identical census command. This is the #91 receipt reproduced -- and it no longer reads
as agreement:

```
verdict: REFUSED
decode  : {('torch._scaled_mm', 'torch_window', 'M1'):  112}
prefill : {('torch._scaled_mm', 'torch_window', 'M64'): 112}
all_required_engaged: False    engaged_modules_max: {'tessera_window_gemv': 0}

PROBLEM: lane 'tessera_window_gemv' was REQUIRED and took 0 of 112 Tessera modules in
any phase (decoder 'window_gemv'; observed {'decode': {'torch_window': 112}, 'prefill':
{'torch_window': 112}}). An arm that requested a route and got zero units on it measured
the fallback, not the lane. 112 of 112 module(s) recorded a load-time refusal, most
commonly: tessera_window_gemv: GrammarError: rates [3] have no lane here (supported
(1, 2, 4)); the materialised FP8 path serves this unit
```

Per-module state on that arm is `served` for all 112, exactly as before -- which is why
the per-module check passed it four times. The engagement field is what refuses it, and
the load-time reason names the rate.

The two arms side by side are the matched pair the original experiment lacked:

| arm | bytes | decode symbol | decoder | engaged | verdict |
|---|---|---|---|---|---|
| R1024 (readable) | 846,726,118 | `tessera_window_gemv::gemv` | `window_gemv` | 112/112 | served |
| R1006 (untouched) | -- | `torch._scaled_mm` | `torch_window` | 0/112 | REFUSED |

## (g) The served pair

Both arms served in the same session on the same box, Tessera's own plugin,
`TESSERA_SERVE_MODE=streamed`, eager, vanilla vLLM 0.28 image, Qwen corpus contract
(`corpus_qwen_n8_s512.json`), against the same `qwen_teacher_bf16_v028` teacher.

**R1024 (the rate-constrained artifact, 4.07 bpp body):**

```
metric=KL-vs-BF16  support=top-1024  partition=teacher-student-intersection
bound=lower bound (data-processing inequality)  regime=prefill  positions=4088
positions=4088  top1_agree=78.57%
  ALL            KL >= 0.150204   (<= 2.893521 at the declared floor 3.72e-44)
                 teacher tail mass outside the compared support: mean 0.028160 max 0.866124
  CONFIDENT      n=1709 (42%)  KL >= 0.099794   (<= 0.644301)
```

**R1006 (the untouched allocated baseline, 4.00 bpp body):**

```
positions=4088  top1_agree=77.47%
  ALL            KL >= 0.174557   (<= 2.954192 at the declared floor 3.72e-44)
                 teacher tail mass outside the compared support: mean 0.028558 max 0.866260
  CONFIDENT      n=1709 (42%)  KL >= 0.114561   (<= 0.664619)
```

| arm | body bpp | served KL (ALL) | confident KL | top-1 agree |
|---|---|---|---|---|
| R1024 -- rate-constrained, lane-readable | 4.07 | **0.150204** | 0.099794 | 78.57% |
| R1006 -- untouched baseline | 4.00 | 0.174557 | 0.114561 | 77.47% |

**Read this carefully, because the sign is easy to misread.** The lane-readable
artifact is 14% *better* on served KL -- but it is also 1.8% larger, and it is a
*higher rate* (root 4.000 against 3.930). The pair is not a demonstration that the
constraint is free; it is the price paid in the only currency the constraint takes at
this knee. **There is no readable rung between root 2 and root 4**, so a producer that
wants this lane at ~4 bpp cannot land on 3.93 -- it must go up to 4.00 and pay 1.8% more
bytes. What it gets for those bytes is a genuinely better artifact.

The cost that does not show up in this pair is the one the screen prices: the loss of
the *continuous* axis for allocation. Uniform-to-uniform the constraint costs nothing
(1.000x at both R512 and R1024); against an allocator free to spend the whole axis it
costs 1.074x-1.138x at the budgets where a readable rung exists, and it is simply
infeasible in the band root in (2, 4).

## What #83 and #102 can now measure that they could not

Both issues were trying to measure the window-GEMV lane, and neither could have:
every checkpoint either held was refused by the lane at load, so the number each
would have recorded was the torch fallback's, reported as a served route.

* **#83** (lane performance) now has an artifact on which the lane actually runs:
  `qwen3-0.6b-uniform-R1024`, 112 of 112 modules on `tessera_window_gemv::gemv` at
  `tile_m 1` in the decode regime. A before/after profile of the lane has a real
  A-side for the first time.
* **#102** (decode-regime KL harness) can now separate "the decode regime" from "the
  torch decode": the same harness on R1024 and on any allocated checkpoint is two
  lane states rather than one wearing two names, and the census will say so in a
  field rather than leaving it to the reader.
* Anything else that wants the lane can ask **before** building: the preflight reads
  the question off bytes somebody else wrote, and `--require-lane` refuses a plan that
  could never answer it.

## Test evidence

```
tests/test_lane_reachability.py tests/test_census_engagement.py -> 50 passed
```

Fail-before on pristine master `42615e4`: 2 collection errors (`rate_set` and
`tessera.serving.census` do not exist).

Impacted set (`tools/impacted_tests.py --ref master...HEAD`, verdict `narrowed`,
87 files), run in this worktree on the pool venv:

```
1634 passed, 8 skipped, 14 warnings in 579.89s (0:09:39)
```

## Reproduce

```
experiments/export_tessera_serving.py /home/rob/models/Qwen3-0.6B <out> \
    --grid E4M3 --q256 1024 --require-lane tessera_window_gemv
tools/tessera_lane_preflight.py <out> --lane tessera_window_gemv
experiments/ts104_gemv_census.sh          # both arms, streamed + eager
experiments/ts104_rate_constraint_cost.py # the rung enumeration and the screen
```

Artifacts: `/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024`;
receipts and logs under `/home/rob/tessera-runs/ts104/`.
