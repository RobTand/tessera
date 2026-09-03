# ts-80: the per-unit `channel_sigma` rule — measured, and recommended against

Worktree `/home/rob/tmp/musefix/ts-80-perunit`, branch `muse/ts-80-perunit`,
commit `44163bc` on top of `c71f37b`. `/home/rob/tessera` untouched.

## Outcome

**Do not ship a per-unit reach rule.** The lever is real and it is small at the
rate the wire ships at; the first-order rule that could have been *derived*
rather than fitted is net negative, and a predictor that beat it would have to
be path-aware -- i.e. run the trellis, which is the per-unit search the brief
ruled out. Issue #80 should close on this measurement.

I made **no encoder change**. `diff -r --brief` against a pristine `c71f37b`
extraction reports exactly three deltas, and `src/` and `tests/` are byte-identical:

- `docs/measurements/tessera-per-unit-reach-2026-09-03.md` (new)
- `experiments/reach_predictor_check.py` (new)
- `docs/issues-snapshot.json` (refreshed -- see below; this one is *required*, not incidental)

Neither new file is importable by the suite, but both are inside its **collected
surface**, which is why the baseline below is a real second tree and not an
argument. `tests/test_experiment_import_roots.py:37` rglobs `experiments/*.py`
and `ast.parse`s each one (my script passes: its `sys.path` insert is derived
from `__file__`, `experiments/reach_predictor_check.py:46`), and
`tests/test_issue_refs.py:74` rglobs `docs/**/*.md` and asserts every `#N` in
prose resolves against `docs/issues-snapshot.json`. That snapshot was stale
(tessera max `#77`), so my doc's references to `#80` and `#84` were the **only 5
dangling refs in the tree** and would have failed that test. `tools/refresh_issues.py`
is the fix the test's own message names ("run it after filing or closing
anything") and I ran it; it added #78-#92 including the three I filed today.
Refreshing can only *shrink* the dangling set, so it cannot mask an unrelated
failure.

## The numbers, with scope

Scope for every number: **weight space**, h-weighted
(`sqrt(sum h_j e_ij^2 / sum h_j w_ij^2)`), diagonal Hessian
`/mnt/shared/tessera-runs/bf16/refs/h_diag.pt`; **8 dense Qwen3-0.6B Linears**,
no GLM experts; E4M3 and BF16, window body, CHANNEL plane, L=14, seed 0; rungs
R1024 (4 b/wt, the E4M3 wire's own rate) and R2048 (8 b/wt, research). **No
served KL, no PPL.** Arms are the 15 already on disk from
`/mnt/shared/tessera-runs/reach/reach_{e4m3,bf16}_{spread,ratio,wide}.json`
(harness `experiments/bf16_l_sigma_sweep.py --stage reach` at `430ca5e`); I ran
no new encodes, so no GPU sweep was spent.

### 1. What a per-unit rule is worth (deliverable 1)

| grid | rung | best single arm (geomean) | **per-unit oracle** |
|---|---|---|---|
| E4M3 | R1024 (ships) | 0.9965 (`m=1.25`) | **0.9723** |
| E4M3 | R2048 | 1.0027 (`m=1.5`) | **0.9640** |
| BF16 | R1024 | 1.0000 (`channel_sigma` axis is a near-gauge) | **0.9999** |
| BF16 | R2048 | 0.9951 (`m=0.75`) | **0.9935** |

The per-unit tables are in the measurements doc, one row per unit, not a
geomean. On BF16 the `channel_sigma` axis is an **exact gauge at dyadic
multipliers** — `m=0.5` and `m=2` are 1.0000 on all eight units at both rungs,
because BF16 is closed under scaling by two and snapping commutes with it — and a
near-gauge otherwise, since a non-dyadic multiplier only changes which grid
points the quantiles snap to: at most 0.05% per unit at R1024 and 1.75% at
R2048. **So the per-unit lever is worth 0.01% on BF16 at R1024 and 0.65% at
R2048, oracle.** BF16's large per-unit effect (oracle 0.7602 at R2048) is on the
`window_sigma/channel_sigma` **ratio** axis, which is #48's already-recorded
finding read per unit, on a route with no serving lane and no served KL.

E4M3's 0.9723 is an **oracle**: picked with the answer in hand, across 15 arms
on 8 units. It is the ceiling, not a rule.

### 2. The derived rule (deliverable 2) — built, tested, fails

A per-unit value found by sweeping is a lookup table, so I built the model the
window table is *built on*: at each position the trellis reaches the `2^R`
entries its `R` new bits select, and those are a permutation of the table's
quantiles, so one position is the nearest of `K = 2^R` draws from the table's own
empirical distribution (Tseng et al.'s random Gaussian codebook):

```
D(t) = E[min_K (t-c)^2] = int_0^inf 2x (1 - F(t+x) + F(t-x))^K dx
pred(arm) = sqrt( sum_rj h_j s_r^2 D(w_rj/s_r) / sum_rj h_j w_rj^2 )
```

with `s_r` the row scale `initial_channel_scale` **actually assigns** under that
arm — the production function called, not modelled, so the reach-aware per-row
start is inside the prediction. `(w, h, codebook, R)` in, an arm out. No fitted
constant, no per-unit search.

| rung | argmin agreement | geomean if the rule is followed | oracle |
|---|---|---|---|
| R1024 | **0/8** | **1.0805** | 0.9723 |
| R2048 | 3/8 | 1.0221 | 0.9640 |

It over-credits a narrow reach on every unit — it prices one position and the
window body's advantage is the path — and on `layers.2.mlp.down_proj` at R1024 it
predicts 0.710 for `m=2` where the encoder measures 1.371. That is the one
misclassification the asymmetry cannot afford: three units gain 5–10%, that one
loses 15%.

The issue's own first candidate, the `over` fraction, separates nothing:
`k_proj` (over 0.202) is helped 10% by `m=1.5`, `o_proj` (over 0.208) is hurt 2%,
and the two lowest-`over` units are both hurt. The only separating statistic is
h-weighted kurtosis, which isolates `layers.2.mlp.down_proj` (1140 vs ≤8.1); as
the sole gate it gives 0.9661 at R2048 and **0.9983 at R1024**.

### 3. Two corrections to the issue as filed

- **The direction language inverts.** `m=1.5` ("a spread 1.5x wider") is **0.78x
  the reach**: the table's outermost entry snaps onto the grid, the grid stops at
  448, so a larger `channel_sigma` *shrinks* `reach_rms` (4.0773 → 3.1712).
- **The two knobs are not one axis.** `channel_sigma` is a pure reach clamp at
  fixed modelled sigma; the `window_sigma/channel_sigma` ratio re-models the
  table and widens the reach until the same 448 stops it at 1.167x (#84). Both
  land on one physical quantity — the body's reach in row-RMS — from opposite
  sides.
- **Two R2048 cells are not measuring reach.** `layers.2.mlp.down_proj` costs
  1.36x under every non-dyadic table sigma and 1.00x under every dyadic one, at
  fixed reach, reproduced across the two arm families that share only the table
  sigma. Filed as #89.

## Acceptance criteria

| criterion | status |
|---|---|
| `encoder_profile_id` byte-identical for every existing recipe | **Yes, and no digest can move.** `diff -r --brief src /pristine/src` and the same for `tests/` report no differing file (only `__pycache__` dirs are unique). No function that computes a digest, a row scale, a table or a manifest was edited; the three changed files are a doc, an experiment script and the offline issue snapshot, none of which any encoder path reads. |
| Default-off: unset flag byte-identical to today | **N/A — no flag was added.** The recommendation is not to add one. |
| `PYTHONPATH=src python -m pytest tests -q`, same line as the unmodified tree | See below. **Two separate runs on two trees** — I first argued one run could serve as both and that argument was wrong: two tests do reach outside `tests/` (`test_experiment_import_roots.py` rglobs `experiments/*.py`, `test_issue_refs.py` rglobs `docs/**/*.md`), so my new files are in the collected surface. The baseline is a pristine `git archive c71f37b` extraction at `/home/rob/tmp/musefix/ts-80-base` (a plain directory, not a `git worktree add`, so no shared repo metadata was mutated). |
| Numbers reported with scope | Yes — grid, unit count, rung, axis, weight space, per-unit tables in the doc and the #80 comment. |
| "Capped and not worth shipping" is a complete answer | That is the answer. |

**pytest final line, this tree:** `1550 passed, 9 skipped, 14 warnings in 2202.93s (0:36:42)`
**pytest final line, pristine `c71f37b`:** `1550 passed, 9 skipped, 14 warnings in 1430.80s (0:23:50)`

Identical counts. Only wall-clock differs, and that is box contention, not the
change: four other agents' suites were running concurrently on sparky while my
run held the lock. Raw logs: `.claude/pytest_run.txt` and
`.claude/pytest_base_c71f37b.txt`.

## Defects filed

- **#87** — `initial_channel_scale` lands its reach *lower bound* with
  round-to-nearest, so 166/1024 rows of `layers.2.mlp.down_proj` land ~3e-4 below
  it and clip the loud weight the bound was raised for. `land_at_least` exists
  for exactly this and is used only by the refit's floor, which is default-off on every shipping path (`export.py:274`, and `export_tessera_serving.py`'s bare `store_true`; no wrapper sets it), so nothing repairs the ulp-low rows today. Lost energy is small
  (1.9e-8 of the h budget); the contract violation is the point. Fixing it moves
  bytes, so it needs a decision.
- **#89** — at R=8 a non-dyadic window-table sigma costs `layers.2.mlp.down_proj`
  1.36x h error at identical reach; not the codebook's own quality (1% apart),
  not clipping (clipped energy *falls*), not general to the unit set. Also says
  `default_channel_sigma`'s dyadic ladder optimises a scalar-RTN objective the
  window table does not use.
- **#90** — #81's sibling: under CHANNEL, `window_sigma` equal to the resolved
  `channel_sigma` is the same encoder (decoded tensor identical) and gets a
  different `encoder_profile_id`, 20 extra bytes and schema minor 5.

## Interaction with #84 and #81

- **#84.** My design does not depend on the fix; everything is measured against
  current behaviour, and the saturation is *reported* as the cap it is (E4M3's
  accessible `reach_rms` is (0, 4.7568], the default at 4.0773). I edited nothing
  in `src/tessera/encode.py`.
- **#81.** I read `_normalize_reach` and `tests/test_profile_reach.py` before
  touching anything that feeds the digest — and then touched nothing that feeds
  it. The reading produced #90.

## Consultations

Advisor (stronger reviewer, full transcript) **twice**.

1. Before any building: it confirmed the quantification was already on disk, set
   the stop rule for the predictor (≤14/16 argmin agreement, or any wrong pick on
   the catastrophic unit → stop and report), flagged the `window_sigma`
   normalisation gap that became #90, and flagged the non-dyadic anomaly that
   became #89.
2. Before returning: it challenged the premise that one pytest run could serve as
   its own baseline. It was right and I was wrong — `grep` found two tests that
   walk `experiments/` and `docs/`, and one of them would have failed on my doc.
   It also caught the "only derived rule" overclaim (now "first-order") and asked
   whether `refit_reach_floor` shrinks #87's scope; it does not on any shipping
   path (`export.py:274` and the exporter's bare `store_true` both default False,
   no wrapper sets it), and I recorded that check on the issue.

I verified each point against the code and the data rather than accepting it —
the #87 scope check reversed nothing, the collected-surface one reversed my own
argument. No `fable-*` consultation was needed.

## What I did not do

- No GLM experts, no E2M1/E2M1x2, no served KL. The verdict is negative, so
  broadening the unit set would only be needed in order to ship something.
- I did not fix #87, #89 or #90: all three move bytes or need a decision, and the
  brief's first acceptance criterion is that no digest moves.
