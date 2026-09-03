# #93 and #96 — a drop and a missing control are now facts, not absences

Branch `muse/ts-93-griddrop` on master `82cdf51`. Two independent fixes, in
separate commits so they read separately.

## What the branch contains

| commit | what |
|---|---|
| `26baa1b` | `experiments/bf16_l_sigma_sweep.py` taken **verbatim** from `muse/ts-18-lsigma` at `77bc1be`. Not authored here. |
| `b76a836` | The fix: `experiments/pair_grid_audit.py` + the call sites in the stage. |
| `fdd3e33` | `tests/test_pair_grid_drop.py` and `tests/test_pair_grid_audit.py`. |
| `26fa1b1` | Comment placement nit. |
| `d770df1` | **#96**: a rung with no contamination check no longer reads like one that passed, plus its tests. |

**Why the transplant.** The pair stage does not exist on master — it is
`+365` lines that live only on `muse/ts-18-lsigma`, so a fix written against
master alone would have had no call site. Master's copy of
`bf16_l_sigma_sweep.py` was **byte-identical to the merge base**
(`c71f37b`), so the transplant is clean and `b76a836` is the only diff #93
owns. Whichever branch lands first, the three-way merge sees the same
content on both sides of `26baa1b` and only `b76a836` needs resolving.
`muse/ts-18-lsigma` was **not** touched.

## The fix

### 1. The denominator comes from the requested grid

`experiments/pair_grid_audit.py` is the one place the rule lives:

    at rung q256 a run owes |{L in pair_bits : L*256 >= q256}| x |pair_ratios| cells

`run_pair_unit` now calls `audit_rung(q, a.pair_bits, a.pair_ratios, cmp,
reasons=dropped)` and prints its lines. The old line took `M` from `len(cmp)`
— the survivors — so it read `N of N` over any drop at all.

### 2. A dropped cell is named, with why it dropped

The stage is the only place that knows *why* a cell is absent, so it records
it rather than leaving a reader to guess. Three reasons, three distinct lines
of code:

* `NO_BYTE_MATCH` — `bytematched_rung` found no integral rung on this shape;
* `REFERENCE_MISSING` — the byte-matched reference arm failed to encode;
* `CANDIDATE_MISSING` — the candidate arm failed to encode.

They are stamped into the JSON as `R{q}_grid_audit` and printed as
`GRID INCOMPLETE at R{q}` followed by one `absent: <label> -- <why>` line per
cell. The `best on <gate>` line — the one that read wrong by 28% on the one
broken artifact — now carries `-- OVER 8 OF 12 CELLS, not the grid` when it
is an argmin over a subset.

### 3. `summarise_pair` re-audits before it summarises

It re-derives every `(unit, rung)` audit from `b.doc["args"]` — the run's own
record, not whatever reached the function — and prints
`REFUSING TO READ R{q} AS A GRID` before any separability verdict. It
**flags rather than raises**: the per-arm rows underneath are the
measurements, and killing a long sweep at the summary step would lose them.

It also picks up a second silent drop in the same function: `if len(g) != n:
continue` removed any arm not measured on *every* unit, without a word. That
now logs `incomplete across units` and records `partial_arms`.

### 4. One implementation of the reader check

`pair_arm_key` moved out of the sweep into `pair_grid_audit`, so the writer
and the reader cannot drift on how a cell is spelled. `pair_grid_audit.py` is
torch-free and imports nothing from `src/`, so the rule is checkable in the
pure interpreter — which matters, because this is a bookkeeping failure that
would otherwise only ever be exercised on a box with a GPU.

The CLI:

```
$ python experiments/pair_grid_audit.py /home/rob/tmp/ts93scratch/broken_pair.json
########## /home/rob/tmp/ts93scratch/broken_pair.json
grid asked for: L=[12, 14, 16] x ratios=[1.0, 1.25, 1.4142, 1.75] at rungs=[1024]; 1 unit(s)
  model.layers.2.self_attn.k_proj R1024:
    byte match: 4 of 12 arms the grid asked for sit at their reference's exact bpp
    GRID INCOMPLETE at R1024: 8 of 12 cells absent
      absent: L=12 r=1 -- the candidate arm was not encoded
      ... (all eight named) ...
    CONTROL MISSING at R1024: no control was recorded at all -- a file written
    before #96, or a rung whose repeat vanished; this rung's arms are unchecked
    for contamination and read exactly like checked ones (#96)

VERDICT: GRID INCOMPLETE -- this file's summary is derived from a subset of
the grid and cannot be read as a comparison over it (#93); and at least one
rung has no passing repeat control, so its arms are not known to be free of
cross-arm contamination (#96)
EXIT=1
```

The input is a hand-built file with the **shape** of the broken artifact:
twelve cells asked for in `args`, only the `L=14` row present.

## Test evidence

### Fails before, passes after — demonstrated, not asserted

`tests/test_pair_grid_drop.py` drives `run_pair_unit` itself (the function
both revisions have) with `encode_arm`/`reach_stats`/`score`/`try_arm`
stubbed, on a 1024x1024 shape where the L=12/L=14 byte delta is exactly −48
q256 steps. Each test removes cells by a different mechanism.

With the stage file checked out at `26baa1b` (pre-fix) and the tests at
`fdd3e33`:

```
6 failed in 0.86s
FAILED test_complete_grid_reports_the_whole_grid
FAILED test_a_width_with_no_byte_match_is_named
FAILED test_a_failed_reference_encode_is_named
FAILED test_a_failed_candidate_encode_is_named
FAILED test_an_argmin_over_a_subset_says_so
FAILED test_summarise_pair_flags_an_incomplete_rung
```

and the captured log they fail over is the defect verbatim:

```
    byte match: 8 of 8 arms sit at their reference's exact bpp
    best on h at matched bytes: R1024 L=14 r=1 at 1.0000x
```

— eight of eight, over a grid of twelve, with `L=12` gone and unmentioned.
With the fix restored: `6 passed`.

`tests/test_pair_grid_audit.py` (14 tests, torch-free) pins the invariant
itself and the reader, including a file shaped like the broken artifact
(12 asked for, 4 present, exit 1, every absent label printed) and a file that
does not record its own grid (refused, not cleared).

### Suite

Baseline on master `82cdf51` and the branch total: **see the numbers appended
at the bottom of this file** — both runs go through `gpulock.sh` and neither
was piped through `tail` in an `&&` chain.

## #96 — the repeat control (commit `d770df1`)

Filed while fixing #93, then taken on the same branch at the coordinator's
ask, as its own commit.

`run_pair_unit` runs the shipped arm first and repeats it last; that repeat is
the stage's only evidence that no arm leaked state into a later one. It was
recorded `if last in res` — and the repeat is the arm most likely to die,
running last after every wide table has churned the allocator. When it died,
no `R{q}_control` key was written, nothing was logged, and the summary never
looked. **A rung with no contamination check read exactly like a rung whose
contamination check passed.**

* The control is recorded either way: `{"ran": False, "reason": ...}`, telling
  a dead repeat (`CONTROL_REPEAT_MISSING`) apart from a dead baseline
  (`CONTROL_BASELINE_MISSING`).
* `CONTROL MISSING at R{q}` in the log, at the rung and again in the summary.
* `summarise_pair` counts controls **over the units the run has**, not over
  the ones that reported — the #93 mistake, avoided here by construction.
* `pair_grid_audit.control_status` keeps three outcomes apart where the code
  had two: ran-and-agreed, ran-and-disagreed, never ran. A gap in the evidence
  and evidence of a bug are different findings, and only one of them means the
  numbers are wrong.
* The reader exits 1 on a rung with no passing control, so a file written
  before this cannot be cleared by a check that did not exist then.

### Fails before, passes after

With `experiments/` checked out at `26fa1b1` (the #93 fix, pre-#96) and the
tests at HEAD:

```
7 failed, 19 passed in 1.10s
FAILED test_pair_grid_drop.py::test_a_control_that_ran_is_recorded_as_having_run
FAILED test_pair_grid_drop.py::test_a_failed_repeat_leaves_a_control_that_says_it_did_not_run
FAILED test_pair_grid_drop.py::test_a_failed_first_arm_names_the_baseline_not_the_repeat
FAILED test_pair_grid_drop.py::test_summarise_pair_counts_controls_over_the_units_it_has
FAILED test_pair_grid_audit.py::test_reader_passes_a_complete_file
FAILED test_pair_grid_audit.py::test_a_control_that_never_ran_is_not_a_control_that_passed
FAILED test_pair_grid_audit.py::test_reader_refuses_a_file_whose_rung_has_no_control
```

The load-bearing one fails on `KeyError: 'R1024_control'` — pre-fix the key is
not merely wrong, it is not there. Restored: `26 passed`. The nineteen that
pass throughout are the #93 tests, which is the check that the two fixes are
independent.

## Issues filed

* **#96** — the pair stage's repeat control disappears if the repeat arm
  fails. Same defect class, different key: `if last in res:` means a run with
  no contamination check reads exactly like one whose check passed. Left out
  of this diff deliberately.
* **#96** — **now fixed on this branch** (`d770df1`), at the coordinator's ask.
* **#97** — land #18's offline `pair_report.py` against `pair_grid_audit` or
  not at all. It carries its own inline copy of the completeness rule; two
  copies of a completeness rule is the shape of the bug #93 was. **Routed to
  the #18 agent, which owns that tree; not touched here.**

## Scope

Harness only. Nothing under `src/` was read for behaviour or edited; no wire,
schema or `encoder_profile_id` is involved. No Fable consultation was needed.
