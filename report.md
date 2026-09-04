# tessera#95 — the LDLQ block floor belongs to the caller's path

Branch `muse/ts-95-floor`, worktree `/home/rob/tmp/musefix/ts-95-floor`, based
on master `82cdf51`.  Two commits: the source change, then the tests.

## The diagnosis holds, verified against the code

`compensate.choose_ldl_block`'s `floor` defaulted to 16 and its docstring
presented 16 as a property of *the wire* ("the smallest block the wire
allows ... `S6B` and `LUT` group 16 weights and so floor at 16").  That
constraint is `compensated_targets`', not the method's, and the path production
takes does not have it.  Three code facts, none of them prose:

1. **The stitching path really does have the floor.**  `compensated_targets`
   calls `encode(target[:, start:stop], start, stop)` — an *independent encode
   per slice*.  Measured, not argued: with `encode_unit(..., group=32)` a slice
   of 32 columns reproduces the whole-matrix span exactly, and a slice of 16 or
   8 is refused by the encoder itself —
   `GrammarError: 16 columns is not a whole number of 32-weight scale groups`.
   The floor there is the encoder's own group (and its rotation block).

2. **The production path does not.**  In `encode.py`'s pass loop the plane is
   read **once per pass, before the block loop** (`scale = current_scale()`,
   with the `for start, stop in block_spans` loop below it) and refit **once,
   after** it (`refit_channel_scale` / `_refit_scales_lut` / `_refit_scales`).
   No scale is ever fit to part of a group, whatever `ldl_block` is.  The only
   floor `encode_unit` imposes is `ldl_block < 1`.  Its docstring already said
   so: "the slice-equals-whole property the standalone
   `compensate.compensated_targets` needs does not arise."

3. **Measured on the plane the docstring said floors at 16.**  On the LUT
   plane (`E2M1x2`, `q256` = 896 the cap and 768 sub-cap, `DEFAULT_HALF` = 16),
   an identity LDL factor at `ldl_block` = 16, 8, 4, 2 **and 1** encodes the
   plain whole-matrix pass **bit for bit** — codes, scale table, refine
   indices and `sse` all equal, on both bodies.  A plane that floored the
   schedule could not do that.  (#60's own sweep had already run b2 and b1 on
   this plane; the chooser was the only thing that could not reach them.)

So the shipped chooser, floored at 16, could not return the dense-Qwen `b8`
and `b4` arms that `1645c23` validated `block_penalty` against in its own
table, and nothing raised.

## What changed

**`src/tessera/compensate.py`**

* `choose_ldl_block(..., floor: int)` — **no default**, keyword-only, in the
  shape of `control.assert_plane_promotion`'s `served_bar`.  The docstring
  names both paths and their floors (stitching: a block the encoder's scale
  group and rotation block both divide; `encode_unit(ldl=...)`: 1), says what going wrong looks like
  (the 16 that silently deletes every block where the win lives), and notes
  that at `floor=1` the "budget the floor cannot meet" refusal is unreachable
  rather than dead, since `block_penalty(H, 1)` is exactly 1.0.
* The refusal message no longer says "the smallest the wire allows" or "lower
  the scale-group floor" — both asserted the stitching path's story.
* `block_ldl`'s sentence now says **whose** floor it describes.
* The module docstring's alignment paragraph ("encoding a column slice is
  bit-identical ... provided the slice is aligned") stated the same
  out-of-scope constraint as a property of compensation itself; it now says the
  requirement is `compensated_targets`' and that `encode_unit(ldl=...)` stitches
  nothing.  Same finding, one paragraph higher, so it is in the same commit.

**`src/tessera/export.py`** — the `DEFAULT_LDLQ_BLOCK` comment already told a
future caller to price its own block with `choose_ldl_block`; it now also
carries the floor that call must pass on this path (1), since that is the one
production call site there will be.

**Tests** — the suite passed `floor=16` at every call site, pinning the wrong
floor as the LUT/S6B convention.  Replacing it with `floor=1` everywhere would
be the same roster with a different number (AGENTS.md rule 3), so what is
pinned is the rule:

* `test_the_floor_has_no_default_because_the_two_callers_disagree_about_it` —
  `inspect.signature` shows no default, and omitting it is a `TypeError`.
* `test_the_block_the_chooser_returns_honours_whatever_floor_it_was_given`,
  parametrised over floors 1/2/8/16/32: `b >= floor`, `b % floor == 0`, and
  `block_ldl` accepts it.
* `test_the_floor_alone_decides_whether_a_small_block_is_reachable` — one
  Hessian, one budget, two callers: a floor above the block the budget wants
  refuses it; a floor of 1 returns exactly that block.  Parametrised over the
  target block, so no plane-to-number table appears anywhere.
* The pre-existing chooser tests are parametrised over the floor instead of
  inheriting 16, and assert against `floor` rather than against 16.
* `tests/test_ldlq_lut_plane.py`: the identity-factor test now also runs at
  `ldl_block=4`, below the plane's `DEFAULT_HALF` group (imported, not
  written as 16), and a new test picks a block with
  `choose_ldl_block(..., floor=1)`, asserts it is below the plane's group, and
  encodes it on this plane.
* `tests/test_compensate.py`: a new CUDA test shows the stitching path's floor
  *is* the encoder's refusal of a sub-group slice, and hands that floor to the
  chooser.

## No bytes move

Nothing in production calls `choose_ldl_block` (`grep`: the tests, and a
cross-reference comment at `DEFAULT_LDLQ_BLOCK`).  An AST comparison of master
against this branch with every docstring stripped shows the *only* semantic
differences in `src/`:

* `kw_defaults` for `floor`: `Constant(16)` -> `None` (no default);
* the two string literals of the refusal message.

`src/tessera/export.py` is AST-identical (the change is a comment).  No encoder
input, no plane, no schedule and no `encoder_profile_id` input is touched.

## Test evidence

Targeted, per AGENTS.md `1f7836c`: the files this diff touches, the files that
import what changed, and the pre-fix failure line for the tests added.  No full
branch suite (one had been started and was killed when that rule landed); no
master baseline (the one I started was killed for the same reason).

**Pre-fix line — the branch's tests against master's source**
(`master_tree` at `82cdf51`, this branch's three test files copied in,
sparklina via `pbrun --gpu`):

```
1 failed, 70 passed, 15 warnings in 423.22s
FAILED tests/test_ldl_block_penalty.py::test_the_floor_has_no_default_because_the_two_callers_disagree_about_it
E       assert 16 is <class 'inspect._empty'>
E        +  where 16 = <Parameter "floor: 'int' = 16">.default
```

That is the whole defect, and it is worth being precise about what the other
new tests say: they **pass on master too**, because master's chooser already
honours a floor it is *given*.  What master cannot do is fail to guess one.  So
the fix is the removed default, and the rest of the new tests are there to keep
the rule from being restated as a number again.

**Branch**

* `tests/test_ldl_block_penalty.py` + `tests/test_compensate.py` — **33 passed**
  (25.75 s).
* `tests/test_ldlq_lut_plane.py` — **38 passed** (102.83 s), including the
  identity factor at `ldl_block=4` on both bodies and the chooser-picked
  sub-group block.
  (Both taken on a scratch copy whose only difference from the committed branch
  is two docstring/comment paragraphs — `diff`-verified, no code differs.)
* `tests/test_audit_byte_baseline.py` — **6 passed** (60.30 s) on the branch
  tree itself.  This is the repo's own no-bytes-move harness.
* A combined re-run of those three plus the other two files that import
  `compensate` (`test_ldlq_window.py`, `test_refit_trailing.py`) was queued
  behind the fleet in the GPU slot and had not been granted it at the time of
  writing; it lands at `/home/rob/tmp/ts95/after_tree/pbrun_result.txt` on
  sparklina.  Nothing in it is expected to move: neither file calls
  `choose_ldl_block`, and the AST diff below bounds what could.

## Off-task fixes

* The `compensate` module docstring's alignment paragraph (above) — the same
  out-of-scope constraint as #95, in the same file, so it is in the #95 commit
  rather than a separate one.

Nothing else was tripped over.  No issues filed.

## Consultations

None.  The advisor (in-session reviewer) was consulted three times: once before
implementing, to check the diagnosis and the test design, and once at the end.
No `fable-<tier>` agent was needed.
