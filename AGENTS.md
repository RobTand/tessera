# Tessera — working rules

Tessera is a wire format, an encoder, decoders, kernels and a vLLM plugin. It
ships bytes that another process must read back exactly, so almost every rule
here is a restatement of one idea: **the thing that is priced must be the thing
that is served.**

`docs/ARCHITECTURE.md` is the current system map. `README.md` is the one-page
statement of what the wire is. This file is the normative contract for anyone —
person or agent — changing the code.

## Core principles

1. **Priced == written == served.** A footprint number, an encoder's cost, the
   bytes on disk and the tiles the kernel executes are one object. A change that
   moves any one of them moves all four or it is a confound. `tests/
   test_audit_byte_baseline.py` is the proof harness; a change that touches
   rendering, planes or layout adds its condition to that matrix or it is
   untested by construction.

2. **No heuristics where an explicit exists.** Derive a threshold from the
   objective, or from a dtype's precision. Never from intuition, and never from
   a round number. An iteration cap is a backstop, not an answer: a descent ends
   at its own fixed point, and the fixed point is what the code tests for.

3. **Pin the rule, not the roster.** A test that restates today's list of
   formats, rungs, planes or lanes passes on the day the list is wrong. Derive
   the expected set from the code that owns it. Pin a roster only when the
   roster *is* the decision.

4. **One rule, one home.** A refusal stated in three modules is three rules that
   will drift. The module that owns the grammar owns the message; everyone else
   calls it. `grammar.require_column_groups` is the shape of this.

5. **Refuse where the bytes are decided.** A width, plane or rate nothing can
   serve is refused at write, by name, with the reason — not at load, as a bare
   reshape error in someone else's process. Fail closed, and say which field.

6. **A claim about another runtime is attested, never asserted.** What vLLM
   executes is read from the contract the plugin publishes, not from prose. A
   `rationale` field explains; it is never the value a gate reads.

7. **A claim needs a receipt, and a screen is not a result.** State no number
   you did not produce. Weight-space and H-weighted numbers are screens; served
   KL against a BF16 teacher at matched bytes is the metric that promotes.
   Say which one you have, and say plainly what you did not measure.

8. **A test must be shown failing before the fix.** A test that passes on the
   unfixed code is worthless. Record the pre-fix failure line.

9. **Measurement is first-class, and telemetry counts as measurement.** Profile
   before and after; the delta is the claim. On GB10 `gpu_utilization` is
   non-diagnostic under load — it reads 96% for a stalled kernel and a saturated
   one alike, and `utilization.memory` is a fake hard 0. Read power against the
   ~140 W envelope and rank by work per joule.

10. **`docs/ARCHITECTURE.md` stays current in the same commit** as any change to
    the wire, the recipe table, the serving lane, the plugin contract or a gate.
    Dated files under `docs/measurements/` are append-only history and never a
    substitute.

11. **Fix the finding where you found it -- file only what you cannot.** A
    defect you noticed and neither fixed nor filed dies with your context. But a
    ticket is not the only record and it is usually the worse one: whoever
    tripped over the defect understands it better than any fresh agent will, and
    that understanding evaporates the moment the task returns. A filed one-liner
    costs a brief, a worktree, a fresh-context ramp-up, a report and a merge, to
    re-derive what somebody already knew. (Rob, 2026-09-03, on a dozen tickets in
    an hour: *"we're just proliferating issues right now that may make sense to
    fix in the context of the way in which they were discovered."*)

    **Default: fix it, on your branch, in a separate commit** (rule 12). That
    buys a readable diff without paying the ticket tax.

    **File instead when the fix is not yours to make:**
    - it needs a decision only Rob prices -- a default moves, bytes move, an
      `encoder_profile_id` moves, a wire changes, a promotion gate is involved;
    - it needs a measurement you are not set up to take -- a serve, a second
      population, the other box;
    - it lives in another agent's live branch (read theirs, never edit theirs);
    - it is large enough to swamp the diff of the task you came for.

    Two bounds, so this is not a licence to widen scope: it covers what you
    **trip over** doing the task you came for, never a hunt for adjacent work;
    and every off-task fix is listed in the report, one line each, so nothing
    lands unannounced.

    **When you do file, timeliness still binds.** Same working session, **before
    starting the next task** -- not at the end of the day, not in a handover, not
    in a summary. A finding held in context for "later" has already failed. The
    bar is **"I believe this is wrong"**, not "I have proved it and scoped the
    fix"; over-filing beats losing a finding, and fixing beats both. State the
    uncertainty rather than withholding until certain.

    **Say which it was.** An issue filed under this rule records whether the
    filer could have fixed it and chose not to, and why. That is how a real
    ticket is told from a deferred one-liner, and without it the backlog stops
    describing the work.

    What an issue owes is otherwise little: the **evidence at `file:line`** (read
    the line; do not repeat a claim), what breaks and under what inputs, a
    **severity**, and what would fix it -- or, when the fix is a judgment call,
    the options and who decides.

    **Severity.** `P0` -- can ship or serve a wrong artifact. `P1` -- a gate that
    cannot catch its own defect, or a wrong or underived number a decision reads.
    `P2` -- provenance, observability, or a claim beyond its evidence. `P3` --
    cleanup with no decision riding on it. Two orthogonal labels:
    `measurement-needed` when a GPU or served A/B decides it, and
    `needs-decision` when the answer is a trade only Rob prices.

    **One exception, narrower than it was.** A finding in prose -- a doc, comment
    or docstring -- is *fixed on sight* and never filed: reading the cited line
    IS the verification, so a stale sentence is a one-line commit. (The former
    second exception, that a delegated worker neither fixes nor files, is
    withdrawn. Workers have `gh`, workers fix, and a worker that files says why
    it did not fix.)

12. **Never widen a commit; widen the branch.** One commit does one thing, so a
    reviewer can take it or drop it on its own -- that is the whole content of
    this rule, and it was never a reason to leave a defect unfixed. A second fix
    on the same branch, in its own commit with its own test and its own pre-fix
    failure line, is exactly right (rule 11). A second fix folded into the first
    commit is what makes a diff unreadable, and that is the only thing forbidden
    here.

## Before finishing

- Targeted tests for every touched module, plus the byte-baseline audit if
  anything about rendering, planes or layout moved.
- **Never compute the master baseline suite to prove your branch is clean.**
  Run the test files your diff touches; for each that fails -- and only those
  -- run that one file against a pristine master checkout. That answers the
  only question a baseline was ever asked ("was this already broken?") in
  seconds instead of a full suite. A delegating coordinator states the known
  baseline in the brief; fifteen agents recomputing one number in parallel put
  a 121 GB box into swap on 2026-09-03.
- **The full suite is the coordinator's, run once on the merge result -- not
  yours, run in isolation.** A branch owes its own targeted evidence, and
  `tools/impacted_tests.py --ref master...HEAD` computes which tests those
  are rather than leaving it to judgement: it walks the import graph, inverts
  the edges, and returns everything reverse-reachable from what you changed.
  Run it **in your own worktree** -- it reads the tree it is standing in, and
  a branch analysed from elsewhere has unreadable edges, which it says. Trust
  its `verdict`: `narrowed` means run the listed files, `full` means it found
  a changed path it cannot reason about and you run everything. It fails open
  by construction, because a selector that silently drops a test is worse
  than no selector. Add the pre-fix failure line for every test you added. It does not owe a whole-tree run.
  Fifteen branches each proving the whole tree green *in isolation* is
  fifteen runs that cannot see the one risk that matters -- what the branches
  do to each other -- while making every one of them slow enough to matter:
  under that load a suite went from minutes to a projected ninety on
  2026-09-03, and the merge queue stalled behind runs that were answering the
  wrong question. Integration risk is caught at the integration point, once.
- **A suite count is meaningless without its device population, and the
  CUDA-gated surface is covered by nothing automatic.** Master was red on three
  CUDA-gated tests while GitHub Actions, the x86 pool suite and a local CPU run
  all read green; not one of the three could collect or run them (tessera#112).
  Every run now says which population it covered -- device, skip count,
  uncollected modules, and the skip reasons verbatim -- so read that block
  before believing a pass count, and quote it whenever you record one. A run
  that must cover the CUDA-gated surface says `--strict-cuda` (or
  `TESSERA_STRICT_CUDA=1`), which refuses a device-less session instead of
  skipping some 450-480 tests and reporting green. `--surface-json PATH` writes
  the same population as a table, which is what a receipt should read.
- The pre-fix failure line for every test added.
- `docs/ARCHITECTURE.md` updated in the same commit if a normative claim moved.
- Every side-finding fixed in its own commit, or filed with the reason it
  was not fixed. Prose is always fixed on sight, never filed.
