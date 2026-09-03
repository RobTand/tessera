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

11. **File the finding before you move on — and err aggressive.** A defect you
    noticed and did not file is a defect that dies with your context. The cost of
    a duplicate issue is thirty seconds and a close; the cost of a lost one is
    that it is rediscovered months later by an artifact. The bar for filing is
    **"I believe this is wrong"**, not "I have proved it and scoped the fix" —
    over-filing is explicitly sanctioned (Rob, 2026-09-03: *"I don't care if
    we're too aggressive"*).

    Timely as well as accurate: file it in the same working session, **before
    starting the next task** — not at the end of the day, not in a handover, not
    in a summary. A finding held in context for "later" has already failed.

    What an issue owes is little: the **evidence at `file:line`** (read the line;
    do not repeat a claim), what breaks and under what inputs, a **severity**,
    and what would fix it — or, when the fix is a judgment call, the options and
    who decides. State the uncertainty rather than withholding until certain.

    **Severity.** `P0` — can ship or serve a wrong artifact. `P1` — a gate that
    cannot catch its own defect, or a wrong or underived number a decision reads.
    `P2` — provenance, observability, or a claim beyond its evidence. `P3` —
    cleanup with no decision riding on it. Two orthogonal labels:
    `measurement-needed` when a GPU or served A/B decides it, and
    `needs-decision` when the answer is a trade only Rob prices.

    **Two narrow exceptions.** (a) A finding in prose — a doc, comment or
    docstring — is *fixed on sight*, not filed: reading the cited line IS the
    verification, so a stale sentence is a one-line commit. (b) A worker inside a
    delegated task does not widen scope and does not file; it records the finding
    in its report with `file:line` and a proposed severity, in a form the
    coordinator can file verbatim — and the **coordinator files it before
    starting the next task**.

12. **Never widen scope inside a change.** Fix what you came for. Everything else
    is a filed issue (rule 11), not a second diff.

## Before finishing

- Targeted tests for every touched module, plus the byte-baseline audit if
  anything about rendering, planes or layout moved.
- The pre-fix failure line for every test added.
- `docs/ARCHITECTURE.md` updated in the same commit if a normative claim moved.
- Every side-finding filed, or fixed on sight if it is prose.
