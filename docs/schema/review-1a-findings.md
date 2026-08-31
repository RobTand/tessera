# Item 1a review — findings

Build order §16 defines item 1a as a **reviewed** byte-level schema and parse
algorithm. §17 item 4 names this review's scope. This file is the review record.

Three independent passes: this session (self), `glm53-flash-high` via Opencode
(adversarial wire-ambiguity framing), Muse Spark (normative-claim conformance
trace). Reviewer findings are merged in below with attribution; **every finding
is verified against the code before it is recorded here** — no reviewer
self-certification is accepted.

Status legend: **OPEN** / **FIXED** (commit) / **REJECTED** (with reason).

**Pass 1 resolution, all in `81ec726`** — 115 tests, up from 101; every fix has
a test in `tests/test_review_1a.py` that exercises the input which used to be
accepted:

| # | Severity as filed | Verified severity | Status |
|---|---|---|---|
| F1 | BLOCKER | BLOCKER | FIXED — plane digests cover real bytes and are verified |
| F2 | BLOCKER | MINOR (over-filed) | FIXED — bound moved into the validator |
| F3 | MAJOR | MAJOR | FIXED — `NORMATIVE_ELEMENT_BITS` binds every descriptor |
| F4 | MAJOR | MAJOR | FIXED — zero padding, alignment bytes and sub-byte slack |
| F5 | MAJOR (misstated) | MAJOR | FIXED — prefix-sum identity, not strict ascent |
| F6 | MINOR | MINOR | FIXED — `WHOLE_PLANE` carries one count |
| F7 | MINOR | MINOR | FIXED — `serialize` runs the accountant |
| F8 | BLOCKER | BLOCKER | FIXED — the prefix shape is validated |
| F9 | BLOCKER | BLOCKER | FIXED — per-terminal `payload_digest` |

Two of my own findings were wrong as filed and are corrected in place rather
than quietly dropped: **F2** was not a fail-open hole (the accountant does
refuse an over-claiming terminal, and `parse` reaches it), and **F5**'s proposed
"strictly ascending" rule would have been an incorrect fix, because a
zero-count granule legitimately repeats an offset.

---

## Pass 1 — this session, independent

### F1 (BLOCKER) — `PlaneDescriptor.content_digest` is dead metadata

`planes.py:127` declares a per-plane `content_digest` and `:140` validates its
*length*. Nothing ever verifies it against the bytes. `container.parse` checks
only the whole-region digest, and only when the region is complete
(`container.py:141-143`).

Consequence: **a truncated artifact has no integrity check at all.** Truncation
is this format's headline feature (§9 legal truncations), so the common case is
the unverified one. A single flipped bit inside a truncated plane region parses
clean. Even a complete artifact never has its planes verified individually, so
a whole-region digest collision or a mis-assembled region is not localised.

Fix: verify each fully-present plane's byte range against its declared
`content_digest` during `parse`, for truncated and complete artifacts alike.

### F2 (MINOR, downgraded on verification) — the terminal/plane-extent bound is enforced late

Originally filed as a blocker. **Wrong.** `footprint.plane_region_bytes` does
raise `FootprintDisagreementError` when `count > descriptor.element_count`, and
`parse` reaches it through `account_terminal` (`container.py:137`). The read
path is fail-closed.

What survives is placement: the invariant lives in the *accountant* rather than
in `Manifest.__post_init__`, so an invalid manifest can be constructed and
carried around, and is only refused when someone prices it. Worth moving
earlier as defence in depth; not a hole.

### F3 (MAJOR) — `element_bits` is unbound from `PlaneKind`

`layout.py:51-61` holds the normative per-kind widths (BODY=1, SCALE_BASE=8,
DIAG_*=16, SCALE_REFINE=4, RELEASE=4, …). That table is consulted only when
`layout.py` *builds* a descriptor. `PlaneDescriptor.__post_init__` accepts any
positive `element_bits`, and `Manifest` does not cross-check.

Consequence: two conforming implementations disagree on bytes for the same
logical unit — a decoder that trusts the descriptor and one that trusts the
schema's fixed widths read different data. For a content-addressed format this
is a divergence, not a nit.

Fix: make the width table normative in `planes.py` and reject any descriptor
that contradicts it.

### F4 (MAJOR) — padding bytes are never checked to be zero

`PlaneDescriptor.byte_length` pads each plane up to `alignment_bytes`
(`planes.py:157-170`), and `bitio.check_padding_zero` exists — but `parse`
never calls it. Padding content is therefore unconstrained.

Two consequences, both real for a content-addressed format: the encoding is
**not canonical** (the same logical content has many legal byte strings, so
identity is not a function of content), and the slack is a covert channel.

Fix: require zero padding and verify it on parse.

### F5 (MAJOR, restated) — `restart_offsets` are a redundant derivable, unpinned

`planes.py:146-148` requires only `sorted()`. My first draft of this finding
proposed *strict* ascent; that is wrong — a granule with a zero count makes two
adjacent offsets legitimately equal.

The real invariant is stronger and exact: `layout.py:174-177` builds the table
as the running prefix sum of `counts`, so `restart_offsets[i] == sum(counts[:i])`
identically. The table is fully derivable, and nothing requires it to agree with
the counts it is derived from. A manifest may therefore declare offsets that
contradict its own counts, and §9's stated purpose for the table — segment-local
random access on the GPU without a host parse — is exactly the consumer that
would then read the wrong range.

Fix: require the prefix-sum identity, which subsumes ascent and bounds together.

### F6 (MINOR) — `counts` arity is unrelated to `count_granularity`

`WHOLE_PLANE` should carry exactly one count; `PER_BLOCK` / `PER_SUPERBLOCK`
should match the geometry. Nothing ties them, so the same plane admits many
`counts` vectors with equal sums. Downstream only `element_count` (the sum) is
read, so this is non-canonicality without a decode divergence.

### F7 (MINOR) — `serialize` can emit an artifact its own `parse` rejects

`container.serialize` binds the region with `payload_digest` but never runs the
accountant over it, so a region whose length matches no declared terminal is
emitted happily and then refused by `parse`. The write side should be as
fail-closed as the read side.

### F8 (BLOCKER) — "a terminal is a prefix" is asserted everywhere and enforced nowhere

`planes.py:51-57` and `manifest.py`'s `TerminalRecord` docstring both state that
a terminal is a **prefix** of the canonical plane order, and `container.py`'s
truncation contract depends on it: `parse` resolves a byte length to a terminal
and hands back `data[manifest_end:]` as that terminal's plane region.

`footprint.plane_region_bytes` sums `byte_length(count)` over the planes in
canonical order **whatever the counts are**. A terminal declaring, say, (full,
0, full) is priced happily. Its `exact_bytes` is a real number, `parse` will
match a region of that length, and the bytes handed back are *not* the bytes
that terminal describes — the region is a byte-prefix of the artifact, but the
terminal's own layout skips a plane in the middle.

So the format's central safety property, that truncating an artifact at a
declared boundary yields exactly the declared terminal, rests on a claim no
code checks.

Fix: enforce the prefix shape in `Manifest.__post_init__` — for each terminal,
counts equal the plane extent up to some index, at most one plane is partial,
and every plane after it is zero.

### F9 (BLOCKER, subsumes F1) — no terminal carries a digest of its own bytes

Following F1: even with per-plane digests, the plane cut at the truncation
boundary can never be verified against a whole-plane digest, so every legal
truncation keeps one unverified plane.

The proportionate fix is a per-terminal `payload_digest` — 32 bytes per
terminal, three terminals in the test ladder — over `plane_region[:exact_bytes]`.
That gives **complete integrity for every legal truncation**, and subsumes the
partial-plane gap entirely.

---

## Pass 2 — `glm53-flash-high` (adversarial ambiguity)

*Pending.*

## Pass 3 — Muse Spark (conformance trace)

*Pending.*
