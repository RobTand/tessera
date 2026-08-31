# Item 1a review — findings

Build order §16 defines item 1a as a **reviewed** byte-level schema and parse
algorithm. §17 item 4 names this review's scope. This file is the review record.

Three independent passes: this session (self), `glm53-flash-high` via Opencode
(adversarial wire-ambiguity framing), Muse Spark (normative-claim conformance
trace). Reviewer findings are merged in below with attribution; **every finding
is verified against the code before it is recorded here** — no reviewer
self-certification is accepted.

Status legend: **OPEN** / **FIXED** (commit) / **REJECTED** (with reason).

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

### F2 (BLOCKER) — terminal plane counts are not bounded by plane extents

`TerminalRecord.__post_init__` (`manifest.py`) checks arity, non-negativity and
the clip-exponent domain, and `Manifest.__post_init__` checks slot and byte
uniqueness. **Nothing checks `terminal.plane_elements[i] <= planes[i].element_count`.**

This is not hypothetical: it is the bug this session already hit during 1b test
bring-up, where a terminal declared 4 released positions against a RELEASE plane
whose extent was 0. It was fixed in the *builder* (`layout.build_planes`
gained `max_released`) — the *invariant* was never added, so any manifest not
built by `layout.py` can still declare a terminal that overruns its own planes.
Fixing a symptom in the producer left the validator fail-open.

Fix: validate the bound in `Manifest.__post_init__`, where it holds for every
manifest however constructed.

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

### F5 (MAJOR) — `restart_offsets` are unbounded and only weakly ordered

`planes.py:146-148` requires the table to be ascending via `sorted()`, which
admits *equal* adjacent offsets, and never checks that an offset lies inside
the plane's own extent. §9's stated purpose for the table is segment-local
random access on the GPU without a host parse — an out-of-range or duplicated
offset is exactly the input that turns that into an out-of-bounds read.

Fix: strict ascent, and bound every offset by the plane's element extent.

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

---

## Pass 2 — `glm53-flash-high` (adversarial ambiguity)

*Pending.*

## Pass 3 — Muse Spark (conformance trace)

*Pending.*
