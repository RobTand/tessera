"""What the encoder *does*, hashed -- the identity nobody has to remember.

``encoder_profile_id`` (``unit_artifact.py``) is input-only by design: it
declares the finite ordered set of terminal slots a reader must reproduce, and
"contains nothing an encode alone can produce" (``manifest.py:12``).  So an
*encoder* change -- same arguments, different bytes out -- moves nothing in it,
and neither ``CONTAINER_VERSION`` (the on-disk container) nor the merge guard's
``activation_aware`` block (settings, not numerics) catches it either.  Two
halves of one checkpoint built either side of such a change compare equal and
merge; a resume picks up half its units from the old encoder.  That is not
hypothetical: it happened once already, and only a uniform ``q896`` made it
recoverable (tessera#78).

This module is the third identity that closes it, and it is **derived, never
declared**.  :func:`encoder_fixture_id` encodes a fixed, tiny fixture set at
fixed arguments and hashes what comes out.  The digest moves exactly when the
encoder's output moves at fixed inputs -- which is the definition of the thing
being caught -- and never when a comment, a docstring or a refactor changes.
Nobody bumps it and nobody can forget to, which is the whole reason it is not
a hand-maintained ``ENCODER_VERSION``: a constant somebody must remember to
bump is a discipline, and a discipline that fails does so silently.

What is hashed, and why exactly that
------------------------------------

Per fixture: the unit's ``payload_digest`` (sha256 of the whole plane region)
and every ``TerminalRecord``'s canonical bytes (the realised per-plane element
counts, the clip-exponent code, the exact byte count, the exact bpp, and the
payload digest again).  Together those are precisely **what the encoder
computed and how the packer laid it out**.

Three things are deliberately outside the hash, each because it already has an
owner and folding it in would make this identity move on a change that is not
the encoder's:

* **The arguments.**  Grid, code, span, body, plane, window width and the reach
  spellings are ``encoder_profile_id``'s, and the fixtures hold them fixed.
* **The container framing.**  Schema id, schema minor, header: ``container.py``'s
  ``SCHEMA_MAJOR``/``SCHEMA_MINOR`` and ``export.CONTAINER_VERSION``.  A future
  minor bump that adds a field must not read as an encoder change.
* **This identity itself.**  It is stamped *into* the manifest, so hashing the
  serialised manifest would be self-referential.  Hashing the payload region and
  the terminal records is not, which is why the fixture build is also the one
  path that stamps nothing (:func:`building`).

The honest limit, stated rather than left to be assumed
-------------------------------------------------------

A fixture hash is exact for what it covers and **blind to what it does not**.
A change that only moves E4M3 bytes is invisible to an E2M1-only fixture, so
the set spans the grids, bodies and scale planes that actually ship --
:func:`shipping_structures` derives that set from ``recipe_table`` over
``SERIALISABLE_GRIDS`` rather than restating it, and
``tests/test_encoder_identity.py`` fails when a structure has no fixture.  That
makes the coverage claim enforced instead of asserted.

Four narrower blind spots, named because a reader would otherwise assume them
covered.  None of them is a *plane* any more: :func:`fixtures` writes every
kind in ``planes.SHARD_PLANE_ORDER``, and
``tests/test_encoder_identity.py`` derives that claim from that tuple rather
than restating it, so a plane added to the wire fails there until a fixture
writes it.  ``experiments/audit_byte_baseline.py`` makes the same claim over
its own matrices; it is the *offline* instrument, run either side of a change
on purpose, and this one is always on.

Five surfaces the wire never selects are closed rather than named
(tessera#143).  ``e2m1-768/s6b`` encodes the plane no
recipe selects, through the same caller-facing ``scale_plane=`` override an S6b
artifact is written by, so ``encode._pack_scales`` and ``encode._refit_scales``
now move this digest; ``e2m1-768/diagonals`` does the same for segment 2a, so
``diagonals.fit_diagonals`` does too; and ``e2m1-256/completion`` spends the
second rate axis at a rung with headroom, so ``encode._completion_choice`` is
offered more than one descendant and its pick reaches bytes;
``e2m1-768/release`` carries a release, which ``encode_linear`` has no keyword
for, so ``encode._canonical_release_order`` moves this digest too; and
``e2m1-768/shard`` cuts the ``e2m1-768/tcq-lut`` bytes the way a rank does at
load, so ``slicing``'s state replay and the shard layout do as well.  Together
they cost +1.36 s cold and +0.80 s warm on the whole set (measured below: four
of the five ride a plan an earlier case already builds, and the fifth builds
its forests one rung lower, where they are small) -- and each re-bases nothing,
because it carries a ``compatibility_baseline`` (the third remaining blind spot
below, and the rule the same paragraph states).

The remaining four:

* **Rate.**  Each structure is encoded at one declared rung, not at every rung
  it covers.  ``window_bits`` varies with the rung on BF16 (L=14/15/16), and it
  is *bound in* ``encoder_profile_id`` -- two builds that disagree on it are
  already distinguishable -- so the fixtures exercise the shared code at one
  width instead of paying for a 65536-entry table.
* **Hessian structure.**  The activation-aware arms use a synthetic PSD
  Hessian.  ``experiments/audit_byte_baseline.py`` records that the CHANNEL
  refit's ``B <= 0`` condition is a property of a *real* H's off-diagonal
  structure and that no synthetic H tried reaches it, so a change confined to
  that branch is invisible here.  The audit harness, with its committed real
  slice, is the instrument for that; this is the cheap always-on one.
* **The fixture set itself.**  The digest binds what the encoder does *on
  these fixtures*, so an ordinary extension re-bases the identity even though
  no encoder changed -- and the coverage rule above requires that cost when a
  new grid, body or plane ships.  A witness added later for a surface the
  encoder *already produced* -- a blind spot inside an existing structure, or
  one outside every shipping structure that only an override or a second
  byte-producing path reaches -- follows the narrower issue-#116 compatibility
  rule instead: its encoded arm-A contribution is recorded once, contributes
  zero bytes while it matches, and contributes its self-delimiting encoded
  bytes when it does not.  What separates the two cases is whether the encoder
  is new, not whether the fixture is: a shipping structure that did not exist
  before is a different encoder and says so, while a surface nobody had looked
  at is the same encoder, looked at harder.  That preserves the identity of unchanged artifacts without hiding the
  newly-covered byte move.  The baseline is measured history and is never
  bumped; a true rollback removes the contribution again, so the identity
  rolls back if and only if every other fixture output does too.
* **Device.**  The fixtures run on CPU by construction, so the digest is one
  value for the fleet rather than one per accelerator.  Nothing about that is
  left to the environment: every accelerated branch dispatches on the *tensor's*
  device, never on ``torch.cuda.is_available()`` (``encode.py`` at the TCQ
  ``device.type != "cuda"`` guard and the window ``targets.is_cuda`` guard), so
  an export running with a GPU visible computes the same digest as a CPU-only
  process -- measured both ways, same value.  The fused GPU window encoder is
  pinned bit-exact against the reference elsewhere
  (``tests/test_window_viterbi_fast.py``); this identity inherits that pin
  rather than re-proving it.

One thing the test suite deliberately cannot catch.  The tests pin the *rule*
-- the field is absent exactly when the computed id equals
:data:`UNTAGGED_ENCODER_ID` -- and never the value, because a test asserting
the digest would have to be edited whenever the encoder legitimately moves,
which is the hand-maintained discipline this replaces.  So a *wrong*
``UNTAGGED_ENCODER_ID`` passes the suite: every artifact would simply declare
minor 6, which is correct behaviour for a different encoder and wrong history
for this one.  ``experiments/audit_byte_baseline.py --diff`` is the instrument
that catches it, by reporting a byte move where there is none.

Cost, measured
--------------

The original seven encodes of a 16x128 unit, on sparky (GB10, CPU only, box
under load): **42.5 s in a cold process, 1.7 s in a warm one**, memoised so any
process pays it once. Issue #116 adds an eighth, baseline-neutral 16x128
witness; its incremental time has not been isolated from that shared cold
start. Almost all of the original 41 s difference is ``_plan_for`` building the
window tables and anchor forests for the five distinct ``(grid, rung)`` pairs
-- work an exporter does anyway -- which is why the cost lands where it does.

Issue #143's five off-wire witnesses were measured against exactly that set, in
fresh processes on sparklina (GB10, CPU only): **eight fixtures 40.74 s cold /
1.32 s warm, thirteen fixtures 42.10 s cold / 2.12 s warm** -- +1.36 s cold and
+0.80 s warm.  Four of the five ride a plan an earlier case already builds, so
they cost an encode each and no forest; only ``e2m1-256/completion`` builds
anything, and one rung lower the forests are small (0.20 s measured, against
the 41 s the rung above it costs).  Where the cost lands is unchanged:


* An **export** computes it, once, before its first unit.  Against an encode
  that runs for hours it is not a cost anyone can measure.
* A **merge guard** computes nothing.  It compares the ``encoder_fixture_id``
  string each part stamped into its ``tessera_config.json``.
* A **resume** would happen inside a process that is about to encode, so it
  pays the same single warm-up that process pays anyway (:func:`resumable`).

That is the honest answer to "make the fixtures tiny": they are not free in a
cold process, and no consumer that has nothing to encode is asked to pay.
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
from dataclasses import dataclass, replace
from types import MappingProxyType

__all__ = [
    "FIXTURE_ROWS",
    "FIXTURE_COLS",
    "UNTAGGED_ENCODER_ID",
    "Fixture",
    "building",
    "encoder_fixture_id",
    "fixture_digests",
    "fixtures",
    "resumable",
    "shipping_structures",
    "stamped_fixture_id",
]

#: What an artifact carrying **no** encoder identity means: the encoder as it
#: stood when this module was written (2026-09-04, tessera#101).
#:
#: This is not the ``ENCODER_VERSION`` the issue refuses, and the difference is
#: the whole point.  Nobody bumps this.  It is a *measured historical fact* --
#: the value :func:`encoder_fixture_id` returned at the commit that introduced
#: it -- and it is never edited again: the moment the encoder moves, the
#: computed id stops matching this and every artifact written from then on
#: carries its own, automatically.  The conditional write is the same one
#: ``encoder_profile_id`` already uses for a span-1 S6b pair and for a default
#: reach record: the default spelling adds no tag, so every artifact already on
#: disk keeps its bytes and its schema minor.
#:
#: Editing it would not "bump a version"; it would re-label every artifact on
#: disk as having been cut by an encoder they were not cut by.  A test pins the
#: *rule* -- absent iff the computed id equals this -- and deliberately does
#: not pin the value, because pinning the value is what turns a derived
#: identity back into a maintained one.
UNTAGGED_ENCODER_ID = bytes.fromhex(
    "b2cee3a2f5998f03747449ba1206c0cd7ad20cc6665e953c71042b5d81816177"
)

#: The fixture unit's shape.  Sixteen rows because the CHANNEL plane's row
#: scales are per row and a single row would exercise no spread across them;
#: 128 columns because the LUT plane fits sixteen entries against the block
#: scales of a whole superblock, and a narrower unit gives the fit nothing to
#: choose between.  Small enough that seven of them run inline.
FIXTURE_ROWS = 16
FIXTURE_COLS = 128

#: The stream's seed.  A declared input, like any fixture's: what must be
#: derived is the *set* of configurations covered (:func:`shipping_structures`),
#: not the arbitrary numbers fed to them.
FIXTURE_SEED = 20260904

#: Rows given a planted outlier, and how far past the row's own spread it sits.
#:
#: The E4M3 window table reaches about 4.08 sigma and a 128-sample Gaussian row
#: has ``amax/rms`` near 3.0, so a plain Gaussian fixture almost never raises a
#: row -- and ``scale_channel.initial_channel_scale``'s raise, and the landing
#: of that raise, are exactly the default-path arithmetic this identity has to
#: see (tessera#87).  Every third row gets an outlier so the fixture carries
#: raised *and* unraised rows: the landing rule is masked to the raised ones,
#: and a fixture with no unraised row could not tell a change that widened the
#: mask from one that did not.
#:
#: The magnitudes **vary per row**, and that is load-bearing rather than
#: decorative.  A raised row's scale is exactly ``amax / reach``, so a single
#: planted constant gives every raised row the identical scale, the identical
#: fp16 word and the identical rounding -- measured: 6 rows raised, 0 landing
#: short, in all three CHANNEL fixtures.  The rule that has to be reachable
#: here is precisely the one that depends on where that scale's mantissa falls
#: between two fp16 words, so the fixture spreads the mantissas.
OUTLIER_STRIDE = 3
OUTLIER_SIGMAS = 6.0
OUTLIER_SPREAD = 6.0

#: A one-time compatibility witness for issue #116.  This is the digest of
#: the boundary fixture contribution under the current arm-A encoder.  That
#: contribution is omitted while it matches, so adding coverage does not
#: re-label artifacts already written by unchanged behaviour; any byte move on
#: the witness contributes its encoded bytes and therefore moves the identity.
#: Like ``UNTAGGED_ENCODER_ID``, this is measured history and is never bumped.
_UNRAISED_BOUNDARY_BASELINE = (
    "a4d6cc3a19556393eb7dacdf3567ad241789900e06f9e42aeabe38982dd8b7f2"
)

#: The same one-time compatibility witness, for the surfaces the *wire* never
#: selects (tessera#143).  ``encoder_fixture_id`` is a digest of the fixture
#: set, so covering a surface that was always there would otherwise re-label
#: every artifact already on disk as cut by a different encoder -- which is
#: false, and is the one thing this identity must never say.  Each digest is
#: the SHA-256 of its fixture's encoded contribution under the encoder that
#: introduced it: measured history, never bumped.
_S6B_BASELINE = (
    "4a35f3e428f409c1ca0c38892f1fc9ee245df33f0b0eede222275561fd499497"
)

#: Segment-2a diagonals, the same way (tessera#143).
_DIAGONALS_BASELINE = (
    "585a857c3c97b618c8d8921325d65c333a5274fc3f016cef119f884f878c5189"
)

#: How many of the fixture's 2048 positions carry a 4-bit release override.
#: A declared input, like the seed and the rungs: one release per eight
#: positions, enough that ``_canonical_release_order``'s descending-magnitude
#: order runs over a non-trivial prefix.  The fixture is 128 columns wide and a
#: superblock is 256, so it holds exactly one -- the quota's *spread* across
#: superblocks is the byte matrix's 512/640/320-column release rows to cover,
#: and this covers the placement and the plane.
FIXTURE_RELEASES = 256

#: The RELEASE plane, the same way (tessera#143).
_RELEASE_BASELINE = (
    "c4e32d4ead731314b1d22081a6a2aeef0a6e1da9671874e5c9c2a6c6812c7075"
)

#: The row extent the shard fixture cuts.  ``r0`` is non-zero on purpose: the
#: identity slice of a unit is that unit, byte for byte, and stores no state at
#: all -- a cut at row 0 would write no INITIAL_STATE plane and watch nothing.
#: Half the fixture's rows, on a boundary that is a whole number of scale
#: blocks, which is what a row cut requires.
FIXTURE_SHARD_ROWS = (FIXTURE_ROWS // 2, FIXTURE_ROWS)

#: Shards, the same way (tessera#143).
_SHARD_BASELINE = (
    "73f5ccd5cb4e5f450b365d17ea3e12bdb95192169fa6c38f12ca1aedb0a227fd"
)

#: The completion axis, the same way (tessera#143).
_COMPLETION_BASELINE = (
    "6eefb0383789d3532c957e9b6a828a8b3a7947686e5d2f6979e383d5520c76ca"
)

_DOMAIN = b"prismaquant.tessera.v1/encoder_fixture_id"

_LOCK = threading.RLock()
_MEMO: "list[bytes]" = []
_BUILDING = threading.local()


def _gaussian_stream(seed: int, count: int) -> "list[float]":
    """``count`` approximately-normal doubles, from Python's own uniform stream.

    ``random.Random.random`` is covered by CPython's compatibility promise and
    produces the same sequence on every platform and version; ``random.gauss``
    and ``torch.randn`` are not promised in the same way, and this identity
    must not move because a runtime upgraded its normal variate.  Twelve
    uniforms minus six is Irwin-Hall: mean 0, variance 1, and every operation
    is IEEE double arithmetic that is bit-identical everywhere.
    """
    rng = random.Random(seed)
    return [sum(rng.random() for _ in range(12)) - 6.0 for _ in range(count)]


def _fixture_weight():
    """The fixture matrix: a fixed Gaussian draw with planted row outliers."""
    import torch

    flat = _gaussian_stream(FIXTURE_SEED, FIXTURE_ROWS * FIXTURE_COLS)
    weight = torch.tensor(flat, dtype=torch.float32, device="cpu").reshape(
        FIXTURE_ROWS, FIXTURE_COLS
    )
    # Placed at a fixed column so the outlier is a property of the fixture and
    # not of the stream's tail, which a change to the stream would move; the
    # magnitude comes off its own uniform stream so no two raised rows share a
    # scale (see OUTLIER_SIGMAS).
    rows = list(range(0, FIXTURE_ROWS, OUTLIER_STRIDE))
    rng = random.Random(FIXTURE_SEED + 2)
    for row in rows:
        weight[row, 0] = OUTLIER_SIGMAS + OUTLIER_SPREAD * rng.random()
    return weight


def _unraised_boundary_fixture():
    """A synthetic E4M3 row in #115's half-ulp residual interval.

    The row is constructed from the fp16 lattice rather than searched for.  Its
    exact RMS scale sits three eighths of one fp16 ulp above 1, while
    ``amax / reach`` sits one quarter ulp above 1.  It is therefore unraised,
    yet round-to-nearest stores 1 and lands below the represented reach.  The
    other rows sit one eighth ulp above 1: they still store 1, while keeping
    the median above the binade boundary so ``channel_global`` is derived as 1.

    Returns ``(weight, sigma, reach)`` so the test can pin the inequalities the
    construction exists to exercise against the same owning arithmetic.
    """
    import torch

    from .alphabet import E4M3_GRID
    from .encode import grid_vector_table, window_table
    from .export import wire_recipe
    from .scale_channel import default_channel_sigma

    recipe = wire_recipe(E4M3_GRID, 1024)
    sigma = (
        float(default_channel_sigma(E4M3_GRID))
        if recipe.channel_sigma is None else float(recipe.channel_sigma)
    )
    table = window_table(
        E4M3_GRID,
        recipe.window_bits,
        sigma=sigma,
        seed=recipe.window_seed,
        half=16,
        device="cpu",
    )
    reach = float(
        grid_vector_table(E4M3_GRID, device="cpu")[table.long()].abs().max()
    )

    lower = torch.tensor(1.0, dtype=torch.float16).float()
    upper = torch.nextafter(
        lower.to(torch.float16),
        torch.tensor(float("inf"), dtype=torch.float16),
    ).float()
    ulp = upper - lower
    floor_scale = lower + ulp / 4
    rms_scale = lower + 3 * ulp / 8
    peak = float(reach * floor_scale)
    target_rms = float(sigma * rms_scale)
    rest = math.sqrt(
        (FIXTURE_COLS * target_rms * target_rms - peak * peak)
        / (FIXTURE_COLS - 1)
    )

    ordinary_scale = lower + ulp / 8
    weight = torch.full(
        (FIXTURE_ROWS, FIXTURE_COLS), float(sigma * ordinary_scale),
        dtype=torch.float32, device="cpu",
    )
    weight[:, 1::2].neg_()
    weight[0].fill_(rest)
    weight[0, 1::2].neg_()
    weight[0, 0] = peak
    return weight, sigma, reach


def _fixture_hessian():
    """A deterministic PSD Hessian for the fixture's columns.

    ``A^T A`` over a fixed 4x-tall draw, so the matrix is a genuine second-
    moment matrix -- full rank, positive definite, with real off-diagonal
    structure -- rather than a diagonal that would leave LDLQ nothing to
    compensate.  It is *not* a real capture, and the module docstring says
    which condition that costs.
    """
    import torch

    tall = FIXTURE_ROWS * 4
    flat = _gaussian_stream(FIXTURE_SEED + 1, tall * FIXTURE_COLS)
    a = torch.tensor(flat, dtype=torch.float32, device="cpu").reshape(
        tall, FIXTURE_COLS
    )
    return (a.T @ a) / tall


@dataclass(frozen=True)
class Fixture:
    """One fixture encode: a shipping structure at a declared rung."""

    label: str
    grid_name: str
    q256: int
    #: ``True`` when the case runs through ``ActivationSource`` -- the
    #: exporter's activation-aware recipe (LDLQ, the metric refits, the reach
    #: floor), which is where most recent byte movers live.
    activation: bool = False
    #: This fixture was added after the identity shipped.  Matching the
    #: recorded contribution is neutral; a mismatch contributes and moves it.
    compatibility_baseline: "str | None" = None
    #: Use the deliberately constructed #115 boundary weights instead of the
    #: general mixed raised/unraised fixture.
    unraised_boundary: bool = False
    #: ``None`` tracks the exporter default.  The boundary witness holds the
    #: initial plane fixed at zero refits so the exact branch reaches bytes.
    scale_refit: "int | None" = None
    #: Extra ``encode_linear`` keywords, for a surface the encoder produces but
    #: no recipe selects -- ``scale_plane=S6B``, ``with_diagonals=True``, a
    #: completion depth.  A fixture that names one is deliberately *not* the
    #: wire for its ``(grid, body, scale plane)``, so it does not count towards
    #: the coverage rule (:attr:`covers_wire`); it exists to reach a plane the
    #: wire never writes.  Spelled unwritable, so the shared class default
    #: cannot be mutated by a caller that thinks it holds its own dict.
    encode: "MappingProxyType" = MappingProxyType({})
    #: Positions to release, or ``None`` for no release at all.  ``encode_linear``
    #: has no ``released_positions`` keyword -- the exporter cannot write a
    #: released unit -- so this one case assembles the exporter's own
    #: ``encode_unit`` call instead, and a test pins that assembly against
    #: ``encode_linear`` at zero releases so it cannot drift from it.
    released_positions: "int | None" = None
    #: ``(r0, r1)``: encode the whole unit, then cut this row extent out of the
    #: bytes with ``slicing.slice_unit``, the way a rank does at load.  A second
    #: byte-producing path, and the only one that writes INITIAL_STATE.
    shard_rows: "tuple[int, int] | None" = None

    @property
    def grid(self):
        from .alphabet import SERIALISABLE_GRIDS

        for grid in SERIALISABLE_GRIDS.values():
            if grid.name == self.grid_name:
                return grid
        raise KeyError(
            f"fixture {self.label!r} names grid {self.grid_name!r}, which is "
            f"not in SERIALISABLE_GRIDS"
        )

    @property
    def covers_wire(self) -> bool:
        """Whether this fixture encodes its structure *as the recipe writes it*.

        Derived from the fields, never declared beside the case: a fixture that
        overrides the encode departs from the wire ``wire_recipe`` resolves, so
        counting it as coverage of that structure would let the coverage rule
        pass on a fixture that writes a different set of planes.  An S6b case
        on the E2M1 wire covers SCALE_BASE and covers ``(E2M1, TCQ, LUT16)``
        not at all.
        """
        return (
            not self.encode
            and self.released_positions is None
            and self.shard_rows is None
        )

    @property
    def structure(self) -> tuple:
        """The ``(grid, body, scale plane)`` key this fixture covers."""
        from .export import wire_recipe

        recipe = wire_recipe(self.grid, self.q256)
        return (self.grid_name, recipe.body, recipe.scale_plane)


def fixtures() -> "tuple[Fixture, ...]":
    """The fixture set.

    One weight-only case per shipping ``(grid, body, scale plane)`` structure,
    plus one activation-aware case per *scale plane*: the plane is what decides
    which refit runs, and running both planes' refits is what makes a change to
    either move this digest. The final E4M3 case is issue #116's
    baseline-neutral witness for the unraised half-ulp reach boundary inside an
    already-covered structure.

    The cases after those reach a surface the *wire* never selects: a plane
    only a caller-facing override writes, or a second byte-producing path.
    Each carries a ``compatibility_baseline`` for the reason the #116 witness
    does -- the surface is not new, only newly watched, so covering it must not
    re-label bytes already written by unchanged behaviour.

    The rungs are declared inputs.  E4M3 at 1024 and E2M1x2 at 768 are the
    rates those wires ship at; E2M1x2 at 896 is its coset-trellis cap (the
    boundary ``wire_recipe`` switches bodies at); E2M1 and BF16 sit at 768 and
    1024, inside the single range each grid's structure covers.
    """
    from .manifest import ScalePlaneKind

    return (
        Fixture("e4m3-1024/window-channel", "E4M3", 1024),
        Fixture("bf16-1024/window-channel", "BF16", 1024),
        Fixture("e2m1x2-768/window-lut", "E2M1x2", 768),
        Fixture("e2m1x2-896/tcq-lut", "E2M1x2", 896),
        Fixture("e2m1-768/tcq-lut", "E2M1", 768),
        Fixture("e4m3-1024/channel+activation", "E4M3", 1024, activation=True),
        Fixture("e2m1x2-768/lut+activation", "E2M1x2", 768, activation=True),
        Fixture(
            "e4m3-1024/window-channel-unraised-boundary",
            "E4M3",
            1024,
            compatibility_baseline=_UNRAISED_BOUNDARY_BASELINE,
            unraised_boundary=True,
            scale_refit=0,
        ),
        # The S6b scale plane (tessera#143).  ``wire_recipe`` selects it
        # nowhere, so no case above writes SCALE_BASE at all -- and yet
        # ``encode_linear_planes(scale_plane=...)`` is a caller-facing override
        # and ``unit_artifact._read_scale_planes`` decodes what it writes, so a
        # change to ``encode._pack_scales`` or ``encode._refit_scales`` moved
        # real bytes at an unmoved identity.  It rides the rung the E2M1 case
        # above already builds a plan for, so it costs one encode and no
        # forest.
        Fixture(
            "e2m1-768/s6b", "E2M1", 768,
            encode=MappingProxyType({"scale_plane": ScalePlaneKind.S6B}),
            compatibility_baseline=_S6B_BASELINE,
        ),
        # Segment-2a diagonals (tessera#143).  ``with_diagonals=`` is the same
        # kind of caller-facing override, and ``diagonals.fit_diagonals`` is a
        # different producer of the DIAG planes than the CHANNEL row scale the
        # E4M3 and BF16 cases fill DIAG_SV with -- so the fit itself, and the
        # fp16 words it stores, moved bytes nothing here could see.  Spelled on
        # a block-plane grid because a CHANNEL plane refuses segment 2a: its
        # row scale *is* the DIAG_SV field.
        Fixture(
            "e2m1-768/diagonals", "E2M1", 768,
            encode=MappingProxyType({"with_diagonals": True}),
            compatibility_baseline=_DIAGONALS_BASELINE,
        ),
        # The completion axis (tessera#143).  Every case above spends the
        # exporter's default of zero completion bits, and so does every rung
        # sitting at its body cap -- E2M1 at 768 and E2M1x2 at 896 both have
        # ``cap - R == 0`` -- so ``encode._completion_choice`` was offered one
        # descendant and could not choose wrongly.  This is the one case at a
        # rung with headroom: 256 leaves two bits of it, which is what puts
        # bytes on the COMPLETION plane at all.  The rung is the declared input
        # here, exactly as it is for every case above.
        Fixture(
            "e2m1-256/completion", "E2M1", 256,
            encode=MappingProxyType({"completion": 2}),
            compatibility_baseline=_COMPLETION_BASELINE,
        ),
        # The RELEASE plane (tessera#143).  ``export.encode_linear`` has no
        # ``released_positions`` keyword at all, so no case above can carry a
        # release and the placement rule in ``encode._canonical_release_order``
        # moved nothing here.  It rides the E2M1 rung the cases above already
        # plan for -- ``tcq_cap_q256(E2M1)`` is that same 768 -- so it costs one
        # encode and no forest.
        Fixture(
            "e2m1-768/release", "E2M1", 768,
            released_positions=FIXTURE_RELEASES,
            compatibility_baseline=_RELEASE_BASELINE,
        ),
        # Shards (tessera#143).  ``slicing.slice_unit`` is a second
        # byte-producing path and nothing an encode alone produces: the
        # INITIAL_STATE plane, ``planes.SHARD_PLANE_ORDER``'s wire order and
        # the tenth ``plane_elements`` entry are written here and nowhere else,
        # so a change to the state replay or to the shard layout moved real
        # bytes at an unmoved identity.  Its parent is the ``e2m1-768/tcq-lut``
        # encode, so the cost is a parse and a cut.
        Fixture(
            "e2m1-768/shard", "E2M1", 768,
            shard_rows=FIXTURE_SHARD_ROWS,
            compatibility_baseline=_SHARD_BASELINE,
        ),
    )


def shipping_structures() -> "frozenset[tuple]":
    """Every ``(grid, body, scale plane)`` an export can write.

    Derived from ``recipe_table`` over ``SERIALISABLE_GRIDS`` -- the two
    modules that own "which grids may be written" and "which wire each rung
    gets" -- so adding a grid, a body or a plane changes this set, and the
    coverage test fails until a fixture covers it.  A restated list would pass
    on the day the list is wrong.
    """
    from .alphabet import SERIALISABLE_GRIDS
    from .export import recipe_table

    return frozenset(
        (grid.name, row.recipe.body, row.recipe.scale_plane)
        for grid in SERIALISABLE_GRIDS.values()
        for row in recipe_table(grid)
    )


def building() -> bool:
    """True while :func:`encoder_fixture_id` is encoding its fixtures.

    The fixture build is the one path that must stamp no identity: the digest
    is not known while it is being computed, and a manifest that carried it
    would make the hash a function of itself.  ``build_unit_artifact`` asks
    :func:`stamped_fixture_id`, which reads this.
    """
    return getattr(_BUILDING, "active", False)


def stamped_fixture_id() -> "bytes | None":
    """The identity to stamp into a manifest, or ``None`` inside a fixture build."""
    if building():
        return None
    return encoder_fixture_id()


def _fixture_blob(case: Fixture) -> bytes:
    """The artifact bytes one fixture encodes to.

    Goes through ``encode_linear``, so what is hashed downstream is the encoder
    an export calls and not a re-implementation of it.  Separated from
    :func:`_encode_fixture` so a test can read the planes the case actually
    wrote off its own manifest, instead of a case declaring which plane it
    covers beside itself.
    """
    from .export import ActivationSource, encode_linear, wire_recipe
    from .manifest import ScalePlaneKind

    if case.shard_rows is not None:
        return _shard_blob(case)
    if case.released_positions is not None:
        return _release_blob(case)
    weight = _unraised_boundary_fixture()[0] if case.unraised_boundary \
        else _fixture_weight()
    kwargs: dict = {}
    if case.scale_refit is not None:
        kwargs["scale_refit"] = case.scale_refit
    if case.activation:
        plane = wire_recipe(case.grid, case.q256).scale_plane
        # Named by naming nothing: a case that spelled ``ldlq_sigma=1.0``
        # would stop tracking ``DEFAULT_LDLQ_SIGMA`` the moment that default
        # moved, and a moved default is a byte change this identity exists to
        # carry.  ``refit_reach_floor`` is the one setting stated, because it
        # is off by default and is the only arm ``land_at_least`` is reachable
        # from -- and it is stated only on the CHANNEL plane, which is the one
        # plane it means anything on (``encode_unit`` refuses it elsewhere).
        source = ActivationSource(
            {"fixture": _fixture_hessian()},
            {"text_sha256": "fixture", "fit_tokens": 0, "fit_ids_sha256": "fixture"},
            refit_reach_floor=plane is ScalePlaneKind.CHANNEL,
        )
        kwargs = source.for_unit("fixture.weight", FIXTURE_COLS, scale_plane=plane)
    exported = encode_linear(
        weight,
        grid=case.grid,
        q256=case.q256,
        name=case.label,
        # The exporter's weighting, not ``encode_unit``'s signature default:
        # this identity is about the encoder an export runs.
        trellis_weighting="scale",
        **kwargs,
        **case.encode,
    )
    return exported.blob


def _release_blob(case: Fixture) -> bytes:
    """A released unit's bytes: the one encode ``encode_linear`` cannot express.

    ``encode_linear_planes`` exposes no ``released_positions``, so this
    assembles the call it would make -- the same recipe, the same plan, the
    same exporter defaults, read off ``export`` rather than restated -- and
    adds the release.  ``test_the_release_fixture_is_the_exporters_own_call``
    pins the assembly by encoding it at zero releases and comparing to
    ``encode_linear`` byte for byte, so a new exporter default cannot leave
    this call behind silently.
    """
    from .encode import encode_unit
    from .export import (
        DEFAULT_CODE, DEFAULT_GROUP, DEFAULT_HALF, DEFAULT_SCALE_REFIT,
        _plan_for, wire_recipe,
    )
    from .manifest import ScalePlaneKind
    from .unit_artifact import build_unit_artifact

    recipe = wire_recipe(case.grid, case.q256)
    sigma = (
        recipe.channel_sigma
        if recipe.scale_plane is ScalePlaneKind.CHANNEL else None
    )
    rates, forests = _plan_for(
        case.grid, case.q256, FIXTURE_COLS, recipe.body, sigma
    )
    unit = encode_unit(
        _fixture_weight(), forests, rates, DEFAULT_CODE,
        completion=0, released_positions=case.released_positions,
        group=DEFAULT_GROUP, half=DEFAULT_HALF,
        scale_refit=DEFAULT_SCALE_REFIT, span=recipe.span,
        scale_plane=recipe.scale_plane,
        # The exporter's weighting, for the reason ``_fixture_blob`` states.
        trellis_weighting="scale",
        body=recipe.body, window_bits=recipe.window_bits,
        window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
        channel_sigma=recipe.channel_sigma,
    )
    _manifest, _region, blob = build_unit_artifact(
        unit, case.label, forests, case.q256 * case.grid.arity, DEFAULT_CODE,
    )
    return blob


def _shard_blob(case: Fixture) -> bytes:
    """A shard's bytes: the second byte-producing path, cut from the first.

    The parent is the ordinary encode of the same case, so the identity binds
    the cut and not a second copy of the encode -- and the cut is made from the
    parent's *bytes*, through ``parse_unit_artifact``, exactly as a rank does
    at load rather than from the encoder object that happens to be in this
    process.
    """
    from .slicing import slice_unit
    from .unit_artifact import build_unit_artifact, parse_unit_artifact

    parent = _fixture_blob(replace(case, shard_rows=None))
    parsed = parse_unit_artifact(parent)
    shard = slice_unit(parsed, rows=case.shard_rows)
    _manifest, _region, blob = build_unit_artifact(
        shard, case.label, parsed.forests,
        case.q256 * case.grid.arity, parsed.code,
    )
    return blob


def _encode_fixture(case: Fixture) -> bytes:
    """One fixture's contribution: its payload digest and terminal records.

    The bytes are read back off the parsed manifest rather than off the blob --
    the blob carries the container framing, which is not this identity's to
    bind.
    """
    from .canonical import Writer
    from .container import parse

    manifest = parse(_fixture_blob(case)).manifest
    writer = Writer()
    writer.text(case.label).digest32(manifest.payload_digest)
    writer.uint(len(manifest.terminals))
    for terminal in manifest.terminals:
        terminal.encode(writer)
    return writer.bytes


def _identity_contribution(case: Fixture) -> bytes:
    """Encoded fixture bytes, neutral only at a recorded compatibility point."""
    encoded = _encode_fixture(case)
    if case.compatibility_baseline is not None:
        if hashlib.sha256(encoded).hexdigest() == case.compatibility_baseline:
            return b""
    return encoded


def fixture_digests() -> "dict[str, str]":
    """Per-fixture hex digests, for a reader diagnosing *which* case moved.

    The identity itself is :func:`encoder_fixture_id`; this is the same work
    reported case by case, so a bisect can say "the LUT plane moved and the
    CHANNEL plane did not" instead of only "the encoder moved".
    """
    with _fixture_build():
        return {
            case.label: hashlib.sha256(_encode_fixture(case)).hexdigest()
            for case in fixtures()
        }


class _fixture_build:
    """Marks the fixture encodes so they stamp no identity, and forbids nesting.

    Re-entry would mean a call path routed the fixture build back through the
    stamping path -- the recursion this guard exists to make impossible -- so
    it raises rather than quietly returning ``None`` twice.
    """

    def __enter__(self):
        if building():
            raise RuntimeError(
                "the encoder fixture build re-entered itself: something asked "
                "for the encoder identity while it was being computed"
            )
        _BUILDING.active = True
        return self

    def __exit__(self, *exc):
        _BUILDING.active = False
        return False


def resumable(manifest) -> bool:
    """Whether a cached unit may be reused by *this* encoder.

    One home for the rule, so a resume, a merge guard and an A/B cannot
    disagree about what "the same encoder" means: a manifest carrying no
    identity was cut by :data:`UNTAGGED_ENCODER_ID`, and anything else must
    match what this process computes.  A cache entry that fails this was
    written by a different encoder and re-encoding it is the only safe move.

    ``cached_unit`` calls this before accepting an exact campaign blob. Its
    receipt additionally binds the source, actual calibration Hessian, complete
    activation settings and producer source seal; numerical compatibility alone
    does not identify the inputs that produced a cached unit.
    """
    stamped = getattr(manifest, "encoder_fixture_id", None)
    return (stamped or UNTAGGED_ENCODER_ID) == encoder_fixture_id()


def encoder_fixture_id() -> bytes:
    """The encoder's behavioural identity: 32 bytes, memoised per process.

    Equal ids mean the encoder produced identical bytes for every fixture, at
    fixed arguments.  Unequal ids mean it did not, and the parts, the resumed
    cache entries or the A/B arms carrying them are two artifacts, not one.

    The build reads the exporter's live defaults -- ``export.wire_recipe`` and
    the encode defaults, at build time, by design, so a moved default moves
    the identity -- and the memo is process-wide.  The corollary binds a test
    that patches one of those defaults: resolve this identity BEFORE patching,
    or the first encode under the patch builds the fixtures at the foreign
    recipe and memoises that encoder for every later encode in the process
    (``tests/test_recipe_table.py`` is the shape of it).
    """
    with _LOCK:
        if _MEMO:
            return _MEMO[0]
        with _fixture_build():
            payload = b"".join(_identity_contribution(case) for case in fixtures())
        _MEMO.append(hashlib.sha256(_DOMAIN + payload).digest())
        return _MEMO[0]
