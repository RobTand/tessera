"""Fail-closed error taxonomy.

Every rejection path in this package raises one of these. The wire contract is
fail-closed: a stream that is not provably well-formed is rejected, never
best-effort parsed (doc S9, S14 validator doctrine).
"""


class TesseraError(Exception):
    """Base class for every Tessera rejection."""


class SchemaError(TesseraError):
    """Header, magic, or schema-version rejection."""


class CanonicalEncodingError(TesseraError):
    """Manifest bytes are not in canonical form (non-minimal varint, trailing
    bytes, out-of-order fields, over-long field)."""


class ManifestError(TesseraError):
    """Manifest is structurally invalid or internally inconsistent."""


class PlaneLayoutError(TesseraError):
    """Plane table violates offset/extent/alignment/contiguity invariants."""


class TruncationError(TesseraError):
    """Byte length does not correspond to any declared terminal.

    Per-superblock quota-boundary truncations within a plane are legal and
    enumerate their own terminal_id; arbitrary interleaved byte-prefixes are
    not terminals (doc S9).

    What that means on an *encoded* artifact today: ``build_unit_artifact``
    writes ``terminals=(terminal,)`` -- one rung, the T-nvfp4 class -- so a
    unit has exactly one legal length and this refuses short or corrupt bytes.
    It never selects a shorter reading, because there is no shorter reading to
    select.  The multi-terminal ladder is a capability of the layout and
    container layers (``layout.build_terminal``, and the terminal match in
    ``container.parse``), exercised by artifacts the tests lay out directly;
    no encoder in this tree produces one.  See tessera#144.
    """


class FootprintDisagreementError(TesseraError):
    """Recomputed exact bytes disagree with the declared or physical bytes.

    The exact-byte authority is the accountant; any disagreement is a defect,
    not a rounding difference (doc S6: provisional arithmetic is admissible for
    headers and padding, never for body bits).
    """


class ScaleCodecError(TesseraError):
    """Base/refinement word is not a legal scale codec word (doc S6b)."""


class GrammarError(TesseraError):
    """Rate schedule, completion level, or descendant map violates the
    refinement grammar (doc S6)."""


class IdentityError(TesseraError):
    """Descriptor/record identity rejection.

    Raised where a human-readable family descriptor is supplied but a concrete
    normative terminal record is required (doc S9, round-7 P1-7).
    """


class ProvenanceError(TesseraError):
    """Input closure failed the ancestry or denylist check (doc S14 ledger
    invariant, build item 11)."""


class ControlNotByteMatchedError(TesseraError):
    """A rate-axis candidate and its uniform control do not weigh the same.

    Two arms labelled "4.0 bpp" that differ by a percent in bytes are not a
    control; the comparison silently prices the difference as quality.  Raised
    by :func:`tessera.control.assert_byte_matched`.
    """


class PromotionRefusedError(TesseraError):
    """A per-plane promotion cleared no bar this gate measures.

    A winning geomean with a losing per-unit record, a served number for an
    arm other than the one promoted, a GLM cross-check above its gate, or a
    served KL that misses its bar each refuse here, by name, with the reason.
    Raised by :func:`tessera.control.assert_plane_promotion` (tessera#65).
    """
