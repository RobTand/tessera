"""The pure wire-arithmetic calculator (build item 11, round-8 P0-1.3).

Round-8 P0-1.3 requires the legacy-plane wire arithmetic to be **re-derived in
a pure calculator artifact before any quotation**.  This module is that
artifact, and it enforces the distinction the document is strict about:

* **DERIVED** -- computed here, now, from exact integer byte counts. Quotable.
* **CITED** -- a number measured elsewhere, carried with its evidence tier.
  Reproduced verbatim with its provenance and *never* presented as derived.

:func:`figure_table` returns both, tagged.  :func:`assert_derivation_matches`
is the honest join: it checks a derived figure against a cited one and reports
agreement, without ever letting the citation stand in for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from .grammar import C_FULL_BITS, bresenham_rate_schedule, root_from_q256
from .layout import TerminalSpec, build_planes, build_terminal
from .manifest import Geometry

__all__ = [
    "Provenance",
    "Figure",
    "EvidenceTier",
    "CITED_FIGURES",
    "plane_rate",
    "terminal_rate",
    "figure_table",
    "assert_derivation_matches",
]


class Provenance(Enum):
    DERIVED = "derived"
    CITED = "cited"


class EvidenceTier(Enum):
    """The document's evidence tiers, carried with every cited figure."""

    IN_TREE = "in-tree"
    BRANCH = "branch"
    BRANCH_PRELIMINARY = "branch-preliminary"
    DERIVED = "derived"
    ARITHMETIC = "arithmetic"
    PROJECTED = "projected"
    CONJECTURE = "conjecture"


@dataclass(frozen=True)
class Figure:
    """One number with its provenance attached, always."""

    name: str
    value: Fraction
    provenance: Provenance
    tier: EvidenceTier
    note: str = ""

    def quotable_as_derived(self) -> Fraction:
        if self.provenance is not Provenance.DERIVED:
            raise ValueError(
                f"{self.name} is {self.provenance.value} at tier "
                f"{self.tier.value}; it may not be quoted as derived arithmetic"
            )
        return self.value


#: Figures the document reports from measurement elsewhere. Reproduced with
#: their tiers so they can be compared against a derivation, never substituted
#: for one. Every one of these is a *citation*, not a result of this package.
CITED_FIGURES: tuple[Figure, ...] = (
    Figure(
        "exact_wire_bpp_q256_512",
        Fraction(25008, 10000),
        Provenance.CITED,
        EvidenceTier.DERIVED,
        "doc S13: 'b = 2.5008 current exact wire' at 4096^2, q256=512",
    ),
    Figure(
        "projected_body_po2_q256_512",
        Fraction(225, 100),
        Provenance.CITED,
        EvidenceTier.PROJECTED,
        "doc S13: 'b = 2.25 projected body+po2'",
    ),
    Figure(
        "body_only_q256_512",
        Fraction(2),
        Provenance.CITED,
        EvidenceTier.DERIVED,
        "doc S13: 'b = 2.0 body-only'",
    ),
    Figure(
        "projected_po2_floor_r0_1",
        Fraction(125, 100),
        Provenance.CITED,
        EvidenceTier.PROJECTED,
        "doc S8/S6: '~1.25 bpp projected floor at r0=1.0' before exact side "
        "overhead",
    ),
    Figure(
        "shaped_rate_ceiling",
        Fraction(396875, 100000),
        Provenance.CITED,
        EvidenceTier.BRANCH,
        "doc S1/S15: structural ceiling 3.96875 -- measured on a branch "
        "artifact, NOT re-derivable here",
    ),
    Figure(
        "expanded_resident_single_pass",
        Fraction(45, 10),
        Provenance.CITED,
        EvidenceTier.BRANCH,
        "doc S7.4/S13: ~4.5 bpp expanded-resident, single pass",
    ),
)


def plane_rate(element_count: int, element_bits: int, quantizable_params: int) -> Fraction:
    """Exact bits per quantizable parameter for one plane."""
    return Fraction(element_count * element_bits, quantizable_params)


def terminal_rate(
    q256: int,
    rows: int,
    columns: int,
    *,
    with_scale_base: bool = True,
    with_scale_refine: bool = False,
    with_diagonals: bool = False,
    completion: int = 0,
    released_positions: int = 0,
    superblock_columns: int = 256,
    cap: int = C_FULL_BITS,
    arity: int = 1,
    span: int = 1,
) -> Fraction:
    """Exact payload bpp for a terminal, from integer byte counts only.

    ``span`` is the trellis super-symbol length (schema minor 1): the BODY
    plane holds ``span * R + span - 1`` bits per super-symbol per column.  A
    LUT scale plane is spelled ``with_scale_base=False, with_scale_refine=True``
    -- its table is manifest side bytes, outside the plane-region rate this
    function states.  The defaults are the minor-0 wire.

    ``cap`` is the family's rate cap -- ``payload_bits - 1``.  It defaults to
    ``C_FULL_BITS`` so every figure derived before families existed is
    reproduced exactly, but it is load-bearing for anything else: the
    completion capacity is ``cap - rate``, and the rate schedule itself cannot
    be built above the cap.  Left at 3, this function silently refused every
    TESSERA-8 rung above 3.0 body bits -- which is most of the ones an 8-bit
    family exists to reach.

    ``arity`` is load-bearing for the same reason on the other axis: BODY and
    COMPLETION hold one entry per *code*, and a code covers ``arity`` rows, so
    at arity 2 they are half the size this function would otherwise predict.
    A predicted bpp that disagrees with the built artifact's is the failure
    this parameter exists to prevent -- the accountant and the wire must agree
    byte for byte, and ``rows`` here is always weight rows.
    """
    geometry = Geometry(
        rows=rows,
        columns=columns,
        superblock_columns=superblock_columns,
        group_weights=32,
        half_weights=16,
        quantizable_params=rows * columns,
    )
    rates = bresenham_rate_schedule(root_from_q256(q256), columns, cap)
    spec = TerminalSpec(
        slot_id="calc",
        completion_bits=tuple(
            min(completion, cap - rate) for rate in rates
        ),
        released_positions=released_positions,
        with_scale_base=with_scale_base,
        with_scale_refine=with_scale_refine,
        with_diagonals=with_diagonals,
    )
    # ``spec`` is what sizes the COMPLETION plane, so the layout is built with
    # it rather than without it: ``unit_artifact`` passes it and this is the
    # second implementation of the same schema, so the two should not differ in
    # what they hand the layout.
    #
    # It is a **no-op today**, verified rather than assumed: 520 configurations
    # (cap 3/7, arity 1/2, every rung at stride 64, completion 0/1/2/full, with
    # and without the refinement plane) are bit-identical with and without it.
    # ``build_terminal`` recomputes the exact byte count from ``spec``, so the
    # plane extent never reaches the returned value.  Kept because a latent
    # divergence between the two accountants is exactly the bug class this
    # function keeps having, not because a rung was ever mispriced here.
    planes = build_planes(geometry, rates, b"", b"", cap=cap, arity=arity,
                          spec=spec, span=span)
    return build_terminal(
        geometry, rates, spec, planes, 0, 0, cap=cap, arity=arity, span=span
    ).exact_bpp


def figure_table(rows: int = 4096, columns: int = 4096) -> tuple[Figure, ...]:
    """Derive the wire figures this package is entitled to quote."""
    derived = (
        Figure(
            "body_only_q256_512",
            terminal_rate(512, rows, columns, with_scale_base=False),
            Provenance.DERIVED,
            EvidenceTier.ARITHMETIC,
            "segment 0 only, exact byte count",
        ),
        Figure(
            "body_po2_q256_512",
            terminal_rate(512, rows, columns),
            Provenance.DERIVED,
            EvidenceTier.ARITHMETIC,
            "body + E8M0 per-32 base",
        ),
        Figure(
            "body_e4m3_16_q256_512",
            terminal_rate(512, rows, columns, with_scale_refine=True),
            Provenance.DERIVED,
            EvidenceTier.ARITHMETIC,
            "body + full segment 2b (wire-rate parity with E4M3/16)",
        ),
        Figure(
            "po2_floor_r0_1",
            terminal_rate(256, rows, columns),
            Provenance.DERIVED,
            EvidenceTier.ARITHMETIC,
            "T-po2 floor at r0=1.0, before side overhead",
        ),
        Figure(
            "scale_plane_full",
            terminal_rate(512, rows, columns, with_scale_refine=True)
            - terminal_rate(512, rows, columns, with_scale_base=False),
            Provenance.DERIVED,
            EvidenceTier.ARITHMETIC,
            "segment 2b total: 16 bits per 32 weights",
        ),
        Figure(
            "diagonals_4096sq",
            terminal_rate(512, rows, columns, with_diagonals=True)
            - terminal_rate(512, rows, columns),
            Provenance.DERIVED,
            EvidenceTier.ARITHMETIC,
            "segment 2a: (n + k) fp16 channel diagonals",
        ),
    )
    return derived + CITED_FIGURES


def assert_derivation_matches(
    derived: Figure, cited: Figure, tolerance: Fraction = Fraction(0)
) -> bool:
    """Compare a derivation against a citation without conflating them."""
    if derived.provenance is not Provenance.DERIVED:
        raise ValueError(f"{derived.name} is not a derived figure")
    if cited.provenance is not Provenance.CITED:
        raise ValueError(f"{cited.name} is not a cited figure")
    return abs(derived.value - cited.value) <= tolerance
