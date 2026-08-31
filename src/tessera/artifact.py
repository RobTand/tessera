"""Full-layout artifact construction at production geometry.

Build item 1b owes a *serializer*, and until now it had only been exercised on
an 8x32 toy geometry inside the tests.  Arm 4b -- the full-layout encoded-read
skeleton, which §16 places immediately after 1b and before any reader, decoder
or kernel work -- needs real artifacts at real shapes to read.  This module is
what produces them.

What it is **not**: an encoder.  Payload bytes here are deterministic filler,
not quantised weights.  Arm 2's minimal measurement encoder is the first gated
ask (§16), and nothing in this file chooses a code, a scale, or a rounding.
What is real is the *layout*: plane extents, superblock granularity, the
terminal ladder, and the exact byte and bpp arithmetic the accountant checks.

The standard terminal ladder mirrors the terminal classes of §6:

``t-po2``
    Body plus the power-of-two scale base.  The floor of the ladder.
``t-c3``
    Adds Stage-C completion to full depth (``c = 3 - R`` per column), which
    §6's partition property makes exactly 3 bits per column from every root.
``t-nvfp4``
    Adds the segment-2b scale refinement, the segment-2a diagonals, and any
    Stage-B released positions -- the T-nvfp4-class terminal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction

from .container import serialize
from .errors import GrammarError
from .grammar import bresenham_rate_schedule, root_from_q256, superblock_quota_ok
from .layout import TerminalSpec, build_plane_region, build_planes, build_terminal
from .manifest import (
    ArrangementMode,
    BranchIdentity,
    ContainerClass,
    Geometry,
    Manifest,
    RotationState,
)
from .planes import PlaneKind

__all__ = [
    "ArtifactPlan",
    "standard_terminal_specs",
    "build_artifact",
    "release_for_bpp",
    "bpp_quantum",
    "GLM_EXPERT_GATE",
    "GLM_EXPERT_DOWN",
]

#: The two GLM-5.3-Flash routed-expert projections, measured from the shipped
#: checkpoint: 288 routed experts per layer, ``hidden_size`` 4096 and
#: ``moe_intermediate_size`` 2048.  Together they carry the overwhelming
#: majority of that model's quantisable parameters, which is why they are the
#: shapes a layout claim has to hold at.
GLM_EXPERT_GATE = (2048, 4096)  # (rows = out, columns = in)
GLM_EXPERT_DOWN = (4096, 2048)


@dataclass(frozen=True)
class ArtifactPlan:
    """Everything a full-layout artifact needs, with nothing inferred."""

    unit_id: str
    rows: int
    columns: int
    q256: int
    superblock_columns: int = 256
    group_weights: int = 32
    half_weights: int = 16
    released_positions: int = 0
    alphabet_bytes: int = 16
    descendant_bytes: int = 32
    alignment_bytes: int = 1
    rotation: RotationState = RotationState.NONE
    container: ContainerClass = ContainerClass.GRIDBOOK

    def geometry(self) -> Geometry:
        return Geometry(
            rows=self.rows,
            columns=self.columns,
            superblock_columns=self.superblock_columns,
            group_weights=self.group_weights,
            half_weights=self.half_weights,
            quantizable_params=self.rows * self.columns,
        )

    def rates(self) -> tuple[int, ...]:
        return bresenham_rate_schedule(root_from_q256(self.q256), self.columns)


def standard_terminal_specs(
    rates: tuple[int, ...], released_positions: int = 0
) -> tuple[TerminalSpec, ...]:
    """The §6 terminal ladder: T-po2, T-C3, T-nvfp4-class.

    At r0 = 3.0 every column is already R=3, so ``c = 3 - R = 0`` and Stage C
    has nothing to add: T-C3 *is* T-po2, byte for byte.  §6 names this case --
    "Sub-mode 'B at r0=3' unifies as R=3, c=0, release-only" -- so the ladder
    there is two rungs, not three.  Emitting both would declare two terminals
    at one byte length, which the manifest rightly refuses: a truncation length
    must identify exactly one terminal.
    """
    c_full = tuple(3 - rate for rate in rates)
    release_only = not any(c_full)
    ladder = [TerminalSpec("t-po2", (0,) * len(rates), with_scale_base=True)]
    if not release_only:
        ladder.append(TerminalSpec("t-c3", c_full, with_scale_base=True))
    return tuple(ladder) + (
        TerminalSpec(
            "t-nvfp4",
            c_full,
            released_positions=released_positions,
            with_scale_base=True,
            with_scale_refine=True,
            with_diagonals=True,
        ),
    )


def _filler(kind: PlaneKind, bits: int) -> bytes:
    """Deterministic, content-addressed filler -- explicitly not encoded data.

    Derived from the plane kind so two planes never share bytes by accident,
    and reproducible so an artifact is a pure function of its plan.

    ``bits`` is the plane's *information* width, not its byte width.  A plane
    whose element count is not a whole number of bytes (a 4-bit RELEASE plane
    with an odd released count, say) leaves slack in its final content byte;
    MSB-first packing puts that slack in the low bits, and it must be zero or
    the same logical content admits many byte strings.  ``verify_plane_region``
    enforces this, so filler that ignored it would be rejected -- correctly.
    """
    length = (bits + 7) // 8
    if length <= 0:
        return b""
    seed = hashlib.sha256(f"tessera.filler.{kind.name}".encode()).digest()
    out = bytearray()
    block = seed
    while len(out) < length:
        out += block
        block = hashlib.sha256(block).digest()
    out = out[:length]
    slack = (-bits) % 8
    if slack:
        out[-1] &= (0xFF << slack) & 0xFF
    return bytes(out)


def build_artifact(plan: ArtifactPlan, encoder_profile: bytes | None = None):
    """Build one full-layout artifact.  Returns ``(manifest, region, blob)``."""
    geometry = plan.geometry()
    rates = plan.rates()
    if plan.columns % plan.superblock_columns:
        raise GrammarError(
            f"columns {plan.columns} is not a whole number of "
            f"{plan.superblock_columns}-column superblocks"
        )
    if not superblock_quota_ok(rates, plan.superblock_columns, root_from_q256(plan.q256)):
        raise GrammarError(
            f"q256={plan.q256} does not admit an exact quota over "
            f"{plan.superblock_columns}-column superblocks"
        )

    alphabet = _filler(PlaneKind.ALPHABET, plan.alphabet_bytes * 8)
    descendant = _filler(PlaneKind.DESCENDANT, plan.descendant_bytes * 8)

    # Two passes: the descriptors fix each plane's extent, and the extents fix
    # how many payload bytes each plane needs.
    probe = build_planes(
        geometry,
        rates,
        alphabet,
        descendant,
        alignment_bytes=plan.alignment_bytes,
        max_released=plan.released_positions,
    )
    payloads = {
        d.kind: _filler(d.kind, d.element_count * d.element_bits) for d in probe
    }
    payloads[PlaneKind.ALPHABET] = alphabet
    payloads[PlaneKind.DESCENDANT] = descendant

    planes = build_planes(
        geometry,
        rates,
        alphabet,
        descendant,
        alignment_bytes=plan.alignment_bytes,
        max_released=plan.released_positions,
        payloads=payloads,
    )
    region = build_plane_region(planes, payloads)
    terminals = tuple(
        build_terminal(
            geometry,
            rates,
            spec,
            planes,
            len(alphabet),
            len(descendant),
            plane_region=region,
        )
        for spec in standard_terminal_specs(rates, plan.released_positions)
    )

    manifest = Manifest(
        encoder_profile_id=encoder_profile
        or hashlib.sha256(b"tessera.layout-only.no-encoder").digest(),
        branch=BranchIdentity(
            unit_id=plan.unit_id,
            root_q256=plan.q256,
            rotation=plan.rotation,
            container=plan.container,
        ),
        geometry=geometry,
        arrangement=ArrangementMode.BRESENHAM,
        rates=rates,
        planes=planes,
        terminals=terminals,
        payload_digest=hashlib.sha256(region).digest(),
    )
    return manifest, region, serialize(manifest, region)


def terminal_table(manifest, region_len: int) -> list[tuple[str, int, Fraction, float]]:
    """`(slot, bytes, exact bpp, bpp as float)` for every declared terminal."""
    return [
        (t.slot_id, t.exact_bytes, t.exact_bpp, float(t.exact_bpp))
        for t in manifest.terminals
    ]


def release_for_bpp(
    plan: ArtifactPlan, target_bpp: Fraction, slot_id: str = "t-nvfp4"
) -> tuple[int, Fraction]:
    """Largest released-position count whose terminal stays within ``target_bpp``.

    §6 gives T-nvfp4-class the rate ``3.0 + 4*eps_B``, so bpp is monotone in the
    released count and a bisection is exact.  Returns ``(released, exact_bpp)``.

    This is how a matched-bpp comparison is set up.  A rate claim against
    another format is only a claim if both sides sit at the *same* bpp, and the
    release dial is fine enough to land on any of them: the RELEASE plane
    stores 4 bits per position, so two released positions buy one byte and the
    bpp quantum is ``8 / quantizable_params`` -- about 1e-6 bpp on a GLM
    expert.  There is no rounding to a nearest rung to disclose.
    """
    positions = plan.rows * plan.columns
    lo, hi, best = 0, positions, None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = replace(plan, released_positions=mid)
        manifest, _, _ = build_artifact(candidate)
        terminal = next(t for t in manifest.terminals if t.slot_id == slot_id)
        if terminal.exact_bpp <= target_bpp:
            best = (mid, terminal.exact_bpp)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise GrammarError(
            f"{slot_id} cannot reach {float(target_bpp):.6f} bpp at this "
            "geometry: even zero release exceeds it"
        )
    return best


def bpp_quantum(plan: ArtifactPlan) -> Fraction:
    """The finest bpp step the release dial can take: one byte over the unit."""
    return Fraction(8, plan.rows * plan.columns)
