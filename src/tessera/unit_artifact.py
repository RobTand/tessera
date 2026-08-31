"""Encoder -> bytes -> decoder.  The seam where 1a/1b and the encoder meet.

Everything else round-trips tensors, which proves self-consistency and not much
else.  A wire format's bugs live in bit order, per-superblock counts, sub-byte
padding, and the question of whether the reader can rebuild the encoder's
context from the manifest alone.  None of those are reachable from a
tensor-level test; all of them are reachable from here.

``read_unit_artifact`` takes **bytes and nothing else**.  If it needed the
encoder's forests, or its scale tensor, or its ConvCode passed alongside, then
the artifact would not be self-describing and the format would not be a format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

import torch

from .alphabet import alphabet_size, build_forest, AnchorForest
from .container import parse, serialize
from .decode import reconstruct_unit
from .encode import EncodedUnit
from .errors import GrammarError
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
from .trellis import ConvCode, _ODS_GENERATORS
from .wire import pack_body, pack_fp16, pack_uniform, unpack_body, unpack_fp16, unpack_uniform

__all__ = ["build_unit_artifact", "read_unit_artifact", "encoder_profile_id"]


def encoder_profile_id(code: ConvCode, rates: "tuple[int, ...]") -> bytes:
    """Digest the decisions a reader must reproduce exactly.

    The convolutional code's memory order and generators are **wire**: two
    encoders that disagree on them emit streams that do not decode to each
    other, silently, into plausible-looking weights.  ``trellis.py`` says they
    are "covered by the encoder profile id" -- this is the function that makes
    that sentence true rather than aspirational.
    """
    payload = "|".join(
        [
            "prismaquant.tessera.v1",
            f"conv:m={code.memory},g={','.join(oct(g) for g in code.generators)}",
            f"forest:build_forest/value-order-dyadic",
            f"rates:{','.join(str(r) for r in sorted(set(rates)))}",
        ]
    )
    return hashlib.sha256(payload.encode()).digest()


def _forest_planes(rates: "tuple[int, ...]", forests: "dict[int, AnchorForest]"):
    """ALPHABET and DESCENDANT, concatenated over the distinct rates present."""
    present = sorted(set(rates))
    alphabet = b"".join(forests[r].alphabet_plane() for r in present)
    descendant = b"".join(forests[r].descendant_plane() for r in present)
    return alphabet, descendant


def _read_forest_planes(rates: "tuple[int, ...]", alphabet: bytes, descendant: bytes):
    """Rebuild the forests from the two charged planes -- not by re-deriving them.

    §6 stores the alphabet and the descendant map *in the artifact* precisely so
    a decoder never has to reproduce the encoder's search.  Rebuilding them here
    by calling ``build_forest`` again would make the planes decorative and would
    hide any disagreement between what was encoded and what was written.
    """
    present = sorted(set(rates))
    out, a_off, d_off = {}, 0, 0
    for rate in present:
        n_anchors = alphabet_size(rate)
        depth = 1 << (3 - rate)
        a_off += n_anchors
        block = descendant[d_off : d_off + n_anchors * depth]
        d_off += n_anchors * depth
        blocks = tuple(
            tuple(block[i * depth : (i + 1) * depth]) for i in range(n_anchors)
        )
        out[rate] = AnchorForest(rate=rate, blocks=blocks)
    if a_off != len(alphabet) or d_off != len(descendant):
        raise GrammarError(
            f"forest planes hold {len(alphabet)}/{len(descendant)} bytes, the "
            f"schedule needs {a_off}/{d_off}"
        )
    return out


def build_unit_artifact(
    unit: EncodedUnit,
    unit_id: str,
    forests: "dict[int, AnchorForest]",
    q256: int,
    code: ConvCode = ConvCode(),
    superblock: int = 256,
    alignment_bytes: int = 1,
    container: ContainerClass = ContainerClass.GRIDBOOK,
):
    """Serialise one encoded Linear.  Returns ``(manifest, region, blob)``."""
    rows, cols = unit.body_bits.shape
    rates = unit.rates
    completion_widths = tuple(3 - r for r in rates)
    geometry = Geometry(
        rows=rows,
        columns=cols,
        superblock_columns=superblock,
        group_weights=unit.group,
        half_weights=unit.half,
        quantizable_params=rows * cols,
    )
    alphabet, descendant = _forest_planes(rates, forests)

    has_diagonals = unit.diagonals is not None
    payloads = {
        PlaneKind.ALPHABET: alphabet,
        PlaneKind.DESCENDANT: descendant,
        PlaneKind.BODY: pack_body(unit.body_bits, rates),
        PlaneKind.SCALE_BASE: pack_uniform(unit.scale_base, 8),
        PlaneKind.COMPLETION: pack_body(unit.completion_bits, completion_widths),
        PlaneKind.SCALE_REFINE: pack_uniform(unit.scale_refine, 4),
        PlaneKind.RELEASE: pack_uniform(unit.release_code, 4),
    }
    if has_diagonals:
        payloads[PlaneKind.DIAG_SU] = pack_fp16(unit.diagonals.su)
        payloads[PlaneKind.DIAG_SV] = pack_fp16(unit.diagonals.sv)
    planes = build_planes(
        geometry,
        rates,
        alphabet,
        descendant,
        alignment_bytes=alignment_bytes,
        max_released=unit.released_positions,
        payloads=payloads,
        with_diagonals=has_diagonals,
    )
    region = build_plane_region(planes, payloads)
    spec = TerminalSpec(
        "t-nvfp4",
        completion_widths,
        released_positions=unit.released_positions,
        with_scale_base=True,
        with_scale_refine=True,
        with_diagonals=has_diagonals,
    )
    terminal = build_terminal(
        geometry, rates, spec, planes, len(alphabet), len(descendant),
        plane_region=region,
    )
    manifest = Manifest(
        encoder_profile_id=encoder_profile_id(code, rates),
        branch=BranchIdentity(
            unit_id=unit_id,
            root_q256=q256,
            rotation=unit.rotation,
            container=container,
        ),
        geometry=geometry,
        arrangement=ArrangementMode.BRESENHAM,
        rates=rates,
        planes=planes,
        terminals=(terminal,),
        payload_digest=hashlib.sha256(region).digest(),
    )
    return manifest, region, serialize(manifest, region)


def read_unit_artifact(blob: bytes, device="cpu") -> torch.Tensor:
    """Decode an artifact to weights from **bytes alone**.

    Nothing here comes from the encoder: the forests come off the ALPHABET and
    DESCENDANT planes, the scales off segment 2b, the convolutional code out of
    the manifest's encoder profile.
    """
    from .container import plane_ranges
    from .diagonals import Diagonals

    art = parse(blob)
    manifest, terminal = art.manifest, art.terminal
    geometry, rates = manifest.geometry, manifest.rates
    rows, cols = geometry.rows, geometry.columns
    completion_widths = tuple(3 - r for r in rates)

    chunks = {}
    for descriptor, offset, content, _total in plane_ranges(manifest, terminal):
        chunks[descriptor.kind] = art.plane_region[offset : offset + content]

    forests = _read_forest_planes(
        rates, chunks[PlaneKind.ALPHABET], chunks[PlaneKind.DESCENDANT]
    )

    # The ConvCode is not stored field-by-field; the profile id commits to it.
    # Recovering it by search over the published orders and checking the digest
    # is a *verification*, not a guess: a mismatch means the artifact was made
    # by an encoder this reader does not implement, and it fails closed.
    code = None
    for memory in sorted(_ODS_GENERATORS):
        candidate = ConvCode(memory=memory)
        if encoder_profile_id(candidate, rates) == manifest.encoder_profile_id:
            code = candidate
            break
    if code is None:
        raise GrammarError(
            "encoder_profile_id matches no convolutional code this reader "
            "implements; the artifact's trellis is not one we can replay"
        )

    from .planes import CANONICAL_PLANE_ORDER

    n_released = terminal.plane_elements[
        CANONICAL_PLANE_ORDER.index(PlaneKind.RELEASE)
    ]
    unit = EncodedUnit(
        rates=rates,
        anchors=torch.zeros(rows, cols, dtype=torch.long, device=device),
        codes=torch.zeros(rows, cols, dtype=torch.long, device=device),
        body_bits=unpack_body(chunks[PlaneKind.BODY], rates, rows, device),
        # BODY stays uint8 -- the replay is bandwidth-bound over it -- but
        # COMPLETION indexes the reachable-descendant table, and a uint8 index
        # tensor is a *boolean mask* in torch, not an integer index.  The two
        # planes share a reader and must not share a dtype.
        completion_bits=unpack_body(
            chunks[PlaneKind.COMPLETION], completion_widths, rows, device
        ).long(),
        scale_base=unpack_uniform(
            chunks[PlaneKind.SCALE_BASE],
            geometry.positions // geometry.group_weights, 8, device,
        ),
        scale_refine=unpack_uniform(
            chunks[PlaneKind.SCALE_REFINE],
            geometry.positions // geometry.half_weights, 4, device,
        ),
        release_index=torch.zeros(0, dtype=torch.long, device=device),
        release_code=torch.zeros(0, dtype=torch.long, device=device),
        sse=0.0,
        rotation=manifest.branch.rotation,
        rotation_block=128,
        diagonals=(
            Diagonals(
                sv=unpack_fp16(chunks[PlaneKind.DIAG_SV], rows, device),
                su=unpack_fp16(chunks[PlaneKind.DIAG_SU], cols, device),
            )
            if chunks.get(PlaneKind.DIAG_SU)
            else None
        ),
        group=geometry.group_weights,
        half=geometry.half_weights,
    )
    if n_released:
        # §9's placement is *derived*, not stored: decode without release,
        # rank by descending decoded magnitude per superblock, and the RELEASE
        # plane's codes land on those positions in that order.
        from .decode import decode_codes_mixed
        from .encode import _canonical_release_order, e2m1_value_table
        from .wire import scales_from_planes

        pre = decode_codes_mixed(unit, forests, code, apply_release=False)
        scale = torch.repeat_interleave(
            scales_from_planes(unit.scale_base, unit.scale_refine,
                               unit.group, unit.half),
            unit.half,
        ).reshape(rows, cols)
        decoded = e2m1_value_table(device)[pre.int()] * scale
        unit.release_index = _canonical_release_order(
            decoded, cols, geometry.superblock_columns, n_released
        )
        unit.release_code = unpack_uniform(
            chunks[PlaneKind.RELEASE], n_released, 4, device
        )
    return reconstruct_unit(unit, forests, code)
