"""Is a trailing partial superblock CORRECT -- and is it even reachable?

Issue #44 asks two things about a unit whose column count is not a whole
number of 256-column superblocks.  They must not be blurred:

  (a) **correctness** -- do the bytes decode to what the encoder meant?  That
      is checkable cheaply, on CPU, unconditionally, and it is what this
      script answers.  The load bearing form is ``serialise -> parse ->
      decode == the encoder's own decode``, not ``encode_unit ->
      decode_codes_mixed``: #22 and #27 are both about the **reader
      regenerating** the release order from the placed count, and a harness
      that never serialises cannot see a disagreement between the two sides.

  (b) **cost** -- a timing claim, which belongs on a quiet box and is
      measured by ``experiments/loadcost_partial.py``.

It also answers a question #44 does not ask and should have: **which partial
widths are reachable at all.**  ``decode.materialize_nvfp4`` needs ``cols %
2 == 0`` and ``kernel._require_column_groups`` needs ``cols % half == 0``
(half = 16), so the ``k*256 + 1`` width the issue names for its throughput
cliff cannot be packed to NVFP4 or fed to a Tessera kernel.  The thinnest
partial superblock that can be served is 16 columns wide, the fattest 240.
Every refusal below is recorded, not asserted.

CPU only, by construction -- it runs while a GPU measurement is in flight.

    python experiments/partial_superblock_roundtrip.py            # table
    python experiments/partial_superblock_roundtrip.py out.json   # + digests
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tessera.alphabet import E2M1_GRID
from tessera.decode import materialize_nvfp4, reconstruct_unit
from tessera.encode import encode_unit
from tessera.errors import GrammarError
from tessera.export import _plan_for, tcq_cap_q256, wire_recipe
from tessera.grammar import release_quota, superblock_count, superblock_widths
from tessera.kernel import _require_column_groups
from tessera.manifest import RotationState, ScalePlaneKind
from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, read_unit_artifact

SUPERBLOCK = 256
HALF = 16
ROWS = 32           # correctness is a SHAPE property; rows only cost time
RELEASE_FRACTION = 0.125    # what experiments/loadcost.py uses

#: Every width is ``k*256 + partial``.  The set is chosen to bracket the
#: question rather than to sample it: two complete widths, the thinnest and
#: the fattest *reachable* partial (16 and 240 columns, the multiples of
#: ``half`` that bound the partial block), and the two widths #44 actually
#: names -- ``k*256 +/- 1`` -- which are here to record their refusal.
WIDTHS = [
    (256, "complete, 1 superblock"),
    (272, "partial, 16 of 256 -- thinnest reachable"),
    (496, "partial, 240 of 256 -- fattest reachable"),
    (512, "complete, 2 superblocks"),
    (640, "partial, 128 of 256 -- the width #22 was found on"),
    (257, "partial, 1 of 256 -- the width #44 names"),
    (511, "partial, 255 of 256 -- the width #44 names"),
    (4864, "complete, 19 superblocks -- timing control"),
    (4880, "partial, 16 of 256 -- thinnest reachable, NOT a 32-group"),
    (4896, "partial, 32 of 256 -- timing arm, thinnest clean partial"),
    (5088, "partial, 224 of 256 -- timing arm, fattest clean partial"),
    (5104, "partial, 240 of 256 -- fattest reachable, NOT a 32-group"),
    (5120, "complete, 20 superblocks -- the shape every published figure covers"),
]


#: The two wires this question has to be answered on, because they are not
#: the same wire.  ``recipe`` is what ``export.wire_recipe`` selects for E2M1
#: at the TCQ cap -- span 2, LUT scale plane -- i.e. what the exporter writes.
#: ``loadcost`` is what ``experiments/loadcost.py`` constructs: ``encode_unit``
#: on its defaults, span 1 and an S6b block plane.  Every published Tessera
#: load figure is on the second, which is a scope limit of that harness and
#: not of the format; correctness is checked on both so neither is assumed
#: from the other.
WIRES = ("recipe", "loadcost")


def _weights(cols: int, rows: int, shared: bool):
    """``shared`` narrows ONE tensor, so a cross-width comparison compares
    encodes of the same numbers.  Seeding per width instead gives every width
    different weights, and then every cross-width disagreement is the seed."""
    torch.manual_seed(
        0xA44 if shared else (zlib.crc32(f"partial-superblock-{cols}".encode()) & 0xFFFF)
    )
    wide = max(c for c, _ in WIDTHS) if shared else cols
    return (torch.randn(rows, wide) * 0.02)[:, :cols].contiguous()


def _encode(cols: int, wire: str = "recipe", rows: int = ROWS, shared: bool = False):
    """One encoded unit at ``cols`` columns on ``wire``, with releases on."""
    grid = E2M1_GRID
    q256 = tcq_cap_q256(grid)
    code = ConvCode(memory=6)
    weight = _weights(cols, rows, shared)
    if wire == "recipe":
        recipe = wire_recipe(grid, q256)
        sigma = (recipe.channel_sigma
                 if recipe.scale_plane is ScalePlaneKind.CHANNEL else None)
        rates, forests = _plan_for(grid, q256, cols, recipe.body, sigma)
        kw = dict(span=recipe.span, scale_plane=recipe.scale_plane,
                  body=recipe.body, window_bits=recipe.window_bits,
                  window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
                  channel_sigma=recipe.channel_sigma, scale_refit=2)
    else:                                   # what loadcost.py actually builds
        rates, forests = _plan_for(grid, q256, cols)
        kw = dict(rotation=RotationState.NONE, with_diagonals=False)
    unit = encode_unit(
        weight, forests, rates, code, completion=0,
        released_positions=int(RELEASE_FRACTION * weight.numel()), **kw,
    )
    return unit, forests, code, q256 * grid.arity, weight


def _pack(unit, codes):
    """``materialize_nvfp4`` called the way the unit's own scale plane needs.

    Dispatching on the plane matters: an S6b unit carries ``scale_base``/
    ``scale_refine`` and a LUT unit carries a table and a global, and passing
    the empty S6b fields of a LUT unit raises a shape error that has nothing
    to do with the width under test.
    """
    kind = ScalePlaneKind(getattr(unit, "scale_plane", ScalePlaneKind.S6B))
    if kind is ScalePlaneKind.LUT:
        return materialize_nvfp4(codes, unit.scale_base, unit.scale_refine,
                                 unit.group, unit.half,
                                 scale_lut=unit.scale_lut,
                                 scale_global=unit.scale_global)
    return materialize_nvfp4(codes, unit.scale_base, unit.scale_refine,
                             unit.group, unit.half)


def _refusal(fn) -> str:
    """``"ok"`` or the exception a guard raises.  A refusal is a result."""
    try:
        fn()
        return "ok"
    except Exception as exc:          # noqa: BLE001 -- recording, not handling
        return f"REFUSED {type(exc).__name__}: {exc}"


def _release_density(unit, cols: int, rows: int) -> dict:
    """Per-superblock release counts and densities, from the placement itself.

    Read off ``release_index`` rather than recomputed from ``release_quota``,
    so a disagreement between the quota and what the encoder actually placed
    shows up here instead of being assumed away.
    """
    widths = superblock_widths(cols, SUPERBLOCK)
    blocks = superblock_count(cols, SUPERBLOCK)
    index = unit.release_index.cpu()
    placed = [0] * blocks
    if index.numel():
        block_of = ((index % cols) // SUPERBLOCK).tolist()
        for b in block_of:
            placed[b] += 1
    return {
        "blocks": blocks,
        "widths": list(widths),
        "quota": list(release_quota(int(index.numel()), cols, SUPERBLOCK)),
        "placed": placed,
        # A release is a per-POSITION object; a block's positions are
        # rows * width.  Equal density across blocks is exactly what #27's
        # width-proportional quota promises, and what the equal-count spread
        # it replaced broke on a partial block.
        "density": [
            round(p / (rows * w), 6) for p, w in zip(placed, widths)
        ],
        # The floor #22 fixed: how many blocks the OLD release order saw.
        "old_floor_blocks": max(1, cols // SUPERBLOCK),
        "trailing_partial_reached": bool(
            cols % SUPERBLOCK and placed[-1] > 0
        ),
    }


def main() -> int:
    out: dict = {"superblock": SUPERBLOCK, "half": HALF, "rows": ROWS,
                 "release_fraction": RELEASE_FRACTION, "wires": {}}
    bad = []
    for wire in WIRES:
        u0, _f, _c, _q, _w = _encode(256, wire)
        plane = ScalePlaneKind(getattr(u0, "scale_plane", ScalePlaneKind.S6B)).name
        span = int(getattr(u0, "span", 1))
        head = (f"wire '{wire}': E2M1 grid, TCQ at cap (q256=768), "
                f"span {span}, {plane} scale plane, releases "
                f"{RELEASE_FRACTION:.1%}, rows={ROWS}")
        print("=" * 108); print(head); print("=" * 108)
        cases: dict = {}
        out["wires"][wire] = {"description": head, "cases": cases}
        print(f"{'cols':>6} {'k*256+p':>11} {'build':>8} {'roundtrip':>11} "
              f"{'nvfp4 pack':>11} {'kernel':>8}  note")
        print("-" * 108)
        for cols, note in WIDTHS:
            partial = cols % SUPERBLOCK
            shape = f"{cols // SUPERBLOCK}*256+{partial}"
            rec: dict = {"note": note, "partial": partial, "shape": shape}
            cases[cols] = rec
            try:
                unit, forests, code, q256_wire, weight = _encode(cols, wire)
                _m, _r, blob = build_unit_artifact(unit, f"u{cols}", forests,
                                                   q256_wire, code)
            except Exception as exc:      # noqa: BLE001
                rec["build"] = f"REFUSED {type(exc).__name__}: {exc}"
                print(f"{cols:>6} {shape:>11} {'REFUSED':>8} {'-':>11} "
                      f"{'-':>11} {'-':>8}  {note}")
                print(f"         {rec['build']}")
                continue
            rec["build"] = "ok"
            rec["bytes"] = len(blob)
            rec["bytes_sha256"] = hashlib.sha256(blob).hexdigest()
            rec["placement_sha256"] = hashlib.sha256(
                unit.release_index.cpu().numpy().tobytes()).hexdigest()
            rec["release"] = _release_density(unit, cols, ROWS)

            # (a) THE correctness question: do the bytes decode to what the
            # encoder meant?  ``read_unit_artifact`` takes NOTHING from the
            # encoder -- forests off the ALPHABET plane, the code out of the
            # manifest, the release order regenerated from the placed count --
            # so equality here is the reader and the writer agreeing about
            # the trailing partial superblock, which is the whole of #22/#27.
            reference = reconstruct_unit(unit, forests, code)
            recovered = read_unit_artifact(blob)
            exact = bool(torch.equal(recovered, reference))
            rec["roundtrip_exact"] = exact
            rec["decoded_sha256"] = hashlib.sha256(
                reference.cpu().numpy().tobytes()).hexdigest()
            if not exact:
                gap = (recovered.float() - reference.float()).abs()
                rec["roundtrip_max_abs"] = float(gap.max())
                rec["roundtrip_n_differing"] = int((gap > 0).sum())
                bad.append((wire, cols))
            codes = torch.zeros(ROWS, cols, dtype=torch.uint8)
            rec["nvfp4_pack"] = _refusal(lambda: _pack(unit, codes))
            rec["kernel_groups"] = _refusal(
                lambda: _require_column_groups(cols, HALF))
            print(f"{cols:>6} {shape:>11} {'ok':>8} "
                  f"{('exact' if exact else 'DIFFERS'):>11} "
                  f"{('ok' if rec['nvfp4_pack'] == 'ok' else 'refused'):>11} "
                  f"{('ok' if rec['kernel_groups'] == 'ok' else 'refused'):>8}"
                  f"  {note}")
            if rec["nvfp4_pack"] != "ok":
                print(f"         pack: {rec['nvfp4_pack']}")
            if rec["kernel_groups"] != "ok":
                print(f"         kernel: {rec['kernel_groups'].split(';')[0]}")

        print()
        print("release density per superblock -- a release is a per-POSITION")
        print("object, so equal DENSITY (not equal count) is what #27's "
              "width-proportional quota promises:")
        for cols, rec in cases.items():
            if "release" not in rec:
                continue
            r = rec["release"]
            lo, hi = min(r["density"]), max(r["density"])
            tail = "" if not rec["partial"] else (
                f"   trailing partial reached: {r['trailing_partial_reached']}"
                f"  (the floor #22 fixed saw {r['old_floor_blocks']} of "
                f"{r['blocks']})")
            print(f"  {cols:>5} {rec['shape']:>11}  blocks={r['blocks']:>3}  "
                  f"density {lo:.5f}..{hi:.5f}  spread {hi - lo:.2e}{tail}")
        print()

    # Cross-width, on the ONE narrowed tensor: does giving a unit a trailing
    # partial superblock disturb the complete superblocks it already had?
    # A diagnostic, reported and not asserted -- the release quota, the block
    # scales and any global scale are all re-fit at the new width, and only
    # the first of those is width-invariant here (at 12.5% of rows*cols, a
    # complete block's share is rows*32 releases at every width).
    print("=" * 108)
    print("cross-width: ONE tensor narrowed.  Do the first 4864 columns "
          "decode the same")
    print("when a trailing partial superblock is appended?  And if not, is "
          "the superblock")
    print("what did it -- or the 32-weight S6b scale group, which varies "
          "over the same set?")
    print("=" * 108)
    for wire in WIRES:
        bu, bf, bc, _q, _w = _encode(4864, wire, shared=True)
        base = reconstruct_unit(bu, bf, bc)
        for cols in (4880, 4896, 5088, 5104, 5120):
            u, f, c, _q, _w = _encode(cols, wire, shared=True)
            wide = reconstruct_unit(u, f, c)[:, :4864]
            agree = float((wide == base).float().mean())
            out.setdefault("cross_width", {}).setdefault(wire, {})[cols] = agree
            kind = "partial" if cols % SUPERBLOCK else "complete"
            # Two alignments vary independently across this set, and only one
            # of them is the superblock.  Printing both is what separates
            # them: the disturbance tracks ``cols % 32`` -- the S6b scale
            # group ``encode._pack_scales`` cuts out of the FLATTENED tensor,
            # which straddles a row boundary when the width is not a whole
            # number of groups -- and is blind to ``cols % 256``.
            print(f"  {wire:>8}  4864 -> {cols:>4} ({kind:>8})  "
                  f"cols%256={cols % SUPERBLOCK:>3}  cols%32={cols % 32:>3}  "
                  f"columns 0..4863 identical on {agree * 100:6.2f}%")

    out["roundtrip_failures"] = bad
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2, sort_keys=True,
                                                default=str))
        print(f"\nwrote {sys.argv[1]}")
    print(f"\nround-trip: {'ALL EXACT' if not bad else 'FAILURES at ' + str(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
