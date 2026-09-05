"""Count what the truncation-terminal ladder could declare on a real encode,
and what the wire and the reader do with each shorter terminal (tessera#144).

Two tables, both on CPU, both on 16-row toy units (seed 0):

1. ``--count``: for every ``(grid, rung)`` in the plan, encode through the
   exporter's own default path (``export.encode_linear_planes``) and list the
   planes that carry elements, the COMPLETION/RELEASE counts the written
   terminal declares, and how many shorter terminals the manifest validator
   would accept -- whole-plane drops plus BODY superblock cuts.
2. ``--rows``: on E2M1 (TCQ body, below the cap, ``completion=None`` so the
   completion axis is actually written), build every shallower completion
   terminal with ``layout.build_terminal``, add it to the manifest, serialise,
   truncate to it, and report where it is refused: the manifest's prefix rule,
   the granule rule, or the reader's geometry-derived counts.

Written up in ``docs/reports/tessera-terminal-ladder-2026-09-04.md`` and
pinned by ``tests/test_audit_container_accounting.py::
test_no_shorter_terminal_survives_the_wire_on_an_encode``.
"""
import argparse
import dataclasses

import torch

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.container import parse, serialize
from tessera.export import encode_linear_planes, tcq_cap_q256, wire_recipe
from tessera.grammar import completion_capacity
from tessera.layout import TerminalSpec, build_terminal
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.planes import CountGranularity, PlaneKind
from tessera.unit_artifact import read_unit_artifact

GRIDS = {g.name: g for g in SERIALISABLE_GRIDS.values()}
PLAN = {
    "E2M1": [256, 512, 768],
    "E2M1x2": [384, 640, 896, 1024],
    "E4M3": [768, 1024, 1536, 1792],
    "BF16": [1024, 2048],
}


def _encode(grid, q256, columns, **kwargs):
    torch.manual_seed(0)
    exported, unit, forests = encode_linear_planes(
        torch.randn(16, columns) * 0.02, grid=grid, q256=q256, name="u", **kwargs
    )
    return parse(exported.blob), unit, forests, exported.blob


def count():
    attempted = written = 0
    for gname, rungs in PLAN.items():
        grid = GRIDS[gname]
        for q256 in rungs:
            attempted += 1
            try:
                art, unit, _, _ = _encode(grid, q256, 256)
            except Exception as e:  # noqa: BLE001 -- the refusal is the row
                print(f"{gname:7s} q256={q256:5d}: encode refused: {type(e).__name__}: {str(e)[:100]}")
                continue
            written += 1
            m = art.manifest
            wire = m.plane_order
            t = m.terminals[0]
            print(f"\n{gname:7s} q256={q256:5d} body={m.body.name} plane={m.scale_plane.kind.name} "
                  f"span={m.span} cap_q256={tcq_cap_q256(grid)} recipe={wire_recipe(grid, q256).body.name} "
                  f"terminals={[x.slot_id for x in m.terminals]} region_bytes={len(art.plane_region)}")
            legal = 0
            planes = []
            for kind in wire:
                d = m.plane(kind)
                ext = d.element_count if d else 0
                if ext == 0:
                    continue
                if d.count_granularity is not CountGranularity.WHOLE_PLANE:
                    cuts, run = {0}, 0
                    for c in d.counts:
                        run += c
                        cuts.add(run)
                    below = [c for c in cuts if c < ext]
                else:
                    below = [0]
                legal += len(below)
                planes.append((kind.name, ext, d.count_granularity.name, len(below)))
            print("   planes (kind, extent, granularity, cuts below extent):", planes)
            print(f"   COMPLETION={t.plane_elements[wire.index(PlaneKind.COMPLETION)]} "
                  f"RELEASE={t.plane_elements[wire.index(PlaneKind.RELEASE)]} "
                  f"released_positions={unit.released_positions}")
            print(f"   shorter terminals the layout could declare: {legal}")
    print(f"\n{attempted} (grid, rung) points attempted, {written} written, {attempted - written} refused")


def rows():
    grid = GRIDS["E2M1"]
    cap = grid.rate_cap

    def offer(columns, scale_plane, label):
        kwargs = dict(completion=None)
        if scale_plane is not None:
            kwargs["scale_plane"] = scale_plane
        art, _, _, blob = _encode(grid, 256, columns, **kwargs)
        m = art.manifest
        wire = m.plane_order
        t = m.terminals[0]
        rates = m.rates
        comp = m.plane(PlaneKind.COMPLETION)
        print(f"\n[{label}] columns={columns} plane={m.scale_plane.kind.name} body={m.body.name}")
        print(f"   COMPLETION extent={comp.element_count} superblock counts={list(comp.counts)} "
              f"terminal counts={t.plane_elements}")
        alphabet = t.plane_elements[wire.index(PlaneKind.ALPHABET)]
        descendant = t.plane_elements[wire.index(PlaneKind.DESCENDANT)]
        variants = []
        for c in range(0, 2):
            variants.append((f"t-c{c}", TerminalSpec(
                f"t-c{c}", tuple(min(c, completion_capacity(r, cap)) for r in rates),
                with_scale_base=t.plane_elements[wire.index(PlaneKind.SCALE_BASE)] > 0,
                with_scale_refine=t.plane_elements[wire.index(PlaneKind.SCALE_REFINE)] > 0,
            )))
        if scale_plane is ScalePlaneKind.S6B:
            variants.append(("t-po2 (base only, completion 0, no refine)", TerminalSpec(
                "t-po2", (0,) * len(rates), with_scale_base=True, with_scale_refine=False)))
            variants.append(("t-c3 (base + full completion, no refine)", TerminalSpec(
                "t-c3", tuple(completion_capacity(r, cap) for r in rates),
                with_scale_base=True, with_scale_refine=False)))
        for name, spec in variants:
            try:
                short = build_terminal(
                    m.geometry, rates, spec, m.planes, alphabet, descendant,
                    plane_region=art.plane_region, cap=cap, arity=grid.arity, span=m.span,
                )
                ladder = serialize(dataclasses.replace(m, terminals=(t, short)), art.plane_region)
            except Exception as e:  # noqa: BLE001 -- the refusal is the row
                print(f"   {name}: manifest REFUSES: {type(e).__name__}: {str(e)[:170]}")
                continue
            print(f"   {name}: counts={short.plane_elements} bytes={short.exact_bytes}/{len(art.plane_region)} "
                  f"manifest +{len(ladder) - len(blob)} B -> manifest ACCEPTS")
            cut = ladder[: len(ladder) - (len(art.plane_region) - short.exact_bytes)]
            try:
                got = read_unit_artifact(cut)
                print(f"      reader: decoded {tuple(got.shape)}")
            except Exception as e:  # noqa: BLE001 -- the failure is the row
                print(f"      reader FAILS: {type(e).__name__}: {str(e)[:160]}")

    offer(256, None, "default LUT plane, one superblock")
    offer(512, None, "default LUT plane, two superblocks")
    offer(512, ScalePlaneKind.S6B, "S6B plane, two superblocks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", action="store_true", help="table 1: the (grid, rung) count")
    ap.add_argument("--rows", action="store_true", help="table 2: shorter terminals offered to the wire")
    args = ap.parse_args()
    if not (args.count or args.rows):
        args.count = args.rows = True
    if args.count:
        count()
    if args.rows:
        rows()
