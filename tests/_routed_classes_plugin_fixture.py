"""Canonical synthetic WINDOW bytes; no encoder search or runtime substitute."""
from __future__ import annotations

from functools import lru_cache
from fractions import Fraction

import torch

from tessera.alphabet import BF16_GRID, E4M3_GRID
from tessera.encode import EncodedUnit
from tessera.fused_frame import pack_fused
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.unit_artifact import build_unit_artifact

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 256, 512, 8, 8
ROLES = (("w13", 0, "w1", "gate_proj", INTERMEDIATE, HIDDEN),
         ("w13", 1, "w3", "up_proj", INTERMEDIATE, HIDDEN),
         ("w2", 0, "w2", "down_proj", HIDDEN, INTERMEDIATE))


def _wire(family, original, role_index, name, rows, columns, q256):
    grid = E4M3_GRID if family == "e4m3" else BF16_GRID
    gen = torch.Generator().manual_seed(1967 + 101 * original + role_index)
    rates = tuple(bresenham_rate_schedule(Fraction(q256, 256), columns, cap=8))
    body = torch.randint(0, 256, (rows, columns), generator=gen).to(torch.int64)
    body &= (1 << torch.tensor(rates)) - 1
    if family == "e4m3":
        # Small finite signed E4M3 values, not NaNs or a zero-output fixture.
        palette = torch.tensor([0x18, 0x20, 0x28, 0x98, 0xA0, 0xA8], dtype=torch.uint8)
        table = palette[torch.randint(0, len(palette), (1 << 14,), generator=gen)]
    else:
        values = (torch.randn(1 << 14, generator=gen) * 0.03).bfloat16()
        table = values.view(torch.int16).to(torch.int32) & 0xFFFF
    empty = torch.empty(0, dtype=torch.uint8)
    unit = EncodedUnit(
        rates=rates, anchors=torch.zeros_like(body), codes=torch.zeros_like(body),
        body_bits=body.to(torch.uint8), completion_bits=torch.zeros_like(body),
        scale_base=empty, scale_refine=empty, release_index=torch.empty(0, dtype=torch.int64),
        release_code=torch.empty(0, dtype=torch.int64), sse=0.0,
        body=BodyKind.WINDOW, window_bits=14, window_codes=table,
        scale_plane=ScalePlaneKind.CHANNEL,
        scale_rows=(torch.rand(rows, generator=gen) * 0.25 + 0.5).half(),
        completion_limit=0)
    _, _, blob = build_unit_artifact(unit, name, grid, q256, fixture_id=None)
    return pack_fused([(name, rows, blob)])


@lru_cache(maxsize=6)
def wire_fixture(family, layout="mixed"):
    """The payload seed names the original expert; the wire slot names storage."""
    if layout == "uniform":
        ids = list(range(EXPERTS))
        profiles = [(0, 8, 1024, 1024)]
    elif layout == "three":
        ids = [1, 5, 0, 3, 7, 2, 4, 6]
        profiles = [(0, 2, 512, 768), (2, 5, 768, 896), (5, 8, 1024, 1024)]
    elif layout == "mixed":
        ids = [1, 3, 5, 7, 0, 2, 4, 6]
        profiles = [(0, 4, 768, 896), (4, 8, 1024, 1024)]
    else:
        raise ValueError(layout)
    classes = [{"start": start, "end": end,
                "q256": {"w13": [gate, gate], "w2": [down]}}
               for start, end, gate, down in profiles]
    rungs = [(gate, down) for start, end, gate, down in profiles for _ in range(start, end)]
    wires = {}
    for storage, original in enumerate(ids):
        for role_index, (group, index, shard, name, rows, columns) in enumerate(ROLES):
            q256 = rungs[storage][int(group == "w2")]
            wires[(storage, shard)] = _wire(family, original, role_index, name, rows, columns, q256)
    groups = {}
    for group, rows, columns, roles in (
            ("w13", 2 * INTERMEDIATE, HIDDEN, [["gate_proj", INTERMEDIATE], ["up_proj", INTERMEDIATE]]),
            ("w2", HIDDEN, INTERMEDIATE, [["down_proj", HIDDEN]])):
        matrix = [([gate, gate] if group == "w13" else [down]) for gate, down in rungs]
        groups[group] = {"rows": rows, "columns": columns, "roles": roles,
                         "q256": matrix if layout != "uniform" else 1024,
                         "wire_stride": max(len(blob) for (slot, shard), blob in wires.items()
                                            if (shard == "w2") == (group == "w2"))}
    scheme = {"family": "TESSERA_FP8" if family == "e4m3" else "TESSERA_BF16",
              "structure": "routed_moe", "grid": "E4M3" if family == "e4m3" else "BF16",
              "body": "WINDOW", "plane": "CHANNEL", "experts": EXPERTS,
              "expert_ids": ids, "expert_classes": classes, "groups": groups}
    return scheme, wires


def route_cases(scheme, tokens, device="cpu"):
    """Original-global routes retain distinct weights at each top-k position."""
    storage_to_global = torch.tensor(scheme["expert_ids"], dtype=torch.int32)
    counts = ([8, 0], [0, 8], [4, 4], [6, 2], [7, 1])
    cycles = [[0] * a + [EXPERTS - 1] * b for a, b in counts]
    if len(scheme["expert_classes"]) == 3:
        cycles += [[0, 1, 2, 3, 4, 5, 6, 7]]  # Three-class population 2/3/3.
    cases = []
    for index, cycle in enumerate(cycles):
        # Different expert values and order make a stale or omitted inverse observable.
        stored = torch.tensor(cycle, dtype=torch.long).roll(index).flip(0).repeat(tokens)
        ids = storage_to_global[stored].reshape(tokens, TOP_K).to(device)
        weights = torch.linspace(0.125, 0.875, tokens * TOP_K).reshape(tokens, TOP_K)
        weights = weights.roll(index + 1, dims=1) * (1.0 - index * 0.0625)
        cases.append((ids, weights.to(device)))
    return cases


def pure_schedule_launch(adapter, mode, x, scale, routing, out, *, weight, swiglu_limit):
    """Call the existing pure native entry with an explicit activation clamp."""
    from tessera import routed_fused as rf

    cls = adapter.classes[0]
    down = mode == 2
    b0, b1 = (cls.down, cls.down) if down else (cls.gate, cls.up)
    w0, w1 = (cls.words_down, cls.words_down) if down else (cls.words_gate, cls.words_up)
    t0, t1 = (cls.table_down, cls.table_down) if down else (cls.table_gate, cls.table_up)
    r0, r1 = (cls.runs_down, cls.runs_down) if down else (cls.runs_gate, cls.runs_up)
    d0, d1 = (cls.bdesc_down, cls.bdesc_down) if down else (cls.bdesc_gate, cls.bdesc_up)
    bm = rf.superblock_rows(adapter.library, mode, routing.tokens)
    counter = torch.zeros(1, dtype=torch.int32, device=x.device)
    empty = torch.empty(0, dtype=torch.float32, device=x.device)
    rf._ext(adapter.library).routed_fused_forward(
        mode, adapter.fp8, x, scale if scale is not None else empty,
        w0, w1, t0, t1, b0.init_all, b1.init_all, b0.has_init, b1.has_init,
        b0.scale_all, b1.scale_all, r0, r1, d0, d1,
        cls.tile_words_down if down else cls.tile_words_gate_up,
        cls.slot_words_down if down else cls.slot_words_gate_up, adapter.piece_major,
        routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.superblocks(bm), counter,
        routing.top_k, 1 if down else 0, weight, swiglu_limit,
        out, torch.cuda.get_device_properties(x.device).multi_processor_count, bm)
