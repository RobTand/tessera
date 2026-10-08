"""Real GLM-5.3 routed units for the register-direct kernel (eng-regdirect-build, stage 1).

Reads one layer's routed expert containers from an exported Tessera artifact, cuts them to one TP2
rank as the serving plan does (gate/up: rows; down: columns), and returns per expert and projection:
the R-bit codes [rows, cols], the per-column rates, the cut start state (the 14-bit window state
before local row 0, from ``compact_prep._window_cut_state``), the E4M3 byte table
``native[codes_of_state]`` and the fp32 row scale (DIAG_SV x global scale).

``reference_weights`` decodes those planes to E4M3 bytes with the window rule alone, so a kernel
decode can be checked bitwise against the disk unit, independent of the fragment repack.
"""
from __future__ import annotations

import dataclasses
import json
import os

import torch

WIN = 14
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclasses.dataclass
class CutUnit:
    codes: torch.Tensor        # uint8 [rows, cols]
    rates: tuple               # per column
    start: torch.Tensor        # int32 [cols]
    table: torch.Tensor        # uint8 [2^14] E4M3 bytes by window state
    scale: torch.Tensor        # float32 [rows]


class Artifact:
    def __init__(self, root):
        from safetensors import safe_open
        self.root = root
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self._open = {}
        self._safe_open = safe_open

    def blob(self, name) -> bytes:
        """The projection's unit container: the checkpoint tensor holds a fused frame
        (``fused_frame.FUSED_MAGIC``) with one member per projection."""
        from tessera.fused_frame import FUSED_MAGIC, parse_fused
        shard = self.index[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(os.path.join(self.root, shard), "pt", device="cpu")
        raw = self._open[shard].get_tensor(name).contiguous().numpy().tobytes()
        if raw[:len(FUSED_MAGIC)] != FUSED_MAGIC:
            return raw
        members = parse_fused(raw)
        if len(members) != 1:
            raise ValueError(f"{name}: {len(members)} fused members, expected one projection")
        return members[0].blob


def cut_unit(blob: bytes, rows_cut, cols_cut) -> CutUnit:
    """One container cut to ``rows_cut`` x ``cols_cut`` (half-open ranges)."""
    from tessera.compact_prep import _window_codes, _window_cut_state
    from tessera.unit_artifact import parse_unit_metadata
    from tessera.planes import PlaneKind
    from tessera.wire import unpack_body, unpack_fp16

    md = parse_unit_metadata(blob, device="cpu")
    rates = tuple(int(r) for r in md.manifest.rates)
    rows, cols = md.manifest.geometry.rows, md.manifest.geometry.columns
    (r0, r1), (c0, c1) = rows_cut, cols_cut
    codes = unpack_body(md.chunks[PlaneKind.BODY], rates, rows, device="cpu")[r0:r1, c0:c1]
    codes = codes.to(torch.uint8).contiguous()
    start = _window_cut_state(md, r0, c0, c1, "cpu", None).to(torch.int32).reshape(-1)
    native = torch.tensor(md.grid.native, dtype=torch.uint8)
    table = native[_window_codes(md).long()].contiguous()
    scale = (unpack_fp16(md.chunks[PlaneKind.DIAG_SV], rows, "cpu")[r0:r1].float()
             * float(md.manifest.scale_plane.global_scale)).reshape(-1).contiguous()
    return CutUnit(codes, rates[c0:c1], start, table, scale)


def layer_units(art: Artifact, layer: int, expert: int, rank: int, tp: int = 2, prefix="model.language_model.layers."):
    """{projection: CutUnit} for one expert of one layer at TP rank ``rank``."""
    out = {}
    for proj in PROJECTIONS:
        blob = art.blob(f"{prefix}{layer}.mlp.experts.{expert}.{proj}.wire")
        from tessera.unit_artifact import parse_unit_metadata
        geo = parse_unit_metadata(blob, device="cpu").manifest.geometry
        if proj == "down_proj":
            step = geo.columns // tp
            out[proj] = cut_unit(blob, (0, geo.rows), (rank * step, (rank + 1) * step))
        else:
            step = geo.rows // tp
            out[proj] = cut_unit(blob, (rank * step, (rank + 1) * step), (0, geo.columns))
    return out


def reference_weights(unit: CutUnit) -> torch.Tensor:
    """E4M3 bytes [rows, cols]: state(n) = last 14 bits of the start state then codes, after row n."""
    codes = unit.codes.to(torch.int64)
    rates = torch.tensor(unit.rates, dtype=torch.int64)
    state = unit.start.to(torch.int64).clone()
    out = torch.empty_like(unit.codes)
    mask = (1 << WIN) - 1
    for n in range(codes.shape[0]):
        state = ((state << rates) | codes[n]) & mask
        out[n] = unit.table[state]
    return out
