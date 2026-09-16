"""The stock NVFP4 reference, kept as a TEST ASSET after the serving wrapper retired.

The A4 lanes decode the compact loader's packed planes in-kernel
(``tessera.kernel_a4``); the INDEPENDENT oracle they are held to is
``tessera.stock.materialize_stock`` on a parsed unit with the roles' LUT tables
moved onto one shared global by ``tessera.fused.shared_lut_global`` -- pure
torch, no serving path.  The retired ``tessera.serving.ops`` wrapper exposed
exactly that pair, and the tests that compare a native decode (or a sliced
unit) against the stock tile import it from here instead of keeping the
production wrapper alive.

Nothing in this module is imported by ``tessera.serving``.
"""
from __future__ import annotations

import dataclasses

import torch

__all__ = ["reference_role_tiles", "reference_tile", "reference_module", "ReferenceModule"]


def reference_role_tiles(parsed_roles, moved_tables, shared, device):
    """Each role's ``(weight_packed, weight_scale bytes)`` through the stock decoder."""
    from tessera.stock import materialize_stock

    parts = []
    for (name, parsed), table in zip(parsed_roles, moved_tables):
        unit = dataclasses.replace(parsed.unit, scale_lut=table.cpu(),
                                   scale_global=float(shared))
        tensors = materialize_stock(unit, parsed.forests, parsed.code)
        parts.append((tensors["weight_packed"].to(device),
                      tensors["weight_scale"].view(torch.uint8).to(device)))
    return parts


def reference_tile(parsed_roles, moved_tables, shared, device):
    """The module's stock NVFP4 pair: ``(packed nibbles, group-16 scale bytes)``."""
    parts = reference_role_tiles(parsed_roles, moved_tables, shared, device)
    return (torch.cat([p for p, _ in parts], 0).contiguous(),
            torch.cat([s for _, s in parts], 0).contiguous())


def reference_module(parsed_roles, device="cuda"):
    """The stock pair for ``parsed_roles``, with the roles' LUT tables moved
    onto one shared global exactly as the serving loader moves them.

    This is the retired wrapper's acceptance path, kept as an oracle: a fused
    group's tables must ride one shared global (``fused.shared_lut_global``,
    the same exact-binade rule), and a role whose table cannot be carried is
    refused by that function, not by a serving gate.
    """
    from tessera.fused import shared_lut_global

    parsed_roles = list(parsed_roles)
    names = [str(name) for name, _parsed in parsed_roles]
    shared, moved = shared_lut_global(
        [parsed.unit.scale_lut for _name, parsed in parsed_roles],
        [float(parsed.unit.scale_global) for _name, parsed in parsed_roles],
        names,
    )
    return ReferenceModule(parsed_roles, moved, shared, device)


class ReferenceModule:
    """The retired wrapper's observable shape, with no serving code behind it.

    ``rows``/``columns``/``row_scale``/``decode`` are what the row-shard and
    rung tests read; the tile is built once, here, from ``materialize_stock``.
    """

    def __init__(self, parsed_roles, moved_tables, shared, device):
        self._parts = reference_role_tiles(parsed_roles, moved_tables, shared, device)
        self.device = torch.device(device)
        rows, columns = 0, None
        for (name, parsed), (packed, _scale) in zip(parsed_roles, self._parts):
            rows += int(packed.shape[0])
            width = int(packed.shape[1]) * 2
            columns = width if columns is None else columns
            if width != columns:
                raise ValueError(f"role {name!r} has {width} input columns, the module {columns}")
        self.rows, self.columns = rows, columns

    def decode(self):
        return (torch.cat([p for p, _ in self._parts], 0).contiguous(),
                torch.cat([s for _, s in self._parts], 0).contiguous())

    def row_scale(self):
        raise NotImplementedError(
            "the reference asset decodes the tile; the row scale is a "
            "materialize_* return value the caller already holds")
