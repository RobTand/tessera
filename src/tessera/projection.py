"""Decode priced weights for direct consumers without importing serving.

Importing this module loads only producer metadata. Torch and the wire decoder
load when a caller requests a weight.
"""
from __future__ import annotations

from .metadata import ROUTES, TESSERA_BF16, direct_consumer_activation_contract

__all__ = ["direct_consumer_weight"]


def direct_consumer_weight(blob, module, role, family, *, device="cpu"):
    """Decode a priced unit with the exact arithmetic of its direct consumer."""
    import torch
    from .decode import materialize_bf16_folded, reconstruct_unit
    from .unit_artifact import parse_unit_artifact

    contract = direct_consumer_activation_contract(module, role)
    if contract is None:
        raise ValueError(f"{module}: role {role!r} has no direct weight consumer")
    parsed = parse_unit_artifact(blob, device=device)
    if family not in ROUTES or parsed.grid.name not in ROUTES[family]["grids"]:
        raise ValueError(f"{module}: {family!r} does not decode grid {parsed.grid.name!r}")
    if family == TESSERA_BF16:
        weight = materialize_bf16_folded(parsed.unit, parsed.forests, parsed.code)
    else:
        weight = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    return weight.to(torch.float32 if contract == "a32" else torch.bfloat16)
