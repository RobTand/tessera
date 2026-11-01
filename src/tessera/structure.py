"""The unit structure: a dense Linear or a routed MoE projection.

Both structures use the same served E2M1x2 recipe. These core names let
``export`` and ``cached_unit`` select the recipe without a serving import.
Historical producer packages keep that import boundary.
"""
from __future__ import annotations

STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = (STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE)
