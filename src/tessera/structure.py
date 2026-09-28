"""What a unit is served AS: a dense Linear or one projection of a routed MoE stack.

The two structures take different decoders, and on the NVFP4 route they carry
different wires below the E2M1x2 cap (``export.served_recipe``). The names live
in this core module so that ``export`` and ``cached_unit`` can state the served
wire without importing the serving plugin layer: a historical producer package
loads ``cached_unit`` and refuses any ``serving`` import
(``historical_producer``). ``serving.scheme`` re-exports them.
"""
from __future__ import annotations

STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = (STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE)
