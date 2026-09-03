"""The checkpoint vocabulary that routes a Tessera wire to one of its routes.

A Tessera unit (``tessera.unit_artifact``) is a self-describing blob: its
manifest binds the grid, the body kind, the scale-plane kind, the rate schedule
and the convolutional code into an ``encoder_profile_id`` and a payload digest,
and Tessera's reader verifies all of it from bytes alone.  The serving plugin
therefore parses nothing of its own: it hands the blob to that reader and hands
the verified planes to its decoder.  What the plugin adds is the serving half:
the sidecar scheme a gate reads without parsing the blob, the refusal when blob
and scheme disagree, and the tile the tensor core runs.

FAMILY = ROUTE.  A Tessera family names what the decoded tile *is* on the
hardware, not the body inside the blob.  Three families, one plugin:

* ``TESSERA_NVFP4`` is any E2M1-based grid over a LUT scale plane, decoded to
  the stock NVFP4 tile (nibble-packed E2M1 codes + group-16 ue4m3 block scales
  + one global) and served through ``torch._scaled_mm`` W4A4.
* ``TESSERA_FP8`` is the scalar E4M3 grid over the CHANNEL scale plane (schema
  minor 3: one fp16 word per output row times a global), decoded to the stock
  per-channel FP8 pair (E4M3 bytes + one fp32 scale per row) and served through
  ``torch._scaled_mm`` W8A8.
* ``TESSERA_BF16`` is the scalar BF16 grid over the same CHANNEL plane and the
  same window body, decoded to a plain bf16 tile and served through the stock
  BF16 GEMM, W16A16.  It exists because the E4M3 *alphabet* floors the body at
  ~0.022 out-space from R = 6 upward while the identical trellis over bf16
  keeps halving; above ~6 bpp an 8-bit tile has nothing left to buy.  Its row
  scale is applied on the GEMM output and never folded into the tile.

The body (span-2 coset trellis, or the window body) is the decoder's business
and the manifest's fact; the scheme carries it only so a receipt can be scoped
to it.  A checkpoint may carry both families, module by module, and one process
serves them at one declared residency.

STRUCTURE.  ``structure`` names what kind of vLLM layer the target is:
``dense`` (a ``LinearBase``, one blob per module) today.  Routed-MoE experts
would be a second value, and it is deliberately not accepted yet -- no served
measurement covers it, so no ``lane_eligibility`` cell names it and the
dispatch refuses by name (``config.get_quant_method``).  The field exists so
that adding the expert route is a value, not a schema change.

WHAT IS ATTESTED.  Only what a ``lane_eligibility`` cell in this package's
``runtime_contract.json`` names.  A cell appears only when a container receipt
covers it (principle 14); absence resolves ``unattested``.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "NVFP4_ACTIVATION_CONTRACT",
    "FP8_ACTIVATION_CONTRACT",
    "BF16_ACTIVATION_CONTRACT",
    "TESSERA_NVFP4",
    "TESSERA_FP8",
    "TESSERA_BF16",
    "TESSERA_FAMILIES",
    "TESSERA_SCHEME_KEY",
    "STRUCTURE_DENSE",
    "STRUCTURES",
    "ROUTES",
    "GROUP_SIZE",
    "is_tessera_scheme",
    "validate_tessera_scheme",
    "parse_tessera_blob_for_scheme",
]

#: The A-side contract each route executes, in the vocabulary the packaged
#: runtime contract publishes.  Defined here rather than beside the telemetry
#: because a producer reads them from the contract on a machine with no torch:
#: this module must stay importable without it.
NVFP4_ACTIVATION_CONTRACT = "e2m1_group16_ue4m3_static"
FP8_ACTIVATION_CONTRACT = "fp8_per_token_dynamic"
#: W16A16.  There is no A-side quantiser to name -- the route hands ``x``
#: to the stock bf16 GEMM as it arrives -- and the honest spelling of that is
#: a contract that says so, not the absence of one.  A gate that reads this
#: field must be able to tell "unquantised, by design" from "nobody filled it
#: in", and only a value can carry that.
BF16_ACTIVATION_CONTRACT = "bf16_unquantized"

TESSERA_NVFP4 = "TESSERA_NVFP4"
TESSERA_FP8 = "TESSERA_FP8"
TESSERA_BF16 = "TESSERA_BF16"

#: A scheme with ``family`` in ``TESSERA_FAMILIES`` is ours.
TESSERA_SCHEME_KEY = "family"

#: The only structure any route serves today.  ``routed_moe`` is the name the
#: expert route will take; it is NOT in this tuple, so a checkpoint declaring
#: it is refused here with the reason rather than served through a dense
#: method that would silently read the wrong tensor rank.
STRUCTURE_DENSE = "dense"
STRUCTURES = (STRUCTURE_DENSE,)

#: What each route can hold, by Tessera's own names (``PayloadGrid.name``,
#: ``ScalePlaneKind.name``).  NVFP4: grids whose codes are E2M1 nibbles (arity
#: 1 or 2 over the E2M1 base) over the LUT plane the tile's ue4m3 block scales
#: come from.  FP8: the scalar E4M3 grid over the CHANNEL plane the tile's
#: per-row fp32 scale comes from.  ``tile`` is the stock tensor the route
#: decodes to, ``columns_multiple`` the K quantum its mainloop needs, and
#: ``activation_contract`` what it executes on the A side.
ROUTES: dict[str, dict] = {
    TESSERA_NVFP4: {
        "grids": ("E2M1", "E2M1x2"), "plane": "LUT",
        "short": "NVFP4",
        "grid_kind": "an E2M1-based",
        "builder": ("tessera.serving.nvfp4_route", "build_tessera_nvfp4_method"),
        "tile": "nvfp4 (packed E2M1 codes, group-16 ue4m3 block scales, one global)",
        "columns_multiple": 16,
        "activation_contract": NVFP4_ACTIVATION_CONTRACT,
    },
    TESSERA_FP8: {
        "grids": ("E4M3",), "plane": "CHANNEL",
        "short": "FP8",
        "grid_kind": "the scalar E4M3",
        "builder": ("tessera.serving.fp8_route", "build_tessera_fp8_method"),
        "tile": "fp8 per-channel (E4M3 bytes, one fp32 scale per row)",
        "columns_multiple": 16,
        "activation_contract": FP8_ACTIVATION_CONTRACT,
    },
    TESSERA_BF16: {
        "grids": ("BF16",), "plane": "CHANNEL",
        "short": "BF16",
        "grid_kind": "the scalar BF16",
        "builder": ("tessera.serving.bf16_route", "build_tessera_bf16_method"),
        "tile": "bf16 (the raw table values, one fp32 scale per row applied on the GEMM output)",
        # No K quantum.  The other two routes decode to a PACKED tile whose
        # mainloop reads groups -- a nibble pair, a group-16 block scale -- and
        # 16 is that group.  A bf16 tile is one word per weight, so the GEMM
        # takes any K, and asserting a quantum here would refuse a geometry
        # this route serves.
        "columns_multiple": 1,
        "activation_contract": BF16_ACTIVATION_CONTRACT,
    },
}

#: DERIVED from ``ROUTES``, never written twice.  FAMILY = ROUTE is the rule,
#: so the set of families a build serves is exactly the set of routes it has:
#: a third family (say a WINDOW/CHANNEL body whose alphabet is snapped to bf16
#: and decoded to a bf16 tile for the stock GEMM) is one route module, one
#: ``ROUTES`` entry naming its builder, and its contract rows -- and nothing
#: here or in ``lane`` has to be edited to admit it.  A hand-written tuple was
#: a fourth place to remember.
TESSERA_FAMILIES = tuple(ROUTES)
_BODIES = ("TCQ", "WINDOW")
_REQUIRED = ("family", "grid", "body", "plane", "q256", "rows", "columns", "wire_bytes", "roles")
GROUP_SIZE = 16


def is_tessera_scheme(scheme: Any) -> bool:
    return isinstance(scheme, Mapping) and scheme.get(TESSERA_SCHEME_KEY) in TESSERA_FAMILIES


def _as_int(scheme: Mapping, field: str, target: str) -> int:
    value = scheme.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tessera target {target!r}: {field} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"tessera target {target!r}: {field} must be positive, got {value}")
    return value


def _refuse_an_unreadable_rung(route: str, grid: str, q256: int, target: str) -> None:
    """The rate has to be one the decoder reads, and the contract has to say so.

    Nothing checked this.  An allocated checkpoint served seven rungs
    (R749/R750/R934/R1006/R1083/R1107/R1262) that the contract's published set
    did not name, and nothing refused -- the run happened to be fine because
    every one of them IS decodable, but a rung the decoder could not read would
    have produced a wrong tensor rather than a refusal, which is the failure
    this gate exists for.

    The published range is derived from the decoder, not from what has been
    exported: each candidate rate was encoded and taken through this very load
    path, and the accepted set is what the contract now states.  Two different
    mechanisms bound it.  On ``E4M3`` the trellis grammar's shaped domain
    (code rate 1..8 over an 8-bit-native alphabet) gives a CONTINUOUS
    ``q256`` in [256, 2048] -- so the contract says continuous, not a list of
    the two rungs anyone had built.  On ``E2M1x2`` the same grammar caps the
    top at 896 (rate 7 of arity-2 native 8) and the native decoder cuts off the
    bottom: it serves the span-2 TCQ body only, and below 896 the recipe writes
    a WINDOW body that has no served decode.  One point, and it is a real one.
    """
    from .contract import reader_accepts, reader_rate_grid

    found = reader_rate_grid(route, grid)
    if found is None:
        raise ValueError(
            f"tessera target {target!r}: this build publishes no decodable rate range for "
            f"{route} on grid {grid!r}, so there is no rung it can promise to read. "
            "runtime_contract.json describes the (route, grid) pairs this plugin serves; a pair "
            "it does not describe is refused rather than served on another pair's numbers.")
    family, low, high, step = found
    if not reader_accepts(q256, low, high, step):
        span = f"[{low}, {high}]" + (f" step {step}" if step != 1 else " (every integer)")
        raise ValueError(
            f"tessera target {target!r}: q256={q256} is outside the rungs this build's decoder "
            f"reads for {family} -- {span}. Serving it would decode bytes the reader does not "
            "promise to understand, which is a wrong tensor rather than a refusal. Re-export at "
            "a rung inside the range, or publish a measured range that contains this one in "
            "runtime_contract.json.")


def validate_tessera_scheme(scheme: Mapping, target: str) -> dict:
    """Resolve a declared Tessera scheme without parsing the blob.

    Returns the normalised scheme; raises ``ValueError`` on anything no route
    can serve, at sidecar-parse time -- before a parameter exists.
    """
    family = scheme.get("family")
    if family not in TESSERA_FAMILIES:
        raise ValueError(
            f"tessera target {target!r}: family must be one of {TESSERA_FAMILIES}, got {family!r}")
    missing = [f for f in _REQUIRED if f not in scheme]
    if missing:
        raise ValueError(
            f"tessera target {target!r}: scheme is missing {missing}; a Tessera scheme "
            "declares its route, grid, body, plane, rate, geometry, byte count and roles")
    # A checkpoint written before the field existed is dense by construction:
    # every wire the plugin has ever served is one blob per vLLM Linear.
    structure = scheme.get("structure", STRUCTURE_DENSE)
    if structure not in STRUCTURES:
        raise ValueError(
            f"tessera target {target!r}: structure {structure!r} is not served; this plugin "
            f"serves {STRUCTURES} today. Routed-MoE expert stacks decode per-expert wires to "
            "the stock packed layouts and run vLLM's own fused-MoE kernels; that route is not "
            "built, carries no lane_eligibility cell and no served measurement, and is refused "
            "here rather than mis-served through the dense method.")
    route = ROUTES[family]
    grid = scheme["grid"]
    if grid not in route["grids"]:
        # The description comes off the route, not off an if-chain here: an
        # if-chain is a second place to remember, and the one that describes
        # every family it has not heard of as the family it was written for.
        raise ValueError(
            f"tessera target {target!r}: {family} holds {route['grid_kind']} grid "
            f"{route['grids']}, got {grid!r}")
    if scheme["plane"] != route["plane"]:
        raise ValueError(
            f"tessera target {target!r}: {family} decodes the {route['plane']} scale plane to its "
            f"{route['tile']} tile; plane {scheme['plane']!r} has no {route['short']} tile")
    body = scheme["body"]
    if body not in _BODIES:
        raise ValueError(f"tessera target {target!r}: body must be one of {_BODIES}, got {body!r}")
    q256 = _as_int(scheme, "q256", target)
    _refuse_an_unreadable_rung(family, grid, q256, target)
    rows = _as_int(scheme, "rows", target)
    columns = _as_int(scheme, "columns", target)
    if columns % route["columns_multiple"]:
        raise ValueError(
            f"tessera target {target!r}: the {family} mainloop needs "
            f"K % {route['columns_multiple']} == 0, got {columns}")
    wire_bytes = _as_int(scheme, "wire_bytes", target)
    roles = scheme["roles"]
    if (not isinstance(roles, (list, tuple)) or not roles
            or any(not (isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[0], str)
                        and isinstance(r[1], int) and not isinstance(r[1], bool) and r[1] > 0)
                   for r in roles)):
        raise ValueError(
            f"tessera target {target!r}: roles must be a non-empty list of [name, rows] pairs, "
            f"got {roles!r}")
    if sum(r[1] for r in roles) != rows:
        raise ValueError(
            f"tessera target {target!r}: roles stack to {sum(r[1] for r in roles)} rows, the "
            f"scheme declares {rows}")
    return {
        "family": family, "structure": structure, "grid": grid, "body": body,
        "plane": route["plane"], "q256": q256, "rows": rows, "columns": columns,
        "wire_bytes": wire_bytes, "roles": [(str(n), int(r)) for n, r in roles],
    }


def parse_tessera_blob_for_scheme(blob: bytes, scheme: Mapping, target: str, device="cpu") -> list:
    """Parse a module's fused container and refuse it unless it IS what the
    scheme declared.  Returns ``[(role, ParsedUnit)]`` in stacking order."""
    from tessera import fused, unit_artifact

    declared = validate_tessera_scheme(scheme, target)
    if len(blob) != declared["wire_bytes"]:
        raise ValueError(
            f"tessera target {target!r}: scheme declares wire_bytes={declared['wire_bytes']} but "
            f"the loaded blob is {len(blob)} bytes")
    members = fused.parse_fused(bytes(blob))
    if [(m.name, m.rows) for m in members] != declared["roles"]:
        raise ValueError(
            f"tessera target {target!r}: the container holds roles "
            f"{[(m.name, m.rows) for m in members]} but the scheme declares {declared['roles']}")
    parsed = []
    for member in members:
        unit = unit_artifact.parse_unit_artifact(member.blob, device=device)
        geometry = unit.manifest.geometry
        # The manifest's root rate is per CODE; a code covers ``arity`` weights
        # (``export.encode_linear_planes`` writes ``q256 * grid.arity``).  The
        # scheme speaks per weight, as the exporter's CLI and ``wire_recipe`` do.
        root = int(unit.manifest.branch.root_q256)
        if root % unit.grid.arity:
            raise ValueError(
                f"tessera target {target!r} role {member.name!r}: root_q256={root} is not a whole "
                f"per-weight rate over an arity-{unit.grid.arity} grid")
        actual = {
            "grid": unit.grid.name, "body": unit.body.name,
            "plane": unit.manifest.scale_plane.kind.name, "q256": root // unit.grid.arity,
            "rows": geometry.rows, "columns": geometry.columns,
        }
        expected = {"grid": declared["grid"], "body": declared["body"], "plane": declared["plane"],
                    "q256": declared["q256"], "rows": member.rows, "columns": declared["columns"]}
        if actual != expected:
            raise ValueError(
                f"tessera target {target!r} role {member.name!r}: the wire is {actual} but the "
                f"sidecar scheme declares {expected}; refusing rather than serving bytes no "
                "receipt describes")
        parsed.append((member.name, unit))
    return parsed
