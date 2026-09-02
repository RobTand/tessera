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
hardware, not the body inside the blob.  Two families, one plugin:

* ``TESSERA_NVFP4`` is any E2M1-based grid over a LUT scale plane, decoded to
  the stock NVFP4 tile (nibble-packed E2M1 codes + group-16 ue4m3 block scales
  + one global) and served through ``torch._scaled_mm`` W4A4.
* ``TESSERA_FP8`` is the scalar E4M3 grid over the CHANNEL scale plane (schema
  minor 3: one fp16 word per output row times a global), decoded to the stock
  per-channel FP8 pair (E4M3 bytes + one fp32 scale per row) and served through
  ``torch._scaled_mm`` W8A8.

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
    "TESSERA_NVFP4",
    "TESSERA_FP8",
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

TESSERA_NVFP4 = "TESSERA_NVFP4"
TESSERA_FP8 = "TESSERA_FP8"

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
        "builder": ("tessera.serving.nvfp4_route", "build_tessera_nvfp4_method"),
        "tile": "nvfp4 (packed E2M1 codes, group-16 ue4m3 block scales, one global)",
        "columns_multiple": 16,
        "activation_contract": NVFP4_ACTIVATION_CONTRACT,
    },
    TESSERA_FP8: {
        "grids": ("E4M3",), "plane": "CHANNEL",
        "short": "FP8",
        "builder": ("tessera.serving.fp8_route", "build_tessera_fp8_method"),
        "tile": "fp8 per-channel (E4M3 bytes, one fp32 scale per row)",
        "columns_multiple": 16,
        "activation_contract": FP8_ACTIVATION_CONTRACT,
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
        kind = "an E2M1-based" if family == TESSERA_NVFP4 else "the scalar E4M3"
        raise ValueError(
            f"tessera target {target!r}: {family} holds {kind} grid {route['grids']}, got {grid!r}")
    if scheme["plane"] != route["plane"]:
        raise ValueError(
            f"tessera target {target!r}: {family} decodes the {route['plane']} scale plane to its "
            f"{route['tile']} tile; plane {scheme['plane']!r} has no {route['short']} tile")
    body = scheme["body"]
    if body not in _BODIES:
        raise ValueError(f"tessera target {target!r}: body must be one of {_BODIES}, got {body!r}")
    q256 = _as_int(scheme, "q256", target)
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
