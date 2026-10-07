"""T-4 route admission against the frozen fused E2M1 interface.

The executive lane plan fixes the interface. This module only reads it.
It never builds a kernel, never loads an extension, and never claims
a serve. Every check runs on the CPU and refuses by name.
"""
from __future__ import annotations

from typing import Any, Mapping

from .structure import STRUCTURES, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE

__all__ = [
    "T4_EXECUTION_MODES",
    "T4_PAYLOAD_FAMILY",
    "T4_PURE_Q256",
    "T4_ROUTE",
    "ATTESTATION_SCHEMA",
    "build_attestation_stub",
    "build_preflight",
    "expected_pairs",
    "require_census_pair",
    "require_dense_geometry",
    "require_execution_mode",
    "require_pure_q256",
    "require_routed_geometry",
    "require_routed_rates",
    "require_served_recipe",
    "require_structure",
    "t4_cell_agreement",
    "t4_census_expected",
]

#: The eight pure pair widths. A rung is q256 per position.
T4_PURE_Q256 = (128, 256, 384, 512, 640, 768, 896, 1024)

#: The two execution paths the admission covers. Graph remains scaffolding.
T4_EXECUTION_MODES = ("eager", "graph")

#: The payload family the T-4 wires carry.
T4_PAYLOAD_FAMILY = "TESSERA_E2M1_K2"

#: The route that serves T-4.
T4_ROUTE = "TESSERA_NVFP4"

#: The attestation stub schema. A stub is never a serving receipt.
ATTESTATION_SCHEMA = "tessera.t4-admission-attestation.v1"


def require_pure_q256(q256: Any) -> int:
    """Return ``q256`` when it names one pure T-4 class."""
    if type(q256) is not int or q256 not in T4_PURE_Q256:
        raise ValueError(
            f"t4_admission: impure rung {q256!r}; "
            f"admit one of {list(T4_PURE_Q256)}"
        )
    return q256


def require_structure(structure: Any) -> str:
    """Return ``structure`` when this build dispatches it."""
    if structure not in STRUCTURES:
        raise ValueError(
            f"t4_admission: unknown structure {structure!r}; "
            f"admit one of {list(STRUCTURES)}"
        )
    return structure


def require_execution_mode(mode: Any) -> str:
    """Return ``mode`` when it names a covered execution path."""
    if mode not in T4_EXECUTION_MODES:
        raise ValueError(
            f"t4_admission: unknown execution mode {mode!r}; "
            f"admit one of {list(T4_EXECUTION_MODES)}"
        )
    return mode


def _field_name(value: Any) -> str:
    name = getattr(value, "name", value)
    return str(name)


def require_served_recipe(recipe: Any) -> Any:
    """Return ``recipe`` when it is the frozen served wire.

    The served wire is WINDOW, span 1, LUT plane, window_bits 14.
    The check reads the recipe object only. It encodes nothing.
    """
    try:
        body = recipe.body if not isinstance(recipe, Mapping) else recipe["body"]
        span = recipe.span if not isinstance(recipe, Mapping) else recipe["span"]
        plane = (
            recipe.scale_plane
            if not isinstance(recipe, Mapping)
            else recipe["scale_plane"]
        )
        window_bits = (
            recipe.window_bits
            if not isinstance(recipe, Mapping)
            else recipe["window_bits"]
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"t4_admission: served recipe lacks a required field: {exc}"
        ) from exc
    if _field_name(body) != "WINDOW":
        raise ValueError(
            f"t4_admission: served recipe body is {_field_name(body)!r}, not 'WINDOW'"
        )
    if int(span) != 1:
        raise ValueError(
            f"t4_admission: served recipe span is {span!r}, not 1"
        )
    if _field_name(plane) != "LUT":
        raise ValueError(
            f"t4_admission: served recipe plane is {_field_name(plane)!r}, not 'LUT'"
        )
    if int(window_bits) != 14:
        raise ValueError(
            f"t4_admission: served recipe window_bits is {window_bits!r}, not 14"
        )
    return recipe


def require_dense_geometry(rows: Any, cols: Any, projection: Any = None) -> None:
    """Refuse a dense shape no native E2M1 launch serves."""
    from .serving.scheme import e2m1_shape_reason

    try:
        rows_i, cols_i = int(rows), int(cols)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"t4_admission: dense shape must be integers, got {rows!r}x{cols!r}"
        ) from exc
    reason = e2m1_shape_reason(
        rows_i, cols_i, structure=STRUCTURE_DENSE, projection=projection
    )
    if reason is not None:
        raise ValueError(f"t4_admission: dense geometry refused: {reason}")


def require_routed_geometry(rows: Any, cols: Any, projection: Any) -> None:
    """Refuse a routed shape no native E2M1 launch serves."""
    from .serving.scheme import e2m1_shape_reason

    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(
            f"t4_admission: unknown routed projection {projection!r}; "
            "admit gate_proj, up_proj, or down_proj"
        )
    try:
        rows_i, cols_i = int(rows), int(cols)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"t4_admission: routed shape must be integers, got {rows!r}x{cols!r}"
        ) from exc
    reason = e2m1_shape_reason(
        rows_i,
        cols_i,
        structure=STRUCTURE_ROUTED_MOE,
        projection=projection,
    )
    if reason is not None:
        raise ValueError(f"t4_admission: routed geometry refused: {reason}")


def require_routed_rates(rungs: Any) -> list:
    """Return the normalized rate matrix when the lane reads it.

    Each row holds one expert in down, gate/up, or gate/up/down order.
    Every rung must name one pure T-4 class. Experts must agree and
    gate and up must share one stride. Anything else refuses by name.
    """
    if not isinstance(rungs, (list, tuple)) or not rungs:
        raise ValueError("t4_admission: routed rates must be a non-empty matrix")
    matrix = []
    width = None
    for row in rungs:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("t4_admission: routed rates rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("t4_admission: routed rates rows disagree in width")
        clean = []
        for rung in row:
            if type(rung) is not int or rung not in T4_PURE_Q256:
                raise ValueError(
                    f"t4_admission: impure routed rung {rung!r}; "
                    f"admit one of {list(T4_PURE_Q256)}"
                )
            clean.append(rung)
        matrix.append(clean)
    from .serving.scheme import e2m1_expert_rate_reason

    reason = e2m1_expert_rate_reason(matrix)
    if reason is not None:
        raise ValueError(f"t4_admission: routed rates refused: {reason}")
    return matrix


def t4_census_expected(structure: str, regime: Any = None) -> set:
    """The ``(symbol, decoder)`` pairs one T-4 structure may report.

    The set derives from the shared launch table. It attests nothing.
    """
    from .serving import scheme as serving_scheme

    require_structure(structure)
    if regime is not None and regime not in ("decode", "batch"):
        raise ValueError(
            f"t4_admission: unknown regime {regime!r}; admit decode, batch, or None"
        )
    narrow: dict = {"structure": structure, "include_experimental": True}
    if regime is not None:
        narrow["regime"] = regime
    return set(serving_scheme.launch_pairs(T4_ROUTE, **narrow))


def expected_pairs(structure: str, regime: Any = None) -> set:
    """Alias for :func:`t4_census_expected` for census callers."""
    return t4_census_expected(structure, regime)


def require_census_pair(
    structure: str, symbol: Any, decoder: Any, regime: Any = None
) -> tuple:
    """Return ``(symbol, decoder)`` when the structure may report it."""
    pairs = t4_census_expected(structure, regime)
    if (symbol, decoder) not in pairs:
        raise ValueError(
            f"t4_admission: unexpected census pair {(symbol, decoder)!r} "
            f"for structure {structure!r}"
        )
    return (symbol, decoder)


def t4_cell_agreement(record: Mapping, expected: set) -> tuple:
    """Check one census record against the expected pairs.

    Returns ``(agrees, problem)``. A record without a symbol and a
    decoder disagrees. The check never executes a kernel.
    """
    if not isinstance(record, Mapping):
        raise ValueError("t4_admission: census record must be a mapping")
    symbol = record.get("symbol")
    decoder = record.get("decoder")
    if symbol is None or decoder is None:
        return (False, "t4_admission: census record names no pair")
    if (symbol, decoder) in expected:
        return (True, None)
    return (False, f"t4_admission: census pair {(symbol, decoder)!r} is unexpected")


def build_attestation_stub(scope: Mapping, execution_mode: str) -> dict:
    """Return a CPU-only attestation stub for one admitted scope.

    The stub records what was checked and states plainly that no GPU
    ran and nothing is qualified. A stub is scaffolding, not evidence.
    """
    if not isinstance(scope, Mapping):
        raise ValueError("t4_admission: attestation scope must be a mapping")
    mode = require_execution_mode(execution_mode)
    structure = require_structure(scope.get("structure"))
    pairs = sorted(t4_census_expected(structure))
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "not_measured",
        "gpu_executed": False,
        "structure": structure,
        "execution_mode": mode,
        "q256": scope.get("q256"),
        "payload_family": T4_PAYLOAD_FAMILY,
        "expected_pairs": [list(pair) for pair in pairs],
        "qualification": "not_measured",
        "serving": "not_attested",
    }


def build_preflight(scope: Mapping) -> dict:
    """Admit one T-4 unit scope on the CPU or refuse it by name.

    ``scope`` holds ``structure``, ``execution_mode``, and either a
    dense ``q256`` with ``rows`` and ``columns`` or routed ``rungs``
    with per-projection shapes. The receipt states the checks that
    passed and carries an attestation stub. It never claims a serve.
    """
    if not isinstance(scope, Mapping):
        raise ValueError("t4_admission: preflight scope must be a mapping")
    structure = require_structure(scope.get("structure"))
    mode = require_execution_mode(scope.get("execution_mode"))
    regime = scope.get("regime")
    if regime is not None and regime not in ("decode", "batch"):
        raise ValueError(
            f"t4_admission: unknown regime {regime!r}; admit decode, batch, or None"
        )
    checks: list = [f"structure:{structure}", f"execution_mode:{mode}"]
    if structure == STRUCTURE_DENSE:
        q256 = require_pure_q256(scope.get("q256"))
        checks.append(f"q256:{q256}")
        if "recipe" in scope and scope["recipe"] is not None:
            require_served_recipe(scope["recipe"])
            checks.append("served_recipe:WINDOW-span1-LUT-L14")
        require_dense_geometry(
            scope.get("rows"), scope.get("columns"), scope.get("projection")
        )
        checks.append(f"geometry:{scope.get('rows')}x{scope.get('columns')}")
        if "symbol" in scope or "decoder" in scope:
            require_census_pair(
                structure, scope.get("symbol"), scope.get("decoder"), regime
            )
            checks.append(f"pair:{scope.get('symbol')}+{scope.get('decoder')}")
        stub_scope = {"structure": structure, "q256": q256}
    elif structure == STRUCTURE_ROUTED_MOE:
        matrix = require_routed_rates(scope.get("rungs"))
        checks.append(f"rungs:{len(matrix)}x{len(matrix[0])}")
        shapes = scope.get("shapes")
        if not isinstance(shapes, Mapping):
            raise ValueError("t4_admission: routed scope needs a shapes mapping")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            if projection not in shapes:
                raise ValueError(
                    f"t4_admission: routed scope lacks {projection} shape"
                )
            rows, cols = shapes[projection]
            require_routed_geometry(rows, cols, projection)
            checks.append(f"geometry:{projection}:{rows}x{cols}")
        if "recipe" in scope and scope["recipe"] is not None:
            require_served_recipe(scope["recipe"])
            checks.append("served_recipe:WINDOW-span1-LUT-L14")
        if "symbol" in scope or "decoder" in scope:
            require_census_pair(
                structure, scope.get("symbol"), scope.get("decoder"), regime
            )
            checks.append(f"pair:{scope.get('symbol')}+{scope.get('decoder')}")
        stub_scope = {"structure": structure, "q256": sorted({r for row in matrix for r in row})}
    else:
        raise ValueError(f"t4_admission: unknown structure {structure!r}")
    stub = build_attestation_stub(stub_scope, mode)
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "admitted_cpu",
        "gpu_executed": False,
        "structure": structure,
        "execution_mode": mode,
        "checks": checks,
        "attestation": stub,
        "qualification": "not_measured",
        "serving": "not_attested",
    }

