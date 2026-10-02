"""CPU-only census planning. Plans are not timing or device receipts (#689)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from . import scheme
from .contract import (EXECUTION_MODES, contract_path, reader_accepts,
                       reader_rate_grid, validate_serving_contract)

PLAN_SCHEMA = "tessera.native_census_plan.v1"
_REQUEST_FIELDS = frozenset({
    "route", "grid", "q256", "structure", "mode", "execution_mode", "regime",
    "tp_degree", "requested_platform", "shape",
})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer, not {value!r}")
    return value


def _request_scope(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("request must be an object")
    missing = _REQUEST_FIELDS - request.keys()
    extra = request.keys() - _REQUEST_FIELDS
    if missing or extra:
        raise ValueError(f"request fields: missing={sorted(missing)}, extra={sorted(extra)}")
    for field in _REQUEST_FIELDS - {"shape", "q256", "tp_degree"}:
        if not isinstance(request[field], str) or not request[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    route = request["route"]
    if route not in scheme.ROUTES:
        raise ValueError(f"route {route!r} is not in the dispatch registry")
    if request["grid"] not in scheme.ROUTES[route]["grids"]:
        raise ValueError(f"grid {request['grid']!r} is not held by route {route!r}")
    structure = request["structure"]
    if structure not in scheme.STRUCTURES:
        raise ValueError(f"structure {structure!r} is not in the dispatch registry")
    modes = {mode for launch in scheme.ROUTE_LAUNCHES[route]
             for mode in launch["modes"]}
    if request["mode"] not in modes:
        raise ValueError(f"mode {request['mode']!r} is not in the dispatch registry")
    if request["execution_mode"] not in EXECUTION_MODES:
        raise ValueError(f"execution_mode {request['execution_mode']!r} is not supported")
    q256 = _positive_integer(request["q256"], "q256")
    tp = _positive_integer(request["tp_degree"], "tp_degree")
    fields = {"M", "N", "K"}
    if structure == scheme.STRUCTURE_ROUTED_MOE:
        fields |= {"experts", "topk"}
    shape = request["shape"]
    if not isinstance(shape, Mapping) or set(shape) != fields:
        raise ValueError(f"shape must contain exactly {sorted(fields)}")
    shape = {field: _positive_integer(shape[field], f"shape.{field}")
             for field in sorted(fields)}
    if structure == scheme.STRUCTURE_ROUTED_MOE and shape["topk"] > shape["experts"]:
        raise ValueError("shape.topk must not exceed shape.experts")
    if request["regime"] != scheme.regime_of_m(shape["M"]):
        raise ValueError("regime disagrees with the shared M-to-regime rule")
    return {**dict(request), "q256": q256, "tp_degree": tp, "shape": shape}


def build_census_plan(requests: Iterable[Mapping[str, Any]], *,
                      raw_contract: bytes | None = None) -> dict[str, Any]:
    """Bind requested rank-local scopes to the current registry, without execution.

    N/K are supplied rank-local dimensions, not a TP geometry derived here.
    The published range is a dense-reader fact, not routed or device admission.
    No image identity, native availability, wire validity, or qualification is
    inferred. Every row still needs actual preparation and device evidence.
    """
    try:
        raw_contract = contract_path().read_bytes() if raw_contract is None else raw_contract
        contract = json.loads(raw_contract)
        validate_serving_contract(contract)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read valid packaged runtime contract: {exc}") from exc
    return _build_validated_census_plan(requests, raw_contract=raw_contract, contract=contract)


def _build_validated_census_plan(requests: Iterable[Mapping[str, Any]], *,
                                 raw_contract: bytes, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Private pure planner after the owning runtime has strictly validated bytes.

    Local callers enter through build_census_plan. External timing callers must
    first obtain the source- and request-bound installed-runtime preflight proof.
    This core neither changes admission nor loads an extension.
    """
    contract_sha = hashlib.sha256(raw_contract).hexdigest()
    registry_sha = hashlib.sha256(_canonical({
        "routes": scheme.ROUTES, "launches": scheme.ROUTE_LAUNCHES,
        "experimental_pairs": sorted(scheme.EXPERIMENTAL_LAUNCHES),
    })).hexdigest()
    rows, seen = [], set()
    for request in requests:
        scope = _request_scope(request)
        identity = {"scope": scope, "contract_sha256": contract_sha,
                    "registry_sha256": registry_sha}
        row_id = hashlib.sha256(_canonical(identity)).hexdigest()
        if row_id in seen:
            raise ValueError(f"duplicate request scope {row_id}")
        seen.add(row_id)
        published = reader_rate_grid(scope["route"], scope["grid"], contract)
        reader = None
        if published is not None:
            family, low, high, step = published
            reader = {"family": family, "low_q256": low, "high_q256": high,
                      "step_q256": step,
                      "accepts_q256": reader_accepts(scope["q256"], low, high, step)}
        pairs = scheme.launch_pairs(scope["route"], structure=scope["structure"],
                                    regime=scope["regime"], mode=scope["mode"])
        rows.append({"id": row_id, "scope": scope,
                     "published_dense_reader": reader,
                     "reader_scope": "dense_range_only_not_routed_admission",
                     "admissible_launch_pairs": [list(pair) for pair in sorted(pairs)],
                     "qualification": "not_measured", "measurement": None})
    if not rows:
        raise ValueError("requests must contain at least one scope")
    return {"schema": PLAN_SCHEMA, "status": "not_executed", "gpu_executed": False,
            "contract_sha256": contract_sha, "registry_sha256": registry_sha,
            "rows": sorted(rows, key=lambda row: row["id"])}


def build_timing_requirements(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Describe missing GPU evidence; never manufacture a timing receipt (#688).

    Rebuild the plan against the current owner tables so stale scopes or supplied
    measurements cannot be laundered into an evidence request. The three-sample
    minimum is the issue's explicit acceptance, not a statistical assertion.
    """
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan must use the current census planning schema")
    try:
        rebuilt = build_census_plan([row["scope"] for row in plan["rows"]])
        if _canonical(plan) != _canonical(rebuilt):
            raise ValueError("plan differs from the current unmeasured owner-derived plan")
    except (KeyError, TypeError) as exc:
        raise ValueError("plan is malformed") from exc
    return {
        "schema": "tessera.kernel_timing_requirements.v1",
        "status": "not_executed",
        "plan_sha256": hashlib.sha256(_canonical(rebuilt)).hexdigest(),
        "minimum_samples": 3,
        "method": "cuda_events",
        "required_evidence": [
            "cuda_event_samples", "torch_profiler", "netdata",
            "observed_runtime_identity", "wire_identity", "native_preparation",
            "route_census",
        ],
        "rows": [{"scope_id": row["id"], "measurement": None}
                 for row in rebuilt["rows"]],
    }
