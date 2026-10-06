"""The #685 baseline comparison: recorded bands against new panel rows (#688).

 tessera#688's third acceptance line -- "The four routed rows reproduce the
 #685 after-run medians within their IQR" -- needs a consumer that reads the
 preserved raw bench tables and one new validated panel, and owns exactly one
 rule: the recorded band is the band the historical bench itself wrote.  That
 bench (``/mnt/shared/tessera-measurements/kernel-640-pact-bench/bench_linears.py``,
 ``summarize``) records nearest-sample quartiles,

    q(p) = s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]

with Python's round (half-to-even), NOT the interpolated
``statistics.quantiles(n=4, method="inclusive")`` the new receipt's
``timing_panel.timing_summary`` uses.  On the same samples the two rules give
different IQRs (after-table experts.T16 M1: recorded 0.003040, inclusive
0.002992), so rebuilding the recorded band with the panel's own summary rule
would judge acceptance against a band nobody measured.  This module
reconstructs the recorded rule verbatim, proves it against every cell of the
table it is about to compare against, and only then judges containment.

The comparison is identity-gated: numbers are compared only after the row's
(structure, module, family, grid, rate) key matches a documented group, so a
dense row can never be read against a routed stack's band.  A median outside
the recorded band is a named gap, never a silent pass, and nothing here
claims a measurement: the verdict is about the comparison, not the hardware.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "tessera.shape_time_baseline_comparison.v1"
#: The M values tessera#688 requests per row; the historical tables also hold
#: M in 2..8 as bench overhead probes, which are not requested rows.
REQUESTED_MS = (1, 512, 2048, 8192)
#: The four #685 comparison groups, exactly as the #688 handoff comment names
#: them (2026-10-01, "Exactly which four baseline records").  The roster IS
#: the decision here: these are the groups the acceptance compares, so they
#: are pinned with the identity the receipt must observe in the table.
BASELINE_GROUPS = (
    {"selector": "experts.T8", "module": "model.language_model.layers.10.mlp.experts",
     "family": "TESSERA_FP8", "grid": "E4M3", "q256": (1024, 1024),
     "structure": "routed_moe", "kind": "routed",
     "rank_local_shape": ((2048, 4096), (4096, 1024)),
     "reference_dispatch": "fused"},
    {"selector": "experts.T16", "module": "model.language_model.layers.10.mlp.experts",
     "family": "TESSERA_BF16", "grid": "BF16", "q256": (1024, 1024),
     "structure": "routed_moe", "kind": "routed",
     "rank_local_shape": ((2048, 4096), (4096, 1024)),
     "reference_dispatch": "fused, folded"},
    {"selector": "rate.experts.E4M3_R896",
     "module": "model.language_model.layers.43.mlp.experts",
     "family": "TESSERA_FP8", "grid": "E4M3", "q256": (896, 896),
     "structure": "routed_moe", "kind": "routed",
     "rank_local_shape": ((2048, 4096), (4096, 1024)),
     "reference_dispatch": "compact; fused lane refuses"},
    {"selector": "experts.T4", "module": "model.language_model.layers.10.mlp.experts",
     "family": "TESSERA_NVFP4", "grid": "E2M1x2", "q256": (896, 896),
     "structure": "routed_moe", "kind": "routed",
     "rank_local_shape": ((2048, 4096), (4096, 1024)),
     "reference_dispatch": "A4 span2 grouped; not the fused identity"},
)
BENCH_RULE = "s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]"
BENCH_RULE_SOURCE = ("kernel-640-pact-bench/bench_linears.py summarize "
                     "(nearest-sample quartile, Python round)")

#: Two vocabularies for one payload family: the preserved bench stamps
#: ``scheme_family`` (``TESSERA_FP8``/``TESSERA_BF16``/``TESSERA_NVFP4``)
#: while the packaged contract's payload families are the route names
#: (``TESSERA_E4M3_K1``/``TESSERA_BF16_K1``/``TESSERA_E2M1_K2``).  The join
#: lives here and nowhere else; an unknown name refuses rather than joining.
FAMILY_ALIASES = {
    "TESSERA_FP8": "TESSERA_FP8", "TESSERA_E4M3_K1": "TESSERA_FP8",
    "TESSERA_BF16": "TESSERA_BF16", "TESSERA_BF16_K1": "TESSERA_BF16",
    "TESSERA_NVFP4": "TESSERA_NVFP4", "TESSERA_E2M1_K2": "TESSERA_NVFP4",
}


def family_key(family: str) -> str:
    """The canonical bench-side family name for a row or contract name."""
    try:
        return FAMILY_ALIASES[str(family)]
    except KeyError:
        raise ValueError(f"payload family {family!r} has no #685 baseline alias") from None


def bench_quartile(sorted_samples: Sequence[float], p: float) -> float:
    """The historical bench's own quartile, verbatim (see BENCH_RULE)."""
    s = sorted_samples
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def bench_summarize(samples: Sequence[float]) -> dict[str, Any]:
    """The recorded median/p25/p75/IQR/min/n of one timing cell."""
    values = []
    for v in samples:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"baseline timing cell: samples_ms entries must be "
                             f"numbers, got {v!r}")
        values.append(float(v))
    s = sorted(values)
    if len(s) < 3:
        raise ValueError("baseline timing cell: requires at least three samples")
    q25, med, q75 = (bench_quartile(s, 0.25), bench_quartile(s, 0.5),
                     bench_quartile(s, 0.75))
    return {"median_ms": med, "p25_ms": q25, "p75_ms": q75, "iqr_ms": q75 - q25,
            "min_ms": s[0], "n": len(s)}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(raw: bytes, where: str) -> dict[str, Any]:
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{where}: not JSON: {exc}") from exc
    if not isinstance(doc, Mapping) or not isinstance(doc.get("results"), list):
        raise ValueError(f"{where}: requires the raw bench table object with results[]")
    return doc


def verify_recorded_statistics(raw: bytes, *, where: str = "baseline") -> dict[str, Any]:
    """Prove the bench rule reproduces EVERY recorded cell of this table.

    The comparison is meaningless over a table whose recorded statistics this
    rule does not regenerate from its own samples, so the proof is a
    precondition, not a flag: any disagreement refuses.
    """
    doc = _load(raw, where)
    cells = 0
    groups = set()
    for group in doc["results"]:
        if not isinstance(group, Mapping) or not isinstance(group.get("timings"), Mapping):
            raise ValueError(f"{where}: group without timings: {group!r}")
        groups.add(str(group.get("group")))
        for m, cell in sorted(group["timings"].items()):
            if not isinstance(cell, Mapping):
                raise ValueError(f"{where}: {group.get('group')} M={m} timing cell "
                                 "must be a JSON object")
            if not isinstance(cell.get("samples_ms"), list):
                raise ValueError(
                    f"{where}: {group.get('group')} M={m} samples_ms must be an array")
            rebuilt = bench_summarize(cell["samples_ms"])
            for key in ("median_ms", "p25_ms", "p75_ms", "iqr_ms", "min_ms", "n"):
                if rebuilt[key] != cell.get(key):
                    raise ValueError(
                        f"{where}: {group.get('group')} M={m} {key}: recorded "
                        f"{cell.get(key)!r}, {BENCH_RULE_SOURCE} rebuilds {rebuilt[key]!r}; "
                        "this table does not follow the recorded rule")
            cells += 1
    return {"cells_checked": cells, "groups": sorted(groups)}


def row_view(*, structure: str, module: str, family: str, grid: str, q256: tuple,
             m: int, median_ms: float, samples_n: int,
             rank_local_shape: tuple | None = None,
             scope_id: str | None = None, panel_row_index: int | None = None) -> dict[str, Any]:
    """One measured row reduced to the #688 comparison key plus its median."""
    if structure not in ("dense", "routed_moe"):
        raise ValueError(f"row structure {structure!r} is not a panel row structure")
    if type(m) is not int or m < 1:
        raise ValueError("row m must be a positive integer")
    if type(samples_n) is not int or samples_n < 1:
        raise ValueError("row samples_n must be a positive integer count")
    q256 = tuple(q256)
    if not q256 or any(type(q) is not int or q < 1 for q in q256):
        raise ValueError("row q256 must be a nonempty tuple of positive rung integers")
    if rank_local_shape is not None:
        rank_local_shape = _checked_rank_local_shape(rank_local_shape, where="row view")
    return {"structure": structure, "module": str(module), "family": str(family),
            "grid": str(grid), "q256": q256, "rank_local_shape": rank_local_shape,
            "m": m, "median_ms": float(median_ms), "samples_n": samples_n,
            "scope_id": scope_id, "panel_row_index": panel_row_index}


def _checked_rank_local_shape(shape: Any, *, where: str) -> tuple:
    """Rank-local geometry is agreement material, so it must be complete and
    well-formed to count: a nonempty sequence of ``(rows, columns)`` pairs of
    positive integers, refused by name otherwise.  ``None`` means unknown and
    is decided by the caller (``compare`` never treats it as agreement); a
    malformed value never reaches a comparison."""
    if isinstance(shape, (str, bytes)) or not isinstance(shape, (tuple, list)):
        raise ValueError(f"{where}: rank-local shape {shape!r} must be a sequence of "
                         "(rows, columns) integer pairs")
    pairs = []
    for pair in shape:
        if (isinstance(pair, (str, bytes)) or not isinstance(pair, (tuple, list))
                or len(pair) != 2):
            raise ValueError(f"{where}: rank-local shape {shape!r} must be a sequence "
                             "of (rows, columns) integer pairs")
        rows, columns = pair
        if (isinstance(rows, bool) or isinstance(columns, bool)
                or not isinstance(rows, int) or not isinstance(columns, int)
                or rows < 1 or columns < 1):
            raise ValueError(f"{where}: rank-local shape {shape!r} requires positive "
                             "integer (rows, columns) pairs")
        pairs.append((rows, columns))
    if not pairs:
        raise ValueError(f"{where}: rank-local shape must carry at least one "
                         "(rows, columns) pair")
    return tuple(pairs)


def _identity(row_or_group: Mapping[str, Any], fields=("structure", "module", "family",
                                                       "grid", "q256")) -> tuple:
    return tuple(row_or_group.get(field) for field in fields)


def panel_row_views(panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project a validated ``tessera.shape_time_panel.v1`` receipt into row views.

    The caller owns validation (``timing_panel.validate_panel``); this reads
    the validated shape only.  Structures the one-row dense schema does not
    project are named, never dropped.
    """
    from .timing_panel import json_bytes, read_bound
    rows = panel.get("rows")
    plan_rows = (panel.get("plan") or {}).get("rows")
    if not isinstance(rows, list) or not isinstance(plan_rows, list):
        raise ValueError("panel_row_views requires a panel with rows[] and plan.rows[]")
    scopes = {row.get("id"): row.get("scope") for row in plan_rows}
    views = []
    for index, row in enumerate(rows):
        scope = scopes.get(row.get("scope_id"))
        if not isinstance(scope, Mapping):
            raise ValueError(f"panel row {index}: plan scope for scope_id "
                             f"{row.get('scope_id')!r} is absent")
        structure = str(scope.get("structure"))
        if structure != "dense":
            raise ValueError(
                f"panel row {index}: structure {structure!r} has no panel projection in "
                "the tessera.shape_time_panel.v1 schema yet; extend panel_row_views when "
                "the schema grows, do not compare a routed stack by dense projection")
        shape = scope["shape"]
        samples = json_bytes(read_bound(panel["evidence"]["samples"]))
        views.append(row_view(
            structure="dense", module=row["prefix"],
            family=family_key(_route_family(scope["route"])), grid=scope["grid"],
            q256=(scope["q256"],),
            rank_local_shape=((shape["N"], shape["K"]),),
            m=shape["M"], median_ms=row["timing"]["median_ms"],
            samples_n=len(samples["samples_ms"]), scope_id=row.get("scope_id"),
            panel_row_index=index))
    return views


def _route_family(route: str) -> str:
    from .contract import PAYLOAD_FAMILY_BY_ROUTE
    try:
        return PAYLOAD_FAMILY_BY_ROUTE[route]
    except KeyError as exc:
        raise ValueError(f"route {route!r} has no payload family") from exc


def _documented_group(doc: Mapping[str, Any], documented: Mapping[str, Any],
                      where: str) -> Mapping[str, Any]:
    for group in doc["results"]:
        if str(group.get("group")) == documented["selector"]:
            observed = {"module": str(group.get("module")),
                        "family": str((group.get("info") or {}).get("scheme_family")),
                        "grid": str((group.get("info") or {}).get("grid")),
                        "kind": str(group.get("kind"))}
            q = (group.get("info") or {}).get("q256") or {}
            observed_q = (q.get("w13"), q.get("w2"))
            expected = {"module": documented["module"], "family": documented["family"],
                        "grid": documented["grid"], "kind": documented["kind"]}
            if observed != expected or observed_q != tuple(documented["q256"]):
                raise ValueError(
                    f"{where}: group {documented['selector']} is not the documented #685 "
                    f"group: documented {expected} q256={tuple(documented['q256'])}, "
                    f"table holds {observed} q256={observed_q}")
            return group
    raise ValueError(f"{where}: documented group {documented['selector']} is absent "
                     "from this table")


def compare(baseline: bytes, rows: Sequence[Mapping[str, Any]], *, requested_ms=REQUESTED_MS,
            baseline_sha256: str | None = None, baseline_path: str | None = None) -> dict[str, Any]:
    """Judge each documented group at each requested M against the new rows.

    ``rows`` are row views (``row_view``, or ``panel_row_views`` over a
    validated panel).  Verdicts per group and M: ``reproduced`` when the new
    median lies inside the recorded [p25, p75]; ``gap`` outside with the
    numbers; ``no_panel_row`` when no row carries that identity;
    ``identity_mismatch`` when a row matched the module/family/grid/rate key
    but disagreed on the documented rank-local shape; ``geometry_missing``
    when the row carries no rank-local geometry of its own -- unknown
    geometry is not agreement, the reference is never borrowed, and no new
    median is compared; ``insufficient_samples`` when the row carries fewer
    than three samples.  Duplicated rows for one (group, M) refuse.
    """
    where = "baseline"
    proof = verify_recorded_statistics(baseline, where=where)
    digest = _sha256(baseline)
    pin = None
    if baseline_sha256 is not None:
        expected = str(baseline_sha256).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("baseline_sha256 pin: requires 64 lowercase hex digits")
        if expected != digest:
            raise ValueError(
                f"pinned baseline digest differs: pinned {expected}, table {digest}")
        pin = {"expected": expected, "matched": True}
    doc = _load(baseline, where)
    requested = tuple(requested_ms)
    if not requested or any(type(m) is not int or m < 1 for m in requested):
        raise ValueError("requested_ms must be positive integers")
    claims: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("median_ms") is None:
            raise ValueError(f"row {index}: not a row view")
        claims.append(dict(row, family=family_key(row["family"])))
    groups_out: dict[str, Any] = {}
    for documented in BASELINE_GROUPS:
        group = _documented_group(doc, documented, where)
        timings = group["timings"]
        per_m = {}
        for m in requested:
            key = str(m)
            cell = timings.get(key)
            if not isinstance(cell, Mapping):
                per_m[key] = {"verdict": "missing_baseline_m",
                              "reason": f"the preserved table holds no M={m} cell "
                                        f"for {documented['selector']}"}
                continue
            matching = [row for row in claims
                        if _identity(row) == _identity(documented) and row["m"] == m]
            if len(matching) > 1:
                raise ValueError(
                    f"{documented['selector']} M={m}: {len(matching)} panel rows claim the "
                    "same identity; duplicates refuse")
            verdict: dict[str, Any] = {
                "identity": {"module": documented["module"], "family": documented["family"],
                             "grid": documented["grid"], "q256": list(documented["q256"]),
                             "structure": documented["structure"],
                             "rank_local_shape": [list(pair) for pair
                                                  in documented["rank_local_shape"]],
                             "reference_dispatch": documented["reference_dispatch"]}}
            if not matching:
                verdict.update({"verdict": "no_panel_row",
                                "reason": "no supplied row carries this group's "
                                          "(structure, module, family, grid, rate) key at "
                                          f"M={m}"})
                per_m[key] = verdict
                continue
            row = matching[0]
            verdict["identity"]["row"] = {
                "module": row["module"], "family": row["family"], "grid": row["grid"],
                "q256": list(row["q256"]), "structure": row["structure"],
                "rank_local_shape": ([list(pair) for pair in row["rank_local_shape"]]
                                     if row["rank_local_shape"] is not None else None),
                "scope_id": row["scope_id"], "panel_row_index": row["panel_row_index"]}
            row_shape = row.get("rank_local_shape")
            if row_shape is None:
                verdict.update({"verdict": "geometry_missing",
                                "reason": "row carries no rank-local geometry; unknown "
                                          "geometry is not agreement and the documented "
                                          "reference is never borrowed"})
                per_m[key] = verdict
                continue
            row_shape = _checked_rank_local_shape(
                row_shape, where=f"{documented['selector']} M={m} row")
            if row_shape != tuple(tuple(pair) for pair in documented["rank_local_shape"]):
                verdict.update({"verdict": "identity_mismatch",
                                "reason": f"row rank-local shape "
                                          f"{[list(p) for p in row_shape]} "
                                          "differs from the documented #685 reference "
                                          f"{[list(p) for p in documented['rank_local_shape']]}"})
                per_m[key] = verdict
                continue
            recorded = {"median_ms": cell["median_ms"], "p25_ms": cell["p25_ms"],
                        "p75_ms": cell["p75_ms"], "iqr_ms": cell["iqr_ms"],
                        "n": cell["n"]}
            verdict["recorded"] = recorded
            verdict["new"] = {"median_ms": row["median_ms"], "samples_n": row["samples_n"]}
            if row["samples_n"] < 3:
                verdict.update({"verdict": "insufficient_samples",
                                "reason": "acceptance requires at least three CUDA-event "
                                          f"samples; the row carries {row['samples_n']}"})
            elif recorded["p25_ms"] <= row["median_ms"] <= recorded["p75_ms"]:
                verdict["verdict"] = "reproduced"
            else:
                verdict["verdict"] = "gap"
                verdict["reason"] = (
                    f"median {row['median_ms']!r} ms is outside the recorded band "
                    f"[{recorded['p25_ms']!r}, {recorded['p75_ms']!r}] ms")
            per_m[key] = verdict
        groups_out[documented["selector"]] = per_m
    return {"schema": SCHEMA, "issue": "RobTand/tessera#688",
            "comparison": "tessera#685 after-run medians within their recorded IQR",
            "baseline": {"path": baseline_path, "bytes": len(baseline), "sha256": digest},
            "baseline_sha256_pin": pin,
            "bench_rule": {"formula": BENCH_RULE, "source": BENCH_RULE_SOURCE,
                           "proven_reproduction": True,
                           "cells_checked": proof["cells_checked"],
                           "groups_in_table": proof["groups"]},
            "requested_ms": list(requested), "groups": groups_out}
