"""Non-serving semantic validation and admission of measured rung tables.

Validation returns its input or raises ValueError. Admission returns a decision
with status allow, wait, hold, excluded, unsupported or failed. Missing evidence
waits; this table never replaces export or serving gates.
"""
from __future__ import annotations

import math
from pathlib import PurePosixPath
import re

STATUSES = frozenset(("pending", "measured", "unsupported", "failed"))


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _text(value):
    return isinstance(value, str) and bool(value)


def _cell(cell):
    _require(isinstance(cell, dict), "cell must be an object")
    _require(_text(cell.get("cell_id")) and _text(cell.get("shape_id")), "cell identifiers missing")
    _require(cell.get("kernel_kind") in ("routed", "dense"), "unknown kernel kind")
    _require(_integer(cell.get("M")) and cell["M"] > 0, "invalid M")
    return tuple(cell[k] for k in ("cell_id", "kernel_kind", "shape_id", "M"))


def validate_index(index):
    """Validate fleet.rung_allowability.index.v1; paths are index-root relative."""
    _require(isinstance(index, dict) and index.get("schema") == "fleet.rung_allowability.index.v1", "index schema")
    _require(isinstance(index.get("formats"), dict), "index formats")
    for fmt, entry in index["formats"].items():
        _require(_text(fmt) and isinstance(entry, dict) and isinstance(entry.get("kernel_builds"), dict), "format entry")
        for build, item in entry["kernel_builds"].items():
            _require(_text(build) and re.fullmatch(r"[A-Za-z0-9_.-]+", build), "build identifier")
            version = item.get("current_version")
            versions = item.get("versions")
            _require(_integer(version) and version >= 1 and isinstance(versions, dict) and str(version) in versions, "current version missing")
            for number, rec in versions.items():
                _require(isinstance(number, str) and number.isdigit() and str(int(number)) == number and int(number) >= 1, "version key")
                _require(isinstance(rec, dict), "version entry")
                path = rec.get("path")
                _require(_text(path) and "\\" not in path and not PurePosixPath(path).is_absolute() and all(p not in ("", ".", "..") for p in path.split("/")), "unsafe table path")
                _require(rec.get("table_schema") == "fleet.rung_allowability.v1" and rec.get("table_status") in ("partial", "complete"), "table schema/status")
    return index


def validate_table(table):
    """Validate complete roster and all measured-row prerequisites, including dominance."""
    _require(isinstance(table, dict) and table.get("schema") == "fleet.rung_allowability.v1", "table schema")
    _require(_integer(table.get("table_version")) and table["table_version"] >= 1, "table version")
    _require(table.get("table_status") in ("partial", "complete") and _text(table.get("format")), "table status/format")
    build = table.get("kernel_build", {})
    _require(all(_text(build.get(k)) for k in ("id", "source_commit", "library_variant", "architecture", "activation_contract")), "build context missing")
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+", build["id"]), "build identifier")
    scope = table.get("scope", {})
    lo, hi, step = (scope.get(k) for k in ("rung_min", "rung_max", "grid_step_q256"))
    _require(all(_integer(v) and v > 0 for v in (lo, hi, step)) and lo <= hi and (hi-lo) % step == 0, "rung grid")
    _require(_text(scope.get("grid_owner")), "grid owner")
    required = scope.get("required_cells")
    _require(isinstance(required, list) and required, "required cells missing")
    keys = [_cell(c) for c in required]
    _require(len(keys) == len(set(keys)) and len({k[0] for k in keys}) == len(keys), "duplicate required cell")
    rows = table.get("rungs")
    _require(isinstance(rows, list), "rung rows")
    by_rung = {}
    for row in rows:
        _require(isinstance(row, dict), "rung object")
        q = row.get("rung")
        _require(_integer(q) and lo <= q <= hi and (q-lo) % step == 0 and q not in by_rung, "duplicate/out-of-grid rung")
        by_rung[q] = row
        _require(row.get("measurement_status") in STATUSES and (row.get("supported") is None or isinstance(row["supported"], bool)), "rung status/support")
        _require(isinstance(row.get("anomaly_flags"), list) and all(_text(f) for f in row["anomaly_flags"]) and len(set(row["anomaly_flags"])) == len(row["anomaly_flags"]), "anomaly flags")
        _require(isinstance(row.get("observations"), list) and isinstance(row.get("quality"), dict) and isinstance(row.get("dominance_evidence"), list), "rung evidence fields")
        _require(isinstance(row.get("excluded"), bool), "exclusion flag")
        measurements = row.get("measurements")
        _require(isinstance(measurements, list), "measurements")
        seen = set()
        for m in measurements:
            key = _cell(m)
            _require(key in keys and key not in seen, "unknown/duplicate measurement cell")
            seen.add(key)
            _require(m.get("measurement_status") in STATUSES and isinstance(m.get("evidence"), dict), "measurement status/evidence")
            t = m.get("kernel_time_us")
            _require(t is None or _number(t), "invalid kernel time")
            if m["measurement_status"] == "measured":
                _require(_number(t) and t > 0 and _text(m.get("kernel_path")), "measured kernel time/path")
                _require(m.get("measurement_build_id") == build["id"], "measurement build scope")
                geom = m.get("geometry")
                _require(isinstance(geom, dict) and all(isinstance(geom.get(k), dict) for k in ("bits_per_256_weight_tile", "alignment", "shared_memory", "register_pressure", "decode_width")), "measured geometry missing")
                bits = geom["bits_per_256_weight_tile"]
                _require(_integer(bits.get("numerator")) and bits["numerator"] > 0 and _integer(bits.get("denominator")) and bits["denominator"] > 0, "tile-bit rational")
                sm = geom["shared_memory"]
                _require(_integer(sm.get("requested_bytes")) and _integer(sm.get("available_bytes")) and 0 < sm["requested_bytes"] <= sm["available_bytes"] and sm.get("fits") is True, "shared-memory fit")
                reg = geom["register_pressure"]
                _require(_integer(reg.get("REG")) and reg["REG"] > 0 and all(_integer(reg.get(k)) and reg[k] >= 0 for k in ("STACK", "LOCAL", "SHARED")), "compiler resources missing")
                _require(isinstance(m.get("pass_times_us"), list) and len(m["pass_times_us"]) == 2 and all(_number(t) and t > 0 for t in m["pass_times_us"]), "paired pass times")
        if row["measurement_status"] == "measured":
            _require(row["supported"] is True and seen == set(keys) and all(m["measurement_status"] == "measured" for m in measurements), "measured row incomplete")
            quality = row["quality"]
            _require(quality.get("measurement_status") == "measured" and quality.get("source_kind") == "actual_sampled_expert_weights" and quality.get("device") == "cpu" and quality.get("samples"), "CPU quality incomplete")
            for sample in quality["samples"]:
                _require(_number(sample.get("relative_sse")) and _number(sample.get("source_squared_norm")) and sample["source_squared_norm"] > 0 and _integer(sample.get("exact_bytes")) and sample["exact_bytes"] > 0 and _text(sample.get("source_sha256")), "quality sample evidence")
        else:
            _require(not row["excluded"], "unmeasured rung excluded")
        if not row["excluded"]:
            _require(row.get("dominating_rung") is None and not row["dominance_evidence"], "spurious dominance")
    _require(set(by_rung) == set(range(lo, hi+1, step)), "missing census rung")
    if table["table_status"] == "complete":
        _require(all(r["measurement_status"] != "pending" for r in rows), "complete table has pending rows")
    for row in rows:
        if not row["excluded"]:
            continue
        q = row["rung"]
        dq = row.get("dominating_rung")
        _require(_integer(dq) and q < dq <= q+step and dq in by_rung, "dominator not adjacent higher")
        higher = by_rung[dq]
        _require(higher["measurement_status"] == "measured" and higher["supported"] is True and not higher["anomaly_flags"], "dominator ineligible")
        evidence = row["dominance_evidence"]
        _require(len(evidence) == len(keys) and {e.get("cell_id") for e in evidence} == {k[0] for k in keys}, "dominance cell coverage")
        for e in evidence:
            low = next(m for m in row["measurements"] if m["cell_id"] == e["cell_id"])
            high = next(m for m in higher["measurements"] if m["cell_id"] == e["cell_id"])
            _require(e.get("lower_time_us") == low["kernel_time_us"] and e.get("higher_time_us") == high["kernel_time_us"] <= low["kernel_time_us"], "dominance times")
            for field in ("comparison_id", "paired_seed_contract", "timing_statistic", "timer"):
                _require(_text(low["evidence"].get(field)) and low["evidence"].get(field) == high["evidence"].get(field), "unpaired dominance: " + field)
            _require(e.get("comparison_id") == low["evidence"]["comparison_id"], "dominance lineage")
    return table


def admit_rung(table, *, format, kernel_build_id, rung, scope=None):
    """Return a structured decision. Invalid evidence raises ValueError, not allow."""
    if table is None:
        return {"status": "wait", "reason": "missing_table", "rung": rung}
    validate_table(table)
    if table["format"] != format or table["kernel_build"]["id"] != kernel_build_id:
        return {"status": "wait", "reason": "unmeasured_format_or_build", "rung": rung}
    if scope is not None and any(table["scope"].get(k) != v for k, v in scope.items()):
        return {"status": "wait", "reason": "unmeasured_scope", "rung": rung}
    row = next((r for r in table["rungs"] if r["rung"] == rung), None)
    if row is None or row["measurement_status"] == "pending":
        status, reason = "wait", "missing_measurement"
    elif row["measurement_status"] in ("unsupported", "failed"):
        status, reason = row["measurement_status"], row["measurement_status"]
    elif row["anomaly_flags"]:
        status, reason = "hold", "quality_or_correctness_anomaly"
    elif row["excluded"]:
        status, reason = "excluded", "adjacent_higher_all_cell_dominance"
    else:
        status, reason = "allow", "measured_supported_no_anomaly"
    return {"status": status, "reason": reason, "rung": rung, "table_version": table["table_version"], "kernel_build_id": kernel_build_id}
