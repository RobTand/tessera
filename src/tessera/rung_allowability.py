"""Non-serving semantic validation and admission of measured rung tables.

Validation returns its input or raises ValueError. Admission returns a decision
with status allow, wait, hold, excluded, unsupported or failed. Missing evidence
waits; this table never replaces export or serving gates.
"""
from __future__ import annotations

import math
import json
from datetime import datetime
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
    _structure(index, INDEX_SCHEMA)
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
    """Validate published structure and measured-row semantics; invalid data refuses."""
    _structure(table, TABLE_SCHEMA)
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
            canonical_low = next(m for m in row["measurements"] if m["cell_id"] == e["cell_id"])
            canonical_high = next(m for m in higher["measurements"] if m["cell_id"] == e["cell_id"])
            low = e.get("lower_measurement", canonical_low)
            high = e.get("higher_measurement", canonical_high)
            # Neighbor-overlap quanta may carry the same higher rung in two
            # paired runs. Its witness is real measured data, not a substituted
            # canonical timing from a different clock window.
            for witness, canonical in ((low, canonical_low), (high, canonical_high)):
                _structure(witness, TABLE_SCHEMA["$defs"]["measurement"], TABLE_SCHEMA)
                _require(_cell(witness) == _cell(canonical), "dominance witness cell scope")
                _require(witness["measurement_status"] == "measured" and witness.get("measurement_build_id") == build["id"], "dominance witness build/status")
                _require(witness["geometry"] == canonical["geometry"] and witness["kernel_path"] == canonical["kernel_path"], "dominance witness kernel/geometry")
                _require(_number(witness["kernel_time_us"]) and witness["kernel_time_us"] > 0, "dominance witness time")
                _require(isinstance(witness.get("pass_times_us"), list) and len(witness["pass_times_us"]) == 2 and all(_number(t) and t > 0 for t in witness["pass_times_us"]), "dominance witness passes")

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

# These are the published structural schemas, not a reduced semantic mirror.
TABLE_SCHEMA = json.loads("{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"$id\":\"fleet.rung_allowability.v1\",\"title\":\"D41 measured rung allowability table\",\"type\":\"object\",\"required\":[\"schema\",\"table_version\",\"table_status\",\"format\",\"kernel_build\",\"generated_at\",\"scope\",\"rungs\"],\"properties\":{\"schema\":{\"const\":\"fleet.rung_allowability.v1\"},\"table_version\":{\"type\":\"integer\",\"minimum\":1},\"table_status\":{\"enum\":[\"partial\",\"complete\"]},\"format\":{\"type\":\"string\",\"minLength\":1},\"generated_at\":{\"type\":\"string\",\"format\":\"date-time\"},\"kernel_build\":{\"type\":\"object\",\"required\":[\"id\",\"source_commit\",\"library_variant\",\"architecture\",\"activation_contract\"],\"properties\":{\"id\":{\"type\":\"string\",\"pattern\":\"^[A-Za-z0-9_.-]+$\"},\"source_commit\":{\"type\":\"string\"},\"library_variant\":{\"type\":\"string\"},\"architecture\":{\"type\":\"string\"},\"activation_contract\":{\"type\":\"string\"},\"metadata\":{\"type\":\"object\"}},\"description\":\"Observed kernel/build and execution context, not a new run-identity seal. A decode-path change republishes a table with affected rows pending until remeasured. Unaffected rows may retain explicit measurement lineage.\"},\"scope\":{\"type\":\"object\",\"required\":[\"rung_min\",\"rung_max\",\"grid_step_q256\",\"grid_owner\",\"required_cells\"],\"properties\":{\"rung_min\":{\"type\":\"integer\",\"minimum\":1},\"rung_max\":{\"type\":\"integer\",\"minimum\":1},\"grid_step_q256\":{\"type\":\"integer\",\"minimum\":1},\"grid_owner\":{\"type\":\"string\"},\"required_cells\":{\"type\":\"array\",\"minItems\":1,\"items\":{\"$ref\":\"#/$defs/cell_key\"}},\"shapes\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"timing_statistic\":{\"type\":\"string\"}},\"description\":\"The scalar q256 domain and exact cell roster are derived from their owners. Initial T-8 scope is R768 through R1152 inclusive, true grid step 1, M 1/16/2048/4096 and 2-3 representative GLM shapes.\"},\"rungs\":{\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/rung\"}},\"evidence\":{\"type\":\"object\"}},\"$defs\":{\"status\":{\"enum\":[\"pending\",\"measured\",\"unsupported\",\"failed\"]},\"cell_key\":{\"type\":\"object\",\"required\":[\"cell_id\",\"kernel_kind\",\"shape_id\",\"M\"],\"properties\":{\"cell_id\":{\"type\":\"string\"},\"kernel_kind\":{\"enum\":[\"routed\",\"dense\"]},\"shape_id\":{\"type\":\"string\"},\"M\":{\"type\":\"integer\",\"minimum\":1}}},\"geometry\":{\"type\":\"object\",\"required\":[\"bits_per_256_weight_tile\",\"alignment\",\"shared_memory\",\"register_pressure\",\"decode_width\"],\"properties\":{\"bits_per_256_weight_tile\":{\"anyOf\":[{\"type\":\"null\"},{\"type\":\"object\",\"required\":[\"numerator\",\"denominator\"],\"properties\":{\"numerator\":{\"type\":\"integer\",\"minimum\":0},\"denominator\":{\"type\":\"integer\",\"minimum\":1}}}]},\"alignment\":{\"type\":[\"object\",\"null\"]},\"shared_memory\":{\"type\":[\"object\",\"null\"]},\"register_pressure\":{\"type\":[\"object\",\"null\"]},\"decode_width\":{\"type\":[\"object\",\"null\"]},\"raw\":{\"type\":\"object\"}},\"description\":\"Actual accountant/launch/compiler fields, per cell when shape or M changes them. Store exact rational tile bits; shared memory records requested/available bytes and fit; registers record compiler resources/spills; decode width records window/value bits, run widths and word stages. Null means missing evidence, never zero-cost evidence.\"},\"measurement\":{\"allOf\":[{\"$ref\":\"#/$defs/cell_key\"}],\"type\":\"object\",\"required\":[\"measurement_status\",\"kernel_time_us\",\"kernel_path\",\"geometry\",\"evidence\"],\"properties\":{\"measurement_status\":{\"$ref\":\"#/$defs/status\"},\"kernel_time_us\":{\"type\":[\"number\",\"null\"],\"minimum\":0},\"kernel_path\":{\"type\":[\"string\",\"null\"]},\"geometry\":{\"anyOf\":[{\"$ref\":\"#/$defs/geometry\"},{\"type\":\"null\"}]},\"evidence\":{\"type\":\"object\"},\"pass_times_us\":{\"type\":\"array\",\"items\":{\"type\":\"number\",\"minimum\":0}},\"measurement_build_id\":{\"type\":\"string\"}}},\"rung\":{\"type\":\"object\",\"required\":[\"rung\",\"measurement_status\",\"supported\",\"anomaly_flags\",\"observations\",\"excluded\",\"dominating_rung\",\"measurements\",\"quality\",\"dominance_evidence\"],\"properties\":{\"rung\":{\"type\":\"integer\",\"minimum\":1},\"measurement_status\":{\"$ref\":\"#/$defs/status\"},\"supported\":{\"type\":[\"boolean\",\"null\"]},\"anomaly_flags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"uniqueItems\":true},\"observations\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"excluded\":{\"type\":\"boolean\"},\"dominating_rung\":{\"type\":[\"integer\",\"null\"],\"minimum\":1},\"measurements\":{\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/measurement\"}},\"quality\":{\"type\":\"object\"},\"dominance_evidence\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"lineage\":{\"type\":\"object\"}},\"allOf\":[{\"if\":{\"properties\":{\"excluded\":{\"const\":true}}},\"then\":{\"properties\":{\"measurement_status\":{\"const\":\"measured\"},\"dominating_rung\":{\"type\":\"integer\",\"minimum\":1}}}}],\"description\":\"Unique scalar q256 rung. measured requires all required cells and geometry plus completed CPU quality screen. excluded is ONLY the measured D41 performance proof: a higher supported rung no farther than one grid step has time <= this rung in every relevant cell, with paired comparable evidence. Quality/correctness anomaly flags separately block production until explained; merely slow or missing-census observations do not fabricate performance exclusion. Missing/unmeasured rows wait. Reader admits only measured, supported, non-excluded rows with empty anomaly_flags, alongside existing serving/export gates. Semantic validation also checks uniqueness, full cell coverage, finite times, same format/build/scope, strict higher-rung/one-step relation and every dominance comparison.\"}}}")
INDEX_SCHEMA = json.loads("{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"$id\":\"fleet.rung_allowability.index.v1\",\"title\":\"Versioned measured rung allowability index\",\"type\":\"object\",\"required\":[\"schema\",\"formats\"],\"properties\":{\"schema\":{\"const\":\"fleet.rung_allowability.index.v1\"},\"formats\":{\"type\":\"object\",\"additionalProperties\":{\"type\":\"object\",\"required\":[\"kernel_builds\"],\"properties\":{\"kernel_builds\":{\"type\":\"object\",\"propertyNames\":{\"pattern\":\"^[A-Za-z0-9_.-]+$\"},\"additionalProperties\":{\"type\":\"object\",\"required\":[\"current_version\",\"versions\"],\"properties\":{\"current_version\":{\"type\":\"integer\",\"minimum\":1},\"versions\":{\"type\":\"object\",\"propertyNames\":{\"pattern\":\"^[1-9][0-9]*$\"},\"additionalProperties\":{\"type\":\"object\",\"required\":[\"path\",\"table_schema\",\"table_status\"],\"properties\":{\"path\":{\"type\":\"string\",\"minLength\":1},\"table_schema\":{\"const\":\"fleet.rung_allowability.v1\"},\"table_status\":{\"enum\":[\"partial\",\"complete\"]}}}}}}}}}}},\"description\":\"Canonical semantic validation is tessera.rung_allowability.validate_index; current_version must select an existing version and table paths must be safe index-root-relative paths.\"}")


def _structure(value, schema, root=None, path="$ "):
    """Evaluate the structural vocabulary used by the two published schemas."""
    root = schema if root is None else root
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        _structure(value, target, root, path)
    for rule in schema.get("allOf", []):
        _structure(value, rule, root, path)
    if "anyOf" in schema:
        for rule in schema["anyOf"]:
            try:
                _structure(value, rule, root, path)
                break
            except ValueError:
                pass
        else:
            raise ValueError(path + ": no schema alternative matches")
    if "if" in schema:
        try:
            _structure(value, schema["if"], root, path)
        except ValueError:
            pass
        else:
            _structure(value, schema.get("then", {}), root, path)
    types = schema.get("type")
    if types:
        types = types if isinstance(types, list) else [types]
        valid = {"object":isinstance(value,dict), "array":isinstance(value,list),
                 "string":isinstance(value,str), "integer":_integer(value),
                 "number":_number(value), "boolean":isinstance(value,bool), "null":value is None}
        _require(any(valid[t] for t in types), path + ": schema type")
    if "const" in schema:
        _require(value == schema["const"], path + ": schema const")
    if "enum" in schema:
        _require(value in schema["enum"], path + ": schema enum")
    if isinstance(value, dict):
        _require(all(k in value for k in schema.get("required", [])), path + ": missing required schema key")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if "propertyNames" in schema:
                _structure(key, schema["propertyNames"], root, path + ".<key>")
            if key in properties:
                _structure(item, properties[key], root, path + "." + key)
            elif isinstance(schema.get("additionalProperties"), dict):
                _structure(item, schema["additionalProperties"], root, path + "." + key)
            elif schema.get("additionalProperties") is False:
                raise ValueError(path + ": extra schema key " + key)
    if isinstance(value, list):
        _require(len(value) >= schema.get("minItems", 0), path + ": schema minimum items")
        if schema.get("uniqueItems"):
            _require(all(item not in value[:i] for i,item in enumerate(value)), path + ": duplicate schema item")
        for i,item in enumerate(value):
            if "items" in schema:
                _structure(item, schema["items"], root, path + f"[{i}]")
    if isinstance(value,str):
        _require(len(value) >= schema.get("minLength",0), path + ": schema minimum length")
        if "pattern" in schema:
            _require(re.search(schema["pattern"],value), path + ": schema pattern")
        if schema.get("format") == "date-time":
            _require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",value), path + ": schema date-time")
            try:
                datetime.fromisoformat(value.upper().replace("Z","+00:00"))
            except ValueError as exc:
                raise ValueError(path + ": schema date-time") from exc
    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        _require(_number(value) and value >= schema["minimum"], path + ": schema minimum")


