"""Passive native dense timing receipts (#688); no torch, vLLM, PQ or PB.

The first slice is one FP8 dense TP1 eager/resident cell. A receipt observes
an operator; it publishes no runtime cell, serving quality or placement.
The native producer parses grid/profile semantics through its ordinary loader.
This replay checks canonical framing/container bytes and their preparation,
runtime, dispatched lane and measured evidence against an independent context.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from collections.abc import Mapping

from ..container import parse
from ..errors import TesseraError
from ..fused_frame import parse_fused
from . import census_plan, scheme
from .contract import (PAYLOAD_FAMILY_BY_ROUTE, cell_covers_rung, cell_runtime_scope,
                       cell_runtime_code, cell_runtime_versions, cell_residency_modes,
                       cell_is_device_backed, refuse_unevaluated_predicates,
                       validate_serving_contract, require_runtime_image)
from .census import cell_launch_agreement

SCHEMA = "tessera.shape_time_panel.v1"
CLAIMS = {"time_claim": "operator_sum_proposal", "certifies_placement": False,
          "served_p95": "not_claimed"}
RUNTIME_FIELDS = {"image", "tessera_commit", "serving_source_sha256", "contract_sha256",
                  "platform", "torch", "vllm", "serve_flags", "execution_mode",
                  "residency", "tp_rank", "tp_degree"}
EVIDENCE = {"runtime", "producer", "contract", "wire", "preparation", "samples",
            "routes", "trace", "telemetry", "native_binary"}
NETDATA_CONTEXTS = {"nvidia_smi.gpu_power_draw", "system.cpu", "system.load",
                    "mem.swapio", "mem.available"}


def canonical(value):
    return census_plan._canonical(value)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _object(value, fields, where):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{where}: requires exactly {sorted(fields)}")
    return value


def _integer(value, where, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: requires integer >= {minimum}")
    return value


def _number(value, where, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{where}: requires a finite {'positive ' if positive else ''}number")
    return float(value)


def _sha(value, where, length=64):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{where}: requires {length} lowercase hex digits")
    return value


def _pairs(items):
    body = {}
    for key, value in items:
        if key in body:
            raise ValueError(f"duplicate JSON key {key!r}")
        body[key] = value
    return body


def json_bytes(raw):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def file_binding(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_bound(bound):
    _object(bound, {"path", "bytes", "sha256"}, "file evidence")
    _integer(bound["bytes"], "file evidence.bytes")
    _sha(bound["sha256"], "file evidence.sha256")
    if not isinstance(bound["path"], str) or not bound["path"]:
        raise ValueError("file evidence.path: requires a nonempty string")
    path = Path(bound["path"])
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("file evidence.path: requires an absolute regular file")
    raw = path.read_bytes()
    if len(raw) != bound["bytes"] or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError(f"file evidence differs: {path}")
    return raw


def timing_summary(samples):
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("samples: requires at least three CUDA-event samples")
    values = [_number(v, "samples_ms") for v in samples]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"method": "cuda_events", "n": len(values), "median_ms": statistics.median(values),
            "p25_ms": q1, "p75_ms": q3, "iqr_ms": q3 - q1,
            "quartiles": "statistics.quantiles.inclusive"}


def runtime_context(value):
    _object(value, RUNTIME_FIELDS, "runtime")
    require_runtime_image(value["image"])
    _sha(value["tessera_commit"], "runtime.tessera_commit", 40)
    for field in ("serving_source_sha256", "contract_sha256"):
        _sha(value[field], "runtime." + field)
    for field in ("platform", "torch", "vllm"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError("runtime." + field + ": requires observed nonempty string")
    if value["execution_mode"] != "eager" or value["residency"] != "resident":
        raise ValueError("first slice supports eager/resident only")
    if type(value["tp_degree"]) is not int or value["tp_degree"] != 1 or type(value["tp_rank"]) is not int or value["tp_rank"] != 0:
        raise ValueError("first slice supports an explicit TP1 rank0 layer only")
    if not isinstance(value["serve_flags"], dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value["serve_flags"].items()):
        raise ValueError("runtime.serve_flags: requires observed string values")
    return dict(value)


def wire_facts(blob, declaration):
    """Canonical byte structure, without importing the tensor grid decoder."""
    _object(declaration, {"family", "structure", "grid", "body", "plane", "q256",
                          "rows", "columns", "roles", "wire_bytes"}, "dense scheme")
    declared = scheme.validate_tessera_scheme(declaration, "shape-panel")
    if declared["family"] != scheme.TESSERA_FP8 or declared["structure"] != "dense" or declared["grid"] != "E4M3":
        raise ValueError("first slice supports E4M3 FP8 dense only")
    if len(set(declared["role_q256"])) != 1:
        raise ValueError("first slice requires one uniform operator rung")
    if len(blob) != declared["wire_bytes"]:
        raise ValueError("wire length differs from dense scheme")
    try:
        members = parse_fused(blob)
    except TesseraError as exc:
        raise ValueError(f"canonical fused wire refuses: {exc}") from exc
    if [[m.name, m.rows] for m in members] != [list(r) for r in declared["roles"]] or len({m.name for m in members}) != len(members):
        raise ValueError("wire roles differ from dense scheme")
    roles = []
    for member, q in zip(members, declared["role_q256"]):
        try:
            parsed = parse(member.blob)
        except TesseraError as exc:
            raise ValueError(f"canonical unit wire refuses: {exc}") from exc
        m = parsed.manifest
        actual = (m.geometry.rows, m.geometry.columns, m.branch.root_q256, m.body.name,
                  m.scale_plane.kind.name, m.span)
        expected = (member.rows, declared["columns"], q, declared["body"], declared["plane"],
                    scheme.ROUTES[declared["family"]]["span"])
        if actual != expected:
            raise ValueError(f"canonical wire geometry/rung/recipe differs: {member.name}")
        counts = dict(zip((kind.name for kind in m.plane_order), parsed.terminal.plane_elements))
        facts = {"rates": list(m.rates), "window_bits": m.window_bits, "body": m.body.name,
                 "plane": m.scale_plane.kind.name, "release_overrides": counts.get("RELEASE", 0),
                 # CHANNEL's DIAG_SV is its row scale, not a rotation diagonal.
                 "diagonals": bool(counts.get("DIAG_SU", 0)),
                 "start_state": bool(m.shard and m.shard.has_initial_state),
                 "rotation": m.branch.rotation.name, "grid_arity": 1, "structure": "dense"}
        roles.append({"name": member.name, "rows": member.rows, "facts": facts,
                      "unit_sha256": hashlib.sha256(member.blob).hexdigest()})
    return declared, roles


def admitted_cell(contract, scope, runtime, pair, roles):
    """Positive coverage, named launch, runtime code and the actual wire predicate."""
    family = PAYLOAD_FAMILY_BY_ROUTE[scope["route"]]
    fmt = next(f for f in contract["formats"] if f["family"] == family)
    candidates = []
    for cell in contract["lane_eligibility"]["cells"]:
        if (cell["platform"], cell["family"], cell["structure"], cell["regime"]) != (runtime["platform"], family, "dense", scope["regime"]):
            continue
        image, modes = cell_runtime_scope(cell)
        if image != runtime["image"] or "eager" not in modes or "resident" not in cell_residency_modes(cell):
            continue
        if not cell_is_device_backed(cell) or not cell_covers_rung(cell, scope["q256"], fmt):
            continue
        if cell_runtime_code(cell) != (runtime["tessera_commit"], runtime["serving_source_sha256"]):
            continue
        if cell_runtime_versions(cell) != (runtime["vllm"], runtime["torch"]):
            continue
        if cell.get("requires_plugin") != "tessera":
            continue
        refuse_unevaluated_predicates(cell)
        flags = runtime["serve_flags"]
        if any("=" not in flag or flags.get(flag.split("=", 1)[0]) not in flag.split("=", 1)[1].split("|") for flag in cell["requires_serve_flags"]):
            continue
        if tuple(pair) not in {(v["symbol"], v["decoder"]) for v in cell["executes"]}:
            continue
        candidates.append(cell)
    if len(candidates) != 1:
        raise ValueError("requires exactly one positively matching backed native cell")
    launches = [v for v in scheme.route_launches(scope["route"], structure="dense", regime=scope["regime"], mode="resident")
                if (v["symbol"], v["decoder"]) == tuple(pair)]
    if len(launches) != 1 or not launches[0]["lane"]:
        raise ValueError("first slice requires a named native extension lane")
    for role in roles:
        report = scheme.lane_wire_report(launches[0]["lane"], role["facts"], contract)
        if not report["readable"]:
            raise ValueError("native lane refuses canonical wire: " + str(report["refusals"]))
    return candidates[0], launches[0]["lane"]


def _validate_panel(panel, *, expected_runtime):
    """Replay one positive receipt; caller provides the independently frozen context."""
    _object(panel, {"schema", "status", "claims", "runtime", "plan", "rows", "evidence", "energy"}, "panel")
    canonical(panel)  # Reject nonfinite values anywhere, including cached summaries.
    if panel["schema"] != SCHEMA or panel["status"] != "measured" or canonical(panel["claims"]) != canonical(CLAIMS):
        raise ValueError("panel schema/status/claims differ")
    runtime = runtime_context(panel["runtime"])
    if runtime != runtime_context(expected_runtime):
        raise ValueError("observed runtime differs from independent expected context")
    evidence = _object(panel["evidence"], EVIDENCE, "evidence")
    raw = {name: read_bound(bound) for name, bound in evidence.items()}
    contract = json_bytes(raw["contract"])
    validate_serving_contract(contract)
    if hashlib.sha256(raw["contract"]).hexdigest() != runtime["contract_sha256"]:
        raise ValueError("raw contract differs from runtime")
    if json_bytes(raw["runtime"]) != runtime:
        raise ValueError("runtime identity evidence differs")
    producer = _object(json_bytes(raw["producer"]), {"commit", "tool_source_sha256"}, "producer")
    _sha(producer["commit"], "producer.commit", 40)
    _sha(producer["tool_source_sha256"], "producer.tool_source_sha256")
    if not isinstance(panel["rows"], list) or len(panel["rows"]) != 1:
        raise ValueError("first slice requires exactly one dense row; duplicates refuse")
    row = _object(panel["rows"][0], {"scope_id", "prefix", "scheme", "timing", "cell_id"}, "row")
    if not isinstance(row["prefix"], str) or not row["prefix"]:
        raise ValueError("row.prefix requires an explicit module name")
    plan = panel["plan"]
    if not isinstance(plan, dict) or len(plan.get("rows", [])) != 1:
        raise ValueError("requires a singleton unmeasured census plan")
    scope = plan["rows"][0]["scope"]
    rebuilt = census_plan.build_census_plan([scope], raw_contract=raw["contract"])
    if canonical(rebuilt) != canonical(plan) or row["scope_id"] != rebuilt["rows"][0]["id"]:
        raise ValueError("plan/scope identity differs")
    if (scope["structure"], scope["route"], scope["mode"], scope["execution_mode"], scope["tp_degree"]) != ("dense", scheme.TESSERA_FP8, "resident", "eager", 1) or scope["requested_platform"] != runtime["platform"]:
        raise ValueError("unsupported dense scope or observed platform")
    declared, roles = wire_facts(raw["wire"], row["scheme"])
    shape = scope["shape"]
    if (declared["rows"], declared["columns"], declared["q256"]) != (shape["N"], shape["K"], scope["q256"]):
        raise ValueError("native wire differs from requested rank-local shape/rung")
    prep = _object(json_bytes(raw["preparation"]), {"builder", "wire_sha256", "roles", "shape", "tp_rank", "tp_degree", "grid", "native_packed_bytes"}, "preparation")
    if prep["builder"] != "tessera.serving.lane.build_tessera_method" or prep["wire_sha256"] != evidence["wire"]["sha256"] or prep["roles"] != roles or prep["shape"] != shape or prep["grid"] != "E4M3" or (type(prep["tp_rank"]), prep["tp_rank"], type(prep["tp_degree"]), prep["tp_degree"]) != (int, 0, int, 1):
        raise ValueError("actual native preparation differs")
    _integer(prep["native_packed_bytes"], "native packed tensor bytes")
    samples = _object(json_bytes(raw["samples"]), {"samples_ms", "warmup_iterations", "interval_unix"}, "samples")
    _integer(samples["warmup_iterations"], "warmup iterations")
    if canonical(row["timing"]) != canonical(timing_summary(samples["samples_ms"])):
        raise ValueError("timing summary differs from actual raw samples")
    interval = samples["interval_unix"]
    if not isinstance(interval, list) or len(interval) != 2 or _number(interval[1], "interval end") <= _number(interval[0], "interval start"):
        raise ValueError("invalid sample interval")
    routes = _object(json_bytes(raw["routes"]), {"records"}, "routes")
    records = routes["records"]
    if not isinstance(records, list) or len(records) != len(samples["samples_ms"]):
        raise ValueError("requires a fresh actual route record for each timed call")
    pairs = set()
    for record in records:
        if not isinstance(record, dict) or record.get("state") != "served" or record.get("kind") != "dense" or record.get("policy") != scheme.TESSERA_FP8 + ":resident" or record.get("platform") != runtime["platform"] or record.get("shape") != f"M{shape['M']}:N{shape['N']}:K{shape['K']}" or record.get("contract") != scheme.ROUTES[scheme.TESSERA_FP8]["activation_contract"]:
            raise ValueError("missing/error/stale or mismatched observed route")
        pairs.add((record.get("symbol"), record.get("decoder")))
    if len(pairs) != 1:
        raise ValueError("timed calls changed native lane")
    pair = next(iter(pairs))
    cell, lane = admitted_cell(contract, scope, runtime, pair, roles)
    if row["cell_id"] != cell["id"]:
        raise ValueError("recorded cell differs from positive cell join")
    agreement, problems = cell_launch_agreement({"timed": {row["prefix"]: records[-1]}}, cells=[cell], phase_regimes={"timed": scope["regime"]}, platform=runtime["platform"], rungs_by_module={row["prefix"]: scope["q256"]}, families_by_route=PAYLOAD_FAMILY_BY_ROUTE, runtime_image=runtime["image"], execution_mode="eager", formats=contract["formats"])
    if problems or agreement["agrees"] is not True or agreement["phases"]["timed"]["covered_by_cell"] != 1:
        raise ValueError("observed launch lacks positive census agreement")
    if not raw["native_binary"].startswith(b"\x7fELF"):
        raise ValueError("native binary evidence is not an ELF object")
    native = next(x for x in contract["native_extensions"] if x["module_name_prefix"] == lane)
    import fnmatch
    if not fnmatch.fnmatch(Path(evidence["native_binary"]["path"]).name, native["filename_glob"]):
        raise ValueError("loaded binary does not name the observed lane")
    trace = json_bytes(gzip.decompress(raw["trace"]))
    kernels = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]
    if not kernels:
        raise ValueError("profiler trace contains no actual CUDA kernel")
    for kernel in kernels:
        _number(kernel.get("dur"), "CUDA kernel duration")
        if not isinstance(kernel.get("name"), str) or not kernel["name"]:
            raise ValueError("CUDA kernel identity is absent")
    telemetry = _object(json_bytes(raw["telemetry"]), {"interval_unix", "fast_power_samples", "netdata"}, "telemetry")
    if telemetry["interval_unix"] != interval:
        raise ValueError("telemetry interval differs from timed window")
    if not isinstance(telemetry["fast_power_samples"], list) or not telemetry["fast_power_samples"]:
        raise ValueError("fast power evidence is absent")
    for stamp, watts in telemetry["fast_power_samples"]:
        if not interval[0] <= _number(stamp, "power stamp") <= interval[1]:
            raise ValueError("power sample outside timed interval")
        _number(watts, "power watts")
    if not isinstance(telemetry["netdata"], dict) or set(telemetry["netdata"]) != {"sparky", "sparklina"}:
        raise ValueError("both boxes' raw Netdata evidence is required")
    for box in telemetry["netdata"].values():
        if not isinstance(box, dict) or set(box) != NETDATA_CONTEXTS:
            raise ValueError("raw Netdata context responses are absent")
        for response in box.values():
            _object(response, {"query", "raw_response", "returned_view"}, "Netdata evidence")
            if not isinstance(response["query"], str) or not response["query"] or not isinstance(response["raw_response"], dict) or response["returned_view"] != response["raw_response"].get("view"):
                raise ValueError("Netdata query/returned view differs")
            result = response["raw_response"].get("result")
            if not isinstance(response["returned_view"], dict) or not isinstance(result, dict) or not isinstance(result.get("labels"), list) or not result["labels"] or not isinstance(result.get("data"), list) or not result["data"]:
                raise ValueError("Netdata raw response contains no result samples")
    if canonical(panel["energy"]) != canonical({"status": "hold", "reason": "cross_host_clock_alignment_unqualified", "reference_w": 140}):
        raise ValueError("energy remains HOLD; no work/J qualification in this slice")
    return {"scope_id": row["scope_id"], "cell_id": cell["id"], "kernel_lane": list(pair),
            "timing": row["timing"], "energy_status": "hold", "claims": dict(CLAIMS)}


def validate_panel(panel, *, expected_runtime):
    """Malformed receipts refuse through one stable passive validation boundary."""
    try:
        return _validate_panel(panel, expected_runtime=expected_runtime)
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration, TesseraError) as exc:
        raise ValueError(f"malformed native timing receipt: {exc}") from exc
