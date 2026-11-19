"""Prepare dense cases on CPU, capture on GPU, and finish through host comparison."""
import copy
import cProfile
from collections import deque
from datetime import datetime, timezone
from itertools import chain, groupby
import json
from pathlib import Path
import pstats
import time

from experiments.numerics_rows.common import error_text, OUTCOMES, BoundaryError, BoundaryRecorder, Invalid, Unattested, sha256, write_json
from experiments.numerics_rows.phase_artifact import Artifact, identities, unit_scope, read, payload, check_capture, cases, records, attempt_cases
from experiments.numerics_rows.phase_lifecycle import Progress
from experiments.numerics_rows.launch_state import json_digest
from experiments.numerics_rows import row
from experiments.numerics_rows.dependency_code import ROSTER_REFUSAL


class HostDense:
    reference = row.Dense.reference

    def __init__(self, entry, cut, module, q1):
        from experiments.numerics_rows.population import store_for
        from experiments.numerics_rows.cases import decode_roles, QuantizerTable
        from tessera.serving.sharding import plan_shard

        store = store_for(entry)
        self.name, self.cut, self.entry = entry["name"], cut, entry
        self.device, self.family, self.library, self.gs = "cpu", "e4m3", entry["library"], 1.0
        scheme = store.schemes[self.name]
        wire = store.wire(self.name).contiguous().numpy().tobytes()
        import hashlib
        if hashlib.sha256(wire).hexdigest() != module["wire_sha256"]:
            raise Invalid("loaded wire differs from the C1 reader artifact")
        size = cut["size"]
        down = self.name.endswith("down_proj")
        partitions = [int(n) for _, n in scheme["roles"]]
        plan = plan_shard(self.name, roles=scheme["roles"], columns=scheme["columns"],
            out_partitions=partitions if down else [n // size for n in partitions],
            in_size=scheme["columns"] // size if down else scheme["columns"],
            tp_rank=cut["rank"], tp_size=size, input_size=scheme["columns"], output_size=sum(partitions))
        self.roles = decode_roles(wire, scheme, self.name, plan=plan)
        self.cols, self.rows = self.roles[0][1].shape[1], sum(r[1].shape[0] for r in self.roles)
        self.quantizer = QuantizerTable(q1, "e4m3")

    def quantized(self, x, recorder):
        """Use Q1 scales or binary32 amax/448 with the Q1 zero-row floor.

        Capture must match these predicted bytes before any served call.
        """
        import torch
        maximum = x.float().abs().amax(1)
        scales = []
        floor = self.quantizer(0.0)[1]
        for value in maximum.tolist():
            try:
                scales.append(self.quantizer(value)[1])
            except KeyError:
                scales.append(max(float(torch.tensor(value, dtype=torch.float32) / 448), floor))
        sf = torch.tensor(scales, dtype=torch.float32).reshape(-1, 1)
        q = (x.float() / sf).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q, sf, q.double(), sf.reshape(-1)


class CaseWriter:
    def __init__(self, artifact, progress, wires):
        self.artifact, self.progress, self.wires = artifact, progress, wires
        self.array_index = 0

    def dense(self, adapter, unit, x, q, sf, reference, mask, bound, *, category, **fields):
        row.finite_reference(reference)
        tensors = {"input": x, "q": q, "sf": sf, "reference": reference,
                   "finite_value": bound.value, "finite_held": bound.held}
        if mask is not None:
            tensors["x_mask"] = mask
        tensors.update({k: fields.pop(k) for k in ("diagnostic_reference", "products_abs") if k in fields})
        case = self.artifact.add(unit_scope(unit, self.wires[unit["module"]]), row.unit_identity(unit)[4], "dense", tensors,
                                 category=category, **fields)
        self.progress.record("diagnostics" if category == "diagnostic" else "gate_cases", unit, case["id"])

    def scale(self, adapter, unit, x, q, sf, coded_sf, role, plan, coding, case, expected, bound, values):
        import torch
        record = self.artifact.add(unit_scope(unit, self.wires[unit["module"]]), row.unit_identity(unit)[4], "scale", {
            "input": x, "q": q, "sf": sf, "coded_sf": coded_sf,
            "reference": torch.tensor([expected], dtype=torch.bfloat16),
            "finite_value": bound.value, "finite_held": bound.held,
            "coded_values": torch.tensor(list(values), dtype=torch.float32)},
            category="scale_index", scale=plan["scale"], role=role["name"],
            output_index=[case["input_row"], case["output_row"]],
            array_index=self.array_index,
            coding=copy.deepcopy(coding), read=case["read"])
        self.progress.record("gate_cases", unit, record["id"])


    def array(self, unit, record):
        self.artifact.record("arrays", {"scope": unit_scope(unit, self.wires[unit["module"]]),
                                      "array_index": self.array_index, "record": record})
        summarize_array(unit, record)
        self.array_index += 1



def case_status(unit):
    """One reason per field. A roster refusal is the cause, so it is kept in preference to a later effect."""
    roster = [r for r in unit["unattested"] if isinstance(r, str) and r.startswith(ROSTER_REFUSAL)]
    return {field: (roster or unit[field])[-1:] if field == "unattested" else unit[field][-1:]
            for field in ("findings", "invalid", "unattested", "cuda_errors")}

def summarize_array(unit, record):
    summary = unit["exact"]["scale_index"]
    summary["streamed"] = True
    for field in ("passes_total", "reads", "reads_covered", "mismatches", "launches"):
        summary[field] = summary.get(field, 0) + record.get(field, 0)
    for field in ("scales", "bases", "families"):
        values = record.get(field, [record["scale"]] if field == "scales" else [record["base"]] if field == "bases" else [])
        summary[field] = sorted(set(summary.get(field, [])) | set(values))
    summary["uncovered_reads"] = record.get("uncovered_reads", [])[:1] or summary.get("uncovered_reads", [])
    summary["arrays"].clear()


def case_ranges(groups):
    """Count the full row order before a job selects its cut or case set."""
    result, offset = {}, 0
    for part in ("gate", "diagnostics"):
        for entry, module, cut, units in groups:
            start = offset
            for unit in units:
                m = unit["m_config"]["M"]
                offset += (9 if part == "diagnostics" else 1 + int(unit["m_config"]["k_split"] > 1)
                    + (module["roles"][0]["cols"] // unit["m_config"]["chunk_size"] - 1)
                    + sum((sum(len(r["values"]) for r in role["separation"]["recipes"]) + m - 1) // m for role in module["roles"])
                    + sum(len({tuple(c["read"]) for c in plan["read_cases"]}) * len(plan["coding"])
                        for role in module["roles"] for plan in role["scale_index"]
                        if plan["array"].get("M", m) == m and plan["reads"] == plan["reads_covered"]))
            result[(entry["name"], cut["size"], cut["rank"], part)] = [start, offset - start]
    return result



def phase_units(groups):
    return [list(unit) for _, unit in groupby(groups, lambda g: (g[0]["name"], g[2]["size"]))]


def unit_range(args, groups, part):
    ranges = [args.case_ranges[(e["name"], c["size"], c["rank"], part)] for e, _, c, _ in groups]
    return [ranges[0][0], sum(count for _, count in ranges)]


def check_range(args, groups, document):
    if not getattr(args, "case_ranges", None) or document["case_set"] == "all":
        return
    ranges = [args.case_ranges[(e["name"], c["size"], c["rank"], document["case_set"])] for e, _, c, _ in groups]
    if (unit_range(args, groups, document["case_set"]) != [document["ordinal_start"], document["streams"]["cases"]["count"]]
            or any(left[0] + left[1] != right[0] for left, right in zip(ranges, ranges[1:]))):
        raise Invalid("phase cases differ from the deterministic full row enumeration")

def scopes(groups):
    return [unit_scope(u, module["wire_sha256"]) for _, module, _, units in groups for u in units]


def check_pair(args, groups, prepared, captured, digest, prepare_root, capture_root):
    check_range(args, groups, prepared)
    return check_capture(prepared, captured, digest, prepare_root, capture_root)


def host_prepare(args, receipt, phases, groups):
    from experiments.numerics_rows.scale_launch import check as scale_checks
    progress = args.progress = Progress(args.out, receipt, phases)
    artifact = Artifact(getattr(args, "phase_root", args.out / "prepare"), "prepare", identities(receipt))
    artifact.document.update(scopes=scopes(groups), case_set=getattr(args, "case_set", "all"))
    if getattr(args, "case_set", "all") != "all":
        start, count = unit_range(args, groups, args.case_set)
        artifact.document.update(ordinal_start=start, expected_cases=count)
    writer = CaseWriter(artifact, progress, {m["name"]: m["wire_sha256"] for _, m, _, _ in groups})
    q1, recorder = json.loads(args.q1.read_text()), BoundaryRecorder(lambda: None, [])
    profile = cProfile.Profile()
    for part in (("gate", "diagnostics") if getattr(args, "case_set", "all") == "all" else (args.case_set,)):
        for entry, module, cut, units in groups:
            adapter = HostDense(entry, cut, module, q1)
            adapter.case_writer = writer
            for unit in units:
                progress.record("diagnostics" if part == "diagnostics" else "gate_cases", unit, checkpoint=True)
                adapter.splits = unit["m_config"]["k_split"]
                if part == "diagnostics":
                    profile.runcall(row.dense_diagnostics, adapter, unit, recorder)
                else:
                    if any(not role["matches_attested_wire"] for role in module["roles"]):
                        raise Invalid("wire stamp differs from attested_wire")
                    row.verify_cases(adapter, unit, module, q1, None)
                    row.run_dense_case(adapter, unit, [], recorder, category="profile")
                    row.dense_cases(adapter, unit, module, recorder)
                    writer.array_index = 0
                    scale_checks(adapter, unit, module, recorder)
            del adapter
        receipt["row"]["gate_preparation_finished"] = part == "gate"
    if profile.getstats():
        profile.dump_stats(str(artifact.root / "diagnostic-reference.prof"))
        with (artifact.root / "diagnostic-reference.txt").open("w") as stream:
            pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(80)
        for name in ("diagnostic-reference.prof", "diagnostic-reference.txt"):
            artifact.file(name)
    if any(unit["invalid"] for unit in receipt["units"]):
        raise Invalid("CPU preparation failed a required case precondition")
    receipt["row"].update(prepare_finished=True, finalization_finished=False, outcome="STOPPED")
    finish_phase(args.out, artifact, receipt, progress)
    write_json(args.out / "numerics_row.json", receipt)


def quantizer_guard(actual, expected):
    import torch
    if len(actual) != len(expected) or any(a.shape != b.shape or a.dtype != b.dtype or
            not torch.equal(a.detach().cpu().contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            for a, b in zip(actual, expected)):
        raise Invalid("actual quantizer bytes differ from prepared bytes before native call")


def preflight_case(root, case):
    """Read and verify the exact case before launching, including finite references."""
    from experiments.numerics_rows.reference import FiniteBound
    tensors = {k: payload(root, v) for k, v in case["tensors"].items()}
    row.finite_reference(tensors["reference"])
    if case["category"] not in ("diagnostic", "scale_index"):
        row.require_x(tensors["x_mask"], sweep=case["category"] in ("sweep", "profile"))
    if "diagnostic_reference" in tensors:
        row.finite_reference(tensors["diagnostic_reference"])
    bound = FiniteBound(tensors["finite_value"], tensors["finite_held"])
    return tensors, bound


def capture_case(adapter, unit, case, tensors, recorder):
    """No reference construction or coverage recomputation in this function."""
    import torch
    from experiments.numerics_rows.scale_launch import dense_launch
    x = tensors["input"].to("cuda")
    q, sf, _, _ = adapter.quantized(x, recorder)
    quantizer_guard((q, sf), (tensors["q"], tensors["sf"]))
    # Check quantizer calls made inside the actual apply as well as the two preflight calls.
    recorder.quantizer_result = lambda result: quantizer_guard(result, (tensors["q"], tensors["sf"]))
    if case["kind"] == "scale":
        result = dense_launch(adapter, {"name": case["role"]}, q, tensors["coded_sf"].to("cuda"), case["scale"],
                              tensors["coded_values"].tolist(), recorder)
        return {"output": result, "quantizer_q": q, "quantizer_sf": sf}
    try:
        got = recorder.call("dense:apply", "apply", lambda: row.opaque_apply(adapter.apply, recorder, x))
    except BoundaryError as error:
        if error.attributed_to is None:
            adapter.staged(q, sf, recorder)
        raise
    staged = adapter.staged(q, sf, recorder)
    outputs = {"output": got, "staged": staged, "quantizer_q": q, "quantizer_sf": sf}
    if case.get("distribution") == "gauss" and case.get("seed") == 0:
        from experiments.numerics_rows.run_routed import graph_record
        launch, buffer = adapter.prepare_staged(q, sf, recorder)
        graph_record(unit, "dense-staged", launch, buffer, recorder)
    return outputs


def device_capture(args, receipt, phases, groups, library, compile_paths):
    import torch
    from experiments import dense_fused_oracle as dense
    from experiments.routed_pair_oracle import kernel_table
    from experiments.numerics_rows.dependency_code import binding
    if args.prepare_artifact is None:
        raise Invalid("capture requires a finished prepare artifact")
    prepare_root, = args.prepare_artifact
    common, expected_scopes = identities(receipt), scopes(groups)
    prepared, prepare_digest = read(prepare_root, "prepare", common, expected_scopes)
    check_range(args, groups, prepared)
    if prepared["case_set"] != getattr(args, "case_set", "all"):
        raise Invalid("capture case set differs from prepare")
    for unit, previous in zip(receipt["units"], prepared["receipt"]["units"]):
        for field in ("findings", "invalid", "unattested"):
            unit[field] = (unit[field] + previous[field])[:1]
    progress = args.progress = Progress(args.out, receipt, phases, args.meter)
    artifact = Artifact(getattr(args, "phase_root", args.out / "capture"), "capture", common)
    artifact.document.update(scopes=expected_scopes, prepare_digest=prepare_digest,
                             ordinal_start=prepared["ordinal_start"], case_set=prepared["case_set"])
    cache, audit = {}, args.code_audit
    native_source = Path(next(r for r in compile_paths if r["key"] == library["key"])["native_source_path"])
    receipt["row"].update(capture_finished=False, finalization_finished=False, outcome="STOPPED")
    complete = True
    with dense.init_vllm_world1():
        grouped = groupby(cases(prepare_root, prepared), lambda c: (c["category"] == "diagnostic", c["scope"]["module"], c["scope"]["cut"]["size"], c["scope"]["cut"]["rank"]))
        for (diagnostic, name, size, rank), selected in grouped:
            entry, module, cut, units = next(g for g in groups if g[0]["name"] == name and g[2] == {"size": size, "rank": rank})
            audit.unit = None
            audit.records.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                adapter = row.Dense(entry, cut, module)
            transfer_events = {}
            preparation_kernels = kernel_table(prof, limit=100000, device_only=True, diagnostics=transfer_events)
            actual, served = row.identify_served(adapter, False, native_source, args.out / "served-binaries", cache)
            preparation_code = list(audit.records)
            del prof
            module_record = row.module_receipt(module, adapter, False)
            if module_record not in receipt["modules"]:
                receipt["modules"].append(module_record)
            adapter.served_identity = served[0] if served else None
            for unit in units:
                row.check_served_identity(unit, served, library)
                row.audit_served(unit, served, library, args.out / "served-comparisons", cache)
                if module["launch_geometry"] != row.actual_geometry(adapter, False):
                    raise Invalid("C1 launch geometry differs from the loaded operands")
                unit["tp"].update(row.tp_facts(adapter, module, False))
                unit["scales"] = row.scale_receipt(adapter, False)
                row.check_preconditions(unit, scale_equal=unit["scales"]["equal"])
            for case in selected:
                unit = next(u for u in units if row.unit_identity(u) == case["scope"]["unit_key"])
                adapter.splits = unit["m_config"]["k_split"]
                progress.record("diagnostics" if diagnostic else "gate_cases", unit, case["id"])
                captured = {k: case[k] for k in ("id", "ordinal", "identities", "scope", "configuration", "kind", "category")}
                captured.update(completion=False, not_run=True, tensors={}, cuda_errors=[])
                if not (unit["invalid"] or unit["unattested"] or unit["cuda_errors"]):
                    tensors, _ = preflight_case(prepare_root, case)
                    unit["boundaries"], audit.records = [], []
                    unit["arm"] = {**{k: v for k, v in unit["arm"].items() if k not in ("profiler_kernels", "profiler_launches")},
                                   "preparation_kernels": preparation_kernels, "preparation_transfer_events": transfer_events}
                    unit["triton_specializations"] = [binding(unit["unit_id"], r["case"], r["identity"]) for r in preparation_code]
                    audit.unit = unit
                    recorder = BoundaryRecorder(torch.cuda.synchronize, unit["boundaries"])
                    recorder.before_call, recorder.code_audit = args.launch_state.check, audit
                    recorder.context = lambda label, case_id=case["id"]: audit.context(case_id + ":" + label)
                    captured.pop("not_run")
                    before = time.monotonic()
                    try:
                        args.launch_state.check()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as case_profile:
                            clock = [time.monotonic()]
                            with row.native_boundaries({library["key"]: actual}, recorder):
                                outputs = capture_case(adapter, unit, case, tensors, recorder)
                            clock.append(time.monotonic())
                        clock.append(time.monotonic())
                        events = kernel_table(case_profile, limit=100000, device_only=True, diagnostics=unit["arm"].setdefault("cuda_transfer_events", {}))
                        clock.append(time.monotonic())
                        row.bind_kernel_profile(adapter, unit, recorder, events,
                            [{"case": case["id"], "route_count": None, "profiler_names": [r["name"] for r in events]}], routed=False)
                        # Measurement only: where the case time goes (profiler start, case work, profiler stop, event table, code binding).
                        captured["timing_seconds"] = dict(zip(("profiler_start", "case", "profiler_stop", "kernel_table", "bind"),
                                                              (b - a for a, b in zip([before, *clock], [*clock, time.monotonic()]))))
                        captured.update(profiler_names=[r["name"] for r in events], emitted_pairs=adapter.pairs(),
                            tensors={k: artifact.tensor(v) for k, v in sorted(outputs.items())}, completion=True)
                        unit["arm"]["case_completed"] = True
                        row.dispatch(unit, entry["pair"], adapter.pairs(), served, adapter.info.get("lane_reason"))
                        captured["code"] = {k: copy.deepcopy(unit[k]) for k in ("unit_id", "arm", "triton_specializations", "kernel_origins", "unresolved_kernel_origins", "dependency_code") if k in unit}
                        captured["kernel_origins"] = [r for r in unit.get("kernel_origins", []) if r["profiler_name"] in captured["profiler_names"]]
                        captured["compiled_kernel_symbols"] = unit["m_config"]["compiled_kernel_symbols"]
                        del outputs, case_profile
                    except BoundaryError as error:
                        row.record_cuda_error(unit, error)
                        captured["cuda_errors"].append({"message": error_text(error), "attributed_to": error.attributed_to})
                    except (Invalid, Unattested) as error:
                        field = "invalid" if isinstance(error, Invalid) else "unattested"
                        unit[field].append(error_text(error))
                    captured.update(elapsed_seconds=time.monotonic() - before, boundaries=unit["boundaries"])
                    recorder.quantizer_result = None
                    del tensors, recorder
                captured["status"] = status = case_status(unit)
                if complete and not captured["completion"]:
                    reason = (status["invalid"] or status["unattested"]
                              or [error["message"] for error in status["cuda_errors"]]
                              or ["required case did not complete"])[0]
                    receipt["row"]["stop_cause"] = f"{case['id']}: {reason}"
                complete &= captured["completion"]
                artifact.record("cases", captured)
                unit.update(captured["status"])
                for field in ("triton_specializations", "kernel_origins", "unresolved_kernel_origins"):
                    unit.pop(field, None)
                unit["boundaries"].clear()
                for field in ("profiler_launches", "profiler_kernels"):
                    unit["arm"].pop(field, None)
                unit["m_config"]["compiled_kernel_symbols"] = []
                audit.records.clear()
                del captured
            _, after = row.identify_served(adapter, False, native_source, args.out / "served-binaries", cache)
            for unit in units:
                row.check_served_after(unit, served, after)
                if unit["arm"].get("case_completed"):
                    row.dispatch(unit, entry["pair"], adapter.pairs(), after, adapter.info.get("lane_reason"))
                unit["completed"] = False
                import sys
                unit["flags_latched_diagnostic"] = args.launch_state.latched_diagnostic(getattr(sys.modules.get("tessera.serving.flags"), "_LATCHED", {}))
                row.finish_unit(unit)
            del adapter
    receipt["runtime"]["environment_diagnostics"] = args.launch_state.diagnostics()
    receipt["row"].update(capture_finished=complete, gate_capture_finished=complete and prepared["case_set"] != "diagnostics",
        outcome="FAIL" if any(u["findings"] for u in receipt["units"]) else "STOPPED", finalization_finished=False,
        utc_end=datetime.now(timezone.utc).isoformat(), exit_status=0 if complete else 3, gpu_minutes=args.meter.finish())
    finish_phase(args.out, artifact, receipt, progress)
    write_json(args.out / "numerics_row.json", receipt)


def compare_case(case, observed, prepare_root, capture_root, unit):
    from experiments.numerics_rows.reference import compare
    tensors, certificate = preflight_case(prepare_root, case)
    if not observed.get("completion"):
        unit["completed"] &= bool(unit["invalid"] or unit["unattested"])
        return
    outputs = {k: payload(capture_root, v) for k, v in observed["tensors"].items()}
    if not observed.get("profiler_names") or {r["profiler_name"] for r in observed.get("kernel_origins", [])} != set(observed["profiler_names"]):
        unit["unattested"].append("case kernel origins are incomplete")
    got = outputs["output"]
    if got.shape != certificate.value.shape or got.dtype != tensors["reference"].dtype:
        unit["findings"].append("captured output has the wrong shape or dtype")
        return
    row.finite_outputs(unit, case["id"], got, certificate)
    if case["kind"] == "scale":
        target = got[tuple(case["output_index"])].reshape(1)
        result = row.compare_output(unit, case["id"], target, tensors["reference"])
        unit["exact"]["scale_index"]["arrays"][0 if unit["exact"]["scale_index"].get("streamed") else case["array_index"]]["mismatches"] += result["mismatches"]
        return
    staged = outputs["staged"]
    if staged.shape != got.shape or staged.dtype != got.dtype:
        unit["findings"].append("captured staged output has the wrong shape or dtype")
        return
    row.finite_outputs(unit, case["id"] + ":staged", staged, certificate)
    equal = compare(got, staged)["equal"]
    unit["exact"]["composition"]["cases"] += 1
    unit["exact"]["composition"]["equal"] &= equal
    if not equal:
        unit["findings"].append("apply differs from the same staged launches")
    if case["category"] == "diagnostic":
        unit.setdefault("diagnostic_composition_equal", []).append(equal)
        unit["diagnostic"].append(row.diagnostic(got, tensors["diagnostic_reference"], tensors["products_abs"],
            tensors["input"].shape[1], case["configuration"]["k_split"], case["distribution"], case["seed"]))
    else:
        result = row.compare_output(unit, case["id"], got, tensors["reference"], tensors["x_mask"])
        if case["category"] in ("sweep", "chunk_pairs"):
            unit["exact"][case["category"]]["mismatches"] += result["mismatches"]


def unit_map(units):
    result = {json.dumps([u["module"], row.unit_identity(u)], sort_keys=True): u for u in units}
    if len(result) != len(units):
        raise Invalid("phase receipt has duplicate units")
    return result


def attempt_paths(directory):
    attempts = list(directory.iterdir()) if directory.is_dir() else []
    if any(not p.is_dir() or p.is_symlink() for p in attempts) or {p.name for p in attempts} != {str(n) for n in range(1, len(attempts) + 1)}:
        raise Invalid(f"attempt ids have a gap or invalid name: {directory.parent.name}")
    return [directory / str(number) for number in range(1, len(attempts) + 1)]


def checked_attempt_receipt(receipt, common, expected, mismatch):
    status = receipt.get("row", {})
    tessera = receipt.get("tessera")
    if not isinstance(tessera, dict) or "commit" not in tessera:
        return status, None
    units = unit_map(receipt["units"])
    if units.keys() != expected or identities(receipt) != common:
        raise Invalid(mismatch)
    return status, units


def lead_cause(attempt, key, number, stopped):
    path = attempt / "lead-cause.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if (set(record) != {"unit", "attempt", "cause"} or record["unit"] != key or type(record["attempt"]) is not int
            or record["attempt"] != number or not isinstance(record["cause"], str) or not record["cause"].strip()):
        raise Invalid(f"invalid lead cause record: {key}/{number}")
    if not stopped:
        raise Invalid(f"lead cause record requires STOPPED attempt: {key}/{number}")
    return {"path": str(path), "sha256": sha256(path)}


def attempt_unit_key(scope):
    return json.dumps([scope["module"], scope["unit_key"]], sort_keys=True)


def failed_case_unit(case, common, expected, *, checked=False):
    if not checked and (case.get("identities") != common or case.get("scope") not in expected):
        raise Invalid("attempt case has another unit or common identity")
    state = case.get("status", case)
    if case.get("findings") or case.get("outcome") == "FAIL" or state.get("findings") or state.get("outcome") == "FAIL":
        return attempt_unit_key(case["scope"])


def attempt_stop_cause(status, torn):
    cause = status.get("stop_cause")
    return "\n".join(([cause] if cause else []) + torn) if torn else cause


def read_compare_attempts(directory, expected, common, running_attempt):
    key = "compare-" + json_digest([expected[0]["module"], "compare"])
    expected_keys = {attempt_unit_key(scope) for scope in expected}
    listing, failures, stopped = [], {}, {}
    for attempt in attempt_paths(directory / key / "attempts"):
        if attempt == running_attempt:
            continue
        number, root = int(attempt.name), attempt / "compare"
        path = root / "numerics_row.json"
        status, units, digest = {}, None, None
        if path.is_file():
            status, units = checked_attempt_receipt(json.loads(path.read_text()), common, expected_keys,
                "compare attempt receipt identities or units differ from the module")
            digest = sha256(path)
        finished = units is not None and status.get("finalization_finished") is True
        found = {identity for identity, unit in (units or {}).items() if unit.get("findings") or unit.get("outcome") == "FAIL"}
        torn = []
        if not finished:
            phase = root / "phase.json"
            document = json.loads(phase.read_text()) if phase.is_file() else {}
            for case in attempt_cases(root, torn, document if document.get("completion") is True else None):
                identity = failed_case_unit(case, common, expected)
                if identity is not None:
                    found.add(identity)
        cause = lead_cause(attempt, key, number, not finished and found != expected_keys)
        if not finished and cause is None:
            stopped.update({identity: f"open STOPPED attempt {key}/{number}" for identity in expected_keys - found})
        for identity in found:
            failures.setdefault(identity, f"FAIL compare attempt {key}/{number}")
        status = {} if finished else status
        listing.append({"unit": key, "attempt": number, "receipt_sha256": digest, "outcome": status.get("outcome"),
            "stop_cause": attempt_stop_cause(status, torn), "cause_record": cause,
            "state": "FAIL" if found else "FINISHED" if finished else "STOPPED"})
    return listing, failures, stopped


def read_attempts(args, groups, prepare_root, capture_root, common, part):
    expected = scopes(groups)
    key = f'tp{expected[0]["cut"]["size"]}-{part}-' + json_digest([expected[0]["module"], expected[0]["cut"]["size"], part])
    directory = prepare_root.parent.parent
    if (part not in ("all", "gate", "diagnostics") or directory.name != "attempts" or directory.parent.name != key or directory.parent.parent.name != "units"
            or prepare_root.name != "prepare" or capture_root != prepare_root.parent / "capture"):
        raise Invalid("phase paths differ from the validated unit attempt layout")
    return inspect_attempts(directory, key, expected, common, part, check_cases=True)


class CompareChecks:
    """Check case bindings during the existing artifact read. Keep only digests."""
    def __init__(self):
        self.pairs, self.arrays, self.group = deque(), deque(), None

    def case(self, phase, case):
        binding = json_digest([case.get(k) for k in ("id", "ordinal", "scope", "configuration", "kind", "identities", "category")])
        names = ("q", "sf") if phase == "prepare" else ("quantizer_q", "quantizer_sf")
        quantizer = json_digest([{k: case["tensors"][name][k] for k in ("sha256", "dtype", "shape")} for name in names]) if phase == "prepare" or case.get("completion") else None
        if phase == "prepare":
            self.pairs.append((binding, quantizer))
            group = json_digest([case["scope"], case.get("array_index")]) if case["kind"] == "scale" else None
            if group is not None and group != self.group:
                self.arrays.append(group)
            self.group = group
            return
        wanted = self.pairs.popleft() if self.pairs else None
        if wanted is None or binding != wanted[0]:
            raise Invalid("capture must contain each prepared case exactly once")
        if not case.get("completion"):
            return
        if quantizer != wanted[1]:
            raise Invalid("actual quantizer bytes differ from prepared bytes before native call")
        boundaries = case.get("boundaries", [])
        if not boundaries or any(b.get("sync_before") is not True or b.get("sync_after") is not True for b in boundaries):
            raise Invalid("completed capture has missing or failed synchronization boundaries")
        if sum(b["kind"] == "quantizer" for b in boundaries) < 2:
            raise Invalid("capture omitted the two synchronized quantizer calls")

    def array(self, array):
        if not self.arrays:
            raise Invalid("prepare has extra scale arrays")
        if self.arrays.popleft() != json_digest([array["scope"], array["array_index"]]):
            raise Invalid("scale case has no matching ordered array")


def inspect_attempts(directory, key, expected, common, part, *, check_cases=False):
    """Read all unit attempts for both feeder holds and module comparison."""
    listing, failures, stopped, newest = [], {}, {}, None
    expected_keys = {attempt_unit_key(scope) for scope in expected}
    for attempt in attempt_paths(directory):
        number = int(attempt.name)
        documents, evidence, found, attempt_row, finished = {}, {}, {}, {}, True
        torn = []
        checks = CompareChecks() if check_cases else None
        for phase in ("prepare", "capture"):
            root = attempt / phase
            path = root / "phase.json"
            document = json.loads(path.read_text()) if path.is_file() else None
            complete = document is not None and document.get("completion") is True
            evidence[phase] = sha256(path) if document is not None else None
            cases = root / "cases.jsonl"
            def inspect_case(case):
                nonlocal finished
                identity = failed_case_unit(case, common, expected, checked=complete)
                if identity is not None:
                    found[identity] = f"FAIL attempt {key}/{number}"
                state = case.get("status", case)
                if phase == "capture" and not case.get("completion") and not any(state.get(f) for f in ("findings", "invalid", "unattested")):
                    finished = False
                if checks is not None and complete and (phase == "prepare" or "prepare" in documents):
                    checks.case(phase, case)
            if document is not None:
                if document.get("case_set") != part:
                    raise Invalid("attempt phase has another case set")
                if complete:
                    if check_cases and document.get("scopes") != expected:
                        raise Invalid("phase artifact has a foreign module or cut; " + row.P5)
                    documents[phase], evidence[phase] = read(root, phase, common, expected, check_case=inspect_case,
                        **({"check_array": checks.array} if checks is not None and phase == "prepare" else {}))
                    if checks is not None and phase == "prepare" and checks.arrays:
                        raise Invalid("scale case has no matching ordered array")
                    if checks is not None and phase == "capture" and "prepare" in documents and checks.pairs:
                        raise Invalid("capture must contain each prepared case exactly once")
                elif document.get("identities") != common or document.get("scopes") != expected:
                    raise Invalid("incomplete attempt phase identities, module or cut differ")
            statuses = [document["receipt"]] if "receipt" in (document or {}) else []
            local_receipt = root / "numerics_row.json"
            if local_receipt.is_file():
                statuses.append(json.loads(local_receipt.read_text()))
                evidence[phase + "_partial"] = sha256(local_receipt)
            for status in statuses:
                phase_row, units = checked_attempt_receipt(status, common, expected_keys,
                    "phase receipt identities or units differ from the selected cut")
                if "outcome" in phase_row or phase_row.get("stop_cause"):
                    attempt_row = phase_row
                found.update({k: f"FAIL attempt {key}/{number}" for k, u in (units or {}).items() if u.get("findings") or u.get("outcome") == "FAIL"})
            finished &= phase in documents
            if not complete:
                for case in attempt_cases(root, torn):
                    inspect_case(case)
            evidence[phase + "_cases"] = sha256(cases) if cases.is_file() else None
        cause = lead_cause(attempt, key, number, not finished and found.keys() != expected_keys)
        if finished:
            attempt_row = {}
            newest = (attempt / "prepare", attempt / "capture", documents["prepare"], documents["capture"], evidence["prepare"], evidence["capture"])
        elif cause is None:
            stopped.update({identity: f"open STOPPED attempt {key}/{number}" for identity in expected_keys - found.keys()})
        failures.update(found)
        listing.append({"unit": key, "attempt": number, "state": "FAIL" if found else "FINISHED" if finished else "STOPPED", "phases": evidence,
            "outcome": attempt_row.get("outcome"), "stop_cause": attempt_stop_cause(attempt_row, torn),
            "cause_record": cause})
    if newest is None:
        stopped.update({identity: f"no finished attempt: {key}" for identity in expected_keys - failures.keys() - stopped.keys()})
    return newest, listing, failures, stopped


def host_compare(args, receipt, phases, groups):
    if args.prepare_artifact is None or args.capture_artifact is None or len(args.prepare_artifact) != len(args.capture_artifact):
        raise Invalid("compare requires both phase artifacts with matching gate and diagnostics artifacts")
    bound, listing, failures, stopped, selected = [], [], {}, {}, {(g[0]["name"], g[2]["size"], g[2]["rank"]): [] for g in groups}
    for prepare_root, capture_root in zip(args.prepare_artifact, args.capture_artifact):
        matched = [(unit, part) for unit in phase_units(groups) for part in ("all", "gate", "diagnostics")
                   if prepare_root.parents[2].name == f'tp{unit[0][2]["size"]}-{part}-' + json_digest([unit[0][0]["name"], unit[0][2]["size"], part])]
        if len(matched) != 1:
            raise Invalid("phase artifact has a foreign module or cut; " + row.P5)
        (unit_groups, part), = matched
        common = identities({**receipt, "inputs": {**receipt.get("inputs", {}), "selected_cuts": [g[2] for g in unit_groups]}})
        for entry, _, cut, _ in unit_groups:
            selected[(entry["name"], cut["size"], cut["rank"])].append(part)
        newest, attempts, failed, unfinished = read_attempts(args, unit_groups, prepare_root, capture_root, common, part)
        listing.extend(attempts)
        failures.update(failed)
        stopped.update(unfinished)
        if newest is not None:
            bound.append((*newest, unit_groups))
    if any(sorted(parts) not in (["all"], ["diagnostics", "gate"]) for parts in selected.values()):
        raise Invalid("compare requires the full gate and diagnostics union exactly once")
    phase_root = getattr(args, "phase_root", None)
    common, module_scopes = identities(receipt), scopes(groups)
    earlier, failed, unfinished = read_compare_attempts(args.prepare_artifact[0].parents[3], module_scopes,
        common, phase_root.parent if phase_root is not None else None)
    listing.extend(earlier)
    failures.update(failed)
    stopped.update(unfinished)
    bound = [(prepare_root, capture_root, prepared, captured, pd, cd,
              unit_map(prepared["receipt"]["units"]), unit_map(captured["receipt"]["units"]),
              check_pair(args, unit_groups, prepared, captured, pd, prepare_root, capture_root))
             for prepare_root, capture_root, prepared, captured, pd, cd, unit_groups in sorted(bound, key=lambda p: p[2]["ordinal_start"])]
    units = unit_map(receipt["units"])
    listing.sort(key=lambda item: (item["unit"], item["attempt"]))
    receipt["attempts"] = {"listing": listing, "sha256": json_digest(listing)}
    compare_row = copy.deepcopy(receipt["row"])
    receipt["row"]["phase_jobs"] = {"compare": compare_row}
    receipt["phase_artifacts"] = {}
    artifact = Artifact(getattr(args, "phase_root", args.out / "compare"), "compare", common)
    artifact.document.update(scopes=module_scopes, ranges=[])
    (artifact.root / "cases.jsonl").touch()
    artifact.document["streams"]["cases"] = {"count": 0}
    progress = args.progress = Progress(args.out, receipt, phases)
    for unit in units.values():
        unit["completed"] = True
    for prepare_root, capture_root, prepared, captured, pd, cd, previous_units, current_units, pairs in bound:
        part = prepared["case_set"]
        receipt["modules"].extend(copy.deepcopy(m) for m in captured["receipt"]["modules"] if m not in receipt["modules"])
        scope = prepared["scopes"][0]
        key = f'{scope["module"]}:tp{scope["cut"]["size"]}r{scope["cut"]["rank"]}:{part}:'
        for phase, root_path, document, digest in (("prepare", prepare_root, prepared, pd), ("capture", capture_root, captured, cd)):
            receipt["phase_artifacts"][key + phase] = {"path": str(root_path / "phase.json"), "sha256": digest}
            receipt["row"]["phase_jobs"][key + phase] = document["receipt"]["row"]
        artifact.document["ranges"].append([prepared["ordinal_start"], prepared["streams"]["cases"]["count"]])
        for identity, current in current_units.items():
            unit, previous = units[identity], previous_units[identity]
            unit.update(copy.deepcopy({k: v for k, v in current.items() if k not in (
                "exact", "completed", "findings", "invalid", "unattested", "cuda_errors", "cuda_unattributed", "graph_replay_equal", "first_mismatch")}))
            if part != "diagnostics":
                unit["exact"] = copy.deepcopy(previous["exact"])
                unit["exact"].update(finite_certificate={"elements": 0, "held": 0, "nonfinite_where_held": 0}, composition={"cases": 0, "equal": True})
                if unit["exact"]["scale_index"].get("streamed"):
                    unit["exact"]["scale_index"] = {"arrays": [], "streamed": True}
            for field in ("findings", "invalid", "unattested", "cuda_errors"):
                unit[field] = (unit[field] + previous[field] + current[field])[:1]
            unit["cuda_unattributed"] |= current["cuda_unattributed"]
            unit["graph_replay_equal"] = current["graph_replay_equal"] or unit["graph_replay_equal"]
        arrays = iter(records(prepare_root, "arrays", prepared)) if "arrays" in prepared["streams"] else iter(())
        for _, batch in groupby(pairs, lambda pair: (pair[0]["scope"]["module"], pair[0]["scope"]["unit_key"], pair[0].get("array_index"))):
            case, observed = next(batch)
            unit = units[attempt_unit_key(case["scope"])]
            array = next(arrays, None) if case["kind"] == "scale" else None
            if case["kind"] == "scale":
                array["record"]["mismatches"] = 0
                unit["exact"]["scale_index"]["arrays"] = [array["record"]]
            for case, observed in chain([(case, observed)], batch):
                progress.record("diagnostics" if case["category"] == "diagnostic" else "gate_cases", unit, case["id"])
                previous = {field: unit[field] for field in ("findings", "invalid", "unattested")}
                unit.update({field: observed.get("status", {}).get(field, []) for field in previous})
                compare_case(case, observed, prepare_root, capture_root, unit)
                result = {k: case[k] for k in ("id", "ordinal", "identities", "scope", "configuration", "kind", "category")}
                result.update({field: unit[field] for field in previous}, identities=common, diagnostic=unit["diagnostic"],
                              first_mismatch=unit["first_mismatch"], completion=observed.get("completion"), code=observed.get("code"),
                              counts={k: v for k, v in unit["exact"].items() if k not in ("scale_index", "separation")},
                              outcome="FAIL" if unit["findings"] else "INVALID" if unit["invalid"] else "UNATTESTED" if unit["unattested"] else "PASS" if observed.get("completion") else "STOPPED")
                artifact.record("cases", result)
                unit.update({field: (previous[field] + unit[field])[:1] for field in previous})
                unit["diagnostic"].clear()
                unit.pop("diagnostic_composition_equal", None)
            if array is not None:
                summarize_array(unit, array["record"])
                artifact.record("arrays", array)
    for identity, reason in failures.items():
        units[identity]["failed_attempts"] = [reason]
        if not units[identity]["findings"]:
            units[identity]["findings"].append(reason)
    for unit in receipt["units"]:
        identity = json.dumps([unit["module"], row.unit_identity(unit)], sort_keys=True)
        if identity in stopped and identity not in failures and not unit["findings"]:
            unit["completed"] = False
        row.finish_unit(unit)
        if unit["outcome"] == "STOPPED" and identity in stopped:
            unit["reasons"].append(stopped[identity])
    compare_row.update(utc_end=datetime.now(timezone.utc).isoformat())
    receipt["row"].update(utc_end=compare_row["utc_end"], outcome=min((u["outcome"] for u in receipt["units"]), key=OUTCOMES.index),
        exit_status=3 if any(u["outcome"] in ("FAIL", "STOPPED", "INVALID") for u in receipt["units"]) else 0)
    digest = finish_phase(args.out, artifact, receipt, progress)
    receipt["phase_artifacts"]["compare"] = {"path": str(artifact.root / "phase.json"), "sha256": digest}
    for unit in receipt["units"]:
        unit["code_stream"] = receipt["phase_artifacts"]["compare"]
    receipt["diagnostic_cells"] = row.diagnostic_cells(receipt["units"])
    compare_row["finalization_finished"] = receipt["row"]["finalization_finished"] = True
    write_json(args.out / "numerics_row.json", receipt)


def run_phase(args, receipt, phases, groups, library, q1, compile_paths):
    if args.image != "N" or any(module["structure"] != "dense" or entry["family"] != "TESSERA_FP8" for entry, module, _, _ in groups):
        raise Invalid("split phases support IMG-N E4M3 dense only")
    if receipt["inputs"]["reference_selftest"]["mismatches"] or not all(c["flagged_exactly"] for c in receipt["negative_controls"]):
        raise Invalid("reference self-test or mutated reference control failed")
    operation = {"prepare": host_prepare, "capture": device_capture, "compare": host_compare}[args.phase]
    operation(args, receipt, phases, groups, **({"library": library, "compile_paths": compile_paths} if args.phase == "capture" else {}))
    raise SystemExit(receipt["row"].get("exit_status") or 0)


def finish_phase(out, artifact, receipt, progress):
    progress.record("setup")
    bind_evidence(out, artifact)
    return artifact.finish(receipt=receipt)


def bind_evidence(out, artifact):
    import shutil
    for name in ("binary-identity", "served-binaries", "served-comparisons", "compiled-code", "native_ops.json", "phase-events.jsonl"):
        source = out / name
        for path in source.rglob("*") if source.is_dir() else [source]:
            if not path.is_file():
                continue
            relative = Path("evidence") / path.relative_to(out)
            destination = artifact.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            artifact.file(str(relative))
