"""A8S E4M3 dense host preparation, device capture and host comparison.

The existing dense and scale-index case generators build the prepare artifact.
Only the capture maps a GPU. It consumes prepared bytes and certificates, and
runs the real serving adapter. Compare alone can finish a row. Environment
values remain in LaunchState's live mapping and never enter phase artifacts.
"""
from __future__ import annotations

from experiments.numerics_rows.common import error_text

import copy
import cProfile
from datetime import datetime, timezone
import json
from pathlib import Path
import pstats
import time

from experiments.numerics_rows.common import OUTCOMES, BoundaryError, BoundaryRecorder, Invalid, Unattested, sha256, write_json
from experiments.numerics_rows.phase_artifact import Artifact, identities, unit_scope, read, payload, check_capture, cases, records
from experiments.numerics_rows.phase_lifecycle import Progress
from experiments.numerics_rows import row


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
        """Predict bytes on CPU. Capture must match them before any served call.

        Q1 supplies exact scales for its alphabet, including the zero-row scale.
        Other rows use binary32 amax/448 and the Q1 zero-row floor. Predictions
        have no device authority: disagreement is INVALID before native use.
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

    def begin(self, unit, case):
        self.progress.record("diagnostics", unit, case, checkpoint=True)

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
        unit["exact"]["scale_index"]["arrays"].clear()
        self.array_index += 1


def summarize_array(unit, record):
    summary = unit["exact"]["scale_index"]
    summary["streamed"] = True
    for field in ("passes_total", "reads", "reads_covered", "mismatches", "launches"):
        summary[field] = summary.get(field, 0) + record.get(field, 0)
    for field in ("scales", "bases", "families"):
        values = record.get(field, [record["scale"]] if field == "scales" else [record["base"]] if field == "bases" else [])
        summary[field] = sorted(set(summary.get(field, [])) | set(values))
    summary["uncovered_reads"] = record.get("uncovered_reads", [])[:1] or summary.get("uncovered_reads", [])


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



def check_range(args, groups, document):
    if getattr(args, "case_ranges", None) and document["case_set"] != "all":
        entry, _, cut, _ = groups[0]
        expected = args.case_ranges[(entry["name"], cut["size"], cut["rank"], document["case_set"])]
        if expected != [document["ordinal_start"], document["streams"]["cases"]["count"]]:
            raise Invalid("phase cases differ from the deterministic full row enumeration")

def scopes(groups):
    return [unit_scope(u, module["wire_sha256"]) for _, module, _, units in groups for u in units]


def host_prepare(args, receipt, phases, groups):
    from experiments.numerics_rows.scale_launch import check as scale_checks
    progress = args.progress = Progress(args.out, receipt, phases)
    artifact = Artifact(args.out / "prepare", "prepare", identities(receipt))
    artifact.document.update(scopes=scopes(groups), case_set=getattr(args, "case_set", "all"))
    if getattr(args, "case_set", "all") != "all":
        entry, _, cut, _ = groups[0]
        start, count = args.case_ranges[(entry["name"], cut["size"], cut["rank"], args.case_set)]
        artifact.document.update(ordinal_start=start, expected_cases=count)
    writer = CaseWriter(artifact, progress, {m["name"]: m["wire_sha256"] for _, m, _, _ in groups})
    q1, recorder = json.loads(args.q1.read_text()), BoundaryRecorder(lambda: None, [])
    profile = writer.profile = cProfile.Profile()
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
    profile.dump_stats(str(artifact.root / "diagnostic-reference.prof"))
    with (artifact.root / "diagnostic-reference.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(80)
    for name in ("diagnostic-reference.prof", "diagnostic-reference.txt"):
        artifact.file(name)
    if any(unit["invalid"] for unit in receipt["units"]):
        raise Invalid("CPU preparation failed a required case precondition")
    receipt["row"].update(prepare_finished=True, finalization_finished=False, outcome="STOPPED")
    progress.record("setup")
    bind_evidence(args.out, artifact)
    artifact.finish(receipt=receipt)
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
    def checked_quantizer(result):
        quantizer_guard(result, (tensors["q"], tensors["sf"]))
    recorder.quantizer_result = checked_quantizer
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
    from itertools import groupby
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
    artifact = Artifact(args.out / "capture", "capture", common)
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
            receipt["modules"] = [row.module_receipt(module, adapter, False)]
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
                    recorder.before_call, recorder.context, recorder.code_audit = args.launch_state.check, audit.context, audit
                    recorder.context = lambda label, case_id=case["id"]: audit.context(case_id + ":" + label)
                    captured.pop("not_run")
                    before = time.monotonic()
                    try:
                        args.launch_state.check()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as case_profile:
                            with row.native_boundaries({library["key"]: actual}, recorder):
                                outputs = capture_case(adapter, unit, case, tensors, recorder)
                        events = kernel_table(case_profile, limit=100000, device_only=True, diagnostics=unit["arm"].setdefault("cuda_transfer_events", {}))
                        row.bind_kernel_profile(adapter, unit, recorder, events,
                            [{"case": case["id"], "route_count": None, "profiler_names": [r["name"] for r in events]}], routed=False)
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
                        captured[field] = error_text(error)
                    captured.update(elapsed_seconds=time.monotonic() - before, boundaries=unit["boundaries"])
                    recorder.quantizer_result = None
                    del tensors, recorder
                complete &= captured["completion"]
                artifact.record("cases", captured)
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
    progress.record("setup")
    bind_evidence(args.out, artifact)
    artifact.finish(receipt=receipt)
    write_json(args.out / "numerics_row.json", receipt)


def compare_case(case, observed, prepare_root, capture_root, unit):
    from experiments.numerics_rows.reference import compare
    tensors, certificate = preflight_case(prepare_root, case)
    if not observed.get("completion"):
        unit["completed"] &= bool(unit["invalid"] or unit["unattested"])
        return
    outputs = {k: payload(capture_root, v) for k, v in observed["tensors"].items()}
    quantizer_guard((outputs["quantizer_q"], outputs["quantizer_sf"]), (tensors["q"], tensors["sf"]))
    boundaries = observed.get("boundaries", [])
    if not boundaries or any(b.get("sync_before") is not True or b.get("sync_after") is not True for b in boundaries):
        raise Invalid("completed capture has missing or failed synchronization boundaries")
    if sum(b["kind"] == "quantizer" for b in boundaries) < 2:
        raise Invalid("capture omitted the two synchronized quantizer calls")
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


def host_compare(args, receipt, phases, groups):
    if args.prepare_artifact is None or args.capture_artifact is None:
        raise Invalid("compare requires both phase artifacts")
    common, expected_scopes = identities(receipt), scopes(groups)
    prepare_paths, capture_paths = args.prepare_artifact, args.capture_artifact
    if len(prepare_paths) != len(capture_paths) or len(prepare_paths) not in (1, 2):
        raise Invalid("compare requires matching gate and diagnostics artifacts")
    bound = []
    for prepare_root, capture_root in zip(prepare_paths, capture_paths):
        prepared, pd = read(prepare_root, "prepare", common, expected_scopes)
        if not (capture_root / "phase.json").is_file():
            partial = capture_root.parent / "numerics_row.json"
            if partial.is_file():
                status = json.loads(partial.read_text())
                if identities(status) != common:
                    raise Invalid("unfinished capture belongs to another row")
                receipt.update(status)
            raise RuntimeError("capture is unfinished; compare cannot finish this row")
        captured, cd = read(capture_root, "capture", common, expected_scopes)
        bound.append((prepare_root, capture_root, prepared, captured, pd, cd))
    for _, _, prepared, _, _, _ in bound:
        check_range(args, groups, prepared)
    selected = sorted(p[2]["case_set"] for p in bound)
    if selected not in (["all"], ["diagnostics", "gate"]):
        raise Invalid("compare requires the full gate and diagnostics union exactly once")
    bound.sort(key=lambda p: p[2]["ordinal_start"])
    compare_row = receipt["row"]
    receipt.update(copy.deepcopy(bound[0][3]["receipt"]))
    receipt["row"]["phase_jobs"] = {"compare": compare_row}
    receipt["phase_artifacts"] = {}
    artifact = Artifact(args.out / "compare", "compare", common)
    artifact.document.update(scopes=expected_scopes, ranges=[])
    progress = args.progress = Progress(args.out, receipt, phases)
    for unit, base in zip(receipt["units"], bound[0][2]["receipt"]["units"]):
        unit["exact"] = copy.deepcopy(base["exact"])
        unit["exact"]["finite_certificate"] = {"elements": 0, "held": 0, "nonfinite_where_held": 0}
        unit["exact"]["composition"] = {"cases": 0, "equal": True}
        if unit["exact"]["scale_index"].get("streamed"):
            unit["exact"]["scale_index"] = {"arrays": [], "streamed": True}
        unit["completed"] = True
    for prepare_root, capture_root, prepared, captured, pd, cd in bound:
        part = prepared["case_set"]
        for phase, root_path, document, digest in (("prepare", prepare_root, prepared, pd), ("capture", capture_root, captured, cd)):
            receipt["phase_artifacts"][part + ":" + phase] = {"path": str(root_path / "phase.json"), "sha256": digest}
            receipt["row"]["phase_jobs"][part + ":" + phase] = document["receipt"]["row"]
        artifact.document["ranges"].append([prepared["ordinal_start"], prepared["streams"]["cases"]["count"]])
        for unit, current, previous in zip(receipt["units"], captured["receipt"]["units"], prepared["receipt"]["units"]):
            for field in ("findings", "invalid", "unattested", "cuda_errors"):
                unit[field] = (unit[field] + previous[field] + current[field])[:1]
            unit["cuda_unattributed"] |= current["cuda_unattributed"]
            unit["graph_replay_equal"] = current["graph_replay_equal"] or unit["graph_replay_equal"]
        arrays = iter(records(prepare_root, "arrays", prepared)) if "arrays" in prepared["streams"] else iter(())
        from itertools import groupby
        pairs = check_capture(prepared, captured, pd, prepare_root, capture_root)
        for _, batch in groupby(pairs, lambda pair: (pair[0]["scope"]["unit_key"], pair[0].get("array_index"))):
            case, observed = next(batch)
            unit = next(u for u in receipt["units"] if row.unit_identity(u) == case["scope"]["unit_key"])
            array = next(arrays, None) if case["kind"] == "scale" else None
            if case["kind"] == "scale":
                if array is None or array["scope"] != case["scope"] or array["array_index"] != case["array_index"]:
                    raise Invalid("scale case has no matching ordered array")
                array["record"]["mismatches"] = 0
                unit["exact"]["scale_index"]["arrays"] = [array["record"]]
            from itertools import chain
            for case, observed in chain([(case, observed)], batch):
                progress.record("diagnostics" if case["category"] == "diagnostic" else "gate_cases", unit, case["id"])
                previous = {field: unit[field] for field in ("findings", "invalid", "unattested")}
                for field in previous:
                    unit[field] = []
                compare_case(case, observed, prepare_root, capture_root, unit)
                result = {k: case[k] for k in ("id", "ordinal", "identities", "scope", "configuration", "kind", "category")}
                result.update({field: unit[field] for field in previous}, diagnostic=unit["diagnostic"],
                              first_mismatch=unit["first_mismatch"], completion=observed.get("completion"), code=observed.get("code"),
                              counts={k: v for k, v in unit["exact"].items() if k not in ("scale_index", "separation")},
                              outcome="FAIL" if unit["findings"] else "UNATTESTED" if unit["unattested"] else "PASS" if observed.get("completion") else "STOPPED")
                artifact.record("cases", result)
                for field in previous:
                    unit[field] = (previous[field] + unit[field])[:1]
                unit["diagnostic"].clear()
                unit.pop("diagnostic_composition_equal", None)
            if array is not None:
                summarize_array(unit, array["record"])
                artifact.record("arrays", array)
                unit["exact"]["scale_index"]["arrays"].clear()
        if next(arrays, None) is not None:
            raise Invalid("prepare has extra scale arrays")
    for unit in receipt["units"]:
        row.finish_unit(unit)
    compare_row.update(utc_end=datetime.now(timezone.utc).isoformat())
    receipt["row"].update(utc_end=compare_row["utc_end"], outcome=min((u["outcome"] for u in receipt["units"]), key=OUTCOMES.index),
        exit_status=3 if any(u["outcome"] in ("FAIL", "STOPPED", "INVALID") for u in receipt["units"]) else 0)
    progress.record("setup")
    bind_evidence(args.out, artifact)
    digest = artifact.finish(receipt=receipt)
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
    for _, module, _, units in groups:
        for unit in units:
            unit["tp"].update(cut_axis=[r["cut_axis"] for r in module["roles"]], cut_range=[r["cut_range"] for r in module["roles"]])
            row.finish_unit(unit)
    if args.phase == "prepare":
        host_prepare(args, receipt, phases, groups)
    elif args.phase == "capture":
        device_capture(args, receipt, phases, groups, library, compile_paths)
    else:
        host_compare(args, receipt, phases, groups)
    raise SystemExit(receipt["row"].get("exit_status") or 0)


def bind_evidence(out, artifact):
    import shutil
    for name in ("binary-identity", "served-binaries", "served-comparisons", "compiled-code", "native_ops.json", "phase-events.jsonl"):
        source = out / name
        for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
            if not path.is_file():
                continue
            relative = Path("evidence") / path.relative_to(out)
            destination = artifact.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            artifact.file(str(relative))
