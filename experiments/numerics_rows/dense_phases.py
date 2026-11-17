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
from experiments.numerics_rows.phase_artifact import Artifact, identities, unit_scope, read, payload, check_capture
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

    def begin(self, unit, case):
        self.progress.record("diagnostics", unit, case, checkpoint=True)

    def dense(self, adapter, unit, x, q, sf, reference, mask, bound, *, category, **fields):
        row.finite_reference(reference)
        tensors = {"input": x, "q": q, "sf": sf, "reference": reference,
                   "finite_value": bound.value, "finite_held": bound.held}
        if mask is not None:
            tensors["x_mask"] = mask
        tensors.update({k: fields.pop(k) for k in ("diagnostic_reference", "products_abs") if k in fields})
        case = self.artifact.add(unit_scope(unit, self.wires[unit["module"]]), clean_config(unit), "dense", tensors,
                                 category=category, **fields)
        self.progress.record("diagnostics" if category == "diagnostic" else "gate_cases", unit, case["id"])
        if category == "diagnostic":
            self.profile.dump_stats(str(self.progress.out / "diagnostic-reference.prof"))

    def scale(self, adapter, unit, x, q, sf, coded_sf, role, plan, coding, case, expected, bound, values):
        import torch
        output = torch.zeros(bound.value.shape, dtype=torch.bfloat16)
        output[case["input_row"], case["output_row"]] = expected
        record = self.artifact.add(unit_scope(unit, self.wires[unit["module"]]), clean_config(unit), "scale", {
            "input": x, "q": q, "sf": sf, "coded_sf": coded_sf,
            "reference": torch.tensor([expected], dtype=torch.bfloat16),
            "finite_value": bound.value, "finite_held": bound.held,
            "coded_values": torch.tensor(list(values), dtype=torch.float32)},
            category="scale_index", scale=plan["scale"], role=role["name"],
            output_index=[case["input_row"], case["output_row"]],
            array_index=len(unit["exact"]["scale_index"]["arrays"]) - 1,
            coding=copy.deepcopy(coding), read=case["read"])
        self.progress.record("gate_cases", unit, record["id"])
        return output


def clean_config(unit):
    return row.unit_identity(unit)[4]


def scopes(groups):
    return [unit_scope(u, module["wire_sha256"]) for _, module, _, units in groups for u in units]


def host_prepare(args, receipt, phases, groups):
    from experiments.numerics_rows.scale_launch import check as scale_checks
    progress = args.progress = Progress(args.out, receipt, phases)
    artifact = Artifact(args.out / "prepare", "prepare", identities(receipt))
    artifact.document["scopes"] = scopes(groups)
    artifact.document["receipt"] = receipt
    writer = CaseWriter(artifact, progress, {m["name"]: m["wire_sha256"] for _, m, _, _ in groups})
    q1 = json.loads(args.q1.read_text())
    recorder = BoundaryRecorder(lambda: None, [])
    profile = writer.profile = cProfile.Profile()
    try:
        for phase in ("gate_cases", "diagnostics"):
            for entry, module, cut, units in groups:
                adapter = HostDense(entry, cut, module, q1)
                adapter.case_writer = writer
                for unit in units:
                    progress.record(phase, unit, checkpoint=True)
                    adapter.splits = unit["m_config"]["k_split"]
                    if phase == "diagnostics":
                        profile.enable()
                        try:
                            row.dense_diagnostics(adapter, unit, recorder)
                        finally:
                            profile.disable()
                            profile.dump_stats(str(args.out / "diagnostic-reference.prof"))
                            with (args.out / "diagnostic-reference.txt").open("w") as stream:
                                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(80)
                        continue
                    if any(not role["matches_attested_wire"] for role in module["roles"]):
                        raise Invalid("wire stamp differs from attested_wire")
                    row.verify_cases(adapter, unit, module, q1, None)
                    # The profiler's zero input is also prepared and finite-checked.
                    row.run_dense_case(adapter, unit, [], recorder, category="profile")
                    row.dense_cases(adapter, unit, module, recorder)
                    scale_checks(adapter, unit, module, recorder)
                del adapter
            if phase == "gate_cases":
                receipt["row"]["gate_preparation_finished"] = True
                progress.record("diagnostics", checkpoint=True)
    finally:
        profile.dump_stats(str(args.out / "diagnostic-reference.prof"))
    import shutil
    shutil.move(str(args.out / "diagnostic-reference.prof"), artifact.root / "diagnostic-reference.prof")
    artifact.document["files"]["diagnostic-reference.prof"] = sha256(artifact.root / "diagnostic-reference.prof")
    if (args.out / "diagnostic-reference.txt").exists():
        shutil.move(str(args.out / "diagnostic-reference.txt"), artifact.root / "diagnostic-reference.txt")
        artifact.document["files"]["diagnostic-reference.txt"] = sha256(artifact.root / "diagnostic-reference.txt")
    progress.record("setup", checkpoint=True)
    if any(unit["invalid"] for unit in receipt["units"]):
        raise Invalid("CPU preparation failed a required case precondition")
    receipt["row"].update(prepare_finished=True, finalization_finished=False, outcome="STOPPED")
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
    from experiments import dense_fused_oracle as dense
    from experiments.routed_pair_oracle import kernel_table
    from experiments.numerics_rows.dependency_code import binding
    if args.prepare_artifact is None:
        raise Invalid("capture requires a finished prepare artifact")
    common, expected_scopes = identities(receipt), scopes(groups)
    prepared, prepare_digest = read(args.prepare_artifact, "prepare", common, expected_scopes)
    progress = args.progress = Progress(args.out, receipt, phases, args.meter)
    artifact = Artifact(args.out / "capture", "capture", common)
    artifact.document.update(scopes=expected_scopes, prepare_digest=prepare_digest)
    captures = {case["id"]: {**{k: copy.deepcopy(case[k]) for k in ("id", "identities", "scope", "configuration", "kind")},
                "completion": False, "not_run": True, "tensors": {}, "cuda_errors": []} for case in prepared["cases"]}
    artifact.document["cases"] = list(captures.values())
    cache, audit = {}, args.code_audit
    native_source = Path(next(r for r in compile_paths if r["key"] == library["key"])["native_source_path"])
    receipt["row"].update(capture_finished=False, finalization_finished=False, outcome="STOPPED")
    progress.record("setup", checkpoint=True)
    try:
        with dense.init_vllm_world1():
            for phase in ("gate_cases", "diagnostics"):
                for entry, module, cut, units in groups:
                    wanted = [case for case in prepared["cases"] if case["scope"]["module"] == entry["name"] and
                              case["scope"]["cut"]["rank"] == cut["rank"] and case["scope"]["cut"]["size"] == cut["size"] and
                              (case["category"] == "diagnostic") == (phase == "diagnostics")]
                    if not wanted:
                        raise Invalid("prepare omitted an entire cut or phase")
                    progress.record("setup", checkpoint=True)
                    audit.unit = None
                    start = len(audit.records)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        adapter = row.Dense(entry, cut, module)
                    transfer_events = {}
                    preparation_kernels = kernel_table(prof, limit=100000, device_only=True, diagnostics=transfer_events)
                    actual, served = row.identify_served(adapter, False, native_source, args.out / "served-binaries", cache)
                    if phase == "gate_cases":
                        receipt["modules"].append(row.module_receipt(module, adapter, False))
                    adapter.served_identity = served[0] if served else None
                    for unit in units:
                        selected = [case for case in wanted if case["scope"]["unit_key"] == row.unit_identity(unit)]
                        if not selected:
                            raise Invalid("prepare omitted a required unit")
                        if unit["invalid"] or unit["cuda_errors"] or unit["unattested"]:
                            continue
                        progress.record("identity_audit", unit, checkpoint=True)
                        row.check_served_identity(unit, served, library)
                        row.audit_served(unit, served, library, args.out / "served-comparisons", cache)
                        if module["launch_geometry"] != row.actual_geometry(adapter, False):
                            raise Invalid("C1 launch geometry differs from the loaded operands")
                        unit["tp"].update(row.tp_facts(adapter, module, False))
                        unit["scales"] = row.scale_receipt(adapter, False)
                        row.check_preconditions(unit, scale_equal=unit["scales"]["equal"])
                        unit["arm"]["preparation_kernels"] = preparation_kernels
                        unit["arm"]["preparation_transfer_events"] = dict(transfer_events)
                        unit["triton_specializations"] = [binding(unit["unit_id"], r["case"], r["identity"]) for r in audit.records[start:]]
                        recorder = BoundaryRecorder(torch.cuda.synchronize, unit["boundaries"])
                        recorder.before_call, recorder.context, recorder.code_audit = args.launch_state.check, audit.context, audit
                        audit.unit = unit
                        adapter.splits = unit["m_config"]["k_split"]
                        case_profiles, unit_captures = [], []
                        for case in selected:
                            progress.record(phase, unit, case["id"], checkpoint=False)
                            # Check the full artifact scope again before the first native call.
                            if case["identities"] != common or case["scope"] not in expected_scopes:
                                raise Invalid("prepared case changed its common bindings")
                            tensors, certificate = preflight_case(args.prepare_artifact, case)
                            captured = captures[case["id"]]
                            captured.pop("not_run")
                            before = time.monotonic()
                            recorder.context = lambda name, case_id=case["id"]: audit.context(case_id + ":" + name)
                            try:
                                args.launch_state.check()
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as case_profile:
                                    with row.native_boundaries({library["key"]: actual}, recorder):
                                        outputs = capture_case(adapter, unit, case, tensors, recorder)
                                records = kernel_table(case_profile, limit=100000, device_only=True, diagnostics=unit["arm"].setdefault("cuda_transfer_events", {}))
                                case_profiles.append({"case": case["id"], "route_count": None, "profiler_names": [r["name"] for r in records], "records": records})
                                captured["profiler_names"] = [r["name"] for r in records]
                                captured["emitted_pairs"] = adapter.pairs()
                                captured["tensors"] = {k: artifact.tensor(v) for k, v in outputs.items()}
                                captured["completion"] = True
                            except BoundaryError as error:
                                row.record_cuda_error(unit, error)
                                captured["cuda_errors"].append({"message": error_text(error), "attributed_to": error.attributed_to})
                            except Invalid as error:
                                unit["invalid"].append(error_text(error))
                                captured["invalid"] = error_text(error)
                            except Unattested as error:
                                unit["unattested"].append(error_text(error))
                                captured["unattested"] = error_text(error)
                            finally:
                                captured["elapsed_seconds"] = time.monotonic() - before
                                captured["boundaries"] = copy.deepcopy(unit["boundaries"])
                                unit["boundaries"].clear()
                                unit_captures.append(captured)
                                recorder.quantizer_result = None
                            if not captured["completion"]:
                                break
                        row.bind_kernel_profile(adapter, unit, recorder,
                            [r for p in case_profiles for r in p["records"]],
                            [{k: v for k, v in p.items() if k != "records"} for p in case_profiles], routed=False)
                        for captured in unit_captures:
                            names = captured.get("profiler_names", [])
                            captured["kernel_origins"] = [r for r in unit.get("kernel_origins", []) if r["profiler_name"] in names]
                            captured["compiled_kernel_symbols"] = [r for r in unit["m_config"]["compiled_kernel_symbols"] if r["profiler_name"] in names]
                            captured["triton_specializations"] = [r for r in unit.get("triton_specializations", []) if r["case"].startswith(captured["id"] + ":")]
                        with phases.measure("identity_audit"):
                            _, after = row.identify_served(adapter, False, native_source, args.out / "served-binaries", cache)
                            row.check_served_after(unit, served, after)
                        row.dispatch(unit, entry["pair"], adapter.pairs(), after, adapter.info.get("lane_reason"))
                        import sys
                        unit["flags_latched_diagnostic"] = args.launch_state.latched_diagnostic(getattr(sys.modules.get("tessera.serving.flags"), "_LATCHED", {}))
                        receipt["runtime"]["environment_diagnostics"] = args.launch_state.diagnostics()
                        unit["completed"] = False  # Capture cannot grant a row verdict.
                        row.finish_unit(unit)
                        progress.record(phase, unit, checkpoint=True)
                    del adapter
                if phase == "gate_cases":
                    expected_gate = [c["id"] for c in prepared["cases"] if c["category"] != "diagnostic"]
                    captured_gate = [c["id"] for c in artifact.document["cases"] if c["completion"]]
                    receipt["row"]["gate_capture_finished"] = set(expected_gate) == set(captured_gate)
                    artifact.document["receipt"] = copy.deepcopy(receipt)
                    write_json(artifact.root / "gate-complete.json", artifact.document)
                    progress.record("diagnostics", checkpoint=True)
            # No reference or diagnostic CPU calculation remains inside the CUDA world.
    finally:
        progress.record("setup", checkpoint=True)
        artifact.document["receipt"] = copy.deepcopy(receipt)
        write_json(artifact.root / "unfinished.json", artifact.document)
    complete = len(artifact.document["cases"]) == len(prepared["cases"]) and all(c["completion"] for c in artifact.document["cases"])
    receipt["row"].update(capture_finished=complete, outcome="FAIL" if any(u["findings"] for u in receipt["units"]) else "STOPPED", finalization_finished=False,
                          utc_end=datetime.now(timezone.utc).isoformat(), exit_status=0 if complete else 3)
    progress.record("setup", checkpoint=True)
    receipt["row"]["gpu_minutes"] = args.meter.finish()
    bind_evidence(args.out, artifact)
    artifact.finish(receipt=receipt)


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
        unit["exact"]["scale_index"]["arrays"][case["array_index"]]["mismatches"] += result["mismatches"]
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
    prepared, prepare_digest = read(args.prepare_artifact, "prepare", common, expected_scopes)
    if not (args.capture_artifact / "phase.json").exists():
        for name in ("unfinished.json", "gate-complete.json"):
            path = args.capture_artifact / name
            if path.is_file():
                partial = json.loads(path.read_text())
                if partial.get("identities") != common or partial.get("scopes") != expected_scopes or partial.get("prepare_digest") != prepare_digest:
                    raise Invalid("unfinished capture belongs to another row")
                receipt.update(partial["receipt"])
                break
        raise RuntimeError("capture is unfinished; compare cannot finish this row")
    captured, capture_digest = read(args.capture_artifact, "capture", common, expected_scopes)
    compare_row = receipt["row"]
    receipt.update(copy.deepcopy(captured["receipt"]))
    pairs = check_capture(prepared, captured, prepare_digest)
    # Reuse verified roster/coverage counts, then replace every device-dependent field.
    base = copy.deepcopy(prepared["receipt"])
    final = copy.deepcopy(captured["receipt"])
    receipt.update(final)
    receipt["row"]["phase_jobs"] = {"prepare": copy.deepcopy(prepared["receipt"]["row"]),
        "capture": copy.deepcopy(captured["receipt"]["row"]), "compare": compare_row}
    for phase, record in receipt['row']['phase_jobs'].items():
        record.update(record.pop('phase_jobs', {}).get(phase, {}))
    progress = args.progress = Progress(args.out, receipt, phases)
    known = {json.dumps(row.unit_identity(u), sort_keys=True): u for u in base["units"]}
    for unit in receipt["units"]:
        previous = known[json.dumps(row.unit_identity(unit), sort_keys=True)]
        unit["exact"] = copy.deepcopy(previous["exact"])
        unit["exact"]["finite_certificate"] = {"elements": 0, "held": 0, "nonfinite_where_held": 0}
        unit["exact"]["composition"] = {"cases": 0, "equal": True}
        unit["completed"] = True
        for case, observed in pairs:
            if case["scope"]["unit_key"] != row.unit_identity(unit):
                continue
            progress.record("diagnostics" if case["category"] == "diagnostic" else "gate_cases", unit, case["id"])
            compare_case(case, observed, args.prepare_artifact, args.capture_artifact, unit)
        row.finish_unit(unit)
    # Only a finished, validated compare creates a joinable receipt.
    receipt["row"]["utc_start"] = prepared["receipt"]["row"]["utc_start"]
    compare_row.update(utc_end=datetime.now(timezone.utc).isoformat())
    receipt["row"].update(utc_end=compare_row["utc_end"],
                          outcome=min((u["outcome"] for u in receipt["units"]), key=OUTCOMES.index),
                          exit_status=3 if any(u["outcome"] in ("FAIL", "STOPPED", "INVALID") for u in receipt["units"]) else 0)
    receipt["phase_artifacts"] = {"prepare": {"path": str(args.prepare_artifact / "phase.json"), "sha256": prepare_digest},
                                  "capture": {"path": str(args.capture_artifact / "phase.json"), "sha256": capture_digest}}
    artifact = Artifact(args.out / "compare", "compare", common)
    artifact.document.update(scopes=expected_scopes, prepare_digest=prepare_digest, capture_digest=capture_digest)
    progress.record("setup", checkpoint=True)
    bind_evidence(args.out, artifact)
    artifact.finish(case_ids=[case["id"] for case, _ in pairs], receipt=receipt)
    receipt["phase_artifacts"]["compare"] = {"path": str(artifact.root / "phase.json"), "sha256": sha256(artifact.root / "phase.json")}
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
            artifact.document["files"][str(relative)] = sha256(destination)
    checkpoint = artifact.root / "gate-complete.json"
    if checkpoint.is_file():
        artifact.document["files"][checkpoint.name] = sha256(checkpoint)
