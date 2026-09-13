"""GPU qualification of the dense receipt producer on retained artifact bytes.

Emits one ``tessera.native_dense_operator_receipt.v1`` receipt (the schema
defined by ``experiments/bench_native_operator.py``) for a single
eager/resident/TP1 dense operator, using an existing original wire blob, its
``tessera.encoding_inputs.v1`` record, and externally rendered reference
tensors, instead of the synthetic encode in
``experiments/qualify_native_operator.py``.

Scope. The panel frozen here carries the wire's own source and calibration
identities, but declares explicit fixture values for the joint cost and probe
identities: a PrismaQuant-frozen panel takes those from a joint AURA cost row
(``prismaquant.joint_aura.operator.v1``), and no such row exists for this
artifact. The result is therefore a producer qualification of timing, route and
native scratch bounds on real bytes -- not a joint-bound runtime price, a
quality claim about the render, or a release admission. Reference tensors are
consumed as supplied; this driver neither renders nor re-scores them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

FIXTURE_PREFIX = "producer qualification ONLY, no joint AURA row: "


def fixture_sha256(label):
    """Declared non-joint panel coordinate; never a real cost/probe digest."""
    return hashlib.sha256((FIXTURE_PREFIX + label).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True,
                        help="tessera.native_dense_request.v1 naming the retained wire/record/tensors")
    parser.add_argument("--library", type=Path, required=True, help="CUPTI collector shared object")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--atol", type=float, required=True, help="predeclared frozen tolerance")
    parser.add_argument("--rtol", type=float, required=True, help="predeclared frozen tolerance")
    args = parser.parse_args(argv)

    request = json.loads(args.request.read_text())
    if request["schema"] != "tessera.native_dense_request.v1":
        raise ValueError("native request schema unsupported")

    def artifact(key):
        path = Path(request[key])
        return path if path.is_absolute() else args.request.parent / path

    from experiments.native_operator_resources import NativeMemoryCollector
    collector = NativeMemoryCollector(args.library)
    import torch
    from safetensors.torch import load_file
    from experiments import bench_native_operator as bench

    if args.out.resolve() in {args.request.resolve(), args.library.resolve(),
                              *(artifact(k).resolve() for k in ("wire_path", "wire_record_path", "tensors_path"))}:
        raise ValueError("receipt output would overwrite an input artifact")

    with bench.native_runtime_context():
        blob = artifact("wire_path").read_bytes()
        record = json.loads(artifact("wire_record_path").read_text())
        tensors = load_file(str(artifact("tensors_path")), device="cuda")
        prepared = bench.prepare_native_operator(blob, record, tensors["source_weight"],
            tensors["rendered_weight"], unit=request["unit"], format_name=request["format"],
            runtime_image=request["runtime_image"], input_global_scale=request["input_global_scale"],
            execution=request["execution"])
        prepared["runtime"]["resource_collector"] = {
            "library_sha256": collector.library_sha256,
            "analysis_source_sha256": hashlib.sha256(
                Path(bench.__file__).with_name("native_operator_resources.py").read_bytes()).hexdigest()}
        operator = prepared["operator"]
        route = operator["declared_route"]
        shape = [operator["scheme"]["rows"], operator["scheme"]["columns"]]
        joint = {"schema": "prismaquant.joint_aura.operator.v1", "qname": request["unit"],
                 "format": request["format"], "probe_identity_sha256": fixture_sha256("probe identity"),
                 "source_weight": operator["source_weight"], "rendered_weight": operator["rendered_weight"],
                 "activation": {"clip_enabled": False,
                                "input_global_scale": operator["input_global_scale"]}}
        phase_tensors, phases = {}, {}
        for phase in bench.PHASES:
            values = {key: tensors[f"{phase}.{key}"] for key in
                      ("input", "reference_qdq", "reference_output")}
            phase_tensors[phase] = values
            phases[phase] = {"m": values["input"].shape[0], "expected_route": route,
                             **{key: bench.tensor_identity(value) for key, value in values.items()}}
        identity = record["identity"]
        panel = {"schema": bench.PANEL_SCHEMA, "unit": request["unit"], "format": request["format"],
                 "shape": shape, "source_sha256": identity["source"]["sha256"],
                 "calibration_sha256": bench.identity_sha256(identity["calibration"]),
                 "cost_sha256": fixture_sha256("cost payload"),
                 "probe_identity_sha256": joint["probe_identity_sha256"],
                 "joint_operator_identity_sha256": bench.identity_sha256(joint),
                 "joint_operator_identity": joint,
                 "wire": {"blob_sha256": record["blob_sha256"], "blob_bytes": record["blob_bytes"],
                          "record": record},
                 "execution": dict(bench.EXECUTION), "runtime": prepared["runtime"],
                 "native_tensors_sha256": bench.identity_sha256(operator["native_tensors"]),
                 "scheme_sha256": operator["scheme_sha256"],
                 "numerics": {"atol": args.atol, "rtol": args.rtol}, "phases": phases}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        panel_path = args.out.with_suffix(args.out.suffix + ".panel.json")
        panel_path.write_text(json.dumps(panel, indent=2, sort_keys=True, allow_nan=False) + "\n")
        trace_path = args.out.with_suffix(args.out.suffix + ".memory.json")
        try:
            receipt = bench.measure_prepared_operator(prepared, panel, phase_tensors,
                warmup_iterations=args.warmup_iterations, iterations=args.iterations,
                resource_collector=collector)
        finally:
            trace = collector.finish(trace_path)
        bench.attach_resource_trace(receipt, trace)
        if receipt["status"] == "resources_observed" and receipt["resources"]["status"] == "complete_operator_bound":
            bench.time_after_resource_collection(prepared, panel, phase_tensors, receipt,
                collector=collector, warmup_iterations=args.warmup_iterations, iterations=args.iterations)
        receipt["qualification_scope"] = (
            "actual retained wire/render/reference bytes; panel cost and probe identities are"
            " declared fixtures, so this is a producer timing/resource qualification and not a"
            " joint AURA runtime price, render-quality result or release admission")
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        published = {"schema": "tessera.native_dense_publication.v1", "status": receipt["status"],
                     "receipt_path": str(args.out),
                     "receipt_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                     "panel_path": str(panel_path),
                     "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                     "memory_trace_path": str(trace_path),
                     "memory_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                     "resources": receipt["resources"]["status"],
                     "phases": {phase: {"numerics": receipt["phases"][phase]["numerics"],
                                        "measurement": receipt["phases"][phase]["measurement"],
                                        "bound": receipt["resources"]["phases"][phase].get("bound")}
                                for phase in bench.PHASES}}
        print(json.dumps(published, sort_keys=True), flush=True)
        return 0 if (receipt["status"] == "timing_admissible"
                     and receipt["resources"]["status"] == "complete_operator_bound") else 2


if __name__ == "__main__":
    raise SystemExit(main())
