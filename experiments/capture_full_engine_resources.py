"""Launch a bound stock-vLLM resource/timing observation, without admission.

Run inside an admitted PrismaBuild action in the attested stock container.
The first process verifies inputs and writes the observer plan. A fresh Python
process enables the early bootstrap for itself and spawned engine workers.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def canonical_roster(census):
    grouped = {member for members in census["anchor_groups"].values() for member in members}
    roster = [{"unit_id": name, "module": name.split(":", 1)[1], "members": members}
              for name, members in census["anchor_groups"].items()]
    roster.extend({"unit_id": "l:" + name, "module": name, "members": [name]}
                  for name in census["dense_targets"] if name not in grouped)
    if not roster or len({row["module"] for row in roster}) != len(roster):
        raise ValueError("canonical census must resolve to distinct nonempty modules")
    return sorted(roster, key=lambda row: row["unit_id"])


def audit_core(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(importlib.util.find_spec("vllm").origin).parent
    actual = {str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size}
              for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    if actual != manifest["files"]:
        raise ValueError("stock vLLM core differs from the immutable runtime manifest")
    return {"root": str(root), "manifest_sha256": digest(manifest_path),
            "unchanged_files": len(actual), "scope": "all installed non-pycache core files"}


def prepare(args):
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("prepare process imported Torch/vLLM before bootstrap")
    config = json.loads(args.config.read_text())
    census = json.loads(args.census.read_text())
    source = census["expert_projection"]["producer"]["source"]
    expected = {**source["files"], **source["auxiliary_sha256"]}
    for name, expected_sha in expected.items():
        if digest(args.model / name) != expected_sha:
            raise ValueError(f"model input differs from canonical census: {name}")
    if json.loads((args.model / "config.json").read_text()).get("quantization_config"):
        raise ValueError("this observer launcher is restricted to source BF16")
    if config["engine_args"]["dtype"] != "bfloat16":
        raise ValueError("source baseline configuration must select BF16")
    runtime = json.loads(args.runtime_evidence.read_text())
    if runtime["registry_base"] != config["runtime_image"]:
        raise ValueError("installed runtime image does not match selected configuration")
    if runtime["upstream_commit"] != config["runtime_identity"]["upstream_vllm_commit"]:
        raise ValueError("installed runtime core identity does not match selected configuration")
    if runtime["core_manifest_sha256"] != digest(args.core_manifest):
        raise ValueError("runtime manifest differs from per-job installation evidence")
    roster = canonical_roster(census)
    by_id = {row["unit_id"]: row for row in roster}
    mode = getattr(args, "observation_mode", "resources")
    units = roster if getattr(args, "all_units", False) else [by_id[name] for name in args.unit]
    if len(units) != len({row["unit_id"] for row in units}):
        raise ValueError("duplicate observed unit")
    assignment = {"schema": "tessera.source_bf16_observer_assignment.v1",
                  "source_sha256": canonical_hash(source),
                  "units": {row["unit_id"]: "source_bf16" for row in roster}}
    reference = None
    if getattr(args, "reference_proof", None) is not None:
        if not args.all_units or args.unit:
            raise ValueError("original-wire reference observation requires --all-units without partial selection")
        from experiments.full_engine_reference import verify_reference_checkpoint
        reference = verify_reference_checkpoint(args.reference_proof, source, roster, digest(args.census))
        assignment = reference["assignment"]
    if args.calibration is not None:
        if digest(args.calibration) != args.calibration_sha256:
            raise ValueError("calibration bytes differ from the declared immutable fixture")
        from safetensors import safe_open
        with safe_open(str(args.calibration), framework="np") as fixture:
            token_ids = fixture.get_tensor("calibration_ids")
        if token_ids.shape != (512, 512) or str(token_ids.dtype) != "int64":
            raise ValueError("resource calibration requires int64[512,512] calibration_ids")
        workload = {"prompt_token_ids": token_ids[0].tolist(),
                    "calibration": {"path": str(args.calibration.resolve()),
                        "sha256": args.calibration_sha256, "key": "calibration_ids", "row": 0},
                    "scope": "canonical first 512-token sequence unchanged; actual generated-token decode, distinct from native boundary decode proxy"}
    else:
        workload = {"messages": [{"role": "user", "content": "Return exactly the word blue."}],
                    "scope": "two scheduled steps for raw allocation observation; no quality claim"}
    workload["sampling"] = {"temperature": 0.0, "seed": 0, "max_tokens": 2, "ignore_eos": True}
    if mode == "timings":
        if args.calibration is None or not args.all_units or args.unit:
            raise ValueError("timing observation requires the canonical calibration and --all-units without a partial --unit selection")
        if type(args.timing_samples) is not int or args.timing_samples < 1:
            raise ValueError("timing_samples must be a positive integer")
        workload["timing_protocol"] = {"samples": args.timing_samples,
            "arms": ["control", "partition"], "cache_state": "reset_prefix_cache before every request",
            "warmup": "one identical request before the interleaved profiled arm pairs",
            "scope": "observer qualification; no admitted timing price"}
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                   text=True).strip().splitlines()
    if len(uuid) != 1:
        raise ValueError("resource observer requires exactly one visible GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "tessera.stock_engine_resource_observer_plan.v1",
            "identity": {"schema": "tessera.full_engine_resource_identity.v1",
                "model_sha256": canonical_hash(source), "configuration_sha256": digest(args.config),
                "runtime_manifest_sha256": digest(args.core_manifest),
                "assignment_sha256": canonical_hash(assignment),
                "canonical_units_sha256": canonical_hash(roster), "workload_sha256": canonical_hash(workload),
                "device_id": 0, "device_uuid": uuid[0]},
            "collector_library": str(args.collector.resolve()),
            "collector_library_sha256": digest(args.collector),
            "output_directory": str(args.output.resolve()), "model": str(args.model.resolve()),
            "selected_configuration": config, "runtime_evidence_sha256": digest(args.runtime_evidence),
            "runtime_evidence": str(args.runtime_evidence.resolve()),
            "core_manifest": str(args.core_manifest.resolve()), "assignment": assignment,
            "canonical_roster": roster, "canonical_modules": [row["module"] for row in roster],
            "observed_units": units, "workload": workload,
            "max_history_entries": 1_000_000, "max_execute_calls": 2,
            "max_invocations_per_unit": 2,
            "max_checkpoints": 7 + 2 * 3 + 2 * 2 * len(units),
            "observer_engine_args": {"worker_cls": "experiments.full_engine_worker.ResourceCaptureWorker"},
            "observer_environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            "scope": "intrusive raw resource capture; no timing, fixed-resource or release admission"}
    plan["observation_mode"] = mode
    plan["unit_boundary"] = "native_apply" if args.all_units else "module_forward"
    if reference is not None:
        plan["reference_checkpoint"] = reference
        plan["model"] = reference["checkpoint"]
        plan["identity"]["source_model_sha256"] = plan["identity"]["model_sha256"]
        plan["identity"]["model_sha256"] = reference["checkpoint_sha256"]
    if mode == "timings":
        plan["timing_samples"] = args.timing_samples
        plan["observer_engine_args"] = {"worker_cls": "experiments.full_engine_timing_worker.TimingCaptureWorker"}
        plan["scope"] = "profiled all-native-unit event partition; observer qualification, no admitted timing or fixed-resource price"
    if args.workspaces is not None:
        plan["blas_workspace_observer"] = {"path": str(args.workspaces.resolve()),
                                          "sha256": digest(args.workspaces)}
    if args.owner_rule is not None:
        plan["native_owner_rule"] = {"path": str(args.owner_rule.resolve()),
                                     "sha256": digest(args.owner_rule)}
    path = args.output / "observer-plan.json"
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(config["environment"])
    env.update(plan["observer_environment"])
    if mode == "resources":
        env.update({"TESSERA_ENGINE_RESOURCE_PLAN": str(path.resolve()),
                    "PYTHONPATH": str(root / "experiments/resource_bootstrap") + os.pathsep + str(root)})
    else:
        env.pop("TESSERA_ENGINE_RESOURCE_PLAN", None)
        env.update({"TESSERA_ENGINE_TIMING_PLAN": str(path.resolve()), "PYTHONPATH": str(root)})
    os.execve(sys.executable, [sys.executable, "-m", "experiments.capture_full_engine_resources",
                             "--run-plan", str(path.resolve())], env)


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    output = Path(plan["output_directory"])
    started = time.time()
    audit_before = audit_core(plan["core_manifest"])
    from vllm import LLM, SamplingParams
    llm = LLM(model=plan["model"], seed=0,
              **plan["selected_configuration"]["engine_args"], **plan["observer_engine_args"])
    if plan.get("observation_mode") == "timings":
        return run_timings(llm, plan, plan_path, started, audit_before)
    armed = llm.collective_rpc("resource_capture_arm")
    workload = plan["workload"]
    sampling = SamplingParams(**workload["sampling"])
    if "prompt_token_ids" in workload:
        responses = llm.generate({"prompt_token_ids": workload["prompt_token_ids"]}, sampling, use_tqdm=False)
        if len(responses) != 1 or responses[0].prompt_token_ids != workload["prompt_token_ids"]:
            raise ValueError("engine changed the explicit calibration TokenPrompt")
    else:
        responses = llm.chat(workload["messages"], sampling, use_tqdm=False)
    workers = llm.collective_rpc("resource_capture_finish")
    audit_after = audit_core(plan["core_manifest"])
    if len(workers) != 1:
        raise ValueError("resource observer expected exactly one worker result")
    result = {"schema": "tessera.stock_engine_raw_resource_run.v1", "scope": plan["scope"],
        "plan_sha256": digest(plan_path), "started_unix": started, "finished_unix": time.time(),
        "workload_arm": armed,
        "core_audit_before": audit_before, "core_audit_after": audit_after, "workers": workers,
        "outputs": [{"prompt_token_ids": response.prompt_token_ids,
                     "outputs": [{"text": item.text, "token_ids": item.token_ids,
                                  "finish_reason": item.finish_reason} for item in response.outputs]}
                    for response in responses],
        "full_model_fixed_resources_complete": False, "timings": None,
        "admission": "not_implemented"}
    (output / "run.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"artifact": str(output / "run.json"), "sha256": digest(output / "run.json"),
                      "worker_count": len(workers), "admission": "not_implemented"}), flush=True)


def run_timings(llm, plan, plan_path, started, audit_before):
    from vllm import SamplingParams
    workload = plan["workload"]
    sampling = SamplingParams(**workload["sampling"])

    def request():
        if llm.reset_prefix_cache() is not True:
            raise RuntimeError("timing request could not establish cold prefix state")
        responses = llm.generate({"prompt_token_ids": workload["prompt_token_ids"]}, sampling, use_tqdm=False)
        if len(responses) != 1 or responses[0].prompt_token_ids != workload["prompt_token_ids"]:
            raise ValueError("timing engine changed the canonical TokenPrompt")
        tokens = [item.token_ids for item in responses[0].outputs]
        if len(tokens) != 1 or len(tokens[0]) != 2:
            raise ValueError("timing request did not generate exactly two tokens")
        return tokens[0]

    warmup_tokens = request()
    arms = []
    for sample in range(plan["timing_samples"]):
        for arm in ("control", "partition"):
            armed = llm.collective_rpc("timing_capture_arm", args=(arm, sample))
            tokens = request()
            workers = llm.collective_rpc("timing_capture_finish")
            if len(workers) != 1 or tokens != warmup_tokens:
                raise ValueError("timing worker population or generated tokens changed between arms")
            arms.append({"arm": arm, "sample": sample, "armed": armed, "tokens": tokens, "workers": workers})
    coverage_verified = all(item["workers"][0]["partition"]["status"] == "observed_same_run_partition"
                            for item in arms if item["arm"] == "partition")
    result = {"schema": "tessera.stock_engine_raw_timing_run.v1", "scope": plan["scope"],
              "plan_sha256": digest(plan_path), "started_unix": started, "finished_unix": time.time(),
              "core_audit_before": audit_before, "core_audit_after": audit_core(plan["core_manifest"]),
              "warmup_tokens": warmup_tokens, "arms": arms, "timings": None,
              "partition_coverage_verified": coverage_verified,
              "full_model_fixed_resources_complete": False, "admission": "not_implemented"}
    path = Path(plan["output_directory"]) / "run.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"artifact": str(path), "sha256": digest(path), "admission": "not_implemented"}), flush=True)
    if not coverage_verified:
        raise RuntimeError("all-unit profiler/event partition remains incomplete; retained raw timing run: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--collector", type=Path)
    parser.add_argument("--core-manifest", type=Path)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--workspaces", type=Path)
    parser.add_argument("--owner-rule", type=Path)
    parser.add_argument("--unit", action="append", default=[])
    parser.add_argument("--all-units", action="store_true")
    parser.add_argument("--observation-mode", choices=("resources", "timings"), default="resources")
    parser.add_argument("--timing-samples", type=int, default=1)
    parser.add_argument("--reference-proof", type=Path)
    args = parser.parse_args()
    if args.run_plan:
        run(args.run_plan)
    else:
        if any(getattr(args, name) is None for name in
               ("config", "census", "model", "collector", "core_manifest", "runtime_evidence", "output")):
            parser.error("prepare requires all input, runtime, collector and output paths")
        if not args.unit and not args.all_units:
            parser.error("select at least one canonical --unit")
        prepare(args)


if __name__ == "__main__":
    main()
