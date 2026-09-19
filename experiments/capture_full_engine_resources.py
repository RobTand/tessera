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

from experiments.full_engine_resources import PRICING_SCOPE


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


CALIBRATION_WIDTH = 512

RECEIPT_SCHEMA = "tessera.native_moe_operator_receipt.v1"

#: The allocator segment policy the reserved extent is a function of, bound in
#: the configuration document's ``environment`` block. ``"unset"`` is an
#: explicit value meaning the capture ran with no ``PYTORCH_CUDA_ALLOC_CONF``
#: in its worker environment -- never a default the launcher fills in
#: silently, and never the literal string handed to the worker.
ALLOCATOR_POLICY_KEY = "PYTORCH_CUDA_ALLOC_CONF"
UNSET_ALLOCATOR_POLICY = "unset"


def require_allocator_policy(config):
    """Read the bound allocator segment policy, or refuse it by name.

    Reserved minus allocated is a function of the allocation sequence and the
    allocator's segment policy, so a reservation witness transfers from a
    capture to a serve only when this value is equal. The configuration digest
    (``configuration_sha256``) is taken over the file that carries it, so the
    binding moves when the policy moves. This is the one home of the
    config-grammar rule: :func:`prepare` calls it before any input is read,
    and the report side enforces the same binding through
    ``allocator_config_of`` plus the assembler's refusal of an unbound claim.
    """
    environment = config.get("environment") if isinstance(config, dict) else None
    value = environment.get(ALLOCATOR_POLICY_KEY) if isinstance(environment, dict) else None
    if type(value) is not str or not value:
        raise ValueError(
            f"selected configuration names no {ALLOCATOR_POLICY_KEY} in its environment "
            "block: the reserved-extent witness transfers only under an equal allocator "
            "segment policy, so the bound value -- including \"unset\" -- is required")
    return value


def read_routed_owner_receipt(path, expected_sha256, *, rank, world_size):
    """The one independent number a startup record cannot derive from itself.

    ``worker_startup_records[0].receipt_resident_bytes`` is the routed-owner
    receipt's own ``resources.resident_bytes``, and the consumer checks it
    against the ledger's fixed-owned resident rows rather than trusting either
    side. The receipt therefore has to be a real artifact of the same rank of
    the same world; a receipt from another rank is a different rank's charge and
    is refused here rather than relabelled.
    """
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError("routed-owner receipt bytes differ from the declared digest")
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"startup receipt must be {RECEIPT_SCHEMA}, not {receipt.get('schema')!r}")
    resources = receipt.get("resources")
    if type(resources) is not dict:
        raise ValueError("routed-owner receipt carries no resources block")
    if resources.get("rank") != rank or resources.get("world_size") != world_size:
        raise ValueError(
            f"routed-owner receipt is rank {resources.get('rank')}/{resources.get('world_size')} "
            f"and this capture is {rank}/{world_size}")
    resident = resources.get("resident_bytes")
    if type(resident) is not int or resident < 0:
        raise ValueError("routed-owner receipt resources.resident_bytes must be a non-negative integer")
    return {"path": str(path.resolve()), "sha256": expected_sha256, "schema": RECEIPT_SCHEMA,
            "rank": rank, "world_size": world_size, "resident_bytes": resident,
            "scope": "the routed-owner receipt's own resources.resident_bytes, checked against "
                     "the ledger's fixed-owned resident rows by the report consumer"}


def read_calibration_prompt(path, expected_sha256):
    """Return the canonical first calibration row and its fixture identity.

    The fixture is ``int64[n, 512]`` ``calibration_ids`` for any ``n >= 1``: the
    first-model fixture carries 512 rows and the artifact fixtures carry the
    eight rows of the serving KL contract, and the observer reads row 0 of
    either. The row count is recorded so the workload digest names the fixture
    shape and not only its bytes.
    """
    if digest(path) != expected_sha256:
        raise ValueError("calibration bytes differ from the declared immutable fixture")
    from safetensors import safe_open
    with safe_open(str(path), framework="np") as fixture:
        token_ids = fixture.get_tensor("calibration_ids")
    if (token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] != CALIBRATION_WIDTH
            or str(token_ids.dtype) != "int64"):
        raise ValueError(f"resource calibration requires int64[n>=1,{CALIBRATION_WIDTH}] calibration_ids")
    return {"prompt_token_ids": token_ids[0].tolist(),
            "calibration": {"path": str(Path(path).resolve()), "sha256": expected_sha256,
                            "key": "calibration_ids", "row": 0, "rows": int(token_ids.shape[0])}}


def prepare(args):
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("prepare process imported Torch/vLLM before bootstrap")
    config = json.loads(args.config.read_text())
    # First, before any input is read: the allocator policy binds the
    # reservation witness, and an unbound configuration cannot produce one.
    allocator_policy = require_allocator_policy(config)
    artifact = None
    if args.artifact:
        # A served Tessera artifact carries its own roster and assignment in
        # tessera_serving_manifest.json; the census-derived source-BF16 roster
        # and its BF16-only refusal do not apply to it.
        from experiments.full_engine_artifact import read_tessera_artifact
        source, roster, assignment = read_tessera_artifact(args.model)
        artifact = {"schema": "tessera.artifact_observer_checkpoint.v1",
                    "path": str(args.model.resolve()), "source_sha256": canonical_hash(source),
                    "manifest_sha256": source["files"]["tessera_serving_manifest.json"],
                    "families": sorted({row["family"] for row in roster}),
                    "candidate_rule": "actual native owner parameters and buffers of each manifest module; external aliases fixed"}
    else:
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
    if artifact is None:
        roster = canonical_roster(census)
    by_id = {row["unit_id"]: row for row in roster}
    mode = getattr(args, "observation_mode", "resources")
    prefix_only = getattr(args, "qualify_first_native_prefix", False)
    world_size, rank = args.world_size, args.rank
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is not inside a world of {world_size}")
    configured_tp = config["engine_args"].get("tensor_parallel_size", 1)
    if world_size != configured_tp:
        raise ValueError(
            f"--world-size {world_size} differs from the selected engine's tensor_parallel_size "
            f"{configured_tp}; one capture observes one rank of the world the configuration runs")
    if mode == "kv":
        if not getattr(args, "all_units", False) or args.unit:
            raise ValueError("the read-only KV pass reads the complete roster's plan identity; "
                             "require --all-units without a partial --unit selection")
        if args.calibration is None:
            raise ValueError("the read-only KV pass must carry the same calibration workload as "
                             "the resource pass it is joined with, so its run identity matches")
        if getattr(args, "reference_proof", None) is not None or prefix_only:
            raise ValueError("the read-only KV pass takes neither an original-wire reference nor "
                             "a first-native prefix; it is a stock engine's own KV observation")
    if prefix_only and (mode != "resources" or not args.all_units or args.unit):
        raise ValueError("first-native prefix qualification requires resource mode and the complete native roster")
    units = roster if getattr(args, "all_units", False) else [by_id[name] for name in args.unit]
    if len(units) != len({row["unit_id"] for row in units}):
        raise ValueError("duplicate observed unit")
    if artifact is None:
        assignment = {"schema": "tessera.source_bf16_observer_assignment.v1",
                      "source_sha256": canonical_hash(source),
                      "units": {row["unit_id"]: "source_bf16" for row in roster}}
    elif not args.all_units or args.unit or prefix_only:
        # tessera#399: an artifact is observed by all three passes -- the
        # intrusive resource ledger, the read-only KV pass and the profiled
        # timing partition -- on its complete manifest roster. The three share
        # one plan identity (configuration, model, roster, workload), which is
        # what lets the report join them; a partial roster or a first-native
        # prefix would give the joined passes different identities.
        raise ValueError("artifact observation requires the complete manifest roster (--all-units) "
                         "without a partial --unit selection or a first-native prefix")
    elif mode == "timings" and not isinstance(config.get("capacity_assertions"), dict):
        # The timing worker asserts the KV capacity before its first step and
        # refuses a run whose runner resolves another; a configuration document
        # that declares none gives it nothing to assert against.
        raise ValueError("timing observation of an artifact requires capacity_assertions in the "
                         "selected configuration document")
    reference = None
    if getattr(args, "reference_proof", None) is not None:
        if artifact is not None:
            raise ValueError("an artifact is its own reference checkpoint; --reference-proof applies to source BF16 only")
        if not args.all_units or args.unit:
            raise ValueError("original-wire reference observation requires --all-units without partial selection")
        from experiments.full_engine_reference import verify_reference_checkpoint
        reference = verify_reference_checkpoint(args.reference_proof, source, roster, digest(args.census))
        assignment = reference["assignment"]
    if args.calibration is not None:
        workload = read_calibration_prompt(args.calibration, args.calibration_sha256)
        workload["scope"] = "canonical first 512-token sequence unchanged; actual generated-token decode, distinct from native boundary decode proxy"
    else:
        workload = {"messages": [{"role": "user", "content": "Return exactly the word blue."}],
                    "scope": "two scheduled steps for raw allocation observation; no quality claim"}
    workload["sampling"] = {"temperature": 0.0, "seed": 0, "max_tokens": 2, "ignore_eos": True}
    if prefix_only:
        workload["resource_qualification"] = {
            "scope": "startup and first native invocation only; remaining request execution is unobserved",
            "native_invocations": 1, "complete_engine_capture": False,
            "profiler": "cProfile around first native observation and capture finalization"}
    if mode == "timings":
        if args.calibration is None or not args.all_units or args.unit:
            raise ValueError("timing observation requires the canonical calibration and --all-units without a partial --unit selection")
        if type(args.timing_samples) is not int or args.timing_samples < 1:
            raise ValueError("timing_samples must be a positive integer")
        workload["timing_protocol"] = {"samples": args.timing_samples,
            "arms": ["control", "partition"], "cache_state": "reset_prefix_cache before every request",
            "warmup": "one identical request before the interleaved profiled arm pairs",
            "scope": "observer qualification; no admitted timing price"}
    # One model forward per generated token invokes every unit once; the
    # engine may execute one more step that runs no forward.
    generated_tokens = workload["sampling"]["max_tokens"]
    declared_steps = generated_tokens + 1
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                   text=True).strip().splitlines()
    if len(uuid) != 1:
        raise ValueError("resource observer requires exactly one visible GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    # The rank scope travels in the plan identity so every downstream record --
    # the ledger, the startup sample, the KV observation -- binds to the rank
    # that measured it rather than to a world total.
    receipt = None
    if args.receipt is not None:
        if mode != "resources":
            raise ValueError("a routed-owner receipt belongs to the intrusive resource pass that "
                             "samples resident-after-load memory")
        receipt = read_routed_owner_receipt(args.receipt, args.receipt_sha256,
                                            rank=rank, world_size=world_size)
    plan = {"schema": "tessera.stock_engine_resource_observer_plan.v1",
            "identity": {"schema": "tessera.full_engine_resource_identity.v1",
                "model_sha256": canonical_hash(source), "configuration_sha256": digest(args.config),
                "runtime_manifest_sha256": digest(args.core_manifest),
                "assignment_sha256": canonical_hash(assignment),
                "canonical_units_sha256": canonical_hash(roster), "workload_sha256": canonical_hash(workload),
                "device_id": 0, "device_uuid": uuid[0],
                "rank": rank, "world_size": world_size},
            "collector_library": str(args.collector.resolve()) if args.collector is not None else None,
            "collector_library_sha256": digest(args.collector) if args.collector is not None else None,
            "output_directory": str(args.output.resolve()), "model": str(args.model.resolve()),
            "selected_configuration": config, "runtime_evidence_sha256": digest(args.runtime_evidence),
            "runtime_evidence": str(args.runtime_evidence.resolve()),
            "core_manifest": str(args.core_manifest.resolve()), "assignment": assignment,
            "canonical_roster": roster, "canonical_modules": [row["module"] for row in roster],
            "observed_units": units, "workload": workload,
            "max_history_entries": 1_000_000, "max_execute_calls": declared_steps,
            "max_invocations_per_unit": generated_tokens,
            "max_checkpoints": 7 + 3 * declared_steps + 2 * generated_tokens * len(units),
            "declared_steps_scope": "one engine step per generated token, plus one: the engine may execute a step it scheduled before it observed the request finish, and a step executed but undeclared leaves coverage partial",
            "observer_engine_args": {"worker_cls": "experiments.full_engine_worker.ResourceCaptureWorker"},
            "observer_environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            "scope": "intrusive raw resource capture; no timing, fixed-resource or release admission"}
    plan["observation_mode"] = mode
    plan["unit_boundary"] = "native_apply" if args.all_units else "module_forward"
    if receipt is not None:
        plan["routed_owner_receipt"] = receipt
    if prefix_only:
        plan["qualification_prefix"] = workload["resource_qualification"]
        plan["scope"] = "bounded first-native resource prefix qualification; incomplete engine capture, no admission"
    if artifact is not None:
        plan["artifact_checkpoint"] = artifact
        plan["scope"] = "intrusive raw resource capture of a served Tessera artifact; no timing, fixed-resource or release admission"
    if reference is not None:
        plan["reference_checkpoint"] = reference
        plan["model"] = reference["checkpoint"]
        plan["identity"]["source_model_sha256"] = plan["identity"]["model_sha256"]
        plan["identity"]["model_sha256"] = reference["checkpoint_sha256"]
    if mode == "timings":
        plan["timing_samples"] = args.timing_samples
        plan["observer_engine_args"] = {"worker_cls": "experiments.full_engine_timing_worker.TimingCaptureWorker"}
        plan["scope"] = "profiled all-native-unit event partition; observer qualification, no admitted timing or fixed-resource price"
    elif mode == "kv":
        # A stock engine and a stock worker: nothing is installed, no recorder
        # is claimed, no snapshot is taken. That is what makes this pass the
        # read-only half of the two-pass pair. Its one RPC is a function, and
        # the stock engine's msgspec transport refuses a function unless the
        # pickle fallback is allowed: the same observer-only transport setting
        # the original-wire launcher records (docs/measurements/original-layer2-
        # wire-checkpoint-2026-09-07.md), in a network-disabled container, kept
        # in the plan's observer environment and never in the configuration
        # document, whose digest it does not touch.
        plan["read_only_kv"] = True
        plan["observer_engine_args"] = {}
        plan["observer_environment"] = dict(plan["observer_environment"],
                                            VLLM_ALLOW_INSECURE_SERIALIZATION="1")
        plan["observer_environment_note"] = ("VLLM_ALLOW_INSECURE_SERIALIZATION=1 is the RPC transport "
                                             "for the read-only pass's function callback; it installs "
                                             "nothing in the worker and is outside the configuration digest")
        plan["scope"] = ("read-only stock-engine KV observation; no resource recorder, no "
                         "synchronized snapshot, no timing claim")
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
    # The worker's effective policy is the bound one, exactly: "unset" means
    # the variable is absent from the worker environment, never the literal
    # string, so a capture that claims "unset" cannot have run under a policy.
    if allocator_policy == UNSET_ALLOCATOR_POLICY:
        env.pop(ALLOCATOR_POLICY_KEY, None)
    env.update(plan["observer_environment"])
    if mode == "resources":
        env.update({"TESSERA_ENGINE_RESOURCE_PLAN": str(path.resolve()),
                    "PYTHONPATH": str(root / "experiments/resource_bootstrap") + os.pathsep + str(root)})
    elif mode == "timings":
        env.pop("TESSERA_ENGINE_RESOURCE_PLAN", None)
        env.update({"TESSERA_ENGINE_TIMING_PLAN": str(path.resolve()), "PYTHONPATH": str(root)})
    else:
        # kv: the plan travels through the run-plan argument only; no bootstrap
        # module may attach to a read-only pass.
        env.pop("TESSERA_ENGINE_RESOURCE_PLAN", None)
        env.update({"PYTHONPATH": str(root)})
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
    if plan.get("read_only_kv"):
        return run_read_only_kv(llm, plan, plan_path, started, audit_before)
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
        "qualification_prefix": plan.get("qualification_prefix"),
        "full_model_fixed_resources_complete": False, "timings": None,
        "admission": None, "pricing_scope": PRICING_SCOPE}
    (output / "run.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"artifact": str(output / "run.json"), "sha256": digest(output / "run.json"),
                      "worker_count": len(workers), "admission": None, "pricing_scope": PRICING_SCOPE}), flush=True)


def run_read_only_kv(llm, plan, plan_path, started, audit_before):
    """The read-only half of the two-pass pair: one stock engine's own KV view.

    Nothing is installed and no snapshot is taken, so the pass evidence this
    writes reports ``read_only: true`` and the projected ``runtime_admission``
    follows. The record binds the same run identity the intrusive resource pass
    carries, because the two are only the same run if the digests agree -- their
    process ids and their pointers are each pass's own and are never compared.
    """
    from experiments.full_engine_kv import (admission_evidence, kv_observation_record,
                                             read_only_kv_observation)
    output = Path(plan["output_directory"])
    workers = llm.collective_rpc(read_only_kv_observation)
    audit_after = audit_core(plan["core_manifest"])
    if len(workers) != 1:
        raise ValueError("the read-only KV pass observes exactly one worker's KV pool")
    identity = plan["identity"]
    evidence = admission_evidence(mode="kv", process_id=workers[0]["process_id"],
                                  recorder_attached=False, snapshot_count=0)
    record = kv_observation_record(
        workers[0]["observation"], evidence=evidence, rank=identity["rank"],
        world_size=identity["world_size"], run_identity=identity,
        scope=("stock engine's own resolved KV descriptors and deduplicated physical "
               "backings, read by a worker RPC that attaches nothing and takes no snapshot"))
    observation_path = output / "kv-observation.json"
    observation_path.write_text(json.dumps(
        {"schema": "tessera.full_engine_read_only_kv_pass.v1",
         "records": [record], "plan_sha256": digest(plan_path),
         "observer_worker_process_id": workers[0]["process_id"]},
        sort_keys=True, indent=2) + "\n")
    result = {"schema": "tessera.stock_engine_read_only_kv_run.v1", "scope": plan["scope"],
              "plan_sha256": digest(plan_path), "started_unix": started, "finished_unix": time.time(),
              "core_audit_before": audit_before, "core_audit_after": audit_after,
              "kv_observation": {"path": str(observation_path),
                                 "sha256": digest(observation_path),
                                 "runtime_admission": record["runtime_admission"]},
              "runtime_admission": record["runtime_admission"],
              "full_model_fixed_resources_complete": False, "timings": None,
              "admission": None, "pricing_scope": PRICING_SCOPE}
    (output / "run.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"artifact": str(output / "run.json"), "sha256": digest(output / "run.json"),
                      "kv_observation_sha256": result["kv_observation"]["sha256"],
                      "runtime_admission": record["runtime_admission"],
                      "admission": None, "pricing_scope": PRICING_SCOPE}), flush=True)


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
              "full_model_fixed_resources_complete": False, "admission": None, "pricing_scope": PRICING_SCOPE}
    path = Path(plan["output_directory"]) / "run.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"artifact": str(path), "sha256": digest(path), "admission": None, "pricing_scope": PRICING_SCOPE}), flush=True)
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
    parser.add_argument("--observation-mode", choices=("resources", "timings", "kv"), default="resources",
                        help="resources: intrusive ledger pass; timings: profiled event partition; "
                             "kv: read-only stock-engine KV pass")
    parser.add_argument("--timing-samples", type=int, default=1)
    parser.add_argument("--reference-proof", type=Path)
    parser.add_argument("--qualify-first-native-prefix", action="store_true")
    parser.add_argument("--rank", type=int, default=0, help="the rank this capture observes")
    parser.add_argument("--world-size", type=int, default=1, help="the tensor-parallel world it belongs to")
    parser.add_argument("--receipt", type=Path,
                        help="the routed-owner receipt whose resources.resident_bytes the startup sample binds to")
    parser.add_argument("--receipt-sha256", help="the declared digest of --receipt")
    parser.add_argument("--artifact", action="store_true",
                        help="observe a served Tessera artifact; roster and assignment come from its serving manifest")
    args = parser.parse_args()
    if args.run_plan:
        run(args.run_plan)
    else:
        required = (("config", "model", "core_manifest", "runtime_evidence", "output")
                    if args.observation_mode == "kv"
                    else ("config", "model", "collector", "core_manifest", "runtime_evidence", "output"))
        if any(getattr(args, name) is None for name in required) or (args.census is None) != args.artifact:
            parser.error("prepare requires all input, runtime, collector and output paths, and a census "
                         "unless --artifact selects the manifest roster")
        if (args.receipt is None) != (args.receipt_sha256 is None):
            parser.error("--receipt and --receipt-sha256 travel together")
        if args.observation_mode == "kv":
            if not args.all_units:
                parser.error("the read-only KV pass requires --all-units to name the plan identity it shares")
        elif not args.unit and not args.all_units:
            parser.error("select at least one canonical --unit")
        prepare(args)


if __name__ == "__main__":
    main()
