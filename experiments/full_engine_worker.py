"""Supported stock-vLLM worker subclass for intrusive raw resource capture.

No serving algorithm is replaced. Synchronized snapshots alter execution and
must never be used as timings or as fixed/KV admission. The owning launcher
audits the installed core before and after this worker runs.
"""
import json
import hashlib
import cProfile
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from vllm.v1.worker.gpu_worker import Worker

from experiments.full_engine_bootstrap import claim
from experiments.full_engine_resources import TensorOwner, BlasWorkspaceObserver
from experiments.full_engine_kv import kv_config_value, kv_configuration_observation, inspect_worker_kv
from experiments.full_engine_timing_boundaries import resolve_apply_boundaries, observe_apply_boundaries


def parameter_category(name, canonical_modules):
    # This launcher accepts source BF16 only. Canonical weight tensors are
    # candidates; router state can also be registered beneath an experts module
    # and remains fixed even when both module paths alias the same Parameter.
    return "candidate" if (name.rsplit(".", 1)[-1] in {"weight", "w13_weight", "w2_weight"}
                           and any(name.startswith(module + ".")
                                   for module in canonical_modules)) else "fixed"


def reference_candidate_tensor_ids(model, boundaries):
    """Read actual native owner parameters/buffers, preserving external aliases.

The reference assignment defines the canonical owners. A registered tensor
also named outside those owners (for example the shared router bias) remains
fixed. Different tensor objects sharing storage still meet the ledger's alias
conflict checks; no partial backing is silently assigned.
    """
    candidate = set()
    prefixes = []
    for row in boundaries:
        owner = row["owner"]
        prefixes.append(row["boundary"].removesuffix(".quant_method.apply") + ".")
        candidate.update(id(tensor) for _, tensor in
                         list(owner.named_parameters(remove_duplicate=False)) + list(owner.named_buffers(remove_duplicate=False)))
    external = {id(tensor) for name, tensor in
                list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
                if not any(name.startswith(prefix) for prefix in prefixes)}
    return candidate - external


def tensor_leaves(value, prefix):
    import torch
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from tensor_leaves(item, f"{prefix}.{key}")


def runtime_tensor_leaves(value, prefix, *, max_nodes=20000):
    """Read existing runtime state without properties, imports or allocations.

    Ancestor cycles stop, while distinct alias paths remain separate witnesses.
    Only explicitly selected runtime state classes are traversed. Model modules
    use their existing named-parameter/buffer census instead.
    """
    import torch
    remaining = max_nodes

    def visit(item, name, ancestors):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or len(ancestors) > 32:
            raise RuntimeError("runtime tensor owner traversal budget exhausted")
        if isinstance(item, torch.Tensor):
            yield name, item
            return
        if id(item) in ancestors or isinstance(item, torch.nn.Module):
            return
        ancestors = ancestors | {id(item)}
        if isinstance(item, (tuple, list)):
            children = [(f"{name}[{index}]", child) for index, child in enumerate(item)]
        elif isinstance(item, dict):
            # Cache keys include (name, torch.device); repr is retained only in
            # an owner label and is never evaluated or treated as storage proof.
            children = [(f"{name}[{key!r}]", child) for key, child in sorted(item.items(), key=lambda row: repr(row[0]))]
        elif (isinstance(item, SimpleNamespace)
                or (type(item).__module__, type(item).__name__) == ("vllm.v1.utils", "CpuGpuBuffer")
                or type(item).__module__.startswith(
                    ("vllm.v1.worker.", "vllm.v1.attention.", "vllm.attention.", "flashinfer."))):
            children = [(f"{name}.{key}", child) for key, child in sorted(vars(item).items())]
        else:
            return
        for child_name, child in children:
            yield from visit(child, child_name, ancestors)

    yield from visit(value, prefix, set())


def persistent_runtime_roots(runner):
    """References only: observe the stock managers' already allocated storage."""
    if runner is not None:
        for name in ("req_states", "input_buffers", "sampler", "model_state", "block_tables",
                     "structured_outputs_worker", "prompt_logprobs_worker", "kv_block_zeroer",
                     "attn_groups", "execute_model_state", "intermediate_tensors", "draft_tokens_handler"):
            if name in vars(runner):
                yield "runner:" + name, vars(runner)[name]
    workspace = sys.modules.get("vllm.v1.worker.workspace")
    manager = vars(workspace).get("_manager") if workspace is not None else None
    if manager is not None:
        yield "vllm:workspace_manager", manager
    flashinfer = sys.modules.get("flashinfer.utils")
    if flashinfer is not None:
        for name in ("_cache_buf", "_cache_buf_retired"):
            if name in vars(flashinfer):
                yield "flashinfer.utils:" + name, vars(flashinfer)[name]


def native_library_observation():
    from experiments.bench_native_operator import _mapped_shared_libraries
    libraries = {}
    errors = []
    for path in sorted(_mapped_shared_libraries()):
        try:
            with path.open("rb") as stream:
                libraries[str(path)] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                        "bytes": path.stat().st_size}
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"scope": "actual mapped shared-object bytes, including generated native libraries",
            "libraries": libraries, "errors": errors, "runtime_admission": False}


def full_engine_runtime_observation(plan):
    """Fresh post-initialization package/source and actual loaded-runtime census."""
    import tessera
    import tessera.cached_unit
    from experiments.bench_native_operator import observe_runtime
    base = observe_runtime(plan["selected_configuration"]["runtime_image"])
    package = Path(tessera.__file__).resolve().parent
    files = {str(path.relative_to(package)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                             "bytes": path.stat().st_size}
             for path in sorted(package.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts}
    installer_bytes = Path(plan["runtime_evidence"]).read_bytes()
    installer_sha256 = hashlib.sha256(installer_bytes).hexdigest()
    if installer_sha256 != plan["runtime_evidence_sha256"]:
        raise ValueError("runtime evidence changed from its planned digest")
    installer = json.loads(installer_bytes)
    if files != installer["plugin_files"]:
        raise ValueError("post-init Tessera package files changed from installer evidence")
    modules = {name: module for name, module in sorted(tuple(sys.modules.items()))
               if name == "tessera" or name.startswith("tessera.")}
    if not {"tessera", "tessera.cached_unit"} <= modules.keys():
        raise ValueError("loaded Tessera package/cached_unit modules are missing")
    module_paths, module_identities = {}, {}
    for name, module in modules.items():
        filename = getattr(module, "__file__", None)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if not isinstance(filename, str) or not filename or not isinstance(origin, str) or not origin:
            raise ValueError("loaded Tessera module has missing file/origin: " + name)
        try:
            path, origin_path = Path(filename).resolve(strict=True), Path(origin).resolve(strict=True)
        except OSError as exc:
            raise ValueError("loaded Tessera module has unverifiable file/origin: " + name) from exc
        if (path != origin_path or not path.is_relative_to(package)
                or str(path.relative_to(package)) not in files):
            raise ValueError("loaded Tessera module file/origin differs from the package roster: " + name)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != files[str(path.relative_to(package))]["sha256"]:
            raise ValueError("loaded Tessera module bytes changed during observation: " + name)
        module_paths[name] = str(path)
        module_identities[name] = {"file": str(path), "origin": str(origin_path), "sha256": actual_sha}
    loaded = {"schema": "tessera.loaded_package_identity.v1",
        "encoder_source_sha256": base["source"]["tessera_package_sha256"],
        "package_path": str(package), "installer_evidence_sha256": installer_sha256,
        "loaded_tessera_modules": module_identities, "module_identity_errors": [],
        "package_files": files, "package_files_unchanged_from_installer": True,
        "tessera_file": tessera.__file__, "cached_unit_file": tessera.cached_unit.__file__,
        "sys_path": list(sys.path),
        "loaded_module_paths": module_paths}
    return {"schema": "tessera.full_engine_runtime.v1", "base": base,
        "loaded_package": loaded,
        "actual_execution": {"mode": os.environ.get("TESSERA_SERVE_MODE", "resident"),
            "execution_mode": "eager" if plan["selected_configuration"]["engine_args"]["enforce_eager"] else "graph",
            "tensor_parallel": plan["selected_configuration"]["engine_args"]["tensor_parallel_size"],
            "expert_parallel": 1},
        "configuration_sha256": plan["identity"]["configuration_sha256"],
        "execution": {"engine_args": plan["selected_configuration"]["engine_args"],
            "environment": plan["selected_configuration"]["environment"],
            "configuration_sha256": plan["identity"]["configuration_sha256"],
            "observer_engine_args": plan["observer_engine_args"],
            "observer_environment": plan["observer_environment"], "scope": plan["scope"]},
        "source": {"full_engine_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "full_engine_resources_sha256": hashlib.sha256(Path(__file__).with_name("full_engine_resources.py").read_bytes()).hexdigest(),
            "full_engine_snapshot_codec_sha256": hashlib.sha256(Path(__file__).with_name("full_engine_snapshot_codec.py").read_bytes()).hexdigest()},
        "instrumentation": {"resource_collector": {"library_sha256": plan["collector_library_sha256"],
                                                   "loaded_path": plan["collector_library"]}
                                                  if plan.get("observation_mode", "resources") == "resources" else None,
            "blas_workspace_observer": {"library_sha256": plan["blas_workspace_observer"]["sha256"],
                                        "loaded_path": plan["blas_workspace_observer"]["path"]}
                                       if plan.get("observation_mode", "resources") == "resources" and plan.get("blas_workspace_observer") else None,
            "native_owner_rule": plan.get("native_owner_rule")
                                 if plan.get("observation_mode", "resources") == "resources" else None}}



class ResourceCaptureWorker(Worker):
    def __init__(self, *args, **kwargs):
        self._resource_recorder, self._resource_plan = claim()
        prefix = self._resource_plan.get("qualification_prefix")
        if prefix is not None:
            if (type(prefix) is not dict or type(prefix.get("native_invocations")) is not int
                    or prefix["native_invocations"] != 1 or prefix.get("complete_engine_capture") is not False
                    or self._resource_plan.get("unit_boundary") != "native_apply"
                    or not self._resource_plan.get("canonical_roster")
                    or self._resource_plan.get("observed_units") != self._resource_plan["canonical_roster"]):
                raise ValueError("prefix qualification requires exactly one invocation of the full native roster")
        self._resource_blas_observer = None
        self._resource_calls = 0
        self._resource_armed = False
        self._resource_startup_calls = 0
        self._resource_scheduler_steps = []
        self._resource_open_step = None
        self._resource_active = False
        self._resource_hooks = []
        self._resource_invocations = {}
        self._resource_scopes = {}
        self._resource_events = []
        self._resource_native_boundaries = None
        self._resource_native_patches = None
        self._resource_prefix_closed = False
        self._resource_prefix_result = None
        super().__init__(*args, **kwargs)

    def _resource_owners(self):
        if self._resource_blas_observer is not None:
            yield from self._resource_blas_observer.owners()
        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is not None:
            reference_ids = (reference_candidate_tensor_ids(model, self._resource_native_boundaries)
                             if self._resource_plan.get("reference_checkpoint") and self._resource_native_boundaries is not None else None)
            for kind, tensors in (("parameter", model.named_parameters(remove_duplicate=False)),
                                  ("buffer", model.named_buffers(remove_duplicate=False))):
                for name, tensor in tensors:
                    if tensor.device.type == "cuda":
                        category = ("candidate" if id(tensor) in reference_ids else "fixed") if reference_ids is not None else parameter_category(
                            name, self._resource_plan["canonical_modules"])
                        yield TensorOwner(f"model:{kind}:{name}", category, tensor,
                            "canonical native owner tensor with external aliases fixed" if reference_ids is not None
                            else "source BF16 canonical weight tensors; other named state fixed")
        if runner is not None:
            for name, tensor in tensor_leaves(getattr(runner, "kv_caches", []), "kv_cache"):
                yield TensorOwner(name, "kv", tensor,
                                 "stock GPUModelRunner.kv_caches; shared storage deduplicated")
        for root, value in persistent_runtime_roots(runner):
            for name, tensor in runtime_tensor_leaves(value, root):
                yield TensorOwner(name, "shared", tensor,
                    "existing runtime state/workspace reference; candidate and workload dependence unresolved")

    def _resource_checkpoint(self, name):
        """Take one checkpoint; report whether it was actually taken.

        The bounded first-native prefix closes the observer mid-step, and every
        later checkpoint is a no-op. A step interval naming a label that was
        never snapshotted would be a boundary over unobserved execution, so the
        caller needs the answer, not just the attempt.
        """
        if self._resource_prefix_closed:
            return False
        self._resource_recorder.snapshot(name, owners=self._resource_owners())
        return True

    def init_device(self):
        result = super().init_device()
        if "blas_workspace_observer" in self._resource_plan:
            spec = self._resource_plan["blas_workspace_observer"]
            self._resource_blas_observer = BlasWorkspaceObserver(spec["path"], spec["sha256"])
        self._resource_checkpoint("device_initialized")
        return result

    def load_model(self, *args, **kwargs):
        self._resource_checkpoint("before_model_load")
        result = super().load_model(*args, **kwargs)
        if self._resource_plan.get("unit_boundary") == "native_apply":
            self._resource_native_boundaries = resolve_apply_boundaries(self.model_runner.model, self._resource_plan["observed_units"])
            self._resource_invocations = {row["module"]: 0 for row in self._resource_native_boundaries}
        self._resource_checkpoint("model_loaded")
        if self._resource_native_boundaries is not None:
            return result
        modules = dict(self.model_runner.model.named_modules())
        for unit in self._resource_plan["observed_units"]:
            name = unit["module"]
            if name not in modules:
                raise RuntimeError(f"canonical observed module absent from stock model: {name}")
            self._resource_invocations[name] = 0
            def before(module, inputs, kwargs, *, name=name, unit_id=unit["unit_id"]):
                if not self._resource_active:
                    return
                invocation = self._resource_invocations[name]
                if invocation >= self._resource_plan["max_invocations_per_unit"]:
                    return
                scope = self._resource_recorder.unit_scope(unit_id)
                scope.__enter__()
                self._resource_scopes[name] = scope
                self._resource_events.append({"unit_id": unit_id, "module": name,
                    "invocation": invocation, "execute_call": self._resource_calls,
                    "input_tensors": [{"name": key, "shape": list(tensor.shape),
                                       "dtype": str(tensor.dtype)}
                                      for key, tensor in tensor_leaves(
                                          {"args": inputs, "kwargs": kwargs}, "input")]})
                self._resource_invocations[name] += 1
            def after(module, inputs, output, *, name=name):
                scope = self._resource_scopes.pop(name, None)
                if scope is not None:
                    scope.__exit__(None, None, None)
            self._resource_hooks.extend([
                modules[name].register_forward_pre_hook(before, with_kwargs=True),
                modules[name].register_forward_hook(after, always_call=True)])
        return result

    @contextmanager
    def _resource_observe_apply(self, boundary, call):
        if not self._resource_active:
            yield
            return
        name, unit_id = boundary["module"], boundary["unit_id"]
        invocation = self._resource_invocations[name]
        if invocation >= self._resource_plan["max_invocations_per_unit"]:
            raise RuntimeError("canonical native unit exceeded the declared invocation budget: " + unit_id)
        self._resource_invocations[name] += 1

        def owners():
            yield from self._resource_owners()
            for kind, value in (("input", call["arguments"]), ("output", call["result"])):
                for label, tensor in tensor_leaves(value, kind):
                    yield TensorOwner(f"native:{unit_id}:{invocation}:{label}", "shared", tensor,
                                      "observed native boundary tensor; carried lifetime retained, not a fixed activation price")

        self._resource_events.append({"unit_id": unit_id, "module": name, "boundary": boundary["boundary"],
            "includes_router": boundary["includes_router"], "invocation": invocation,
            "execute_call": self._resource_calls,
            "input_tensors": [{"name": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                              for key, tensor in tensor_leaves(call["arguments"], "input")]})
        profiler = cProfile.Profile() if self._resource_plan.get("qualification_prefix") else None
        if profiler is not None:
            profiler.enable()
        try:
            with self._resource_recorder.unit_scope(unit_id, owners=owners):
                yield
            if self._resource_plan.get("qualification_prefix"):
                self._resource_active = False
                self._resource_prefix_closed = True
                self._resource_recorder._errors.append(
                    "bounded first-native prefix qualification; the remaining engine execution is unobserved")
                self._resource_prefix_result = self._resource_write_capture(prefix_only=True)
        finally:
            if profiler is not None:
                profiler.disable()
                path = Path(self._resource_plan["output_directory"]) / f"prefix-observer-worker-{os.getpid()}.pstats"
                profiler.dump_stats(str(path))

    def initialize_from_config(self, kv_cache_config):
        self._resource_checkpoint("before_kv_allocation")
        result = super().initialize_from_config(kv_cache_config)
        self._resource_checkpoint("kv_allocated")
        expected = self._resource_plan.get("selected_configuration", {}).get("capacity_assertions")
        if hasattr(self.model_runner, "kv_cache_config"):
            self._resource_kv_description = inspect_worker_kv(self, expected, received=kv_cache_config)
        elif expected is not None:
            raise RuntimeError("selected KV assertions require the actual runner-resolved configuration")
        else:
            self._resource_kv_description = kv_configuration_observation(self, kv_cache_config)
        if expected is not None:
            path = Path(self._resource_plan["output_directory"]) / f"kv-worker-{os.getpid()}.json"
            with path.open("x") as stream:
                json.dump(self._resource_kv_description, stream, sort_keys=True)
            if not self._resource_kv_description["capacity_assertions"]["passed"]:
                self._resource_recorder._errors.append("actual resolved KV capacity differs from selected assertions")
                raise RuntimeError(f"actual KV capacity failed selected assertions; observations: {path}")
        return result

    def execute_model(self, scheduler_output):
        if not self._resource_armed:
            self._resource_startup_calls += 1
            return super().execute_model(scheduler_output)
        self._resource_calls += 1
        self._resource_active = (not self._resource_prefix_closed
                                 and self._resource_calls <= self._resource_plan["max_execute_calls"])
        if self._resource_active:
            self._resource_scheduler_steps.append({
                "execute_call": self._resource_calls,
                "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                "num_scheduled_tokens": scheduler_output.num_scheduled_tokens,
                "new_requests": [{"req_id": request.req_id,
                                  "num_computed_tokens": request.num_computed_tokens,
                                  "prompt_token_count": len(request.prompt_token_ids)}
                                 for request in scheduler_output.scheduled_new_reqs],
                "cached_requests": {"req_ids": scheduler_output.scheduled_cached_reqs.req_ids,
                    "num_computed_tokens": scheduler_output.scheduled_cached_reqs.num_computed_tokens}})
            begin = f"execute:{self._resource_calls}:begin"
            if self._resource_checkpoint(begin):
                self._resource_open_step = (self._resource_calls, begin)
        try:
            return super().execute_model(scheduler_output)
        finally:
            if self._resource_active:
                self._resource_checkpoint(f"execute:{self._resource_calls}:end")
            self._resource_active = False

    def sample_tokens(self, grammar_output):
        result = super().sample_tokens(grammar_output)
        if self._resource_armed and self._resource_calls <= self._resource_plan["max_execute_calls"]:
            end = f"sample:{self._resource_calls}:end"
            taken = self._resource_checkpoint(end)
            open_step = self._resource_open_step
            if taken and open_step is not None:
                self._resource_open_step = None
                if open_step[0] != self._resource_calls:
                    # The sampler that closed is not the execute call that
                    # opened. Rather than pair two halves of different steps,
                    # drop the declaration and say so: the unmatched step leaves
                    # declared below executed, which holds the whole coverage
                    # claim open instead of publishing a wrong interval.
                    self._resource_recorder._errors.append(
                        f"engine step {open_step[0]} was never closed by its own sampler")
                else:
                    # The step spans the sampler: sampling is per-step work, and
                    # a step ending at execute:N:end would place the sampler's
                    # allocations outside every step. There is no fallback to
                    # that label -- a step whose sampler never ran stays
                    # undeclared and fails the coverage claim closed.
                    self._resource_recorder.declare_step_interval(
                        f"step:{open_step[0]}", begin=open_step[1], end=end)
        return result

    def resource_capture_arm(self):
        if self._resource_armed:
            raise RuntimeError("resource workload was already armed")
        self._resource_checkpoint("ready_for_workload")
        if self._resource_native_boundaries is not None:
            self._resource_native_patches = observe_apply_boundaries(self._resource_native_boundaries, self._resource_observe_apply)
            self._resource_native_patches.__enter__()
        self._resource_armed = True
        return {"pid": os.getpid(), "startup_execute_calls": self._resource_startup_calls,
                "scope": "subsequent execution belongs to the explicit observation workload"}

    def resource_capture_finish(self):
        if self._resource_plan.get("qualification_prefix"):
            if self._resource_prefix_result is None:
                raise RuntimeError("first-native prefix qualification did not produce its bounded capture")
            return self._resource_prefix_result
        return self._resource_write_capture()

    def _resource_write_capture(self, *, prefix_only=False):
        if not self._resource_armed:
            raise RuntimeError("cannot finish before the observation workload was armed")
        for handle in self._resource_hooks:
            handle.remove()
        self._resource_hooks.clear()
        if self._resource_native_patches is not None:
            self._resource_native_patches.__exit__(None, None, None)
            self._resource_native_patches = None
        for name, count in self._resource_invocations.items():
            if count != self._resource_plan["max_invocations_per_unit"]:
                self._resource_recorder._errors.append(
                    f"observed unit {name} had {count} invocations, expected "
                    f"{self._resource_plan['max_invocations_per_unit']}")
        directory = Path(self._resource_plan["output_directory"]) / f"worker-{os.getpid()}"
        runtime = full_engine_runtime_observation(self._resource_plan)
        native_libraries = native_library_observation()
        native_evidence = []
        if "native_owner_rule" in self._resource_plan:
            from experiments.full_engine_native_owners import capture_rule_evidence, validate_rule_evidence
            spec = self._resource_plan["native_owner_rule"]
            content = Path(spec["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != spec["sha256"]:
                raise ValueError("native ownership rule changed after plan preparation")
            evidence = capture_rule_evidence(json.loads(content), native_libraries["libraries"])
            try:
                validate_rule_evidence(evidence)
            except ValueError as exc:
                # Preserve the failed observation. The ledger validates this
                # same evidence again before assigning any native ownership.
                self._resource_recorder._errors.append(f"native ownership evidence invalid: {exc}")
            native_evidence.append(evidence)
        result = self._resource_recorder.finish(directory, owners=self._resource_owners(),
                                                native_ownership_evidence=native_evidence,
                                                executed_steps=self._resource_calls,
                                                measured_runtime_sha256=hashlib.sha256(json.dumps(runtime,
                                                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        (directory / "post-native-package.json").write_text(json.dumps(runtime["loaded_package"], sort_keys=True, indent=2) + "\n")
        (directory / "runtime-observation.json").write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\n")
        (directory / "worker-observations.json").write_text(json.dumps({
            "worker_class": f"{type(self).__module__}.{type(self).__name__}",
            "model_runner_class": f"{type(self.model_runner).__module__}.{type(self.model_runner).__name__}",
            "native_libraries": native_libraries,
            "execute_calls": self._resource_calls, "units": self._resource_events,
            "startup_execute_calls": self._resource_startup_calls,
            "scheduler_steps": self._resource_scheduler_steps,
            "kv_configuration": getattr(self, "_resource_kv_description", None),
            "qualification_prefix": self._resource_plan.get("qualification_prefix") if prefix_only else None,
            "scope": ("bounded startup/first-native prefix only; subsequent request execution unobserved"
                      if prefix_only else "intrusive raw engine resource pass; timing and admission ineligible")
        }, sort_keys=True))
        return {"directory": str(directory), "receipt": result}
