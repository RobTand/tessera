"""Supported stock-vLLM worker subclass for intrusive raw resource capture.

No serving algorithm is replaced. Synchronized snapshots alter execution and
must never be used as timings or as fixed/KV admission. The owning launcher
audits the installed core before and after this worker runs.
"""
import json
import hashlib
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from experiments.full_engine_bootstrap import claim
from experiments.full_engine_resources import TensorOwner


def parameter_category(name, canonical_modules):
    return "candidate" if any(name.startswith(module + ".")
                              for module in canonical_modules) else "fixed"


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


class ResourceCaptureWorker(Worker):
    def __init__(self, *args, **kwargs):
        self._resource_recorder, self._resource_plan = claim()
        self._resource_calls = 0
        self._resource_armed = False
        self._resource_startup_calls = 0
        self._resource_scheduler_steps = []
        self._resource_active = False
        self._resource_hooks = []
        self._resource_invocations = {}
        self._resource_scopes = {}
        self._resource_events = []
        super().__init__(*args, **kwargs)

    def _resource_owners(self):
        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is not None:
            for kind, tensors in (("parameter", model.named_parameters(remove_duplicate=False)),
                                  ("buffer", model.named_buffers(remove_duplicate=False))):
                for name, tensor in tensors:
                    if tensor.device.type == "cuda":
                        yield TensorOwner(f"model:{kind}:{name}", parameter_category(
                            name, self._resource_plan["canonical_modules"]), tensor,
                            "stock model named tensors; canonical-module prefix classification")
        if runner is not None:
            for name, tensor in tensor_leaves(getattr(runner, "kv_caches", []), "kv_cache"):
                yield TensorOwner(name, "kv", tensor,
                                 "stock GPUModelRunner.kv_caches; shared storage deduplicated")

    def _resource_checkpoint(self, name):
        self._resource_recorder.snapshot(name, owners=self._resource_owners())

    def init_device(self):
        result = super().init_device()
        self._resource_checkpoint("device_initialized")
        return result

    def load_model(self, *args, **kwargs):
        self._resource_checkpoint("before_model_load")
        result = super().load_model(*args, **kwargs)
        self._resource_checkpoint("model_loaded")
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

    def initialize_from_config(self, kv_cache_config):
        self._resource_checkpoint("before_kv_allocation")
        result = super().initialize_from_config(kv_cache_config)
        self._resource_checkpoint("kv_allocated")
        self._resource_kv_description = {
            "num_blocks": kv_cache_config.num_blocks,
            "tensors": [{"size": tensor.size, "layers": tensor.layers,
                         "layer_stride": tensor.layer_stride,
                         "block_stride": tensor.block_stride, "offset": tensor.offset}
                        for tensor in kv_cache_config.kv_cache_tensors],
            "scope": "stock KV configuration descriptors; do not sum as physical storage"}
        return result

    def execute_model(self, scheduler_output):
        if not self._resource_armed:
            self._resource_startup_calls += 1
            return super().execute_model(scheduler_output)
        self._resource_calls += 1
        self._resource_active = self._resource_calls <= self._resource_plan["max_execute_calls"]
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
            self._resource_checkpoint(f"execute:{self._resource_calls}:begin")
        try:
            return super().execute_model(scheduler_output)
        finally:
            if self._resource_active:
                self._resource_checkpoint(f"execute:{self._resource_calls}:end")
            self._resource_active = False

    def sample_tokens(self, grammar_output):
        result = super().sample_tokens(grammar_output)
        if self._resource_armed and self._resource_calls <= self._resource_plan["max_execute_calls"]:
            self._resource_checkpoint(f"sample:{self._resource_calls}:end")
        return result

    def resource_capture_arm(self):
        if self._resource_armed:
            raise RuntimeError("resource workload was already armed")
        self._resource_checkpoint("ready_for_workload")
        self._resource_armed = True
        return {"pid": os.getpid(), "startup_execute_calls": self._resource_startup_calls,
                "scope": "subsequent execution belongs to the explicit observation workload"}

    def resource_capture_finish(self):
        if not self._resource_armed:
            raise RuntimeError("cannot finish before the observation workload was armed")
        for handle in self._resource_hooks:
            handle.remove()
        self._resource_hooks.clear()
        for name, count in self._resource_invocations.items():
            if count != self._resource_plan["max_invocations_per_unit"]:
                self._resource_recorder._errors.append(
                    f"observed unit {name} had {count} invocations, expected "
                    f"{self._resource_plan['max_invocations_per_unit']}")
        directory = Path(self._resource_plan["output_directory"]) / f"worker-{os.getpid()}"
        native_libraries = native_library_observation()
        result = self._resource_recorder.finish(directory, owners=self._resource_owners())
        (directory / "worker-observations.json").write_text(json.dumps({
            "worker_class": f"{type(self).__module__}.{type(self).__name__}",
            "model_runner_class": f"{type(self.model_runner).__module__}.{type(self.model_runner).__name__}",
            "native_libraries": native_libraries,
            "execute_calls": self._resource_calls, "units": self._resource_events,
            "startup_execute_calls": self._resource_startup_calls,
            "scheduler_steps": self._resource_scheduler_steps,
            "kv_configuration": getattr(self, "_resource_kv_description", None),
            "scope": "intrusive raw source-BF16 resource pass; timing and admission ineligible"
        }, sort_keys=True))
        return {"directory": str(directory), "receipt": result}
