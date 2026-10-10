"""tessera#1185 TP2 prefill profiler hook (OFF REPO, sparky-local only).

Loaded as ``usercustomize`` (this directory on PYTHONPATH at container start).
Only the profiler below is active, and only when TP2TRACE_PROF_DIR and
TP2TRACE_TRIGGER are both set. It wraps both model runner classes'
execute_model (the default V2 runner and the legacy V1 runner): after the
trigger file appears it profiles the next TP2TRACE_STEPS steps with work
(default 4: four 2048-token prefill chunks; max_tokens=1 serves no decode
steps) with a fresh torch.profiler per step, record_shapes on. Each step
-- its key_averages table grouped by input shape, sorted by self CUDA
time -- and the run ends with steps.json (per-step tokens and host time).
It then removes the trigger.

Per-step profilers (not one profiler over all steps) keep every chunk's
self-device-time separate, so the categorizer reads numbers, never trace
timestamps. Differs from the #508 hook (experiments/glm53_508_graph_qual):
record_shapes is on and each profiled step owns its table.
"""

import os

_PROF_DIR = os.environ.get("TP2TRACE_PROF_DIR")
_PROF_TRIGGER = os.environ.get("TP2TRACE_TRIGGER")

if _PROF_DIR and _PROF_TRIGGER:
    import json
    import sys
    import time

    _SKIP = int(os.environ.get("TP2TRACE_SKIP", "0"))
    _STEPS = int(os.environ.get("TP2TRACE_STEPS", "4"))
    _STATE = {"state": "idle", "skipped": 0, "chunk": 0, "prof": None,
              "walls": [], "run": None}

    def _log(msg):
        print(f"[tp2trace] {msg}", file=sys.stderr, flush=True)

    def _write_chunk_table(prof, chunk, work):
        ka = prof.key_averages(group_by_input_shape=True)
        path = os.path.join(_PROF_DIR, f"chunk{chunk}.txt")
        with open(path, "w") as fh:
            fh.write(f"# chunk={chunk} tokens={work}\n")
            fh.write(ka.table(sort_by="self_cuda_time_total", row_limit=400))

    def _before(sched_out):
        if _STATE["state"] == "idle" and os.path.exists(_PROF_TRIGGER):
            _STATE.update(state="active", skipped=0, chunk=0, walls=[])
            _log(f"trigger seen; skipping {_SKIP}, profiling {_STEPS}")
        work = int(getattr(sched_out, "total_num_scheduled_tokens", 0) or 0)
        if _STATE["state"] != "active" or work == 0:
            return work if _STATE["state"] == "active" else None
        if _STATE["skipped"] < _SKIP:
            _STATE["skipped"] += 1
            return None
        if _STATE["chunk"] >= _STEPS:
            return work
        import torch

        run = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=False)
        run.__enter__()
        _STATE["run"] = run
        return work

    def _after(work, wall):
        if work is None or _STATE["run"] is None:
            return
        import torch

        run = _STATE["run"]
        _STATE["run"] = None
        chunk = _STATE["chunk"]
        torch.cuda.synchronize()
        run.__exit__(None, None, None)
        os.makedirs(_PROF_DIR, exist_ok=True)
        _write_chunk_table(run, chunk, work)
        _STATE["walls"].append(dict(chunk=chunk, tokens=work,
                                    execute_model_host_s=wall))
        _STATE["chunk"] += 1
        if _STATE["chunk"] >= _STEPS:
            with open(os.path.join(_PROF_DIR, "steps.json"), "w") as fh:
                json.dump(dict(steps=_STATE["walls"], skip=_SKIP), fh,
                          indent=1)
            _STATE["state"] = "idle"
            try:
                os.remove(_PROF_TRIGGER)
            except OSError:
                pass
            _log(f"wrote {_PROF_DIR} ({_STEPS} chunks)")

    def _wrap_class(cls, label):
        orig_exec = cls.execute_model

        def execute_model(self, scheduler_output, *args, **kwargs):
            if kwargs.get("dummy_run", False):
                return orig_exec(self, scheduler_output, *args, **kwargs)
            work = _before(scheduler_output)
            t0 = time.perf_counter()
            try:
                return orig_exec(self, scheduler_output, *args, **kwargs)
            finally:
                if work is not None and _STATE["state"] == "active":
                    _after(work, time.perf_counter() - t0)

        cls.execute_model = execute_model
        _log(f"runner wrap installed on {label}")

    def _install():
        # vLLM serves the V2 runner (vllm.v1.worker.gpu.model_runner) by
        # default and keeps the V1 runner
        # (vllm.v1.worker.gpu_model_runner) for opt-outs. A V1-only wrap
        # never fires under V2, so both classes are patched.
        patched = []
        for module, label in (
                ("vllm.v1.worker.gpu.model_runner", "V2"),
                ("vllm.v1.worker.gpu_model_runner", "V1")):
            try:
                mod = __import__(module, fromlist=["GPUModelRunner"])
            except ImportError as exc:
                _log(f"no {label} model runner ({exc!r})")
                continue
            _wrap_class(mod.GPUModelRunner, label)
            patched.append(label)
        if not patched:
            _log("no model runner class found; profiler stays idle")
            return
        _log(f"dir={_PROF_DIR} skip={_SKIP} steps={_STEPS}")

    _install()
    _log("loaded")
