"""tessera#508 research hooks: per-module digests, MoE finalize order, profiler.

Loaded by Python at start-up when this directory is on PYTHONPATH (``usercustomize``
is imported after the image's own ``sitecustomize``, which it therefore leaves
alone). Research only: never on a serving path, never in src/. Every feature is
inactive unless its variable is set:

``T508_DIGEST=<jsonl>``
    Per-module digests of every forward, one JSON line per forward.
    Every module under the model root (the outermost module whose class name
    ends in ``ForCausalLM`` or ``ForConditionalGeneration``), down to
    ``T508_DIGEST_DEPTH`` name components, gets a forward hook on its first
    tensor input and first tensor output. The module names recorded are
    written once to ``<jsonl>.names.json``.

    Buffer mode (default): the hook copies the first ``T508_DIGEST_ROWS`` token
    rows into persistent device buffers. The copies are ordinary stream work,
    so a PIECEWISE or FULL capture records them and every replay repeats them.
    The readout (one device sync, one line) runs at the end of the root
    forward through ``eager_break_during_capture`` in eager and PIECEWISE
    runs; a FULL replay never calls the root forward, so the model runner's
    ``execute_model`` is wrapped to read out after a step that replayed a FULL
    graph, with the step's real token count from the scheduler output. A line
    written during a capture is skipped. Buffers are allocated on first eager
    sight; a module first seen inside a capture is recorded as missing.

    Whole-tensor mode (``T508_DIGEST_FULL=1``, eager only): the hook computes
    exact integer checksums of the whole input and output (and of their first
    and last rows) on the device, so a divergence in any row is visible.

``T508_MOE_DETERMINISTIC=1``
    FlashInfer's ``cutlass_fused_moe`` fuses the top-k expert reduction into
    the second GEMM's epilogue with atomics by default; its own docstring says
    the result is then not deterministic run to run. This replaces vLLM's
    per-call binding of the function so every call passes
    ``use_fused_finalize=False`` (the non-fused, deterministic finalize) and
    logs the first call per weight dtype.

``T508_PROF_DIR=<dir>`` with ``T508_PROF_TRIGGER=<file>``
    When the trigger file exists, the runner skips ``T508_PROF_SKIP`` steps with
    work (default 2), then runs ``torch.profiler`` (CPU and CUDA, no stacks)
    over ``T508_PROF_STEPS`` steps (default 8), writes a Chrome trace and two
    ``key_averages`` tables into the directory, records the host-side time of
    each profiled ``execute_model`` call (no added sync), and removes the
    trigger. vLLM's /start_profile route is broken in the
    pinned image (``AsyncLLM`` has no ``profiler`` attribute).

``T508_CAPTURE_LOG=<jsonl>``
    Prices CUDA-graph capture. Wraps the V2 runner's ``CudaGraphManager.capture``
    and writes one JSON line per call: the phase (``profile`` for vLLM's
    memory-profiling capture into a throwaway pool, ``real`` for the serving
    capture), and for every batch descriptor captured its token count, mode, the
    wall time of its warmup plus capture, and the change in the caching
    allocator's reserved and allocated bytes over it, measured between device
    syncs. Reserved bytes include the graph's private pool, so the sum over a
    phase is the capture's memory cost. vLLM's own "took X GiB" line is the
    drop in device free memory, which on GB10's unified memory moves with the
    page cache and every other process on the box.
"""
import os

_DIGEST_PATH = os.environ.get("T508_DIGEST")
_MOE_DET = os.environ.get("T508_MOE_DETERMINISTIC") == "1"
_PROF_DIR = os.environ.get("T508_PROF_DIR")
_PROF_TRIGGER = os.environ.get("T508_PROF_TRIGGER")
_CAPTURE_LOG = os.environ.get("T508_CAPTURE_LOG")

if _DIGEST_PATH or _MOE_DET or (_PROF_DIR and _PROF_TRIGGER) or _CAPTURE_LOG:
    import hashlib
    import json
    import sys
    import time

    import torch

    _ROOT_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
    _ROWS = int(os.environ.get("T508_DIGEST_ROWS", "8"))
    _DEPTH = int(os.environ.get("T508_DIGEST_DEPTH", "6"))
    _FULL = os.environ.get("T508_DIGEST_FULL") == "1"
    _S = {"root": None, "names": {}, "bufs": {}, "order": [], "step": 0, "missing": set(),
          "pending": [], "installed": False, "full_replayed": None, "seen": 0}

    def _log(msg):
        print(f"t508-hooks[{os.getpid()}]: {msg}", file=sys.stderr, flush=True)

    _log(f"loaded: digest={_DIGEST_PATH} full={os.environ.get('T508_DIGEST_FULL') == '1'} "
         f"moe_deterministic={_MOE_DET} prof={_PROF_DIR} capture_log={_CAPTURE_LOG}")

    # ------------------------------------------------------------------ digest
    def _first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = _first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, dict):
            for item in value.values():
                found = _first_tensor(item)
                if found is not None:
                    return found
        return None

    def _rows_view(t):
        """(tokens, features) view: token dim 0, or 1 for a leading batch of 1."""
        if t.dim() == 0:
            return None
        if t.dim() >= 3 and t.shape[0] == 1:
            t = t[0]
        return t.reshape(t.shape[0], -1)

    _INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

    def _cksum(t):
        """Exact integer checksums (sum, position-weighted sum, count) of t's bits."""
        flat = t.detach().contiguous().reshape(-1)
        view = _INT_VIEW.get(flat.element_size())
        if view is None or flat.numel() == 0:
            return None
        if flat.dtype == torch.bool:
            flat = flat.to(torch.uint8)
        xi = flat.view(view).to(torch.int64)
        w = torch.arange(1, xi.numel() + 1, device=xi.device, dtype=torch.int64)
        return torch.stack([xi.sum(), (xi * w).sum(),
                            torch.full((), xi.numel(), device=xi.device, dtype=torch.int64)])

    def _record(key, t):
        if t is None or not t.is_cuda:
            return
        rows = _rows_view(t)
        if rows is None or rows.shape[1] == 0:
            return
        capturing = torch.cuda.is_current_stream_capturing()
        if _FULL:
            if capturing:
                _S["missing"].add(key)
                return
            parts = [_cksum(rows), _cksum(rows[:1]), _cksum(rows[-1:])]
            if any(p is None for p in parts):
                return
            if key not in _S["bufs"]:
                _S["bufs"][key] = True
                _S["order"].append(key)
            _S["pending"].append((key, torch.cat(parts)))
            return
        buf = _S["bufs"].get(key)
        if buf is None:
            if capturing:
                _S["missing"].add(key)
                return
            buf = torch.zeros((_ROWS, rows.shape[1]), dtype=rows.dtype, device=rows.device)
            _S["bufs"][key] = buf
            _S["order"].append(key)
        if buf.shape[1] != rows.shape[1] or buf.dtype != rows.dtype:
            return
        n = min(rows.shape[0], _ROWS)
        buf[:n].copy_(rows[:n])

    def _real_tokens():
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None, None, None
        ctx = get_forward_context()
        mode = getattr(ctx, "cudagraph_runtime_mode", None)
        desc = getattr(ctx, "batch_descriptor", None)
        padded = getattr(desc, "num_tokens", None)
        meta = ctx.attn_metadata
        values = meta.values() if isinstance(meta, dict) else ([meta] if meta is not None else [])
        real = None
        for m in values:
            if isinstance(m, list):
                m = m[0] if m else None
            if m is None:
                continue
            if getattr(m, "num_actual_tokens", None) is not None:
                real = int(m.num_actual_tokens)
                break
            if getattr(m, "num_decode_tokens", None) is not None:
                real = int(m.num_decode_tokens) + int(getattr(m, "num_prefill_tokens", 0) or 0)
                break
        return real, padded, getattr(mode, "name", str(mode))

    def _write(real, padded, mode):
        _S["step"] += 1
        digests = {}
        if _FULL:
            pending, _S["pending"] = _S["pending"], []
            if pending:
                values = torch.stack([v for _, v in pending]).cpu().tolist()
                seen = {}
                for (key, _), vals in zip(pending, values):
                    seen[key] = seen.get(key, 0) + 1
                    name = key if seen[key] == 1 else f"{key}#{seen[key]}"
                    digests[name] = "%x.%x.%x|%x.%x|%x.%x" % (
                        vals[0] & 0xFFFFFFFFFFFFFFFF, vals[1] & 0xFFFFFFFFFFFFFFFF, vals[2],
                        vals[3] & 0xFFFFFFFFFFFFFFFF, vals[4] & 0xFFFFFFFFFFFFFFFF,
                        vals[6] & 0xFFFFFFFFFFFFFFFF, vals[7] & 0xFFFFFFFFFFFFFFFF)
            n = real
        else:
            n = min(real if real is not None else _ROWS, _ROWS)
            for key in _S["order"]:
                data = _S["bufs"][key][:n].contiguous().view(torch.uint8).cpu().numpy().tobytes()
                digests[key] = hashlib.sha256(data).hexdigest()[:16]
        with open(_DIGEST_PATH, "a") as fh:
            fh.write(json.dumps(dict(step=_S["step"], real=real, padded=padded, mode=mode, rows=n,
                                     full=_FULL, missing=sorted(_S["missing"]),
                                     digests=digests)) + "\n")

    def _readout():
        if torch.cuda.is_current_stream_capturing():
            _S["pending"] = []
            return
        real, padded, mode = _real_tokens()
        # Device-wide: a replayed segment and this readout need not share a stream.
        torch.cuda.synchronize()
        _write(real, padded, mode)

    def _readout_full(real, padded):
        """Readout after a FULL graph replay (no root forward ran)."""
        torch.cuda.synchronize()
        _write(real, padded, "FULL")

    _READOUT = None

    def _readout_break():
        global _READOUT
        if _READOUT is None:
            from vllm.compilation.breakable_cudagraph import eager_break_during_capture

            _READOUT = eager_break_during_capture(_readout)
        _READOUT()

    # ------------------------------------------------------- MoE finalize order
    def _install_moe_deterministic():
        """Route every vLLM FlashInfer-CUTLASS MoE call through the deterministic finalize.

        vLLM's experts module binds ``flashinfer_cutlass_fused_moe`` (a lazy
        wrapper that forwards keyword arguments to FlashInfer) into its own
        globals and looks it up per call, so replacing that global reaches every
        later call however the lazy wrapper has cached FlashInfer's function.
        """
        try:
            from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe as fcm
        except Exception as exc:  # recorded, never silent
            _log(f"MOE_DETERMINISTIC: vLLM FlashInfer-CUTLASS experts module not importable: {exc!r}")
            return
        orig = fcm.flashinfer_cutlass_fused_moe
        if getattr(orig, "_t508_wrapped", False):
            return
        seen = {}

        def flashinfer_cutlass_fused_moe(*args, **kwargs):
            kwargs["use_fused_finalize"] = False
            w = kwargs.get("fc1_expert_weights")
            tag = str(getattr(w, "dtype", None))
            seen[tag] = seen.get(tag, 0) + 1
            if seen[tag] in (1, 100, 10000):
                _log(f"MOE_DETERMINISTIC: fc1 dtype={tag} call {seen[tag]} use_fused_finalize=False")
            return orig(*args, **kwargs)

        flashinfer_cutlass_fused_moe._t508_wrapped = True
        fcm.flashinfer_cutlass_fused_moe = flashinfer_cutlass_fused_moe
        _log("MOE_DETERMINISTIC: vLLM flashinfer_cutlass_fused_moe wrapped (use_fused_finalize=False)")

    # ----------------------------------------------------------- runner wrap
    _PROF = {"state": "idle", "skip": int(os.environ.get("T508_PROF_SKIP", "2")),
             "steps": int(os.environ.get("T508_PROF_STEPS", "8")), "count": 0, "prof": None,
             "walls": []}

    def _prof_before(scheduler_output):
        if not (_PROF_DIR and _PROF_TRIGGER):
            return
        work = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        if _PROF["state"] == "idle" and os.path.exists(_PROF_TRIGGER):
            _PROF.update(state="skipping", count=0, walls=[])
            _log(f"PROF: trigger seen; skipping {_PROF['skip']} steps with work")
        if work == 0:
            return
        if _PROF["state"] == "skipping" and _PROF["count"] >= _PROF["skip"]:
            _PROF["prof"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False)
            _PROF["prof"].__enter__()
            _PROF.update(state="profiling", count=0)
            _log(f"PROF: profiling {_PROF['steps']} steps")

    def _prof_after(scheduler_output, wall):
        if not (_PROF_DIR and _PROF_TRIGGER) or _PROF["state"] == "idle":
            return
        work = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        if work == 0:
            return
        _PROF["count"] += 1
        if _PROF["state"] == "profiling":
            _PROF["walls"].append(dict(tokens=work, execute_model_host_s=wall))
            if _PROF["count"] >= _PROF["steps"]:
                torch.cuda.synchronize()
                prof = _PROF["prof"]
                prof.__exit__(None, None, None)
                os.makedirs(_PROF_DIR, exist_ok=True)
                prof.export_chrome_trace(os.path.join(_PROF_DIR, "trace.json"))
                ka = prof.key_averages()
                with open(os.path.join(_PROF_DIR, "by_cuda.txt"), "w") as fh:
                    fh.write(ka.table(sort_by="self_cuda_time_total", row_limit=60))
                with open(os.path.join(_PROF_DIR, "by_cpu.txt"), "w") as fh:
                    fh.write(ka.table(sort_by="cpu_time_total", row_limit=60))
                with open(os.path.join(_PROF_DIR, "steps.json"), "w") as fh:
                    json.dump(dict(steps=_PROF["walls"], skip=_PROF["skip"]), fh, indent=1)
                _PROF.update(state="idle", prof=None)
                try:
                    os.remove(_PROF_TRIGGER)
                except OSError:
                    pass
                _log(f"PROF: wrote {_PROF_DIR}")

    def _install_runner_wrap():
        try:
            from vllm.v1.worker.gpu import cudagraph_utils as cu
            from vllm.v1.worker.gpu import model_runner as mr
        except Exception as exc:
            _log(f"runner wrap unavailable: {exc!r}")
            return
        for cls in (cu.CudaGraphManager, getattr(cu, "ModelCudaGraphManager", None)):
            if cls is None or "run_fullgraph" not in cls.__dict__:
                continue
            orig_full = cls.__dict__["run_fullgraph"]

            def run_fullgraph(self, desc, _orig=orig_full):
                out = _orig(self, desc)
                _S["full_replayed"] = getattr(desc, "num_tokens", -1)
                return out

            cls.run_fullgraph = run_fullgraph
        orig_exec = mr.GPUModelRunner.execute_model

        def execute_model(self, scheduler_output, *args, **kwargs):
            dummy = bool(kwargs.get("dummy_run", False))
            _S["full_replayed"] = None
            if not dummy:
                _prof_before(scheduler_output)
            t0 = time.perf_counter()
            out = orig_exec(self, scheduler_output, *args, **kwargs)
            if not dummy:
                if _DIGEST_PATH and not _FULL and _S["full_replayed"] is not None:
                    _readout_full(int(scheduler_output.total_num_scheduled_tokens),
                                  _S["full_replayed"])
                if _PROF_DIR and _PROF_TRIGGER and _PROF["state"] != "idle":
                    # Host-side time of the call (no sync: the step stays asynchronous).
                    _prof_after(scheduler_output, time.perf_counter() - t0)
            return out

        mr.GPUModelRunner.execute_model = execute_model
        _log("runner wrap installed (FULL readout / profiler)")

    # ------------------------------------------------------- capture pricing
    def _mem():
        torch.cuda.synchronize()
        return time.perf_counter(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated()

    def _install_capture_log():
        try:
            from vllm.v1.worker.gpu import cudagraph_utils as cu
        except Exception as exc:
            _log(f"capture log unavailable: {exc!r}")
            return
        orig_capture = cu.CudaGraphManager.capture

        def capture(self, create_forward_fn, *args, **kwargs):
            phase = "profile" if getattr(self, "_capture_mem_samples", None) is not None else "real"
            graphs, cur = [], {}

            def close():
                if cur:
                    t1, r1, a1 = _mem()
                    graphs.append(dict(num_tokens=cur["num_tokens"], mode=cur["mode"],
                                       desc=cur["desc"], wall_s=t1 - cur["t0"],
                                       reserved_delta=r1 - cur["r0"],
                                       allocated_delta=a1 - cur["a0"]))
                    cur.clear()

            def counted(desc, warmup=False):
                # Every descriptor starts with its warmup call; that call ends the
                # previous descriptor's warmup and capture.
                if warmup:
                    close()
                    t0, r0, a0 = _mem()
                    cur.update(num_tokens=int(getattr(desc, "num_tokens", -1)),
                               mode=getattr(getattr(desc, "cg_mode", None), "name", "?"),
                               desc=repr(desc)[:200], t0=t0, r0=r0, a0=a0)
                return create_forward_fn(desc, warmup=warmup)

            t0, r0, a0 = _mem()
            try:
                return orig_capture(self, counted, *args, **kwargs)
            finally:
                close()
                t1, r1, a1 = _mem()
                rec = dict(phase=phase, manager=type(self).__name__, pid=os.getpid(),
                           n_graphs=len(graphs), wall_s=t1 - t0, reserved_delta=r1 - r0,
                           allocated_delta=a1 - a0, reserved_after=r1, graphs=graphs)
                with open(_CAPTURE_LOG, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                _log(f"CAPTURE {phase}: {len(graphs)} graphs in {t1 - t0:.2f} s, "
                     f"reserved +{(r1 - r0) / 2**20:.1f} MiB")

        cu.CudaGraphManager.capture = capture
        _log("capture log installed")

    # --------------------------------------------------------------- hooks
    def _pre(module, args):
        # The global pre-hook takes no kwargs in this torch.
        if not _S["installed"]:
            _S["installed"] = True
            if _MOE_DET:
                _install_moe_deterministic()
            if (_DIGEST_PATH and not _FULL) or (_PROF_DIR and _PROF_TRIGGER):
                _install_runner_wrap()
            if _CAPTURE_LOG:
                _install_capture_log()
        if _DIGEST_PATH and _S["root"] is None and type(module).__name__.endswith(_ROOT_SUFFIXES):
            _S["root"] = module
            _S["names"] = {id(m): n for n, m in module.named_modules()
                           if n and n.count(".") < _DEPTH}
            with open(_DIGEST_PATH + ".names.json", "w") as fh:
                json.dump(dict(root=type(module).__name__, depth=_DEPTH,
                               recorded=sorted(_S["names"].values()),
                               deeper=sorted(n for n, _ in module.named_modules()
                                             if n and n.count(".") >= _DEPTH)), fh, indent=0)
            _log(f"digest root {type(module).__name__}: {len(_S['names'])} modules at depth < {_DEPTH}")
        return None

    def _post(module, args, kwargs, output):
        if not _DIGEST_PATH:
            return None
        if module is _S["root"]:
            _record("root|out", _first_tensor(output))
            _readout_break()
            return None
        name = _S["names"].get(id(module))
        if name is not None:
            # Inputs are read after the forward: a module that updates its input
            # in place shows the updated value, identically in both arms.
            t = _first_tensor(args) if args else None
            if t is None and kwargs:
                t = kwargs.get("hidden_states")
                if t is None:
                    t = _first_tensor(kwargs)
            _record(name + "|in", t)
            _record(name + "|out", _first_tensor(output))
        return None

    torch.nn.modules.module.register_module_forward_pre_hook(_pre)
    torch.nn.modules.module.register_module_forward_hook(_post, with_kwargs=True)
