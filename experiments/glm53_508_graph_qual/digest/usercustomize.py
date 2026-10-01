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

``T695_GC_BEFORE_DRAFTER=1`` (tessera#695)
    Runs ``gc.collect()`` and ``torch.cuda.empty_cache()`` once before the MTP
    drafter loads, and prints MemAvailable before and after. The V2 runner loads
    the target and the drafter inside one ``DeviceMemoryProfiler`` block, whose
    exit is the first full collection after the target's load; on the four-layer
    stub the target's load leaves about 33 GiB collectable until then (MemAvailable
    110 -> 32.5 GiB during an eager load, 74 GiB right after the block exits), so
    the drafter loaded on top of it and the box watchdog stopped both serves at
    its 16 GiB floor. Memory management only: no arithmetic changes.

``T695_DRAFT_LOG=<prefix>`` (tessera#695)
    Records what the MTP drafter proposed, so a graph arm's drafts can be
    compared with an eager arm's at the same context. Acceptance cannot do this
    on the stub: its target emits near-random tokens, so no draft is accepted.
    Each engine process appends to ``<prefix>.<pid>.jsonl``:
    ``{"ev": "new", "req", "prompt_len", "prompt_sha"}`` for every request the
    runner adds, and ``{"ev": "draft", "reqs", "k", "rows"}`` for every serving
    ``propose`` call, one row per request: its k draft tokens, then the token
    the target sampled last (the drafter's first input), how many tokens the
    target emitted this step, how many drafts it rejected, and the sequence
    length. Dummy, profiling and capturing calls are not recorded. The rows are
    copied device-to-host without blocking, behind an event, and written once
    the event completes (at the next call or within a second), so the runner's
    host-device overlap is kept. The copies add a gather and a few small
    kernels per step; the arithmetic of every forward is unchanged.
    ``T695_DRAFT_LOG_TRIGGER=<path>`` records drafts only while that file
    exists (checked at most once a second), so one serve can time its
    latency cells unrecorded and record its equality probes. Requests are
    recorded either way.
"""
import os

_T695_GC = os.environ.get("T695_GC_BEFORE_DRAFTER") == "1"
_T695_DRAFT_LOG = os.environ.get("T695_DRAFT_LOG")

if _T695_GC or _T695_DRAFT_LOG:
    import gc
    import importlib.abc
    import sys

    def _t695_available_mib():
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        return -1

    class _T695DraftLog:
        """Deferred device-to-host copies of the drafter's proposals (see the docstring)."""

        def __init__(self, prefix, trigger=None):
            self.prefix = prefix
            self.trigger = trigger
            self.on = trigger is None
            self.checked = 0.0
            self.owner = None
            self.failed = False

        def active(self):
            """Record this step? Always without a trigger; else while the trigger file exists."""
            if self.trigger is None:
                return True
            import time
            now = time.monotonic()
            if now - self.checked >= 1.0:
                self.checked = now
                self.on = os.path.exists(self.trigger)
            return self.on

        def _ensure(self):
            """Per process: the state, the drain thread and the file belong to the process
            that records. A forked child re-creates them (a thread does not survive a
            fork, and a lock copied while held would never be released)."""
            if self.owner == os.getpid():
                return
            import atexit
            import collections
            import threading
            self.owner = os.getpid()
            self.fh = None
            self.pending = collections.deque()
            self.lock = threading.Lock()
            threading.Thread(target=self._drain_loop, name="t695-draft-log", daemon=True).start()
            atexit.register(self.flush, True)

        def _write(self, rec):
            import json
            if self.fh is None:
                self.fh = open(f"{self.prefix}.{os.getpid()}.jsonl", "a", buffering=1)
            self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

        def new_requests(self, scheduler_output):
            import hashlib
            import json
            self._ensure()
            recs = []
            for req in getattr(scheduler_output, "scheduled_new_reqs", None) or []:
                prompt_len = getattr(req, "prompt_len", None)
                toks = req.prompt_token_ids
                if toks is None:
                    toks = (req.prefill_token_ids or [])[:prompt_len]
                toks = [int(t) for t in toks]
                recs.append(dict(ev="new", req=req.req_id, prompt_len=len(toks),
                                 prompt_sha=hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]))
            if recs:
                with self.lock:
                    for rec in recs:
                        self._write(rec)

        def propose(self, input_batch, bound, drafts):
            import torch
            n = int(input_batch.num_reqs)
            if n <= 0:
                return
            self._ensure()
            args = bound.arguments
            idx = input_batch.idx_mapping[:n].long()
            # Every column as [n, width]: the runner's last-sampled buffer is
            # [max_num_reqs, 1], the per-request counters and lengths are [n].
            cols = [drafts[:n].reshape(n, -1).to(torch.int64),
                    args["last_sampled"][idx].reshape(n, -1)[:, -1:].to(torch.int64),
                    args["num_sampled"][:n].reshape(n, 1).to(torch.int64),
                    args["num_rejected"][:n].reshape(n, 1).to(torch.int64),
                    input_batch.seq_lens[:n].reshape(n, 1).to(torch.int64)]
            packed = torch.cat(cols, dim=1)
            host = torch.empty(packed.shape, dtype=torch.int64, device="cpu", pin_memory=True)
            host.copy_(packed, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            meta = dict(ev="draft", reqs=list(input_batch.req_ids[:n]), k=int(drafts.shape[1]))
            with self.lock:
                self.pending.append((event, host, meta))
            self.flush(False)

        def flush(self, wait):
            if self.owner != os.getpid():
                return
            with self.lock:
                while self.pending:
                    event, host, meta = self.pending[0]
                    if wait:
                        event.synchronize()
                    elif not event.query():
                        break
                    self.pending.popleft()
                    meta["rows"] = host.tolist()
                    self._write(meta)

        def _drain_loop(self):
            import time
            while True:
                time.sleep(1.0)
                try:
                    self.flush(False)
                except Exception as exc:  # research hook: report once, never stop the serve
                    if not self.failed:
                        self.failed = True
                        print(f"[t695] draft log flush failed: {exc!r}", file=sys.stderr, flush=True)

    _t695_draft_log = (_T695DraftLog(_T695_DRAFT_LOG, os.environ.get("T695_DRAFT_LOG_TRIGGER") or None)
                       if _T695_DRAFT_LOG else None)

    def _t695_patch_speculator(module):
        cls = getattr(module, "MTPSpeculator", None)
        if cls is None or getattr(cls, "_t695", False):
            return
        cls._t695 = True
        if _T695_GC:
            original_load = cls.load_model

            def load_model(self, *args, **kwargs):
                before = _t695_available_mib()
                collected = gc.collect()
                import torch
                torch.cuda.empty_cache()
                print(f"[t695] gc before the drafter loads: {collected} objects collected, "
                      f"MemAvailable {before} -> {_t695_available_mib()} MiB", file=sys.stderr, flush=True)
                # The drafter load's own torch peak (tessera#695/#749): the peak
                # counters restart here, so the readout after the load is the
                # most the caching allocator held while the drafter was built,
                # on top of the target it shares with.
                torch.cuda.synchronize()
                alloc0, resv0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                out = original_load(self, *args, **kwargs)
                torch.cuda.synchronize()
                gib = 2 ** 30
                print(f"[t695] drafter load peak: allocated {alloc0 / gib:.3f} -> peak "
                      f"{torch.cuda.max_memory_allocated() / gib:.3f} -> "
                      f"{torch.cuda.memory_allocated() / gib:.3f} GiB; reserved {resv0 / gib:.3f} -> "
                      f"peak {torch.cuda.max_memory_reserved() / gib:.3f} -> "
                      f"{torch.cuda.memory_reserved() / gib:.3f} GiB; MemAvailable "
                      f"{_t695_available_mib()} MiB", file=sys.stderr, flush=True)
                return out

            cls.load_model = load_model
        if _t695_draft_log is not None:
            import inspect
            original_propose = cls.propose
            signature = inspect.signature(original_propose)

            def propose(self, input_batch, *args, **kwargs):
                drafts = original_propose(self, input_batch, *args, **kwargs)
                import torch
                if (kwargs.get("dummy_run") or kwargs.get("is_profile")
                        or torch.cuda.is_current_stream_capturing() or _t695_draft_log.failed
                        or not _t695_draft_log.active()):
                    return drafts
                try:
                    bound = signature.bind(self, input_batch, *args, **kwargs)
                    _t695_draft_log.propose(input_batch, bound, drafts)
                except Exception as exc:  # research hook: report once, never stop the serve
                    _t695_draft_log.failed = True
                    print(f"[t695] draft log disabled: {exc!r}", file=sys.stderr, flush=True)
                return drafts

            cls.propose = propose
        print(f"[t695] MTPSpeculator patched: gc={_T695_GC} draft_log={bool(_t695_draft_log)}",
              file=sys.stderr, flush=True)

    def _t695_patch_runner(module):
        cls = getattr(module, "GPUModelRunner", None)
        if cls is None or _t695_draft_log is None or getattr(cls, "_t695", False):
            return
        cls._t695 = True
        original_add = cls.add_requests

        def add_requests(self, scheduler_output, *args, **kwargs):
            if not _t695_draft_log.failed:
                try:
                    _t695_draft_log.new_requests(scheduler_output)
                except Exception as exc:  # research hook: report once, never stop the serve
                    _t695_draft_log.failed = True
                    print(f"[t695] draft log disabled: {exc!r}", file=sys.stderr, flush=True)
            return original_add(self, scheduler_output, *args, **kwargs)

        cls.add_requests = add_requests

    _T695_PATCHES = {
        "vllm.v1.worker.gpu.spec_decode.mtp.speculator": _t695_patch_speculator,
        "vllm.v1.worker.gpu.model_runner": _t695_patch_runner,
    }

    class _T695Finder(importlib.abc.MetaPathFinder):
        """Patch the MTP speculator and the V2 runner right after their modules execute."""

        # Another patching finder (``_GAFinder``) delegates down the same
        # meta path, so without a guard the two call each other for a module
        # both patch (the V2 runner) until the recursion limit.  A re-entered
        # finder steps aside; the outer call still wraps the spec it returns.
        # Per instance, so a second instance never sees the first one's names.
        def __init__(self):
            self._busy = set()

        def find_spec(self, name, path, target=None):
            patch = _T695_PATCHES.get(name)
            if patch is None or name in self._busy:
                return None
            self._busy.add(name)
            try:
                for finder in sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(name, path, target)
                    if spec is None or spec.loader is None:
                        continue
                    run = spec.loader.exec_module

                    def exec_module(module, _run=run, _patch=patch):
                        _run(module)
                        _patch(module)

                    spec.loader.exec_module = exec_module
                    return spec
                return None
            finally:
                self._busy.discard(name)

    sys.meta_path.insert(0, _T695Finder())

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


# ---------------------------------------------------------------- tessera#702
# ``GA_DISPATCH_LOG=<prefix>`` (tessera#702, graph attestation on the vLLM nightly)
#     Counts what the V2 runner's CUDA-graph managers dispatch on serving steps:
#     every ``CudaGraphManager.dispatch`` result inside a non-dummy
#     ``GPUModelRunner.execute_model`` call, keyed by manager class, graph mode,
#     token count and request count, plus each manager's captured token counts
#     after its capture. Each engine process rewrites ``<prefix>.<pid>.json`` with
#     the running totals at most once a second and after every capture, so an
#     equality arm can show that each captured size was replayed. Host-side
#     bookkeeping only: no device work, no arithmetic change.
_GA_DISPATCH_LOG = os.environ.get("GA_DISPATCH_LOG")

if _GA_DISPATCH_LOG:
    import importlib.abc as _ga_abc
    import json as _ga_json
    import sys as _ga_sys
    import threading as _ga_threading
    import time as _ga_time

    _GA = {"counts": {}, "captured": {}, "serving": _ga_threading.local(), "last": 0.0,
           "lock": _ga_threading.Lock(), "dirty": False}

    def _ga_flush(force=False):
        now = _ga_time.monotonic()
        if not force and now - _GA["last"] < 1.0:
            return
        _GA["last"] = now
        with _GA["lock"]:
            rec = {"pid": os.getpid(), "captured": dict(_GA["captured"]),
                   "counts": dict(_GA["counts"])}
            _GA["dirty"] = False
        path = f"{_GA_DISPATCH_LOG}.{os.getpid()}.json"
        with open(path + ".tmp", "w") as fh:
            _ga_json.dump(rec, fh, indent=0, sort_keys=True)
        os.replace(path + ".tmp", path)

    def _ga_patch_cudagraph(module):
        cls = module.CudaGraphManager
        orig_dispatch = cls.dispatch
        orig_capture = cls.capture

        def dispatch(self, *args, **kwargs):
            desc = orig_dispatch(self, *args, **kwargs)
            if getattr(_GA["serving"], "on", False):
                mode = getattr(getattr(desc, "cg_mode", None), "name", "?")
                key = (f"{type(self).__name__}|{mode}|tokens={getattr(desc, 'num_tokens', -1)}"
                       f"|reqs={getattr(desc, 'num_reqs', -1)}")
                with _GA["lock"]:
                    _GA["counts"][key] = _GA["counts"].get(key, 0) + 1
                    _GA["dirty"] = True
                _ga_flush()
            return desc

        def capture(self, *args, **kwargs):
            out = orig_capture(self, *args, **kwargs)
            try:
                sizes = self.captured_token_counts()
            except Exception as exc:  # recorded, never fatal
                sizes = f"unreadable: {exc!r}"
            with _GA["lock"]:
                _GA["captured"][f"{type(self).__name__}@{id(self):x}"] = sizes
            _ga_flush(force=True)
            return out

        cls.dispatch = dispatch
        cls.capture = capture
        _ga_threading.Thread(target=_ga_flusher, name="ga702-flush", daemon=True).start()
        print(f"[ga702] CudaGraphManager patched (pid {os.getpid()})", flush=True)

    def _ga_patch_runner(module):
        # The target forward runs in execute_model; the drafter's propose runs in
        # sample_tokens. Dummy and profiling runs pass dummy_run=True.
        cls = module.GPUModelRunner
        orig_exec, orig_sample = cls.execute_model, cls.sample_tokens

        def execute_model(self, *args, **kwargs):
            if kwargs.get("dummy_run", False):
                return orig_exec(self, *args, **kwargs)
            _GA["serving"].on = True
            try:
                return orig_exec(self, *args, **kwargs)
            finally:
                _GA["serving"].on = False

        def sample_tokens(self, *args, **kwargs):
            _GA["serving"].on = True
            try:
                return orig_sample(self, *args, **kwargs)
            finally:
                _GA["serving"].on = False

        cls.execute_model = execute_model
        cls.sample_tokens = sample_tokens
        print(f"[ga702] GPUModelRunner.execute_model/sample_tokens patched (pid {os.getpid()})",
              flush=True)

    def _ga_flusher():
        # The last dispatches of a burst land inside the one-second window; a
        # daemon writes them once the burst is over.
        while True:
            _ga_time.sleep(1.0)
            if _GA["dirty"]:
                try:
                    _ga_flush(force=True)
                except Exception:  # a full disk must not stop the engine
                    pass

    _GA_PATCHES = {
        "vllm.v1.worker.gpu.cudagraph_utils": _ga_patch_cudagraph,
        "vllm.v1.worker.gpu.model_runner": _ga_patch_runner,
    }

    class _GAFinder(_ga_abc.MetaPathFinder):
        # Re-entrancy guard, per instance: see ``_T695Finder``; both patch
        # the V2 runner.
        def __init__(self):
            self._busy = set()

        def find_spec(self, name, path, target=None):
            patch = _GA_PATCHES.get(name)
            if patch is None or name in self._busy:
                return None
            self._busy.add(name)
            try:
                for finder in _ga_sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(name, path, target)
                    if spec is None or spec.loader is None:
                        continue
                    run = spec.loader.exec_module

                    def exec_module(module, _run=run, _patch=patch):
                        _run(module)
                        _patch(module)

                    spec.loader.exec_module = exec_module
                    return spec
                return None
            finally:
                self._busy.discard(name)

    _ga_sys.meta_path.insert(0, _GAFinder())
