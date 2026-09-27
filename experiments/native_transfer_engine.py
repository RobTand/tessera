"""Native engine binding for the transfer passes (TP1 dense, #654).

Binds the pass runners in ``native_resource_passes`` to the real bench
machinery: ``prepare_native_operator`` for preparation, ``apply_complete`` for
the full apply (prime), ``time_apply`` for decision timing, and the fixture
layout produced by the qualification encode path. TP1 only: world 1, no
distributed context. Every heavy import stays inside a method so CPU protocol
tests can substitute the bench functions and the collector can start before
Torch is mapped; no GPU claim is made here.

Rates are WIRE RATE NAMES (``TESSERA_BF16_K1_R512``), the same keys the pass
runners mark; ``load`` refuses a fixture whose request names another format,
so a mis-mapped directory cannot produce self-consistent legs under the wrong
rate label. Applies and timed applies run under ``torch.inference_mode()``,
exactly like the fresh-process measurement path.

The timer for pass T (and the r=5 fresh band) is ``torch.cuda.Event`` pairs in
``time_apply``; ``eps`` for the band is the measured minimal positive
back-to-back event elapsed time on the run's device, recorded by the driver in
``eps_source`` -- never a constant.
"""
from __future__ import annotations

import json
from pathlib import Path

PHASES = ("prefill", "decode")


def _cuda_settle():
    """Synchronize the device and release the caching allocator's free blocks."""
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _process_identity():
    """Process-unique identity: pid, boot id, and start jiffies from /proc.

    ``comm`` can contain spaces, so the field split starts after the last
    ``)`` of the stat line, never at whitespace alone.
    """
    try:
        raw = Path("/proc/self/stat").read_text()
        tail = raw[raw.rfind(")") + 1:].split()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        pid, start_ticks = int(raw.split(maxsplit=1)[0]), int(tail[19])
    except (OSError, IndexError, ValueError) as error:
        raise ValueError(f"process identity is unavailable: {error}") from error
    return {"pid": pid, "boot_id": boot_id, "start_ticks": start_ticks}


class NativeTransferEngine:
    """Runner-protocol adapter over the bench: one fixture directory per rate.

    ``fixtures`` maps a wire rate name to a directory holding the encode-path
    layout (``fixture.tessera``, ``wire-record.json``, ``tensors.safetensors``,
    ``request.json``) produced once, ahead of every leg, so all legs consume
    byte-identical inputs. ``prime`` runs the complete apply for every phase --
    for TP1 that is ``prepared['method'].apply`` with no reduction.
    """

    def __init__(self, fixtures, *, warmup_iterations=8, iterations=8):
        if not isinstance(fixtures, dict) or not fixtures:
            raise ValueError("engine requires at least one fixture rate")
        if any(not isinstance(rate, str) or not rate for rate in fixtures):
            raise ValueError("engine fixture keys are wire rate names, not integers")
        self._fixtures = {rate: Path(path) for rate, path in fixtures.items()}
        self._warmup = warmup_iterations
        self._iterations = iterations

    def family(self, fmt):
        """Enter the native runtime context for one format family."""
        from experiments import bench_native_operator as bench
        return bench.native_runtime_context()

    def warmup(self, rate):
        """One unmarked load/prepare/prime/evict cycle on the first rate.

        Lazy one-time allocations (runtime imports, context creation) land in
        the initialization baseline ahead of the first marker instead of
        contaminating the first rate window. The payload and prepared objects
        are dropped BEFORE settle: ``evict`` can only delete its own frame's
        references, and a settle with this frame's tensors still live would
        leave cached segments in the pre-begin baseline that the first rate's
        own settle then frees -- a baseline change every window refuses.
        """
        payload = self.load(rate)
        prepared = self.prepare(rate, payload)
        self.prime(prepared, payload)
        self.evict(prepared, payload)
        del prepared, payload
        self.settle()

    def load(self, rate):
        """Materialize a rate's fixture: wire bytes, record, tensors, request.

        The request's format must name this wire rate; a mis-mapped directory
        refuses here rather than producing a leg under the wrong rate label.
        """
        directory = self._fixtures[rate]
        try:
            request = json.loads((directory / "request.json").read_text())
            record = json.loads((directory / "wire-record.json").read_text())
            blob = (directory / "fixture.tessera").read_bytes()
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"fixture for rate {rate} could not be read back: {error}") from error
        if not isinstance(request, dict) or request.get("format") != rate:
            raise ValueError(f"fixture for rate {rate} names format "
                             f"{request.get('format') if isinstance(request, dict) else None!r}")
        from safetensors import torch as safetensors_torch
        tensors = safetensors_torch.load_file(str(directory / "tensors.safetensors"), device="cuda")
        return {"request": request, "blob": blob, "record": record, "tensors": tensors}

    def prepare(self, rate, payload):
        """Prepare the native operator exactly as the fresh-process mode does."""
        from experiments import bench_native_operator as bench
        request = payload["request"]
        tensors = payload["tensors"]
        return bench.prepare_native_operator(
            payload["blob"], payload["record"], tensors["source_weight"],
            tensors["rendered_weight"], unit=request["unit"], format_name=request["format"],
            runtime_image=request["runtime_image"],
            input_global_scale=request.get("input_global_scale"))

    def prime(self, prepared, payload):
        """Complete applies for every phase -- the transfer's real work."""
        import torch

        from experiments import bench_native_operator as bench
        outputs = {}
        for phase in PHASES:
            with torch.inference_mode():
                outputs[phase] = bench.apply_complete(prepared,
                                                      payload["tensors"][f"{phase}.input"])
        return outputs

    def identity(self, prepared, payload):
        """Contract binding: what any leg must re-derive identically.

        Built only from the bench's own real output shapes: the operator's
        byte-hashed ``wire_sha256``, its full ``tensor_identity`` dicts, the
        whole native-tensor mapping digested, the declared route, and the
        untimed phase-input identities -- plus the request's format, which
        ``load`` already proved names this rate.
        """
        from experiments import bench_native_operator as bench
        operator = prepared["operator"]
        return {"format": payload["request"]["format"],
                "operator": {
                    "wire_sha256": operator["wire_sha256"],
                    "wire_record_sha256": operator["wire_record_sha256"],
                    "source_weight": operator["source_weight"],
                    "rendered_weight": operator["rendered_weight"],
                    "native_tensors_sha256": bench.identity_sha256(operator["native_tensors"]),
                    "scheme_sha256": operator["scheme_sha256"],
                    "declared_route": operator["declared_route"],
                    "input_global_scale": operator["input_global_scale"]},
                "runtime": {"execution": dict(prepared["runtime"]["execution"])},
                "phase_tensors": {phase: bench.tensor_identity(payload["tensors"][f"{phase}.input"])
                                  for phase in PHASES}}

    def settle(self):
        _cuda_settle()

    def evict(self, prepared, payload):
        """Drop this frame's references and release the allocator's free blocks.

        Only the CALLER's references keep the objects alive; ``evict`` deletes
        its own parameters, so callers must drop theirs too (``del prepared,
        payload``) before the settle that follows.
        """
        del prepared, payload
        _cuda_settle()

    def process(self):
        return _process_identity()

    def time(self, prepared, payload):
        """Decision timing: one complete apply per CUDA event pair, per phase.

        ``inference_mode`` is entered once, outside the timed region, the way
        the bench does around its own ``time_apply`` loop; entering it inside
        the apply would charge mode transitions to the timed samples.
        """
        import torch

        from experiments import bench_native_operator as bench
        samples = {}
        with torch.inference_mode():
            for phase in PHASES:
                def apply_once(value=payload["tensors"][f"{phase}.input"]):
                    return bench.apply_complete(prepared, value)
                result = bench.time_apply(apply_once, warmup_iterations=self._warmup,
                                          iterations=self._iterations)
                samples[phase] = result["samples_ms"]
        return samples
