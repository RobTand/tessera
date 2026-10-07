"""Kernel-neutral routing work intervals and two-stream class orchestration.

Kernel bindings own packed payloads, activation buffers, optional persistent
scratch, work geometry, grid/shared memory and the native launch. A payload may
cover one class or several classes. Rate profiles and absolute word offsets are
packing planes inside that payload; this module does not interpret them.
"""
from __future__ import annotations

from typing import Protocol

import torch


class RoutedClassKernel(Protocol):
    """A loaded kernel binding. No model weights live in the resource registry."""

    def work_shape(self, mode: int, tokens: int, index: int, parameters: dict) -> tuple[int, int]:
        """Return superblock rows and work units per superblock, including K parts."""
        ...

    def prepare_input(self, x, a_scale, rows: int, family: str, device):
        """Return kernel-owned operands, including its zero sentinel row if needed."""
        ...

    def launch(self, mode, x, a_scale, *, index, start, end, prefix, counter,
               routing, parameters, bm, work_units, empty_scale, a_row_mode,
               mul_weight, limit, out):
        """Consume absolute bounds on the active CUDA stream; counter may be None.

        Packing planes and persistent K-part scratch belong to the binding.
        Scratch indexes absolute work units, never class-local work numbers.
        """
        ...


def declared_route_widths(kernel, tokens, issue_order, parameters, modes=(0, 2)):
    """Return every bound-kernel width before route prefixes enter the stream DAG."""
    return tuple(dict.fromkeys(kernel.work_shape(mode, tokens, c, parameters)[0]
                              for c in issue_order for mode in modes))


def initialize_class_counter(counter: torch.Tensor, prefix: torch.Tensor,
                             start: int, work_units: int) -> None:
    """Recompute the device start from this invocation or captured replay."""
    counter.copy_(prefix[start:start + 1])
    counter.mul_(work_units)


def dispatch_class_projection(mode, x, a_scale, routing, *, parameters, starts,
        ends, issue_order, counters, resources, mul_weight, limit, a_row_mode, out):
    """Issue declared expert spans with live prefixes and a fixed stream DAG.

    Today each span is one class. The launch contract also accepts a wider span;
    grouping spans is a separate dispatcher decision, not a kernel fallback.
    The current two-stream dispatcher remains the selected production policy.
    """
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    caller = torch.cuda.current_stream(index)
    streams = resources.streams
    kernel = resources.kernel
    resources.ready.record(caller)
    for stream in streams:
        stream.wait_event(resources.ready)
    projection = 1 if mode == 2 else 0
    for c in issue_order:
        with torch.cuda.stream(streams[c % 2]):
            bm, work_units = kernel.work_shape(mode, routing.tokens, c, parameters)
            prefix = routing.superblocks(bm)
            counter = None if counters is None else counters[c, projection:projection + 1]
            if counter is not None:
                initialize_class_counter(counter, prefix, starts[c], work_units)
            kernel.launch(mode, x, a_scale, index=c, start=starts[c], end=ends[c],
                prefix=prefix, counter=counter, routing=routing, parameters=parameters,
                bm=bm, work_units=work_units, empty_scale=resources.empty,
                a_row_mode=a_row_mode, mul_weight=mul_weight, limit=limit, out=out)
            # A binding may request a different prefix width for a later span.
            # Each used prefix records its real consuming stream, without a host read.
            prefix.record_stream(streams[c % 2])
    for stream, finished in zip(streams, resources.finished):
        finished.record(stream)
        caller.wait_event(finished)
    for tensor in (x, a_scale, routing.offsets, routing.flat_sorted, routing.rw_sorted, out, resources.empty):
        if tensor is not None:
            for stream in streams:
                tensor.record_stream(stream)
