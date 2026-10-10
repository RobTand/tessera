"""The resident-tensor declaration every Tessera route makes (#399, #580).

A route prepares its weights into objects vLLM does not register: a slotted
``PreparedDenseNativeModule`` on the dense window routes, lists of
``A4Unit`` and epilogue tensors on the NVFP4 route, ``A4UnitStack`` planes on
the NVFP4 MoE route.  Registering them as buffers would change what vLLM loads
and moves, so a resource observer cannot find them through
``named_parameters()``/``named_buffers()``.  Instead each route's quant method
declares them:

* ``quant_method.resident_tensors(layer)`` yields ``(name, tensor)`` for every
  device tensor the route holds for ``layer`` outside registered state, by
  reference.  Names are relative to the layer.
* A prepared object the route stores exposes ``named_tensors()`` over its own
  tensor references.  The walk here expands tensors, lists/tuples, and objects
  that declare ``named_tensors()``; an object that declares nothing is refused
  by name rather than skipped, because an unnamed resident tensor is a byte no
  observer can charge.

The declaration copies nothing, allocates nothing and moves nothing.  Shared
storage is not deduplicated here: an observer joins every declared view to its
backing allocation and charges that allocation once.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import torch


def named_resident_tensors(value, prefix: str) -> Iterator[Tuple[str, torch.Tensor]]:
    """``(name, tensor)`` references reachable from one declared attribute value."""
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from named_resident_tensors(item, f"{prefix}.{index}")
    elif callable(getattr(value, "named_tensors", None)):
        for name, tensor in value.named_tensors():
            yield f"{prefix}.{name}", tensor
    else:
        raise TypeError(f"{prefix}: {type(value).__name__} declares no named_tensors(), so "
                        "its resident tensors cannot be attributed")


def layer_resident_tensors(layer, attributes: Iterable[str]) -> Iterator[Tuple[str, torch.Tensor]]:
    """The declared attributes of ``layer``; an attribute not yet set yields nothing."""
    for attribute in attributes:
        yield from named_resident_tensors(vars(layer).get(attribute), attribute)
