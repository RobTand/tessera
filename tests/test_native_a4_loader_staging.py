"""The native A4 loader's bounded staging: reusable buffers and a packed axis.

The loader runs under the runtime's ``max_split_size_mb=20`` allocator context
(``vllm/v1/worker/gpu_worker.py``), where a fresh per-wire device transfer of a
few megabytes left a dead 20 MiB allocator slab per wire.  Three ownership
contracts keep the load bounded (measured in
``docs/measurements/tessera-a4-loader-staging-20260916.md``):

* ``compact_prep._plane_u8`` and ``kernel_bits._plane_words`` fill a
  **caller-owned** reusable buffer when one is handed in, return exactly the
  requested length whatever capacity the buffer has grown to, and keep the
  original single-allocation path when ``scratch`` is ``None``;
* ``serving.native_a4.A4ExpertAxis`` allocates its stacked planes once on the
  first ``put`` and copies each expert into its own slot, so per-expert
  temporaries are never retained and ``finish`` copies nothing;
* reusing a scratch buffer never mutates an expert already written.

These are device tests (CUDA paths).  Run them in the pinned image through
PrismaBuild, not on a CPU-only host.
"""
from __future__ import annotations

import torch

import pytest

from tessera.kernel_bits import _plane_words
from tessera.compact_prep import _plane_u8
from tessera.errors import GrammarError
from tessera.kernel_a4 import A4Unit, A4UnitStack
from tessera.serving.native_a4 import A4ExpertAxis

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the loader staging paths are CUDA paths")


def _unit(rows: int = 8, cols: int = 64, seed: int = 0) -> A4Unit:
    """A small A4Unit whose plane shapes are the only contract the axis checks."""
    g = torch.Generator().manual_seed(seed)
    return A4Unit(
        select=torch.randint(0, 256, (rows // 2,), dtype=torch.uint8, generator=g).cuda(),
        label=torch.randint(0, 256, (cols // 4,), dtype=torch.uint8, generator=g).cuda(),
        point=torch.randint(0, 256, (cols * 3,), dtype=torch.uint8, generator=g).cuda(),
        nibbles=torch.randint(0, 256, (rows // 2, cols // 16), dtype=torch.uint8,
                              generator=g).cuda(),
        lut_bytes=torch.randint(0, 127, (16,), dtype=torch.uint8, generator=g).cuda(),
        label_lut=torch.randint(0, 64, (16,), dtype=torch.int32, generator=g).cuda(),
        subset_nibbles=torch.randint(0, 255, (16,), dtype=torch.uint8, generator=g).cuda(),
        code_nibbles=torch.randint(0, 255, (32,), dtype=torch.uint8, generator=g).cuda(),
        rows=rows, cols=cols, rate=3, arity=2, memory=4, half=16,
        global_scale=float(seed + 1),
    )


@cuda
def test_plane_u8_scratch_reuses_storage_and_exact_length():
    scratch: dict = {}
    small = bytes(range(64))
    large = bytes(range(256)) * 4
    first = _plane_u8(small, "cuda", scratch, "body")
    assert int(first.numel()) == len(small)
    assert bool(torch.equal(first, torch.frombuffer(bytearray(small), dtype=torch.uint8).cuda()))
    pointer = first.data_ptr()

    grown = _plane_u8(large, "cuda", scratch, "body")
    assert int(grown.numel()) == len(large)
    assert grown.data_ptr() == pointer, "the scratch buffer was reallocated for a larger plane"
    assert bool(torch.equal(grown, torch.frombuffer(bytearray(large), dtype=torch.uint8).cuda()))

    shrunk = _plane_u8(small, "cuda", scratch, "body")
    assert int(shrunk.numel()) == len(small), "a shrunk reuse must return the exact request"
    assert shrunk.data_ptr() == pointer
    assert bool(torch.equal(shrunk, torch.frombuffer(bytearray(small), dtype=torch.uint8).cuda()))


@cuda
def test_plane_u8_without_scratch_matches_the_scratch_path():
    data = bytes(range(200))
    plain = _plane_u8(data, "cuda")
    staged = _plane_u8(data, "cuda", {}, "body")
    assert bool(torch.equal(plain, staged))


@cuda
def test_plane_words_scratch_grow_shrink_empty_and_identity():
    scratch: dict = {}
    for n in (0, 3, 8, 4096, 11, 0):
        plane = torch.arange(n, dtype=torch.uint8)
        plain = _plane_words(plane)
        staged = _plane_words(plane, scratch)
        assert bool(torch.equal(plain, staged)), f"n={n}: scratch path changed the words"
        assert int(staged.numel()) == int(plain.numel()), \
            f"n={n}: scratch path returned cached capacity, not the requested words"
    # Empty planes keep the one-word pad semantics of the original path.
    assert bool(torch.equal(_plane_words(torch.empty(0, dtype=torch.uint8)),
                            _plane_words(torch.empty(0, dtype=torch.uint8), scratch)))


@cuda
def test_axis_preallocates_once_and_finish_copies_nothing():
    axis = A4ExpertAxis(4)
    units = [_unit(seed=index) for index in range(4)]
    axis.put(2, units[2])
    pointers = {field: axis._planes[field].data_ptr() for field in axis._FIELDS}
    storages = {field: axis._planes[field].untyped_storage().nbytes()
                for field in axis._FIELDS}
    for expert in (0, 1, 3):
        axis.put(expert, units[expert])
    for field in axis._FIELDS:
        assert axis._planes[field].data_ptr() == pointers[field], \
            "a later put reallocated a preallocated axis plane"
        assert axis._planes[field].untyped_storage().nbytes() == storages[field]

    stack = axis.finish()
    assert isinstance(stack, A4UnitStack)
    for field in axis._FIELDS:
        stored = getattr(stack, field)
        assert stored.data_ptr() == pointers[field], \
            f"{field}: finish copied instead of returning the preallocated buffer"
    for expert, unit in enumerate(units):
        for field in axis._FIELDS:
            expected = getattr(unit, field)
            assert bool(torch.equal(getattr(stack, field)[expert], expected))
    expected_globals = torch.tensor([unit.global_scale for unit in units])
    assert bool(torch.equal(stack.globals.cpu(), expected_globals))
    assert stack.globals.data_ptr() != 0


@cuda
def test_axis_refuses_a_second_put_and_a_foreign_layout():
    axis = A4ExpertAxis(3)
    axis.put(0, _unit(seed=0))
    with pytest.raises(GrammarError):
        axis.put(0, _unit(seed=1))
    with pytest.raises(GrammarError):
        axis.put(1, _unit(rows=16, seed=2))
    with pytest.raises(GrammarError):
        axis.put(3, _unit(seed=3))


@cuda
def test_scratch_reuse_leaves_written_experts_untouched():
    """The ownership invariant the probe also checks end to end."""
    scratch: dict = {}
    plane = torch.arange(4096, dtype=torch.uint8)
    axis = A4ExpertAxis(2)
    axis.put(0, _unit(seed=0))
    snapshot = {field: axis._planes[field][0].clone() for field in axis._FIELDS}
    # A larger plane through the same scratch, then a second expert.
    _plane_words(plane, scratch)
    _plane_words(plane[:7], scratch)
    _plane_u8(bytes(range(255)) * 8, "cuda", scratch, "body")
    axis.put(1, _unit(seed=1))
    for field in axis._FIELDS:
        assert bool(torch.equal(snapshot[field], axis._planes[field][0])), \
            f"{field}: scratch reuse mutated an expert already written"
