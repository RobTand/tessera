"""The prepared module's fingerprint guard must not break a compiled forward.

vLLM's default serve traces the model forward with Dynamo; the streamed mode
decodes inside it, and a ``data_ptr`` comparison is untraceable
(``torch._dynamo.exc.Unsupported: Data pointer comparison``, seen at engine
start on 2026-09-02).  Eager calls keep the check.

Ported from Gridbook's ``test_tessera_ops_compile.py``.  The sibling test for
``trellis_ops.PreparedTrellisWireCuda`` is gone with that class: this package
has one prepared module, not two.
"""
import pytest
import torch

from tessera.serving.ops import PreparedTesseraModule, _PreparedRole


def _module():
    planes = tuple(torch.zeros(4, 4, dtype=torch.uint8) for _ in range(3))
    role = _PreparedRole("test", 0, planes, {"rows": 8, "cols": 32})
    return PreparedTesseraModule([role], rows=8, columns=32, global_scale=1.0,
                                 device=torch.device("cpu"), body="TCQ")


def test_eager_guard_refuses_a_replaced_plane():
    m = _module()
    m._require_unchanged()
    role = m._PreparedTesseraModule__roles[0]
    role.planes = (torch.zeros(4, 4, dtype=torch.uint8),) + role.planes[1:]
    with pytest.raises(RuntimeError, match="changed after preparation"):
        m._require_unchanged()


def test_compiled_forward_traces_through_the_guard():
    m = _module()

    def _forward(x):
        m._require_unchanged()
        return x + 1

    compiled = torch.compile(_forward, fullgraph=True)
    assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 2.0))


def test_decode_is_a_custom_op_the_compiled_forward_traces(monkeypatch):
    """The forward's decode is ``tessera::nvfp4_decode_module``, a functional
    custom op: opaque to Dynamo (a pybind symbol is a function it marks as
    skipped), owning the tile it returns (so no aliased graph input is
    mutated, the thing that broke the streamed compiled serve), executed at
    call time.  The native symbol sits behind ``_decode_impl``; a recorder
    stands in for it so the trace can be exercised on any device.  The served
    receipt is the CUDA check."""
    from tessera.serving import ops

    calls = []

    def _record(*args):
        calls.append(args)

    monkeypatch.setattr(ops, "_decode_impl", _record)
    planes = tuple(torch.zeros(4, 4, dtype=torch.uint8) for _ in range(7))
    role = _PreparedRole("q", 0, planes, {"rows": 8, "cols": 32, "rate": 7, "arity": 2,
                                          "memory": 6, "half": 16})
    role2 = _PreparedRole("k", 8, planes, {"rows": 4, "cols": 32, "rate": 7, "arity": 2,
                                           "memory": 6, "half": 16})
    m = PreparedTesseraModule([role, role2], rows=12, columns=32, global_scale=1.0,
                              device=torch.device("cpu"), body="TCQ")

    def _forward(x):
        packed, scales = m.decode()
        return x + packed[0, 0].to(x.dtype) + scales[0, 0].to(x.dtype)

    compiled = torch.compile(_forward, fullgraph=True)
    compiled(torch.ones(2))
    assert len(calls) == 2, "one native call per role"
    assert calls[0][7:13] == (8, 32, 7, 2, 6, 16) and calls[1][7] == 4
    assert tuple(calls[0][13].shape) == (8, 16) and tuple(calls[1][13].shape) == (4, 16)
    assert tuple(calls[0][14].shape) == (8, 2)
    # the two roles' slices are row slices of ONE fresh tile each
    assert calls[0][13].untyped_storage().data_ptr() == calls[1][13].untyped_storage().data_ptr()


def test_the_decode_ops_live_in_the_tessera_namespace():
    """The op namespace moved with the code: ``prismaquant::`` was Gridbook's
    producer, and a Tessera serve must not register under it."""
    assert hasattr(torch.ops.tessera, "nvfp4_decode_module")
    assert hasattr(torch.ops.tessera, "nvfp4_decode_span2_out")


def test_decode_out_still_serves_a_caller_that_brings_its_own_tile(monkeypatch):
    from tessera.serving import ops

    calls = []
    monkeypatch.setattr(ops, "_decode_impl", lambda *a: calls.append(a))
    planes = tuple(torch.zeros(4, 4, dtype=torch.uint8) for _ in range(7))
    role = _PreparedRole("test", 0, planes, {"rows": 8, "cols": 32, "rate": 7, "arity": 2,
                                             "memory": 6, "half": 16})
    m = PreparedTesseraModule([role], rows=8, columns=32, global_scale=1.0,
                              device=torch.device("cpu"), body="TCQ")
    packed, scales = m.empty_tile()
    m.decode_out(packed, scales)
    assert len(calls) == 1 and calls[0][7:9] == (8, 32)
