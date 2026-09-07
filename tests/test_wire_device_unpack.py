"""Device BODY unpacking preserves the CPU byte oracle and canonical refusals."""
import pytest
import torch
from tessera import wire
from tessera.errors import GrammarError

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='BODY device unpack requires CUDA')

@cuda
def test_cuda_body_does_not_reconstruct_fields_on_cpu(monkeypatch):
    blob = bytes([0x1b, 0xe4])
    expected = wire.unpack_body(blob, (2, 2), 4)
    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA BODY called CPU field reconstruction')
    monkeypatch.setattr(wire, '_from_bits', forbidden)
    actual = wire.unpack_body(blob, (2, 2), 4, 'cuda')
    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected)

@cuda
@pytest.mark.parametrize('span', [1, 2, 3, 8])
@pytest.mark.parametrize('rows_per_span', [0, 1, 17, 257])
def test_cuda_body_matches_cpu_for_mixed_unaligned_fields(span, rows_per_span):
    rates = tuple(range(1, 9 if span == 1 else 8)) + ((0,) if span == 1 else ())
    rows = span * rows_per_span
    values = torch.empty((rows, len(rates)), dtype=torch.int64)
    for j, rate in enumerate(rates):
        widths = wire.field_widths(rate, span)
        for p, width in enumerate(widths):
            values[p::span, j] = torch.arange(rows_per_span) * 13 % (1 << width)
    blob = wire.pack_body(values, rates, span)
    expected = wire.unpack_body(blob, rates, rows, span=span)
    actual = wire.unpack_body(blob, rates, rows, 'cuda', span)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.cpu(), expected)

@cuda
@pytest.mark.parametrize('blob,rates,rows,span', [(b'\xff',(3,),1,1),(b'',(3,),1,1),(b'\x00\x01',(1,),1,1),(b'',(1,),3,2)])
def test_cuda_body_keeps_canonical_refusals(blob, rates, rows, span):
    with pytest.raises(GrammarError) as cpu:
        wire.unpack_body(blob, rates, rows, span=span)
    with pytest.raises(GrammarError) as gpu:
        wire.unpack_body(blob, rates, rows, 'cuda', span)
    assert str(cpu.value) == str(gpu.value)

@cuda
def test_cuda_body_remains_available_without_optional_triton(monkeypatch):
    import sys
    # Initialise torch's CUDA context before blocking the optional kernel import.
    torch.empty(1, device='cuda')
    blob = bytes([0x1b, 0xe4])
    expected = wire.unpack_body(blob, (2, 2), 4)
    monkeypatch.delitem(sys.modules, 'tessera.kernel_wire', raising=False)
    monkeypatch.setitem(sys.modules, 'triton', None)
    actual = wire.unpack_body(blob, (2, 2), 4, 'cuda')
    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected)
