"""Selected kernel reads existing prepared bytes, including shard pads."""
import subprocess
import sys

import pytest
import torch

from tessera.serving.window import PreparedWindow, prepare_window

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='selected window kernel requires CUDA')


def _batch(dtype, window_bits=14, steps=35, rates=(8, 1, 7, 2, 6, 3, 5, 4) * 5):
    rates = tuple(min(r, window_bits) for r in rates)
    windows = []
    for expert in range(4):
        generator = torch.Generator().manual_seed(135 + expert)
        body = torch.stack([torch.randint(1 << r, (steps,), generator=generator)
                            for r in rates], 1).to(torch.uint8)
        table = torch.randint(256, (1 << window_bits,), generator=generator).to(dtype)
        if dtype.is_floating_point:
            table = (table / 32 - 4).to(dtype)
        initial = torch.randint(1 << window_bits, (len(rates),), generator=generator)
        windows.append(prepare_window(body, rates, window_bits, table, 'cuda', initial_state=initial))
    return PreparedWindow.stack(windows)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('window_bits', [4, 14, 20])
def test_fused_selection_equals_torch_across_rates_tables_and_shard_pads(dtype, window_bits):
    batch = _batch(dtype, window_bits)
    # Strided int32 IDs, permutations, repeated experts, all experts, and empty.
    for values in ([3, 0, 2, 3, 1], [0, 1, 2, 3], []):
        ids = torch.tensor([v for value in values for v in (value, 0)],
                           device='cuda', dtype=torch.int32)[::2]
        expected = batch.decode(ids, max_experts_per_chunk=2)
        for chunk in (1, 3, 8):
            actual = batch.decode(ids, max_experts_per_chunk=chunk, backend='triton')
            assert actual.dtype == dtype
            assert actual.is_contiguous()
            assert torch.equal(actual, expected)


def test_fused_scratch_is_output_plus_selection_metadata():
    batch = _batch(torch.uint8, steps=513, rates=(2, 3) * 65)
    ids = torch.tensor([3, 0, 2, 3, 1] * 7, device='cuda')
    expected = batch.decode(ids, max_experts_per_chunk=3)
    warm = batch.decode(ids, max_experts_per_chunk=3, backend='triton')
    assert torch.equal(warm, expected)
    del warm
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    requested_before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    torch.cuda.reset_peak_memory_stats()
    actual = batch.decode(ids, max_experts_per_chunk=3, backend='triton')
    torch.cuda.synchronize()
    peak = torch.cuda.memory_stats()["requested_bytes.all.peak"] - requested_before
    # Requested bytes exclude allocator reuse/rounding of oversized blocks.
    # The bound excludes full-sized integer gathers and chunk output copies.
    assert peak <= actual.numel() * actual.element_size() + 64 * 1024
    assert torch.equal(actual, expected)
    del actual
    assert torch.cuda.memory_allocated() == before


@pytest.mark.parametrize('invalid', [-1, 4, 2**62])
def test_invalid_ids_fail_before_any_out_of_bounds_read(invalid):
    # A CUDA device assertion poisons its context, so isolate this refusal.
    code = f'''
import torch
from tessera.serving.window import PreparedWindow, prepare_window
w = prepare_window(torch.zeros((3, 2), dtype=torch.uint8), [2, 2], 4,
                   torch.arange(16, dtype=torch.uint8), 'cuda')
b = PreparedWindow.stack([w] * 4)
b.decode(torch.tensor([{invalid}], device='cuda'), max_experts_per_chunk=1, backend='triton')
torch.cuda.synchronize()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'device-side assert' in result.stderr or 'expert IDs out of range' in result.stderr


def test_fp8_batch_forwards_backend_and_preserves_role_bytes_and_scales():
    from tessera.serving.fp8_route import PreparedTesseraFp8Batch
    windows = [_batch(torch.uint8, steps=35), _batch(torch.uint8, steps=35)]
    scales = torch.arange(4 * 70, device='cuda', dtype=torch.float32).reshape(4, 70)
    batch = PreparedTesseraFp8Batch(windows, scales, ('gate', 'up'), 70, 40, scales.device)
    ids = torch.tensor([3, 1, 3, 0], device='cuda')
    assert torch.equal(batch.decode(ids, max_experts_per_chunk=2, backend='triton'),
                       batch.decode(ids, max_experts_per_chunk=2, backend='torch'))
    assert torch.equal(batch.row_scale(ids), scales.index_select(0, ids))
