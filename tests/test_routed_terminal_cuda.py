"""Finite #855 native termination coverage; submit explicitly through PB.

The native entry is called even at M=0, so the empty-work case exercises
terminal CTAs instead of a Python fast return. Work counters independently
bind completed item claims plus one terminal claim per launched CTA.
"""
import json
import os
from pathlib import Path
import uuid

import pytest
import torch

from tessera import routed_fused as rf
from tessera import routed_fused_e2m1 as fe
from tessera.kernel_a4 import a4_quantize_activation

import test_dense_fused_window as dense
import test_routed_fused_e2m1 as fp4


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='native terminal CUDA coverage')

CASES = [
    pytest.param(0, 4, 1, 3, id='zero-work'),
    pytest.param(1, 4, 1, 1, id='single-item'),
    pytest.param(1, 4, 1, 4, id='idle-ctas'),
    pytest.param(1, 4, 2, 1, id='minimum-split'),
    pytest.param(1, 6, 2, 3, id='odd-split'),
    pytest.param(129, 5, 1, 2, id='mixed-odd-even'),
]


def record_loaded_native(lib, library):
    from tessera._dev.native_identity import loaded_native_identity

    source = Path(rf.__file__).parent / 'serving/csrc/routed_fused_window.cu'
    identity = dict(library=library, **loaded_native_identity(lib, source))
    directory = os.environ.get('TERMINAL_NATIVE_IDENTITY_DIR')
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / (uuid.uuid4().hex + '.json')).open('x') as handle:
            json.dump(identity, handle)
    print('TERMINAL_NATIVE_IDENTITY ' + json.dumps(identity), flush=True)


@pytest.mark.parametrize('library', ['value', 'e4m3', 'e4m3mma', 'e2m1'])
@pytest.mark.parametrize('m,chunks,split,grid', CASES)
def test_dense_terminal_ctas(monkeypatch, library, m, chunks, split, grid):
    """Small K, both parities, empty CTAs, repeat bits and independent decode."""
    is_fp4 = library == 'e2m1'
    cols = chunks * (64 if is_fp4 else rf.BK)
    rows = 256 if is_fp4 else rf.BN
    x = torch.zeros((m, cols), dtype=torch.bfloat16, device='cuda')
    if m:
        x[torch.arange(m, device='cuda'), torch.arange(m, device='cuda') % cols] = 1
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    out = torch.empty((m, rows), dtype=torch.bfloat16, device='cuda')
    empty = torch.empty(0, dtype=torch.float32, device='cuda')
    partial = torch.empty((split, m, rows), dtype=torch.float32, device='cuda') if split > 1 else empty
    if is_fp4:
        blob = fp4._encode(rows, cols, 512, 855)
        gs = 448.0 * 6.0
        role = fe.prepare_dense_role(fp4._unit(blob), gs)
        lib = fe._ext()
        if m:
            codes, scales = a4_quantize_activation(x, role.gs)
        else:
            codes = torch.empty((0, cols // 2), dtype=torch.uint8, device='cuda')
            scales = torch.empty((0, cols // 16), dtype=torch.uint8, device='cuda')

        def launch():
            lib.dense_forward_fp4(codes.contiguous(), scales.view(torch.uint8).contiguous(),
                role.words, role.codes, role.init, role.has_init, role.plane, role.lut,
                role.ratio, role.runs, role.desc, role.rows, role.tile_words,
                role.slot_words, counter, split, partial, out, grid)
    else:
        monkeypatch.setenv(rf.ENV_E4M3_MMA, 'e4m3' if library == 'e4m3mma' else 'f16')
        family = 'value' if library == 'value' else 'e4m3'
        expert, bundle = dense._role(family, rows=rows, cols=cols, seed=855)
        role = rf.prepare_dense_role(bundle)
        assert role.library == library
        lib = rf._ext(library)
        xq, scale = dense._quant(x) if family == 'e4m3' and m else (x, None)
        if family == 'e4m3' and not m:
            xq = x.to(torch.float8_e4m3fn)
            scale = empty
        if scale is not None:
            scale = scale.reshape(-1).contiguous().float()

        def launch():
            lib.dense_forward(role.fp8, xq.contiguous(), scale if scale is not None else empty,
                role.words, role.table16, role.init, role.has_init, role.wscale,
                role.runs, role.bdesc, role.tile_words, role.slot_words,
                counter, split, partial, out, grid, rf.BM)

    record_loaded_native(lib, library)
    previous = None
    items = ((m + rf.BM - 1) // rf.BM) * split
    for repeat in range(2):
        counter.zero_()
        out.fill_(float('nan'))
        launch()
        torch.cuda.synchronize()
        assert int(counter.item()) == items + grid, (library, m, chunks, split, grid)
        assert bool(torch.isfinite(out).all())
        if previous is not None:
            assert torch.equal(out.view(torch.int16), previous.view(torch.int16))
        previous = out.clone()
    if m and is_fp4:
        a = fp4._a_deq(x, gs)
        weight = fp4._ref_weight(blob, None)
        ref = (a @ weight.T).float().mul(role.ratio[0]).double()
        fp4._check(out, ref, torch.zeros_like(ref), True, 'terminal FP4 onehot')
    elif m:
        dense._within(out, dense._bound(expert, family, xq, scale, split), 'terminal dense')
