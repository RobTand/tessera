"""``regdirect_routed.layer_stacks`` reads the class build's construction planes exactly.

Bundles are laid out as the routed loader lays them (tile-order words of the documented layout,
rate-sorted permutation, runs, permuted start states, composed byte tables).  The fragment stacks
built from them must decode, through the fragment planes alone, to the window replay of the
original codes: mixed R3/R4 k-steps, one expert with a TP-cut start state.  CPU only.
"""
from __future__ import annotations

import os
import sys

import torch

from tessera import regdirect_routed as rr
from tessera.decode import replay_window
from tessera.window_gemm_grouped import PreparedGroupedWindowGemm
from window_pack_reference import pack_bitstream

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "regdirect_stage1"))
from fragment_synth import reference_decode  # noqa: E402

E = 3
SHAPES = {"gate": (256, 256), "up": (256, 256), "down": (512, 128)}


def _rates(cols, block, seed):
    g = torch.Generator().manual_seed(seed)
    per = [3 + int(torch.randint(0, 2, (1,), generator=g)) for _ in range(cols // block)]
    return tuple(r for r in per for _ in range(block))


def _bundle(name, rates_per_expert, seed):
    rows, cols = SHAPES[name]
    g = torch.Generator().manual_seed(seed)
    reps, codes, starts, has = [], [], [], []
    for e, rates in enumerate(rates_per_expert):
        body = torch.stack([torch.randint(0, 1 << r, (rows,), generator=g) for r in rates], 1)
        reps.append(pack_bitstream(body, rates))
        codes.append(body)
        start = torch.randint(0, 1 << 14, (cols,), generator=g) if e == 1 else torch.zeros(cols, dtype=torch.int64)
        starts.append(start)
        has.append(int(e == 1))
    word_off = torch.tensor([0] + [r.words.numel() for r in reps][:-1]).cumsum(0).to(torch.int32)
    run_len = torch.tensor([r.runs.reshape(-1, 4).shape[0] for r in reps])
    run_off = torch.cat([torch.zeros(1, dtype=torch.int64), run_len.cumsum(0)]).to(torch.int32)
    native = torch.randint(0, 256, (E, 256), generator=g, dtype=torch.int64)
    native[(native & 127) == 127] = 126
    bundle = PreparedGroupedWindowGemm(
        words_all=torch.cat([r.words for r in reps]), table_all=None,
        codes_all=torch.randint(0, 256, (E, 1 << 14), generator=g, dtype=torch.int64).to(torch.uint8),
        native_all=native.to(torch.uint8), scale_all=torch.rand(E, rows, generator=g),
        runs_all=torch.cat([r.runs.reshape(-1, 4) for r in reps]),
        init_all=torch.stack([s[r.perm.long()] for s, r in zip(starts, reps)]).to(torch.int32),
        has_init=torch.tensor(has, dtype=torch.int32), word_off=word_off,
        tile_words=torch.tensor([r.tile_words for r in reps], dtype=torch.int32),
        total_words=torch.tensor([r.words.numel() for r in reps], dtype=torch.int32), run_off=run_off,
        perm_all=torch.stack([r.perm for r in reps]), rows=rows, cols=cols, experts=E, window_bits=14,
        family="e4m3", block_m=64, block_n=64, block_k=64, arithmetic="epilogue")
    return bundle, codes, starts


def _table(bundle):
    return torch.gather(bundle.native_all, 1, bundle.codes_all.long()).contiguous()


def _expected(table, codes, rates, start):
    out = torch.empty(codes.shape, dtype=torch.uint8)
    for rate in sorted(set(rates)):
        cols = torch.tensor([j for j, r in enumerate(rates) if r == rate])
        state = replay_window(codes[:, cols], 14, rate, start[cols])
        out[:, cols] = table[state]
    return out


def test_layer_stacks_decode_to_the_window_replay_of_the_bundle_codes():
    gu_rates = [_rates(256, 32, 10 + e) for e in range(E)]
    down_rates = [_rates(128, 64, 20 + e) for e in range(E)]
    (gate, gc, gs), (up, uc, us) = _bundle("gate", gu_rates, 1), _bundle("up", gu_rates, 2)
    down, dc, ds = _bundle("down", down_rates, 3)
    tables = tuple(_table(b) for b in (gate, up, down))
    stacks = rr.layer_stacks(gate, up, down, tables)
    for e in range(E):
        got = reference_decode(stacks[0], e)
        assert torch.equal(got[0], _expected(tables[0][e], gc[e], gu_rates[e], gs[e]))
        assert torch.equal(got[1], _expected(tables[1][e], uc[e], gu_rates[e], us[e]))
        got = reference_decode(stacks[2], e)
        assert torch.equal(got[0], _expected(tables[2][e], dc[e], down_rates[e], ds[e]))
    assert torch.equal(stacks[0]["wscale"][1, 0], gate.scale_all[1])


def test_layer_stacks_refuse_a_table_the_register_direct_kernel_does_not_read():
    import pytest
    from tessera.errors import GrammarError
    rates = [_rates(256, 32, 10 + e) for e in range(E)]
    gate, up = _bundle("gate", rates, 1)[0], _bundle("up", rates, 2)[0]
    down = _bundle("down", [_rates(128, 64, 20 + e) for e in range(E)], 3)[0]
    f16 = tuple(torch.zeros(E, 1 << 14, dtype=torch.int16) for _ in range(3))
    with pytest.raises(GrammarError, match="compose_table8"):
        rr.layer_stacks(gate, up, down, f16)
