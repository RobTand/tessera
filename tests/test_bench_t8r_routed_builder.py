"""The T8R benchmark's routed builder serves router IDs in storage order.

CPU control flow only: a recording adapter stands in for the native owner,
so no CUDA arithmetic or timing is claimed."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def builder(**constants):
    torch = pytest.importorskip('torch')
    path = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/bench_t8r.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == 'build_routed')
    scope = {'torch': torch, 'os': os, 'hashlib': hashlib, 'TOP_K': 2, 'TP_RANK': 0, 'TP_SIZE': 1,
             'SWIGLU_LIMIT': 10.0, **constants}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['build_routed']


def test_routed_builder_maps_router_ids_through_the_storage_inverse(monkeypatch):
    torch = pytest.importorskip('torch')
    from tessera.serving import moe_route, scheme

    # Storage s holds global expert expert_ids[s]: a nonidentity class sort.
    declared = {'expert_ids': [1, 2, 0]}
    monkeypatch.setattr(scheme, 'validate_tessera_moe_scheme', lambda s, m: declared)
    calls = []

    class Native:
        library = None

        def __call__(self, x, ids, w, **unused):
            calls.append(('forward', ids.clone(), w.clone()))
            return x

        def gate_up(self, x, ids, w):
            calls.append(('gate_up', ids.clone(), w.clone()))
            return x

    class Packed:
        device = torch.device('cpu')
        gate = up = down = SimpleNamespace(word_layout='legacy')

        def adapter(self):
            return Native()

    class Intake:
        def __init__(self, *unused):
            pass

        def load(self, *unused, **unused_kw):
            pass

        def finish(self, *unused):
            return Packed()

    monkeypatch.setattr(moe_route, '_RankLocalPackedIntake', Intake)
    store = SimpleNamespace(schemes={'m': {'family': 'e4m3', 'groups': {}}},
                            get=lambda name: torch.zeros(4, dtype=torch.uint8))
    fn, _info, _packed, _touched = builder(EXPERTS=3)(store, 'm')
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    storage = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    fn(x, ids, weights)
    assert calls[-1][0] == 'forward' and torch.equal(calls[-1][1], storage)
    fn.gate_up(x, ids, weights)
    assert [c[0] for c in calls] == ['forward', 'gate_up']
    for _entry, got, got_weights in calls:
        assert torch.equal(got, storage)
        assert torch.equal(got_weights, weights)
