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
    tree = ast.parse(path.read_text())
    wanted = {'build_routed', '_legacy_uniform_scheme_metadata', '_check_routed_runs_against_classes'}
    scope = {'torch': torch, 'os': os, 'hashlib': hashlib, 'TOP_K': 2, 'TP_RANK': 0, 'TP_SIZE': 1,
             'SWIGLU_LIMIT': 10.0, **constants}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    # Pre-fix files carry no helpers: the old tests then exercise the old
    # behaviour, and the new tests fail on the missing derivation.
    scope.setdefault('_legacy_uniform_scheme_metadata', lambda s, m: s)
    scope.setdefault('_check_routed_runs_against_classes', lambda *a: None)
    return scope['build_routed']


def _legacy_scheme(q13=1024, q2=1024, experts=3):
    return {'family': 'e4m3', 'experts': experts,
            'groups': {'w13': {'q256': q13, 'roles': [['gate_proj', 2], ['up_proj', 2]]},
                       'w2': {'q256': q2, 'roles': [['down_proj', 4]]}}}


def _runs_bundle(torch, runs, experts=3, cols=256, width=100):
    runs = torch.tensor(runs, dtype=torch.int32)
    assert runs.shape[0] == experts
    return SimpleNamespace(word_layout='legacy', runs_all=runs,
                           word_off=torch.arange(experts, dtype=torch.int32) * width,
                           experts=experts, cols=cols)


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


def _stub_stack(monkeypatch, torch, seen, runs, experts=3):
    from tessera.serving import moe_route, scheme

    def validate(s, m):
        seen['scheme'] = s
        return {'expert_ids': s.get('expert_ids'), 'expert_classes': s.get('expert_classes')}

    monkeypatch.setattr(scheme, 'validate_tessera_moe_scheme', validate)

    class Native:
        library = None

        def __call__(self, x, ids, w, **unused):
            return x

        def gate_up(self, x, ids, w):
            return x

        def adapter(self):
            return self

    class Packed:
        device = torch.device('cpu')

        def __init__(self):
            self.gate = _runs_bundle(torch, runs, experts=experts)
            self.up = _runs_bundle(torch, runs, experts=experts)
            self.down = _runs_bundle(torch, runs, experts=experts)

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


def test_legacy_classless_scheme_reaches_validate_with_identity_classes(monkeypatch):
    torch = pytest.importorskip('torch')
    seen = {}
    runs = [[[4, 0, 256, 0]]] * 3
    _stub_stack(monkeypatch, torch, seen, runs)
    store = SimpleNamespace(schemes={'m': _legacy_scheme()}, get=lambda name: torch.zeros(4, dtype=torch.uint8))
    builder(EXPERTS=3)(store, 'm')
    got = seen['scheme']
    assert got.get('expert_ids') == [0, 1, 2]
    assert [(c['start'], c['end']) for c in got.get('expert_classes', [])] == [(0, 3)]


def test_half_present_expert_metadata_refuses(monkeypatch):
    pytest.importorskip('torch')
    legacy = _legacy_scheme()
    legacy['expert_ids'] = [0, 1, 2]
    store = SimpleNamespace(schemes={'m': legacy}, get=lambda name: None)
    with pytest.raises(ValueError, match='come as a pair'):
        builder(EXPERTS=3)(store, 'm')


def test_loaded_runs_must_match_the_asserted_rungs(monkeypatch):
    torch = pytest.importorskip('torch')
    seen = {}
    declared = {'expert_ids': [0, 1],
                'expert_classes': [{'start': 0, 'end': 2,
                                    'q256': {'w13': [1088, 1088], 'w2': [1088]}}]}
    same = [[[4, 0, 192, 0], [5, 192, 64, 12288]]] * 2
    _stub_stack(monkeypatch, torch, seen, same, experts=2)
    from tessera.serving import scheme
    monkeypatch.setattr(scheme, 'validate_tessera_moe_scheme', lambda s, m: declared)
    store = SimpleNamespace(schemes={'m': _legacy_scheme(1088, 1088, experts=2)},
                            get=lambda name: torch.zeros(4, dtype=torch.uint8))
    builder(EXPERTS=2)(store, 'm')
    off = [[[4, 0, 256, 0]]] * 2
    _stub_stack(monkeypatch, torch, seen, off, experts=2)
    with pytest.raises(ValueError, match='run tables sum'):
        builder(EXPERTS=2)(store, 'm')
