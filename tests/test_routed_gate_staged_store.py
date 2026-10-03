"""The exact replay Store consumes owned staged bytes, never original paths."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def store_class():
    path = ROOT / 'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Store')
    scope = {'os': __import__('os'), 'json': __import__('json'), 'safe_open': forbidden_origin}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['Store']


def forbidden_origin(*args, **kwargs):
    raise PermissionError('origin read forbidden by regression tripwire')


def test_existing_store_uses_owned_wire_under_origin_denial():
    cls = store_class()
    store = cls.__new__(cls)
    store.root = '/forbidden-origin'
    store.index = {'wire': 'shard.safetensors'}
    store._open = {}
    store.inputs = type('Inputs', (), {'tensor': lambda self, root, key, **kwargs: b'owned-wire'})()
    assert store.get('wire') == b'owned-wire'


def test_existing_store_uses_owned_metadata_under_origin_denial(monkeypatch):
    monkeypatch.setattr('builtins.open', forbidden_origin)
    cls = store_class()
    store = cls.__new__(cls)
    store.root = '/forbidden-origin'
    store.inputs = type('Inputs', (), {'json': lambda self, path: {'config_groups': {}}})()
    assert store.metadata('config.json') == {'config_groups': {}}


def test_legacy_store_keeps_origin_behavior():
    cls = store_class()
    store = cls.__new__(cls)
    store.root = '/forbidden-origin'
    store.index = {'wire': 'shard.safetensors'}
    store._open = {}
    store.inputs = None
    with pytest.raises(PermissionError, match='origin read forbidden'):
        store.get('wire')


def reader_class():
    path = ROOT / 'experiments/t8r_speed/pb_staged_store.py'
    spec = importlib.util.spec_from_file_location('pb_staged_store', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StagedInputs
















def canonical_replay_paths():
    """Read the closed input authority used by the unchanged real validator.

    These values are passed only to the pure admission function; no box path
    is copied into a fixture or opened. Ambiguous source grammar fails here.
    """
    tree = ast.parse((ROOT / 'experiments/t8r_speed/bench_t8r.py').read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'require_single_replay_options')
    def argument(n, name):
        return (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == 'args' and n.attr == name)
    artifacts = [n.comparators[0].value for n in ast.walk(owner)
                 if isinstance(n, ast.Compare) and argument(n.left, 'artifact')
                 and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Constant)
                 and isinstance(n.comparators[0].value, str)]
    native_names = [n.comparators[0].id for n in ast.walk(owner)
                    if isinstance(n, ast.Compare) and argument(n.left, 'profile_native_file')
                    and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Name)]
    assert len(artifacts) == len(native_names) == 1, 'closed replay owner changed'
    natives = [n.value.value for n in ast.walk(owner)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == native_names[0]
               and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    assert len(natives) == 1, 'closed native owner changed'
    return {'artifact': artifacts[0], 'native': natives[0]}


def options():
    return SimpleNamespace(single_routing_file='recorded.pt',
        groups='experts.R1024.L10', ms='2048', input_manifest='sealed.json',
        routing=None, no_graph=True, power_s=30, warmup=10, iters=30,
        artifact=canonical_replay_paths()['artifact'])


def require_options(args, **kwargs):
    path = ROOT / 'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'require_single_replay_options')
    scope = {'ARTIFACT': '/legacy'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    scope[node.name](args, **kwargs)


def test_exact_single_options_admit():
    require_options(options())


@pytest.mark.parametrize('field,value', [('groups','all'), ('ms','512,2048'),
    ('input_manifest',None), ('routing','unsealed-dir'), ('no_graph',False),
    ('power_s',2), ('artifact','older-T8R-artifact')])
def test_single_options_refuse_menu_or_unqualified_inputs(field, value):
    args = options()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        require_options(args)


def test_single_options_refuse_stubbed_runtime():
    with pytest.raises(ValueError, match='stubbed'):
        require_options(options(), stubbed=True)


def test_main_releases_pin_if_metadata_admission_fails(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    path = ROOT / 'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    args = options()
    args.out = str(tmp_path/'out')
    closed = []
    reader = SimpleNamespace(close=lambda: closed.append(True))
    module = ModuleType('pb_staged_store')
    module.StagedInputs = lambda p: reader
    monkeypatch.setitem(sys.modules, 'pb_staged_store', module)
    parser = SimpleNamespace(add_argument=lambda *a,**k: None, parse_args=lambda: args)
    def failed_store(*args):
        raise ValueError('metadata admission refused')
    scope = {'argparse':SimpleNamespace(ArgumentParser=lambda:parser),
        'ARTIFACT':'/legacy', 'VLLM_STUBBED':False,
        'require_single_replay_options':lambda *a,**k:None,
        'os':os, 'torch':SimpleNamespace(manual_seed=lambda *a:None,device=lambda *a:None),
        'Store':failed_store}
    exec(compile(ast.Module(body=[node], type_ignores=[]),str(path),'exec'),scope)
    with pytest.raises(ValueError,match='metadata admission refused'):
        scope['main']()
    assert closed == [True]










def real_roster():
    return [{'expert':e,'role':r,'tensor':
             f'model.language_model.layers.10.mlp.experts.{e}.{r}.weight'}
            for e in range(288) for r in ('gate_proj','up_proj','down_proj')]


def test_actual_publisher_projection_names_bind_complete_l10_roster():
    cls = reader_class()
    reader = cls.__new__(cls)
    reader.bind_roles('/unused',real_roster(), module='model.language_model.layers.10.mlp.experts')
    assert len(reader.roles) == 864
    assert 'model.language_model.layers.10.mlp.experts.0.gate_proj.wire' in reader.roles


@pytest.mark.parametrize('fault',['missing','duplicate','foreign'])
def test_incomplete_or_foreign_roster_refuses(fault):
    roles=real_roster()
    if fault=='missing': roles.pop()
    elif fault=='duplicate': roles[-1]=roles[0]
    else: roles[-1]['tensor']='foreign.287.down_proj.weight'
    cls=reader_class(); reader=cls.__new__(cls)
    with pytest.raises(ValueError,match='roster'):
        reader.bind_roles('/unused',roles, module='model.language_model.layers.10.mlp.experts')






@pytest.mark.parametrize('ncu,path',[(False,'canonical'),(True,'/foreign/module.so')])
def test_native_input_requires_closed_counter_mode(ncu,path):
    args=options();args.ncu=ncu;args.profile_native_file=canonical_replay_paths()['native'] if path=='canonical' else path
    with pytest.raises(ValueError,match='counter-only'):
        require_options(args)

