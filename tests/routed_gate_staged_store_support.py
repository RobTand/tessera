"""Shared staged-store readers and closed replay-admission inputs."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace


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


def reader_class():
    path = ROOT / 'experiments/t8r_speed/declared_inputs.py'
    spec = importlib.util.spec_from_file_location('declared_inputs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeclaredInputs


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
    constants = {n.targets[0].id: n.value.value for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
                 and isinstance(n.value.value, str)}
    artifacts = {n.comparators[0].value if isinstance(n.comparators[0], ast.Constant)
                 else constants[n.comparators[0].id] for n in ast.walk(owner)
                 if isinstance(n, ast.Compare) and argument(n.left, 'artifact')
                 and len(n.comparators) == 1
                 and ((isinstance(n.comparators[0], ast.Constant)
                       and isinstance(n.comparators[0].value, str))
                      or (isinstance(n.comparators[0], ast.Name)
                          and n.comparators[0].id == 'REPLAY_ARTIFACT'))}
    native_names = [n.comparators[0].id for n in ast.walk(owner)
                    if isinstance(n, ast.Compare) and argument(n.left, 'profile_native_file')
                    and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Name)]
    assert len(artifacts) == len(native_names) == 1, 'closed replay owner changed'
    natives = [n.value.value for n in ast.walk(owner)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == native_names[0]
               and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    assert len(natives) == 1, 'closed native owner changed'
    return {'artifact': next(iter(artifacts)), 'native': natives[0]}


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
    scope = {'ARTIFACT': '/legacy', 'REPLAY_ARTIFACT': canonical_replay_paths()['artifact']}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    scope[node.name](args, **kwargs)


def real_roster():
    return [{'expert':e,'role':r,'tensor':
             f'model.language_model.layers.10.mlp.experts.{e}.{r}.weight'}
            for e in range(288) for r in ('gate_proj','up_proj','down_proj')]
