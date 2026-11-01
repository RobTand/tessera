"""Exercise the production main parser/validator before its first device access."""
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from box_artifacts import ROOTS

ROOT = Path(__file__).resolve().parents[1]
A8SE = str(Path(ROOTS["shared_runs"].default) /
           "moe/glm53-a8-bf16menu-20260930/release/exported")

@pytest.fixture(autouse=True)
def preserve_runtime_mode(monkeypatch):
    monkeypatch.setenv("TESSERA_SERVE_MODE", os.environ.get("TESSERA_SERVE_MODE", "resident"))



class DeviceBoundaryReached(Exception):
    pass


def main_scope(*, stubbed=False):
    path = ROOT / 'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in ('main', 'require_single_replay_options'))
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id in ('ARTIFACT', 'REPLAY_ARTIFACT') for t in n.targets))]
    def device(*args):
        raise DeviceBoundaryReached('argument admission reached actual main device boundary')
    scope = dict(argparse=argparse, os=os, _install_vllm_stubs=lambda: stubbed,
                 torch=SimpleNamespace(manual_seed=lambda seed: None, device=device))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def options(tmp_path):
    return ['--out', str(tmp_path / 'out'), '--artifact', A8SE,
            '--groups', 'experts.R1024.L10', '--ms', '1,2048',
            '--no-graph', '--outputs-only']


def test_exact_output_screen_reaches_main_device_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', *options(tmp_path)])
    with pytest.raises(DeviceBoundaryReached):
        main_scope()['main']()


@pytest.mark.parametrize('field,value', [
    ('--artifact', '/other-artifact'), ('--groups', 'all'), ('--groups', 'experts.R1024.L11'),
    ('--ms', '2048'), ('--ms', '1,2048,2049'), ('--routing', 'routing-dir'),
    ('--single-routing-file', 'routing.pt'), ('--input-manifest', 'readset.json'),
    ('--profile-native-file', 'native.so'),
])
def test_other_screen_inputs_refuse_before_device(monkeypatch, tmp_path, field, value):
    argv = options(tmp_path)
    if field in argv:
        argv[argv.index(field) + 1] = value
    else:
        argv += [field, value]
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', *argv])
    with pytest.raises(ValueError):
        main_scope()['main']()


@pytest.mark.parametrize('remove', ['--outputs-only', '--no-graph'])
def test_timing_or_graph_cannot_use_screen_artifact_override(monkeypatch, tmp_path, remove):
    argv = options(tmp_path)
    argv.remove(remove)
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', *argv])
    with pytest.raises(ValueError):
        main_scope()['main']()


def test_profile_cannot_use_output_screen(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', *options(tmp_path), '--ncu'])
    with pytest.raises(SystemExit) as exc:
        main_scope()['main']()
    assert exc.value.code == 2


def test_screen_refuses_stubbed_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', *options(tmp_path)])
    with pytest.raises(ValueError, match='stubbed'):
        main_scope(stubbed=True)['main']()


def test_default_benchmark_admission_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', '--out', str(tmp_path / 'out')])
    with pytest.raises(DeviceBoundaryReached):
        main_scope()['main']()


def test_finite_comparison_parser_exists_before_device(monkeypatch, tmp_path):
    # The AST-isolated main must resolve siblings as direct script execution does.
    monkeypatch.syspath_prepend(str(ROOT / "experiments/t8r_speed"))
    argv = ['--out', str(tmp_path / 'out'), '--comparison-protocol', 'missing.json',
            '--comparison-phase', 'numeric']
    monkeypatch.setattr('sys.argv', ['bench_t8r.py', *argv])
    # The old parser exits with an unknown-argument error; the new owner must
    # read the explicit protocol before touching a device.
    with pytest.raises(FileNotFoundError):
        main_scope()['main']()


def test_benchmark_import_does_not_advertise_a_missing_runtime():
    pytest.importorskip("torch")
    import subprocess
    import sys

    script = r"""
import importlib.abc
import importlib.util
import os
import sys

class MissingRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise ModuleNotFoundError("The runtime is absent for this control.")

sys.meta_path.insert(0, MissingRuntime())
import bench_class_dispatch
try:
    runtime = importlib.util.find_spec("vllm")
except ModuleNotFoundError:
    runtime = None
assert runtime is None
assert "vllm" not in sys.modules
assert "TESSERA_SERVE_MODE" not in os.environ
"""
    env = dict(os.environ)
    env.pop("TESSERA_SERVE_MODE", None)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"),
        str(ROOT / "experiments/t8r_speed"), env.get("PYTHONPATH", "")))
    done = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
