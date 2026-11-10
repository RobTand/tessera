"""LFM stages share the reviewed serve owner instead of legacy containment."""
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from box_artifacts import FLUX_RENDER_ROOT
from tools import _flux_serve as policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('stage', ['teacher', 'student'])
def test_lfm_stage_has_a_reviewed_renderer_and_refuses_on_a_bare_host(tmp_path, stage):
    wrapper = ROOT / 'experiments/lfm_bound_served.sh'
    spec = {'gpu_memory_utilization': '.35'}
    argv = policy.render_serving([str(wrapper), stage], spec,
                                 out=str(FLUX_RENDER_ROOT / 'lfm-stage'), spec_path=tmp_path / 'spec.json')
    assert '--resources-shape=slot/[core=4;memory=47;gpu=1]' in argv
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('FLUX_', 'TESSERA_SERVE_'))}
    result = subprocess.run(['bash', str(wrapper), stage], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'OUTSIDE_FLUX_JOB' in result.stderr


@pytest.mark.parametrize('name', ['ts5_lfm_teacher_bound.py', 'ts5_lfm_served_bound.py'])
def test_lfm_serve_does_not_override_the_reviewed_memory_bound(name):
    tree = ast.parse((ROOT / 'experiments' / name).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {'TESSERA_KL_DOCKER_EXTRA', 'TESSERA_LANE_DOCKER_EXTRA'}:
                    found.append(value)
                    assert '--memory' not in ast.unparse(value)
    assert found


def test_lfm_stage_finalizer_stops_through_its_supervisor(monkeypatch):
    events = []
    def rpc(action, arguments):
        events.append((action, arguments))
    monkeypatch.setattr(policy, 'rpc', rpc)
    stop = SimpleNamespace(set=lambda: events.append('stop-telemetry'))
    monitor = SimpleNamespace(join=lambda **kwargs: events.append('join-telemetry'))
    positive = policy.cleanup_scoring_stage('obsolete-name', launched=True, completed=True,
                                           stop=stop, monitor=monitor)
    assert positive['safe_to_release']
    assert events == ['stop-telemetry', 'join-telemetry', ('stop', [])]
    def failed(*args):
        raise policy.Refused('CLEANUP_UNVERIFIED')
    monkeypatch.setattr(policy, 'rpc', failed)
    negative = policy.cleanup_scoring_stage('obsolete-name', launched=True, completed=False,
                                           stop=stop, monitor=monitor)
    assert not negative['safe_to_release']
    assert 'CLEANUP_UNVERIFIED' in negative['error']
