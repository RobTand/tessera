"""A torch gate needs an actual torch population and an honest summary."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


MODULES = {
    'gate_actions': ROOT / 'tools/ci/gate_actions.py',
}


def load(name):
    spec = importlib.util.spec_from_file_location(name, MODULES[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('surface', [
    {'not_collected': ['test_gpu.py'], 'device': 'torch 2.11 reports no CUDA device'},
    {'not_collected': [], 'device': 'no torch in this interpreter (ImportError)'},
])
def test_aggregate_refuses_missing_torch_or_uncollected_modules(surface):
    actions = load('gate_actions')
    plan = {'population': {'version': 'v'}, 'mode': 'full', 'base_sha': 'a', 'head_sha': 'b'}
    record = {**plan, 'population_version': 'v', 'lane': 'torch', 'files': ['tests/test_a.py'],
              'verdict': 'pass', 'pytest_exit_code': 0, 'budget': {'status': 'pass'},
              'collection': {'status': 'pass'}, 'surface': {
                  'role': 'population', 'cuda': False, 'cuda_surface': {'executed': 0}, **surface}}
    assert actions._verdict(plan, record['files'], record, 'torch') is None


def test_workflow_names_mergify_batch_drafts_exactly():
    """gate runs on Mergify's batch drafts and pr-gate on everything else; the
    draft is the same repository, a merge-queue branch and Mergify's app user,
    as tools/ci/gate_select.py's full_gate says."""
    workflow = (ROOT / '.github/workflows/gate.yml').read_text()
    identity = ("github.event.pull_request.head.repo.full_name == github.repository && "
                "startsWith(github.head_ref, 'mergify/merge-queue/') && "
                "github.event.pull_request.user.login == 'mergify[bot]' && "
                "github.event.pull_request.user.id == 37929162")
    assert workflow.count(identity) == 3  # BATCH_DRAFT, pr-gate's negation, gate's condition
    assert '!(' + identity + ')' in workflow


def test_final_summary_quotes_skip_uncollected_and_cuda_population():
    actions = load('gate_actions')
    record = {'lane': 'bytes', 'surface': {'counts': {'passed': 10, 'skipped': 3},
              'not_collected': ['test_gpu.py'], 'skip_reasons': {'requires CUDA': 3},
              'device': 'no torch in this interpreter (ImportError)',
              'cuda_surface': {'executed': 0}}}
    text = actions.population_summary([record])
    for value in ['10 passed', '3 skipped', '1 uncollected', 'CUDA not covered', '0 allocated', 'requires CUDA']:
        assert value in text
