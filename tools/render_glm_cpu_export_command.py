"""Validate actual export bindings and emit a reviewable PB argv; never submit.

The row is agent-submitted work, so it takes the agent priority band (-10):
it yields to campaign work at 0 and never displaces it in the queue.

The interpreter is derived, not named: ``/home/rob/venvs/pq-cpu312-tessera-<c>``
where ``<c>`` is the first eight hex digits of the Tessera commit the
submitted checkout holds -- the fleet's naming for the CPU venv provisioned
at that pin.  The runner executes the checkout's own ``src`` (``PYTHONPATH``),
so the venv supplies the dependencies that pin was qualified with, and a
checkout that is not a clean commit is refused: its venv name would describe
code the row does not run.  The renderer cannot see the placement box, so the
result records the derived path as a precondition to check there.
"""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
from run_glm_cached_cpu_export import INPUTS, read_bound

#: The agent band (PrismaBuild #362): below campaign work, never ahead of it.
AGENT_PRIORITY = -10
VENV_ROOT = Path('/home/rob/venvs')
PLACEMENT_TAG = 'dl380g10'


def checkout_commit(checkout) -> str:
    """The full commit a clean checkout holds; a dirty or non-git tree is refused."""
    def git(*args):
        return subprocess.run(['git', '-C', str(checkout), *args], check=True,
                              capture_output=True, text=True).stdout
    commit = git('rev-parse', 'HEAD').strip()
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError(f'checkout {checkout} has no commit HEAD: {commit!r}')
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise ValueError(f'checkout {checkout} has uncommitted changes; the derived venv '
                         f'names commit {commit[:8]}, which is not the code this row would run')
    return commit


def pinned_python(commit: str) -> Path:
    """The CPU venv interpreter the fleet provisions for Tessera ``commit``."""
    return VENV_ROOT / f'pq-cpu312-tessera-{commit[:8]}' / 'bin' / 'python'


def render(bindings, bindings_sha256, checkout) -> dict:
    bound = {'path': bindings, 'sha256': bindings_sha256}
    doc = json.loads(read_bound(bound))
    if doc.get('schema') != 'prismaquant.glm_cached_cpu_export_bindings.v1' or set(doc.get('inputs', {})) != set(INPUTS):
        raise ValueError('actual allocation/PACT/export bindings are incomplete')
    for value in doc['inputs'].values():
        read_bound(value)
    selected = json.loads(read_bound(doc['inputs']['selected_manifest']))
    if selected.get('schema') != 'tessera.cached_units.v2':
        raise ValueError('mixed selected manifest required')
    wire_bytes = sum(record['blob_bytes'] for record in selected['units'].values())
    if doc['intake_threads'] != 7 or doc['intake_window_bytes'] != 8 << 30:
        raise ValueError('resource settings differ from reviewed row')
    checkout = Path(checkout).resolve()
    commit = checkout_commit(checkout)
    python = pinned_python(commit)
    argv = ['python3', '/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py', '--cwd', str(checkout),
            '--tag', PLACEMENT_TAG, '--cpus', '8', '--demand', 'mem_gb=48', '--priority', str(AGENT_PRIORITY),
            '--env', 'OMP_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1',
            '--env', 'PYTHONPATH=src', '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--progress', 'source_verify=900', '--progress', 'export_shards=600', '--progress', 'publish=600', '--',
            str(python), 'tools/run_glm_cached_cpu_export.py',
            '--bindings', bindings, '--bindings-sha256', bindings_sha256]
    return {'schema': 'prismaquant.reviewable_cpu_export_submission.v1', 'submitted': False, 'argv': argv,
            'tessera_commit': commit,
            'interpreter': {'path': str(python), 'derived_from': 'tessera_commit',
                            'precondition': f'must exist on {PLACEMENT_TAG}; this renderer cannot see that box'},
            'source': doc['source'], 'output': doc['output'], 'selected_wire_bytes': wire_bytes,
            'required_free_bytes': doc['required_free_bytes'], 'binding': bound,
            'remaining_gate': 'root review of actual PACT result, selected assignment and complete scientific scope'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', required=True); parser.add_argument('--bindings-sha256', required=True)
    parser.add_argument('--checkout', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = render(args.bindings, args.bindings_sha256, args.checkout)
    with Path(args.out).open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
