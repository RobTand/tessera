"""Legacy Docker paths refuse in a Flux job, before doing any work."""
import os
import ast
import re
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('job_id', ['fABC', ''])
def test_legacy_entry_refuses_inside_flux(tmp_path, job_id):
    fake = tmp_path / 'bin'
    fake.mkdir()
    docker = fake / 'docker'
    docker.write_text('#!/bin/sh\necho legacy-docker-invoked >&2\nexit 42\n')
    docker.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'experiments/container_limits_probe.sh')],
                            env={**os.environ, 'FLUX_JOB_ID': job_id, 'PATH': str(fake) + ':' + os.environ['PATH']},
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'ROOT_DOCKER_REQUIRES_REVIEWED_SERVING_WRAPPER' in result.stderr
    assert 'anything else goes to the lead' in result.stderr
    assert 'legacy-docker-invoked' not in result.stderr


def test_reviewed_owner_exemption_is_an_exact_path(tmp_path):
    directory = tmp_path / 'other'
    directory.mkdir()
    (directory / '_flux_serve.py').write_text("import subprocess\nsubprocess.run(['docker', 'run', 'image'])\n")
    assert unreviewed_docker_paths(tmp_path) == ['other/_flux_serve.py']


def test_python_shell_docker_is_checked(tmp_path):
    (tmp_path / 'entry.py').write_text("import subprocess\nsubprocess.run('docker run image', shell=True)\n")
    assert unreviewed_docker_paths(tmp_path) == ['entry.py']


def test_a_guard_after_docker_does_not_protect_it(tmp_path):
    (tmp_path / 'entry.sh').write_text('docker run image\npython3 tools/_flux_docker_guard.py || exit $?\n')
    assert unreviewed_docker_paths(tmp_path) == ['entry.sh']


def unreviewed_docker_paths(root):
    """Scan live shell and Python commands; archived and test payloads are inert.

    Legacy commands must call the shared refusal guard in their own file.
    Comments and docstrings are excluded, so documentation is not a bypass list.
    """
    failures = []
    for path in sorted(Path(root).rglob('*')):
        relative = path.relative_to(root)
        if any(part in {'archive', '.git', 'tests', '__pycache__'} for part in relative.parts):
            continue
        if not path.is_file() or relative.as_posix() == 'tools/_flux_serve.py':
            continue
        if path.suffix not in {'.py', '.sh', ''}:
            continue
        text = path.read_text()
        shell = path.suffix == '.sh' or (not path.suffix and re.match(r'#!.*(?:/sh|bash)', text))
        if shell:
            code = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
            direct = re.search(r'\bdocker\s+(?:run|create)\b', code)
            guard = re.search(r'^python3 .*_flux_docker_guard\.py.*\|\| exit', code, re.MULTILINE)
            guarded = guard is not None and (direct is None or guard.start() < direct.start())
        else:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            direct_lines = []
            for node in ast.walk(tree):
                if (isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 2 and
                        isinstance(node.elts[0], ast.Constant) and node.elts[0].value == 'docker' and
                        isinstance(node.elts[1], ast.Constant) and node.elts[1].value in {'run', 'create'}):
                    direct_lines.append(node.lineno)
                function = getattr(node, 'func', None)
                name = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', '')
                if isinstance(node, ast.Call) and name in {
                        'run', 'Popen', 'check_call', 'check_output', 'call', 'system', 'execvp', 'execvpe'} and any(
                        isinstance(arg, ast.Constant) and isinstance(arg.value, str) and
                        re.search(r'\bdocker\s+(?:run|create)\b', arg.value) for arg in node.args):
                    direct_lines.append(node.lineno)
            guards = [node.lineno for node in tree.body if isinstance(node, ast.Expr) and
                      isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                      node.value.func.id == 'refuse_unreviewed_docker']
            direct = bool(direct_lines)
            guarded = bool(guards) and (not direct_lines or min(guards) < min(direct_lines))
        if direct and not guarded:
            failures.append(relative.as_posix())
    return failures


def test_live_docker_paths_are_fenced():
    assert unreviewed_docker_paths(ROOT) == []


def test_new_docker_path_is_rejected(tmp_path):
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/bypass.sh').write_text('#!/bin/bash\ndocker run --rm image\n')
    assert unreviewed_docker_paths(tmp_path) == ['tools/bypass.sh']


def test_guard_comment_is_not_a_guard(tmp_path):
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/bypass.sh').write_text('#!/bin/bash\n# _flux_docker_guard\ndocker run --rm image\n')
    assert unreviewed_docker_paths(tmp_path) == ['tools/bypass.sh']


def test_extensionless_docker_path_is_checked(tmp_path):
    (tmp_path / 'tools').mkdir()
    path = tmp_path / 'tools/bypass'
    path.write_text('#!/bin/bash\ndocker run --rm image\n')
    path.chmod(0o755)
    assert unreviewed_docker_paths(tmp_path) == ['tools/bypass']


