"""Remote gate clients hydrate history and blobs before any Bazel work."""
import importlib.util
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote_history_gate', ROOT / 'tools/ci/remote_gate.py')
remote_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_gate)


def git(root, *arguments):
    return subprocess.run(['git', *arguments], cwd=root, check=True, text=True,
                          capture_output=True).stdout.strip()


def test_full_history_hydrates_a_real_shallow_partial_clone(tmp_path, monkeypatch):
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--initial-branch=master')
    git(origin, 'config', 'user.email', 'fixture@example.invalid')
    git(origin, 'config', 'user.name', 'Fixture')
    git(origin, 'config', 'uploadpack.allowFilter', 'true')
    (origin / 'old.txt').write_text('historical blob\n')
    git(origin, 'add', 'old.txt')
    git(origin, 'commit', '-m', 'old')
    old = git(origin, 'rev-parse', 'HEAD')
    (origin / 'old.txt').unlink()
    (origin / 'new.txt').write_text('current blob\n')
    git(origin, 'add', '-A')
    git(origin, 'commit', '-m', 'new')
    head = git(origin, 'rev-parse', 'HEAD')
    clone = tmp_path / 'clone'
    git(tmp_path, 'clone', '--depth=1', '--filter=blob:none', '--no-checkout',
        origin.as_uri(), str(clone))
    assert git(clone, 'rev-parse', '--is-shallow-repository') == 'true'
    assert git(clone, 'config', '--get', 'remote.origin.promisor') == 'true'
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'fake-fixture-key')
    monkeypatch.setenv('UNRELATED_TOKEN', 'fake-fixture-token')
    observed_secret_names = []
    run = subprocess.run

    def observe(*arguments, **keywords):
        observed_secret_names.extend(name for name in keywords['env']
                                     if name.endswith(('_API_KEY', '_TOKEN')))
        return run(*arguments, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, 'run', observe)
        assert remote_gate.prepare_full_history(head, root=clone)
    assert observed_secret_names == []
    assert git(clone, 'rev-parse', 'HEAD') == head
    assert git(clone, 'rev-parse', '--is-shallow-repository') == 'false'
    assert git(clone, 'show', f'{old}:old.txt') == 'historical blob'
    assert not any(line.startswith('?') for line in
                   git(clone, 'rev-list', '--objects', '--missing=print', '--all', 'HEAD').splitlines())
    assert not list((clone / '.git/objects/pack').glob('*.promisor'))
    result = subprocess.run(['git', 'config', '--get-regexp',
                             r'^(remote\..*\.(promisor|partialclonefilter)|extensions\.partialclone)$'],
                            cwd=clone, capture_output=True)
    assert result.returncode == 1


def test_full_history_refuses_a_different_requested_head(tmp_path):
    git(tmp_path, 'init', '--initial-branch=master')
    assert not remote_gate.prepare_full_history('not-a-full-commit', root=tmp_path)
