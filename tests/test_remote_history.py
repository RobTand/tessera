"""Remote gate clients hydrate history and blobs before any Bazel work."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote_history_gate', ROOT / 'tools/ci/remote_gate.py')
remote_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_gate)


def git(root, *arguments):
    return subprocess.run(['git', *arguments], cwd=root, check=True, text=True,
                          capture_output=True).stdout.strip()


@pytest.fixture
def partial_clone(tmp_path):
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
    # Remote Bazel has checked out the requested revision before this preflight.
    # Match bbrun's real fixture, including materializing the current tree.
    git(clone, 'checkout', '--detach', head)
    assert git(clone, 'rev-parse', '--is-shallow-repository') == 'true'
    assert git(clone, 'config', '--get', 'remote.origin.promisor') == 'true'
    return clone, head, old


def test_full_history_hydrates_a_real_shallow_partial_clone(partial_clone, monkeypatch):
    clone, head, old = partial_clone
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


def test_fetch_disables_automatic_repository_maintenance(partial_clone, monkeypatch):
    clone, head, _ = partial_clone
    commands = []
    run = subprocess.run

    def observe(command, **kwargs):
        commands.append(command)
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', observe)
    assert remote_gate.prepare_full_history(head, root=clone)
    fetch, = [command for command in commands if 'fetch' in command]
    assert fetch[:6] == ['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false', 'fetch']


def test_marker_removed_after_enumeration_is_already_clean(partial_clone, monkeypatch):
    clone, head, _ = partial_clone
    pack = clone / '.git/objects/pack'
    glob = Path.glob
    removed = []

    def competing_cleanup(path, pattern):
        entries = list(glob(path, pattern))
        if path == pack and pattern == '*.promisor':
            # Deterministic interleaving: another actor deletes each enumerated
            # marker before this preflight attempts to remove it. No sleeps.
            for entry in entries:
                entry.unlink()
                removed.append(entry.name)
        return iter(entries)

    monkeypatch.setattr(Path, 'glob', competing_cleanup)
    assert remote_gate.prepare_full_history(head, root=clone)
    assert removed
    assert git(clone, 'rev-parse', 'HEAD') == head
    assert git(clone, 'rev-parse', '--is-shallow-repository') == 'false'
    assert not list(pack.glob('*.promisor'))
    assert not any(line.startswith('?') for line in
                   git(clone, 'rev-list', '--objects', '--missing=print', '--all', 'HEAD').splitlines())


def test_other_marker_errors_fail_with_safe_stage_and_type(partial_clone, monkeypatch, capsys):
    clone, head, _ = partial_clone
    unlink = Path.unlink
    private_message = 'fixture-private-diagnostic-never-print'

    def denied(path, *args, **kwargs):
        if path.suffix == '.promisor':
            raise PermissionError(private_message)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', denied)
    assert not remote_gate.prepare_full_history(head, root=clone)
    output = capsys.readouterr()
    assert 'promisor-cleanup (PermissionError)' in output.err
    assert private_message not in output.err + output.out
    assert list((clone / '.git/objects/pack').glob('*.promisor'))


def test_history_preflight_does_not_swallow_interrupts(partial_clone, monkeypatch):
    clone, head, _ = partial_clone

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, 'run', interrupted)
    with pytest.raises(KeyboardInterrupt):
        remote_gate.prepare_full_history(head, root=clone)
