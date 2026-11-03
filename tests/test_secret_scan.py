"""The vendored secret scanner is PrismaQuant's, byte for byte, and every upload runs it first."""
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_vendored_scanner_is_the_recorded_copy():
    recorded = (ROOT / 'tools/ci/check_no_secret_leak.sha256').read_text()
    assert '71c36dc8a10803a72f1a2a67e96a88af8e728420' in recorded
    assert recorded.split()[0] == 'decd9ce671299ece1b245811af459349ed9932ee9b81b2c4c7a13cf96b94f609'
    assert hashlib.sha256((ROOT / 'tools/ci/check_no_secret_leak.py').read_bytes()).hexdigest() == recorded.split()[0]


def test_every_artifact_upload_is_scanned_first():
    """Each upload-artifact step follows a scan of what it uploads, with the secrets to look for."""
    for workflow in sorted((ROOT / '.github/workflows').glob('*.yml')):
        steps = re.split(r'\n      - ', workflow.read_text())
        for index, step in enumerate(steps):
            if 'actions/upload-artifact@' not in step:
                continue
            scan = steps[index - 1]
            assert ('python3 tools/ci/check_no_secret_leak.py ' in scan
                    or 'python3 tools/ci/bazel_gate.py scan-artifacts ' in scan), \
                f'{workflow.name}: an upload is not scanned first'
            # --require both selects the variable for scanning and refuses a missing value.
            assert '--require BUILDBUDDY_API_KEY' in scan and '--require MERGIFY_TOKEN' in scan, workflow.name


@pytest.mark.parametrize('name', ['BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN'])
def test_required_variables_are_scanned_without_a_duplicate_env_flag(name, monkeypatch, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location('required_secret_scanner', ROOT / 'tools/ci/check_no_secret_leak.py')
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    monkeypatch.setenv(name, 'fake-required-only-secret')
    (tmp_path / 'diagnostic.log').write_text('fake-required-only-secret')
    assert scanner.main([str(tmp_path), '--require', name]) == 1
    assert f'{name}: secret detected' in capsys.readouterr().out


@pytest.mark.parametrize('value', [None, ''])
def test_a_missing_required_key_fails(value, monkeypatch, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location('required_secret_scanner', ROOT / 'tools/ci/check_no_secret_leak.py')
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    if value is None:
        monkeypatch.delenv('BUILDBUDDY_API_KEY', raising=False)
    else:
        monkeypatch.setenv('BUILDBUDDY_API_KEY', value)
    assert scanner.main([str(tmp_path), '--env', 'BUILDBUDDY_API_KEY', '--require', 'BUILDBUDDY_API_KEY']) == 75
    assert 'BUILDBUDDY_API_KEY: required but not available' in capsys.readouterr().out
