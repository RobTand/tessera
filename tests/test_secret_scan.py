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


# Keep the vendor-aware directory-read checks here. test_impacted_tests is
# imported by source-execution tests; scanner fixtures must not enter their
# graph through that shared module.
spec = importlib.util.spec_from_file_location('scanner_impacted_tests', ROOT / 'tools/impacted_tests.py')
impacted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impacted)


# Issue 1014 classifies every site in docs/reports/unnamed-directory-reads-1014.md.
# select() measured 115 modules and 169 sites on base 83a1f38c4965.
# Literal loop alternatives name the conftest's three import roots, so the
# measured ceiling now holds at 114 modules and 168 sites.
# Runtime parameters, external box paths, and temporary test fixtures retain
# the accepted unnamed-base limit. A new read must name its base or raise
# both relevant limits here with its reason (PB1496).
# The endpoint producer consumes the loader's file list and adds no directory read.
# graph_equality.load_arm adds one class-2 runtime-parameter site: directory.iterdir()
# enumerates caller-supplied receipt evidence to reject extra as well as missing raw
# cases. It has no fixed repository base; keep it reported, not guessed or hidden.
# docs/reports/unnamed-directory-reads-1014.md records this justified increment.
# The Bazel test runner (#1589) adds two more class-2 runtime-parameter sites:
# tools/bazel/runtime.py globs the per-test venv it creates under TEST_TMPDIR
# for its site-packages, and tools/bazel/python_runner.py lists the trace
# records each traced process writes to a directory under TEST_TMPDIR. Both
# directories exist only at run time; neither has a repository base.
_UNNAMED_DIRECTORY_READ_MODULES = 117
_UNNAMED_DIRECTORY_READ_SITES = 171


def _ratcheted_directory_reads(root, listed):
    # Pinned vendored third-party code, not repo code. Both this exact path
    # and the byte-for-byte scanner pin are required; no other exemption.
    import hashlib
    relative = 'tools/ci/check_no_secret_leak.py'
    try:
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    except OSError:
        return listed
    if digest != 'decd9ce671299ece1b245811af459349ed9932ee9b81b2c4c7a13cf96b94f609':
        return listed
    return {path: lines for path, lines in listed.items() if path != relative}


def _exceeds_unnamed_directory_read_ceiling(modules: int, sites: int) -> bool:
    return modules > _UNNAMED_DIRECTORY_READ_MODULES or sites > _UNNAMED_DIRECTORY_READ_SITES


@pytest.mark.parametrize("modules, sites, exceeds", [
    (_UNNAMED_DIRECTORY_READ_MODULES, _UNNAMED_DIRECTORY_READ_SITES, False),
    (_UNNAMED_DIRECTORY_READ_MODULES - 1, _UNNAMED_DIRECTORY_READ_SITES, False),
    (_UNNAMED_DIRECTORY_READ_MODULES, _UNNAMED_DIRECTORY_READ_SITES - 1, False),
    (_UNNAMED_DIRECTORY_READ_MODULES + 1, _UNNAMED_DIRECTORY_READ_SITES, True),
    (_UNNAMED_DIRECTORY_READ_MODULES, _UNNAMED_DIRECTORY_READ_SITES + 1, True),
    # Fewer modules with more sites is still a gained read; a tuple comparison
    # orders on the module count first and lets it through.
    (_UNNAMED_DIRECTORY_READ_MODULES - 1, _UNNAMED_DIRECTORY_READ_SITES + 1, True),
    (1, _UNNAMED_DIRECTORY_READ_SITES + 1, True),
    (_UNNAMED_DIRECTORY_READ_MODULES + 1, 1, True),
])
def test_the_unnamed_directory_read_ceiling_bounds_modules_and_sites_independently(
        modules, sites, exceeds):
    assert _exceeds_unnamed_directory_read_ceiling(modules, sites) is exceeds


def test_this_repository_does_not_gain_an_unnamed_directory_read():
    listed = _ratcheted_directory_reads(
        ROOT, impacted.select(ROOT, ["README.md"])["unnamed_directory_reads"])
    sites = sum(len(lines) for lines in listed.values())
    assert not _exceeds_unnamed_directory_read_ceiling(len(listed), sites), (
        "a directory read whose base cannot be named was added; name its base, "
        "or raise the ceiling here with the reason", len(listed), sites, sorted(listed))


def test_only_the_hash_pinned_vendored_scanner_is_exempt(tmp_path):
    relative = 'tools/ci/check_no_secret_leak.py'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / relative).read_bytes())
    listed = {relative: [152], 'tools/own_reader.py': [7]}
    assert _ratcheted_directory_reads(tmp_path, listed) == {'tools/own_reader.py': [7]}


def test_a_one_byte_vendor_change_counts_again(tmp_path):
    relative = 'tools/ci/check_no_secret_leak.py'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / relative).read_bytes() + b'\n')
    listed = {relative: [152]}
    assert _ratcheted_directory_reads(tmp_path, listed) == listed


def test_the_same_vendor_bytes_at_an_unpinned_path_count_again(tmp_path):
    relative = 'other/check_no_secret_leak.py'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / 'tools/ci/check_no_secret_leak.py').read_bytes())
    listed = {relative: [152]}
    assert _ratcheted_directory_reads(tmp_path, listed) == listed
