"""Repository-wide directory-read census requires the runner's complete Git checkout."""
import ast
from collections import defaultdict
from pathlib import Path
import subprocess

import pytest

from tessera._dev.source_dependencies import file_imports, source_execution_modules

ROOT = Path(__file__).resolve().parents[1]


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
# tessera#1701 deleted guarded_reexport, whose ast.parse call marked the resolver as
# source-executing: its 12 existing reads with caller-supplied bases now count (no new read).
_UNNAMED_DIRECTORY_READ_MODULES = 118
_UNNAMED_DIRECTORY_READ_SITES = 183


def _unnamed_directory_reads(root):
    """{path: lines} of each tracked module's directory reads whose base nothing names.

    A module that runs source, itself or through an imported helper, keeps
    such a read as a wildcard, so it is not listed. Every dotted suffix of a
    module's name resolves to it, as the import roots on sys.path allow.
    """
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=root).decode().split('\0')[:-1]
    trees = {root / path: ast.parse((root / path).read_text()) for path in tracked}
    names, spellings = {}, defaultdict(set)
    for path in trees:
        parts = list(path.relative_to(root).with_suffix('').parts)
        parts = parts[1:] if parts[0] == 'src' else parts
        names[path] = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        for start in range(len(parts)):
            spellings['.'.join(parts[start:])].add(path)
    executing = source_execution_modules(trees, names, lambda spelling: tuple(sorted(spellings.get(spelling, ()))))
    found = {}
    for path, tree in trees.items():
        lines = []
        file_imports(tree, path, root, executes_source=path in executing, unnamed=lines)
        if lines:
            found[path.relative_to(root).as_posix()] = lines
    return found


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
    checkout = subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip()
    assert Path(checkout).resolve() == ROOT.resolve(), 'census must use the repository checkout'
    listed = _ratcheted_directory_reads(ROOT, _unnamed_directory_reads(ROOT))
    sites = sum(len(lines) for lines in listed.values())
    print(f'directory census: modules={len(listed)} sites={sites}')
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


def test_the_repository_census_uses_the_full_checkout_runner():
    entry = f'"//tests:{Path(__file__).stem}": struct(tags = ["runner"]'
    assert entry in (ROOT / 'tools/bazel/nonhermetic.bzl').read_text(), (
        'a repository-wide census must run against the complete Git checkout, not partial runfiles')
