"""Keep known kernel telemetry out of repository dependency uncertainty."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kernel', ['/proc/sys/kernel/random/boot_id', '/proc/self/maps', '/proc/self/mountinfo'])
def test_kernel_telemetry_is_not_an_unknown_repository_read(tmp_path, kernel):
    select = load(ROOT / 'tools/impacted_tests.py')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'source.py').write_text(f'from pathlib import Path\ndef read(): return Path({kernel!r}).read_text()\n')
    (tmp_path / 'tests/test_reader.py').write_text('import source\ndef test_read(): pass\n')
    (tmp_path / 'other.py').write_text('VALUE = 1\n')
    result = select.select(tmp_path, ['other.py'])
    assert result['verdict'] == 'none'
    assert result['tests'] == []
    assert select.select(tmp_path, ['source.py'])['tests'] == ['tests/test_reader.py']


@pytest.mark.parametrize('external', ['/outside/data.json', '/proc/self/root/repo/data.json'])
def test_unmapped_reads_keep_their_conservative_consumers(tmp_path, external):
    select = load(ROOT / 'tools/impacted_tests.py')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'source.py').write_text(f'from pathlib import Path\ndef read(): return Path({external!r}).read_text()\n')
    (tmp_path / 'tests/test_reader.py').write_text('import source\ndef test_read(): pass\n')
    (tmp_path / 'other.py').write_text('VALUE = 1\n')
    assert select.select(tmp_path, ['other.py'])['tests'] == ['tests/test_reader.py']


@pytest.mark.parametrize('path', ['README.md', 'docs/ci_gate.md'])
def test_documentation_uses_only_the_fast_boundary_tier(tmp_path, path):
    gate = load(ROOT / 'tools/ci/gate_select.py')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_standalone_boundary.py').write_text('')
    (tmp_path / 'tests/test_heavy.py').write_text('')
    result = gate.select_files(tmp_path, 'base', 'head',
        execute=lambda command, **kwargs: SimpleNamespace(returncode=0, stdout=path+'\n', stderr=''))
    assert result['verdict'] == 'narrowed'
    assert result['tests'] == ['tests/test_standalone_boundary.py']


def test_unknown_data_read_does_not_activate_a_guarded_lazy_import(tmp_path):
    select = load(ROOT / 'tools/impacted_tests.py')
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/__init__.py').write_text('')
    (tmp_path / 'pkg/layout.py').write_text(
        'BASE = 1\n_NAMES = frozenset({"slice"})\n'
        'def __getattr__(name):\n'
        '    if name in _NAMES:\n'
        '        from . import slicing as _slicing\n'
        '        return getattr(_slicing, name)\n'
        '    raise AttributeError("no such attribute")\n')
    (tmp_path / 'pkg/slicing.py').write_text(
        'from pathlib import Path\ndef slice(): return Path("/outside/data.json").read_text()\n')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_base.py').write_text('from pkg.layout import BASE\ndef test_base(): assert BASE == 1\n')
    (tmp_path / 'tests/test_slice.py').write_text('from pkg.layout import slice\ndef test_slice(): pass\n')
    (tmp_path / 'other.py').write_text('VALUE = 1\n')
    assert select.select(tmp_path, ['other.py'])['tests'] == ['tests/test_slice.py']
    assert 'tests/test_slice.py' in select.select(tmp_path, ['pkg/slicing.py'])['tests']
