"""History-free frozen-selector acceptance for issue #1524.

Reuses the fixture identity and snapshot approach from the recoverable #1315
work (PR #1481), not an alternative historical runtime. The loader seam is
_load_frozen_runtime(destination, *, archive_path=..., manifest_path=...).
These checks prove historical source custody and real selector execution,
not CUDA numerical or performance qualification.

Test-author guidance: commit historical bytes with independently checked
provenance; never require old Git objects in a test. Keep genuine dependency
initialization outside an external-command guard: CUDA Torch may run ldconfig.
Never replace selectors with mocks or skip a broken fixture/history probe.

Research RED on base 17b3fdb396410698c9a142615345f046d2f94a5d:
- Original named target on admitted x86: 1 failed, no skips/uncollected modules,
  zero device allocations; git archive exits 128, 'fatal: not a tree object'.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"75c37fcaee353b814736b76e0d2018fec34f8304da9cb8d379e3a215bbecbfe8","size":6044}},"operation":"e8d022db028ecce8de630fd08989bf0733c8d6f07b038b0d826473e3f60a9951","receipt":"0be7bfaab3002fa137b6d6533fe1a045002aaca1b4378c593a3529f9320eaa8b"}]
- New acceptance plus entire eager-levers module and both repository guards,
  admitted x86 with two workstealing workers: 6 failed, 247 passed, no skips or
  uncollected modules, zero device allocations. All five new cases fail for
  missing fixture/loader or the forbidden historical git invocation; original
  target fails for the unavailable historical tree. Both guards and all other
  eager-levers cases pass. Retained stdout was read. No GPU run repeated this
  CPU-established defect; configured-GPU post-fix execution is still owed.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"71dc034efc09a9ded354b196fa940bdbddb9fe23af9b324d85f5a1dcc37fd186","size":15238}},"operation":"a67f299a4082f781435d18b5e924a49987af816a7816d14180242eb24db87bbe","receipt":"74bcccb485c07f23367f97c1b0239fef0a90d471a2334d8e9e9b48e8095c953d"}]
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

import test_graph_attest_eager_levers as levers

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/frozen_eager_2dbac191"
COMMIT = "2dbac1910c88254d9c6391f02a34c4b07e516803"
# Independently observed exact historical export, not an editable manifest pin.
EXPORT_SHA256 = "91f3a340e9e14068e448bbc150a702d16e913cdf70fc487e7acbe0a17fb272fb"


def loader():
    value = getattr(levers, "_load_frozen_runtime", None)
    assert callable(value), "replace historical Git export with the authenticated fixture loader"
    return value


def fixture_bytes():
    archive = FIXTURE / "runtime.tar.gz"
    manifest = FIXTURE / "manifest.json"
    assert archive.is_file(), "missing committed historical runtime (not an optional dependency)"
    assert manifest.is_file(), "missing committed historical provenance"
    return archive.read_bytes(), json.loads(manifest.read_text())


def test_fixture_is_the_exact_historical_export(tmp_path):
    compressed, manifest = fixture_bytes()
    assert levers.benchmark.RUNTIME_COMMIT == COMMIT
    assert manifest["source_commit"] == COMMIT
    assert manifest["archive_sha256"] == hashlib.sha256(compressed).hexdigest()
    exported = gzip.decompress(compressed)
    assert manifest["export_sha256"] == EXPORT_SHA256 == hashlib.sha256(exported).hexdigest()
    destination = tmp_path / "runtime"
    destination.mkdir()
    loader()(destination)
    # Authentication above anchors this comparison to independently known bytes.
    with tarfile.open(fileobj=io.BytesIO(exported)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                assert (destination / member.name).read_bytes() == tar.extractfile(member).read()


@pytest.mark.parametrize("fault", ["missing_archive", "corrupt", "wrong_commit"])
def test_loader_refuses_invalid_fixture_before_extracting(tmp_path, fault):
    load = loader()
    compressed, manifest = fixture_bytes()
    if fault == "corrupt":
        compressed = compressed[:-1] + bytes([compressed[-1] ^ 1])
    if fault == "wrong_commit":
        manifest["source_commit"] = "0" * 40
    archive = tmp_path / "runtime.tar.gz"
    metadata = tmp_path / "manifest.json"
    archive.write_bytes(compressed)
    metadata.write_text(json.dumps(manifest))
    if fault == "missing_archive":
        archive.unlink()
    destination = tmp_path / "runtime"
    destination.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        load(destination, archive_path=archive, manifest_path=metadata)
    destination.rmdir()  # Fails if the loader wrote anything before rejecting.


# Initialize genuine prerequisites under network denial, then forbid external
# commands before any historical source import or selector work. In particular,
# do not block CUDA Torch's dependency initialization or substitute fake Torch.
CHILD_BOOTSTRAP = r'''
import importlib
import sys

def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise AssertionError('network is forbidden: ' + event)
sys.addaudithook(offline)
for prerequisite in ('torch', 'triton'):
    importlib.import_module(prerequisite)
assert not any(name == 'tessera' or name.startswith('tessera.')
               or name == 'test_piece_major_reader_boundaries' for name in sys.modules)

def no_external(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
        raise AssertionError('external command in frozen child')
sys.addaudithook(no_external)
'''

SNAPSHOT_PROBE = r'''
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / 'tests'), str(root / 'src')]
assert not (root / '.git').exists()

def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise AssertionError('network is forbidden: ' + event)
sys.addaudithook(offline)
# The controller also imports Torch in main's original target. Bootstrap real
# dependencies before controlling its subprocess boundary, not frozen source.
for prerequisite in ('torch', 'triton'):
    importlib.import_module(prerequisite)

prefix = @CHILD_BOOTSTRAP@
suffix = r"""
from pathlib import Path
required = ('tessera.serving.e4m3_prefill', 'tessera.serving.flags',
            'tessera.serving.glm53_prefill', 'tessera.serving.moe_route',
            'tessera.serving.native_window', 'test_piece_major_reader_boundaries')
for name in required:
    assert name in sys.modules, 'missing frozen reader: ' + name
for name, module in list(sys.modules.items()):
    if name == 'tessera' or name.startswith('tessera.') or name == 'test_piece_major_reader_boundaries':
        source = getattr(module, '__file__', None)
        if source:
            expected = Path.cwd() / ('tests' if name == 'test_piece_major_reader_boundaries' else 'src')
            assert Path(source).resolve().is_relative_to(expected.resolve()), (name, source, expected)
print('FROZEN_IMPORTS_VERIFIED')
"""
seen = []
original = subprocess.Popen
class CheckedChild(original):
    def __init__(self, args, *positional, **kwargs):
        assert Path(args[0]).resolve() == Path(sys.executable).resolve(), ('historical/external command forbidden', args)
        assert '-c' in args, args
        assert Path(kwargs['cwd']).resolve() == (root / 'probe-runtime').resolve(), kwargs
        args = list(args)
        index = args.index('-c') + 1
        args[index] = prefix + '\n' + args[index] + '\n' + suffix
        super().__init__(args, *positional, **kwargs)
    def communicate(self, *args, **kwargs):
        out, err = super().communicate(*args, **kwargs)
        if self.returncode == 0:
            assert 'FROZEN_IMPORTS_VERIFIED' in out, (out, err)
            assert 'all three flags read' in out, (out, err)
            seen.append(True)
        return out, err
subprocess.Popen = CheckedChild
namespace = runpy.run_path(str(root / 'tests/test_graph_attest_eager_levers.py'))
work = root / 'probe-runtime'
work.mkdir()
try:
    namespace['test_frozen_2db_flags_reach_different_runtime_selectors'](work)
except pytest.skip.Exception as exc:
    raise AssertionError('frozen selector must execute, not skip: ' + str(exc)) from exc
assert len(seen) == 1, seen
print('FROZEN_RESULT=' + json.dumps(dict(status='executed')))
'''.replace('@CHILD_BOOTSTRAP@', repr(CHILD_BOOTSTRAP))


def test_actual_target_executes_offline_without_history(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "tests").mkdir(parents=True)
    shutil.copy2(ROOT / "pyproject.toml", snapshot / "pyproject.toml")
    # Named directory edges are visible to impacted_tests. Controller source is
    # distinct from the fresh child's verified historical import roots.
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "experiments/graph_attest_702", snapshot / "experiments/graph_attest_702",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "tests/test_graph_attest_eager_levers.py", snapshot / "tests")
    if FIXTURE.exists():
        shutil.copytree(FIXTURE, snapshot / "tests/fixtures/frozen_eager_2dbac191")
    script = tmp_path / "offline_probe.py"
    script.write_text(SNAPSHOT_PROBE)
    done = subprocess.run([sys.executable, str(script), str(snapshot)], cwd=snapshot,
                          env=dict(os.environ), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    records = [json.loads(line.removeprefix("FROZEN_RESULT=")) for line in done.stdout.splitlines()
               if line.startswith("FROZEN_RESULT=")]
    assert records == [{"status": "executed"}], done.stdout + done.stderr
