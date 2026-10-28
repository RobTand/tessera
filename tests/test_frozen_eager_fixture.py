"""Acceptance for #1315: authenticated historical test data, not GPU performance.

Loader seam required in test_graph_attest_eager_levers:
_load_frozen_runtime(destination, *, archive_path=..., manifest_path=...).
It extracts or raises FileNotFoundError/ValueError; it never skips.
Fixture: tests/fixtures/frozen_eager_2dbac191/{runtime.tar.gz,manifest.json}.
Manifest fields: source_commit, archive_sha256 (compressed), export_sha256
(uncompressed git archive), generation_command. Authentication must not trust
an editable manifest alone.

Research source: git archive from the readable Tessera bare repository,
2dbac1910c88254d9c6391f02a34c4b07e516803, selection exactly src pyproject.toml
tests/test_piece_major_reader_boundaries.py. Export SHA-256 below was observed
during research; implementation must independently compare its export to it.

Research evidence at base a2269f9b42310d18ff1250504177bf1ea25b2048:
- Original target in the admitted x86 history-free snapshot: 1 failed, no
  skips; CalledProcessError 128, 'fatal: not a tree object: 2dbac191...'.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"47e3f5c62466debe30cfa20f65e543f20b5697c38fee3a1bba17e8a0e8ba6b95","size":6160}},"operation":"06a65f9b56207fddbebb7cb7c31ad8afa332894063d6114772d13c8bcef5d2db","receipt":"4e440e35dddd7aaad1c89a46231abe558f9767e4d7ad2479ee62ab0a6fff16cd"}]
- Final acceptance plus complete eager-levers module on x86: 13 failed,
  35 passed, no skips, zero device allocations. All 12 new cases fail for
  absent fixture/loader or forbidden Git export; original target also fails.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0cb11e5d04c2deb94f42be36ed813a580506f1c6999b9ff2757e3ea95b1d4a6a","size":23257}},"operation":"7cc101bb5bddd44486999a4c97010b542e7b6623a02e82a92a22080499935917","receipt":"5215079a82df4ebba2281232b187d29b10ca4cd0e04f11f1339ef83197360455"}]
- Corrected harness plus issue-reference tests and unnamed-directory guard:
  13 expected failures, 39 passed. An earlier full impacted-tests run exposed
  the two temporary iterdir calls; these were replaced by rmdir emptiness
  assertions. Earlier snapshot setup omitted controller src/pyproject.toml;
  corrected before the final defect-demonstrating runs. No GPU run was used
  to repeat this CPU-established defect; GPU execution remains required after
  implementation. Probes establish selector wiring, not numerical/performance
  qualification.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ca14beecf91a1a5cd44db3be892630834e145e218ce8e1b00a72a95ca283e476","size":23231}},"operation":"84e88f3c42d4022a12b09d0bce34d8825895fa66eb426ace36c8e713043772cf","receipt":"6438ac1197b033b1ef5526e73006e4be585458c683e090d01422a37ff0f855fc"}]

Replan at main 4897b63469acea4fc39342060280d380f197c8db:
- Prior implementation 914e032fe0b23986cb761624ce0bae476219e8de is
  recoverable, not landed. Its acceptance harness denied subprocesses before
  CUDA Torch initialization and consequently rejected dependency ldconfig.
  This replan changes only acceptance tests: deny networking throughout,
  initialize genuine Torch/Triton narrowly, then deny all commands before
  any frozen imports. No ldconfig allowlist, fake Torch, or relaxed selectors.
- Original target reproduced in the admitted history-free x86 snapshot:
  1 failed, no skips; Git exit 128, not a tree object. Read retained stdout.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6799611417554c42e3a0a86ee791737724dc1126af99d8dc21d4fbb6390d5333","size":6345}},"operation":"44edb389c119c626d539f796f2271db82675cd13a1b2f3b27b1d24d389b78463","receipt":"b21b8468daf55981b4e1aa3d3c42bcf6269759bc900a3f2c46cfbe9209a9185e"}]
- Acceptance + complete eager-levers + both guards on x86: 13 expected
  failures, 247 passed, no skips/uncollected modules/device allocations.
  Fixture and loader are absent, target still uses Git. FROZEN_GUARD_READY
  confirms genuine prerequisite bootstrap and guard assertions completed;
  snapshot then refuses Git. Every new acceptance case fails for unmet work,
  not import/discovery/setup errors. Read retained stdout, including traceback.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ab07f526422ea754358c128bc323fed7d5b58ff917b0fb2612e7335c5a0866df","size":24888}},"operation":"70d150e261a2cf453eff724472e87bcec1e31348b56f790e86e4beb39f94302b","receipt":"a274e5fbb96ab08459e68693347bfd9210ff121c249834c24610a775ed98e299"}]
- GPU run budget before submission: the single snapshot acceptance decides
  whether the corrected guard boundary admits genuine CUDA Torch/Triton
  initialization. CPU Torch cannot decide the CUDA dependency-import path.
  Expect less than one GPU minute; no kernels/device allocations intended.
  Main must still fail on forbidden Git; this is harness validation, not a
  claim that fixture implementation or frozen selector execution passed.
- First GPU attempt: 1 expected-red test, but an additional controller-side
  bootstrap boundary was discovered. FROZEN_GUARD_READY proves the fresh-child
  bootstrap succeeded; main then imports Torch in the controller, invoking
  ldconfig under CheckedChild. Read the retained traceback. The controller now
  also initializes genuine dependencies under network denial before wrapping
  Popen. No fixture/selector work is prewarmed in either interpreter.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1e89452204b4adc7298f03a8ed87691c5aa1c6450ed0508ad330c1b7b29662c4","size":6290}},"operation":"003bcf7c7954854e46506685778c17c10b087a4103e7a8fbf819867158d23266","receipt":"695129667d309bcdd3ae97420bed83453ce7e47a52716f77c40d5d1769b68320"}]
- Second GPU run budget before submission: same single snapshot case, less
  than one GPU minute, to verify controller and child prerequisite boundaries
  both reach the real pre-fix Git refusal. CPU cannot validate CUDA imports.
- Independent source inspection (not a test receipt): local historical tree
  4d0a7c9abc8c699d29d3772aa99bdbaab1754cd7 at 2dbac191 exported with the
  exact selections is byte-for-byte equal to the recovered fixture tar:
  4,362,240 bytes, 126 files/6 directories, SHA256 equal to EXPORT_SHA256 below.
  Recovered gzip is 1,236,761 bytes with SHA256 equal to ARCHIVE_SHA256 below.
  No network or fetch was used. Implementer must preserve/re-record identity
  verification when recovering these artifacts; checks must never use Git.
- Final GPU targeted re-run: 1 failed in 5.17s, no skips/uncollected modules,
  zero device allocations; Torch 2.13.0+cu130 on GB10. Guard marker precedes
  refusal of the actual Git archive invocation, with no ldconfig error.
  This verifies the revised harness boundary, not post-fix selector success.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"61e94075b553e884280e4a4fa79efd54b8a5bbd280d3dde83e7ac69332dccdc5","size":4484}},"operation":"2bca49036c2e3d3879d3574de16f42f041ffa30ffcbf7796083c2ee309219a4b","receipt":"d7c32cfe8aa412ced6e37286b5b5935a9bfcb8c043b772aae6a2d39bafe70b59"}]
- Final x86 acceptance + entire eager-levers module: 13 failed, 35 passed in
  20.73s, no skips/uncollected modules/device allocations. Same expected
  missing-fixture/loader/Git failures. Both standard x86 guards had passed
  in the earlier combined run; they were not repeated for bootstrap-only
  edits. Retained stdout for each cited run was read, not merely submitted.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"896a13bab76ca34f63de05a67d5916db00ee6fa156023f4530cfdbe343325297","size":24719}},"operation":"64907e972fb41435b2033bf23067a16da67bd6f71a4d9d9242c60827f552bedd","receipt":"b6114119a702b03e20d980591872bb277f38f11c86c610c681770f8acb1ded79"}]
- Impacted selector research: Git index staging is unavailable (index.lock
  permission denied). Called tools/impacted_tests.py's select API on the
  explicit untracked test at the stated base: verdict narrowed, includes
  this module and eager-levers. Broad additional selection is not an executed
  suite. Implementation should run the normal CLI against the exact base
  after its changes are tracked; full integration belongs to the coordinator.
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
EXPORT_SHA256 = "91f3a340e9e14068e448bbc150a702d16e913cdf70fc487e7acbe0a17fb272fb"


def loader():
    value = getattr(levers, "_load_frozen_runtime", None)
    assert callable(value), "historical Git export has not been replaced by an authenticated fixture loader"
    return value


def fixture_bytes():
    archive = FIXTURE / "runtime.tar.gz"
    manifest = FIXTURE / "manifest.json"
    assert archive.is_file(), "missing committed frozen runtime archive (not an optional dependency)"
    assert manifest.is_file(), "missing committed frozen runtime provenance"
    return archive.read_bytes(), json.loads(manifest.read_text())


def test_fixture_is_the_exact_historical_export(tmp_path):
    compressed, manifest = fixture_bytes()
    assert levers.benchmark.RUNTIME_COMMIT == COMMIT
    assert manifest["source_commit"] == COMMIT
    assert manifest["archive_sha256"] == hashlib.sha256(compressed).hexdigest()
    exported = gzip.decompress(compressed)
    assert manifest["export_sha256"] == EXPORT_SHA256 == hashlib.sha256(exported).hexdigest()
    command = manifest["generation_command"]
    assert COMMIT in command and "git" in command and "archive" in command
    for selection in ("src", "pyproject.toml", "tests/test_piece_major_reader_boundaries.py"):
        assert selection in command
    destination = tmp_path / "runtime"
    destination.mkdir()
    loader()(destination)
    # Verify actual loaded bytes, not just an identity label or a subset export.
    with tarfile.open(fileobj=io.BytesIO(exported)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                assert (destination / member.name).read_bytes() == tar.extractfile(member).read()


@pytest.mark.parametrize("fault", ["missing_archive", "missing_manifest", "corrupt", "wrong_commit", "wrong_digest", "runtime_pin_moved"])
def test_loader_refuses_invalid_fixture_before_extracting(tmp_path, monkeypatch, fault):
    load = loader()
    compressed, manifest = fixture_bytes()
    archive = tmp_path / "runtime.tar.gz"
    metadata = tmp_path / "manifest.json"
    if fault == "corrupt":
        compressed = compressed[:-1] + bytes([compressed[-1] ^ 1])
    if fault == "wrong_commit":
        manifest["source_commit"] = "0" * 40
    if fault == "wrong_digest":
        manifest["archive_sha256"] = "0" * 64
    if fault == "runtime_pin_moved":
        monkeypatch.setattr(levers.benchmark, "RUNTIME_COMMIT", "0" * 40)
    archive.write_bytes(compressed)
    metadata.write_text(json.dumps(manifest))
    if fault == "missing_archive":
        archive.unlink()
    if fault == "missing_manifest":
        metadata.unlink()
    destination = tmp_path / "runtime"
    destination.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        load(destination, archive_path=archive, manifest_path=metadata)
    # rmdir asserts emptiness without enumerating an unnameable scratch tree.
    destination.rmdir()


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink", "hardlink"])
def test_loader_rejects_rebound_unsafe_archive(tmp_path, attack):
    load = loader()
    _, manifest = fixture_bytes()
    buffer = io.BytesIO()
    outside = tmp_path / "escaped"
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        member = tarfile.TarInfo(str(outside) if attack == "absolute" else "../escaped" if attack == "traversal" else "src/link")
        if attack in ("symlink", "hardlink"):
            member.type = tarfile.SYMTYPE if attack == "symlink" else tarfile.LNKTYPE
            member.linkname = str(outside)
        else:
            member.size = 4
        tar.addfile(member, io.BytesIO(b"evil") if member.isfile() else None)
    exported = buffer.getvalue()
    compressed = gzip.compress(exported, mtime=0)
    # Rebinding both digest labels must not authorize arbitrary new source.
    manifest.update(archive_sha256=hashlib.sha256(compressed).hexdigest(),
                    export_sha256=hashlib.sha256(exported).hexdigest())
    archive = tmp_path / "runtime.tar.gz"
    metadata = tmp_path / "manifest.json"
    archive.write_bytes(compressed)
    metadata.write_text(json.dumps(manifest))
    destination = tmp_path / "runtime"
    destination.mkdir()
    with pytest.raises(ValueError):
        load(destination, archive_path=archive, manifest_path=metadata)
    assert not outside.exists()
    # rmdir asserts emptiness without enumerating an unnameable scratch tree.
    destination.rmdir()


# Networking is denied from fresh-child startup. Only genuine third-party
# prerequisite initialization precedes command denial: CUDA Torch can invoke
# ldconfig while importing cuda.pathfinder. No frozen Tessera source may be
# prewarmed. Absence is left to the actual target's narrow optional-dependency
# policy; transitive failures are never suppressed here.
CHILD_BOOTSTRAP = r'''
import importlib
import sys

def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise AssertionError('network is forbidden: ' + event)
sys.addaudithook(offline)

def require_denied(event, message):
    try:
        sys.audit(event)
    except AssertionError as exc:
        assert message in str(exc), (event, exc)
    else:
        raise AssertionError('audit guard did not deny ' + event)

for event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
    require_denied(event, 'network is forbidden')
# Synthetic event, not an actual external command. This boundary is intentional.
sys.audit('subprocess.Popen')
for prerequisite in ('torch', 'triton'):
    try:
        importlib.import_module(prerequisite)
    except ModuleNotFoundError as exc:
        if exc.name != prerequisite:
            raise
        break
assert not any(name == 'tessera' or name.startswith('tessera.')
               or name == 'test_piece_major_reader_boundaries' for name in sys.modules)

def no_external(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
        raise AssertionError('external command in frozen child')
sys.addaudithook(no_external)
for event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
    require_denied(event, 'network is forbidden')
for event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
    require_denied(event, 'external command in frozen child')
'''

# Exercise the actual target unchanged through a history-free mini-snapshot.
# Audit hooks establish Python-level denial, not an OS-level sandbox claim.
SNAPSHOT_PROBE = r'''
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / "tests"), str(root / "src")]
assert not (root / ".git").exists()

def offline(event, args):
    if event.startswith("socket.") and event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise AssertionError("network is forbidden: " + event)
sys.addaudithook(offline)

# Main's pre-fix target also imports Torch in the controller. Initialize only
# genuine prerequisites before replacing Popen there, under network denial.
# The separate selector child still initializes its own genuine dependencies.
for prerequisite in ('torch', 'triton'):
    try:
        __import__('importlib').import_module(prerequisite)
    except ModuleNotFoundError as exc:
        if exc.name != prerequisite:
            raise
        break

prefix = @CHILD_BOOTSTRAP@
# Validate the harness independently even when the target still attempts Git.
# This is a guard-only child, not a second selector process or a fake Torch.
guard = subprocess.run([sys.executable, '-c', prefix + "\nprint('FROZEN_GUARD_READY')"],
                       cwd=root, capture_output=True, text=True, timeout=30)
assert guard.returncode == 0, guard.stdout + guard.stderr
assert 'FROZEN_GUARD_READY' in guard.stdout, guard.stdout + guard.stderr
print(guard.stdout, end='', flush=True)
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
        assert Path(args[0]).resolve() == Path(sys.executable).resolve(), ('external command forbidden', args)
        assert '-c' in args, args
        assert Path(kwargs['cwd']).resolve() == (root / 'probe-runtime').resolve(), kwargs
        args = list(args)
        program_index = args.index('-c') + 1
        args[program_index] = prefix + '\n' + args[program_index] + '\n' + suffix
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
    reason = str(exc)
    assert 'torch' in reason.lower() or 'triton' in reason.lower(), reason
    print('FROZEN_RESULT=' + json.dumps(dict(status='skip', reason=reason)))
else:
    assert len(seen) == 1, seen
    print('FROZEN_RESULT=' + json.dumps(dict(status='executed')))
'''.replace('@CHILD_BOOTSTRAP@', repr(CHILD_BOOTSTRAP))


def test_actual_target_executes_offline_without_history(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "tests").mkdir(parents=True)
    shutil.copy2(ROOT / "pyproject.toml", snapshot / "pyproject.toml")
    # Controller imports use current src; the fresh child's imports are checked
    # separately against the frozen extraction, never this controller source.
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    # Named directories keep these data edges visible to impacted_tests.
    shutil.copytree(ROOT / "experiments/graph_attest_702", snapshot / "experiments/graph_attest_702",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "tests/test_graph_attest_eager_levers.py", snapshot / "tests")
    if FIXTURE.exists():
        shutil.copytree(FIXTURE, snapshot / "tests/fixtures/frozen_eager_2dbac191")
    script = tmp_path / "offline_probe.py"
    script.write_text(SNAPSHOT_PROBE)
    done = subprocess.run([sys.executable, str(script), str(snapshot)], cwd=snapshot,
                          env=dict(os.environ), capture_output=True,
                          text=True, timeout=90)
    assert done.returncode == 0, done.stdout + done.stderr
    records = [json.loads(line.removeprefix("FROZEN_RESULT=")) for line in done.stdout.splitlines()
               if line.startswith("FROZEN_RESULT=")]
    assert len(records) == 1, done.stdout + done.stderr
    if records[0]["status"] == "skip":
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
        else:
            assert not torch.cuda.is_available(), "configured GPU target must execute, not skip: " + records[0]["reason"]
        pytest.skip("CPU frozen-selector dependency: " + records[0]["reason"])
    assert records[0]["status"] == "executed"
