"""Native container authority, starting with independently known commands."""
import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import _suite_container as container
import merge_suite as merge
import _suite_flux as flux


def composed(tmp_path, targets=None, cpus=2):
    arm = {**merge.ARMS["gpu"], "python": "/usr/bin/python3", "container": True,
           "fans_out": True, "targets": targets or ["tests/test_grammar.py"]}
    surface = tmp_path / "output/surface.gpu.json"
    inner = merge._command(arm, surface, [], cpus)
    inner += ["--junitxml", str(surface.parent / "junit.gpu.xml")]
    return ["/usr/bin/python3", "-s", container.RUNNER,
            "--image", "sha256:" + "a" * 64,
            "--deps-site", str(tmp_path / "dependencies"), "--deps-sha256", "b" * 64,
            "--surface-dir", str(surface.parent), "--cache-dir", str(tmp_path / "cache"),
            "--", *inner]


def test_actual_generated_command_passes_finite_parser(tmp_path):
    command = composed(tmp_path)
    spec = container.command_spec(command)
    assert spec["workers"] == 2
    assert spec["surface"] == str(tmp_path / "output/surface.gpu.json")
    assert spec["junit"] == str(tmp_path / "output/junit.gpu.xml")
    assert merge._effective_surface_json(merge._timed_command(command, 60)) == (spec["surface"], None)


@pytest.mark.parametrize("shadow", [
    ["-o", "pythonpath=/foreign"], ["-o", "addopts="], ["-c", "/foreign"],
    ["--junitxml", "/foreign.xml"], ["--surface-json", "/foreign.json"],
    ["-n", "1"], ["-p", "xdist.plugin"], ["-p", "foreign.plugin"],
    ["tests/test_other.py"], ["tests/test_grammar.py::test_x"],
])
def test_composed_controls_cannot_be_shadowed(tmp_path, shadow):
    command = composed(tmp_path)
    assert container.command_spec(command)["workers"] == 2
    with pytest.raises(ValueError):
        container.command_spec(command + shadow)


@pytest.mark.parametrize("targets", [
    ["tests/a.py", "tests/a.py"], ["tests/./a.py"],
    ["tests/a", "tests/a-else.py", "tests/a/b.py"],
    ["tests/a.py::test_a", "tests/a.py::test_a-other", "tests/a.py::test_a[param]"],
])
def test_parser_refuses_target_aliases_and_nonadjacent_overlap(tmp_path, targets):
    command = composed(tmp_path)
    assert container.command_spec(command)["workers"] == 2
    start = command.index("pytest") + 1
    command[start:start + 1] = targets
    with pytest.raises(ValueError, match="target"):
        container.command_spec(command)


@pytest.fixture
def native_record(tmp_path):
    """Known hand-authored execution, independent of producer functions."""
    import _suite_flux as flux
    directory = tmp_path / "native"
    directory.mkdir()
    source = {"verification": "verified", "snapshot_commit": "c" * 40, "sha256": "d" * 64}
    intent = {"container": {"--image": "sha256:" + "a" * 64, "--deps-sha256": "b" * 64},
              "source": source, "resources": {"architecture": "x86_64", "gpus": 0}}
    entry = {"schema": "tessera.suite_container_runtime.v1", "job_id": "ƒ123",
             "launch_sha256": "e" * 64, "source": source, "dependencies": "b" * 64,
             "affinity": [2, 42], "driver": None,
             "runtime": {"schema": "tessera.suite_runtime.v1", "sha256": "f" * 64}}
    record = {"schema": "tessera.suite_container_execution.v1", "job_id": "f123",
              "launch_sha256": "e" * 64, "source": source,
              "image": {"Id": "sha256:" + "a" * 64, "Architecture": "amd64", "Os": "linux"},
              "gpus": 0, "affinity": [2, 42], "dependencies": "b" * 64,
              "removed": True, "cleanup_errors": [], "entry": entry, "exit": copy.deepcopy(entry),
              "driver": None, "cid": "1" * 64,
              "inspection": {"Id": "1" * 64, "Image": "sha256:" + "a" * 64, "CpusetCpus": "2,42"}}
    def publish():
        for phase in ("entry", "exit"):
            flux.save(directory / f"container.{phase}.json", record[phase])
            record[phase + "_sha256"] = flux.hashlib.sha256((directory / f"container.{phase}.json").read_bytes()).hexdigest()
        flux.save(directory / "container.json", record)
        digest = flux.hashlib.sha256((directory / "container.json").read_bytes()).hexdigest()
        (directory / "stdout").write_text(f"Tessera container evidence {directory}/container.json sha256={digest}\n")
    publish()
    assert container.verify_record(directory, intent, "e" * 64, "ƒ123", [2, 42]) == record
    return directory, intent, record, publish


@pytest.mark.parametrize("fact", ["source", "image", "image-arch", "deps", "job", "launch", "cpus", "gpu", "cid",
                                   "runtime", "inside-source", "inside-deps", "inside-cpus", "driver", "cleanup", "removed",
                                   "missing", "replaced", "symlink"])
def test_native_verifier_accepts_positive_before_mutation(native_record, fact):
    directory, intent, record, publish = native_record
    if fact in ("source", "image", "deps", "job", "launch", "cpus", "gpu", "cid", "image-arch"):
        if fact == "source": record["source"] = {"sha256": "0" * 64}
        elif fact == "image": record["image"]["Id"] = "sha256:" + "0" * 64
        elif fact == "image-arch": record["image"]["Architecture"] = "arm64"
        elif fact == "deps": record["dependencies"] = "0" * 64
        elif fact == "job": record["job_id"] = "f124"
        elif fact == "launch": record["launch_sha256"] = "0" * 64
        elif fact == "cpus": record["affinity"] = [3, 43]
        elif fact == "gpu": record["gpus"] = 1
        elif fact == "cid": record["cid"] = "2" * 64
    elif fact == "runtime": record["exit"]["runtime"]["sha256"] = "0" * 64
    elif fact.startswith("inside-") or fact == "driver":
        key = {"inside-source": "source", "inside-deps": "dependencies", "inside-cpus": "affinity", "driver": "driver"}[fact]
        record["entry"][key] = ["unexpected driver"] if fact == "driver" else None
        record["exit"][key] = ["unexpected driver"] if fact == "driver" else None
    elif fact == "cleanup": record["cleanup_errors"] = ["failed to delete owned container"]
    elif fact == "removed": record["removed"] = False
    publish()
    path = directory / "container.json"
    if fact == "missing": path.unlink()
    elif fact == "replaced": path.write_text(path.read_text() + " ")
    elif fact == "symlink":
        target = directory / "replacement.json"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        container.verify_record(directory, intent, "e" * 64, "f123", [2, 42])


def test_native_source_authentication_and_mutation_replace_bundle_fixture(tmp_path, monkeypatch):
    import subprocess
    import _suite_flux as flux
    from tessera._dev.suite_source import measured_source
    checkout = tmp_path / "source"
    checkout.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], stderr=subprocess.DEVNULL)
    git("init", "--quiet")
    (checkout / "runner.py").write_text("print('original authenticated source')\n")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@local", "commit", "--quiet", "-m", "original")
    git("checkout", "--detach", "HEAD")
    identity = measured_source(checkout, verifier=None)
    assert identity["verification"] == "verified"
    directory = tmp_path / "record"
    directory.mkdir()
    intent = {"directory": str(directory), "checkout": str(checkout), "source": identity,
              "qualification": {"profile": "fixture"},
              "command": ["/usr/bin/python3", container.RUNNER],
              "resources": {"architecture": "x86_64", "cores": 1, "memory_gib": 1, "gpus": 0}}
    flux.save(directory / "launch.json", intent)
    digest = flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest()
    flux.save(directory / "submission.json", {"state": "submitted", "job_id": "f123", "launch_sha256": digest})
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(flux, "source", lambda root: measured_source(root, verifier=None))
    _mock_native_admission(monkeypatch, directory, intent, digest)
    assert container.admitted_context(directory, digest, {"FLUX_JOB_ID": "ƒ123"}) == intent
    (checkout / "runner.py").write_text("print('forged source')\n")
    with pytest.raises(ValueError, match="source authentication"):
        container.admitted_context(directory, digest, {"FLUX_JOB_ID": "f123"})
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@local", "commit", "--quiet", "-m", "replacement")
    with pytest.raises(ValueError, match="source authentication"):
        container.admitted_context(directory, digest, {"FLUX_JOB_ID": "f123"})
    with pytest.raises(ValueError, match="job ID"):
        container.admitted_context(directory, digest, {"FLUX_JOB_ID": "f124"})


@pytest.mark.parametrize("gpus,arch", [(2, "aarch64"), (0.5, "aarch64"), (True, "aarch64"), (1, "x86_64"), (0, "arm")])
def test_finite_resource_architecture_and_whole_gpu_intent(gpus, arch):
    import _suite_flux as flux
    assert flux.resources(2, 16, 600, gpus=1, architecture="aarch64") == {
        "schema": "tessera.suite_resources.v1", "cores": 2, "memory_gib": 16,
        "timeout_s": 600, "gpus": 1, "architecture": "aarch64"}
    with pytest.raises(ValueError):
        flux.resources(2, 16, 600, gpus=gpus, architecture=arch)


def test_cleanup_failure_is_retained_even_when_remove_fails(tmp_path, monkeypatch):
    cid = "1" * 64
    path = tmp_path / "container.cid"
    path.write_text(cid)
    record = {"cidfile": str(path), "cid": cid, "cleanup_errors": [], "removed": False}
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=125, stderr="owned cleanup denied")
    monkeypatch.setattr(container.subprocess, "run", run)
    container.cleanup_cid(record)
    assert len(record["cleanup_errors"]) == 3
    assert all(argv[-1] == cid for argv in calls)
    assert record["removed"] is False


def test_original_native_wrapper_authenticates_launch_bytes():
    import _suite_flux as flux
    intent = {"directory": "/owned/native", "command": ["/named/python", "payload"],
              "resources": {"cores": 2}}
    original = {"tasks": [{"command": ["/named/python", "-s", "tools/_suite_flux.py", "payload", "/owned/native", "a" * 64]}]}
    flux.require_original_launch(intent, "a" * 64, original)
    with pytest.raises(ValueError, match="launch digest"):
        flux.require_original_launch(intent, "b" * 64, original)
    with pytest.raises(ValueError, match="launch digest"):
        flux.require_original_launch({**intent, "command": ["/other/python"]}, "a" * 64, original)


def _allocation(core_properties):
    core = {"type": "core", "rank": 0, "id": 2, "properties": {"os-pus": "2,42", **core_properties}}
    return {"execution": {"R_lite": [{"rank": "0", "children": {"core": "2"}}], "nodelist": ["dl380g10"]},
            "scheduling": {"graph": {"nodes": [{"metadata": item} for item in [
        core, {"type": "memory", "rank": 0, "id": 0},
        {"type": "node", "rank": 0, "name": "dl380g10", "properties": {"arch": "x86_64"}}]]}}}


@pytest.mark.parametrize("fact", ["cores", "memory_gib", "gpus", "architecture", "affinity", "host"])
def test_actual_allocation_refuses_different_resource_ownership(fact):
    import _suite_flux as flux
    demand = {"architecture": "x86_64", "cores": 1, "memory_gib": 1, "gpus": 0, "host": "dl380g10"}
    allocation = _allocation({"core-type": "perf"})
    flux.require_allocation(allocation, demand, [2, 42], [2, 42])
    if fact in ("cores", "memory_gib", "gpus"):
        demand[fact] += 1
    elif fact == "architecture": demand[fact] = "aarch64"
    elif fact == "host": demand["host"] = "sparky"
    with pytest.raises(ValueError, match="allocation"):
        flux.require_allocation(allocation, demand, [3, 43] if fact == "affinity" else [2, 42], [2, 42])


@pytest.mark.parametrize("properties", [{"core-type": "perf"}, {"core-type": "efficiency"}, {}])
def test_actual_allocation_is_class_agnostic(properties):
    """Pre-fix: an efficiency core, or a core with no class, was refused."""
    import _suite_flux as flux
    demand = {"architecture": "x86_64", "cores": 1, "memory_gib": 1, "gpus": 0, "host": "dl380g10"}
    flux.require_allocation(_allocation(properties), demand, [2, 42], [2, 42])


@pytest.mark.parametrize("reported", ["a" * 64, "sha256:" + "a" * 64])
def test_actual_podman_image_id_forms_bind_same_immutable_content(monkeypatch, reported):
    payload = [{"Id": reported, "Architecture": "amd64", "Os": "linux"}]
    monkeypatch.setattr(container.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    assert container.image_identity("sha256:" + "a" * 64) == {"Id": "sha256:" + "a" * 64, "Architecture": "amd64", "Os": "linux"}
    with pytest.raises(ValueError, match="content"):
        container.image_identity("sha256:" + "b" * 64)


@pytest.mark.parametrize("boundary", ["interrupted-create", "failed-start", "cleanup-fails"])
def test_cid_registration_precedes_creation_and_errors_survive(tmp_path, monkeypatch, boundary):
    import signal
    import _suite_flux as flux
    command = composed(tmp_path, cpus=1)
    spec = container.command_spec(command)
    spec["strict_cuda"] = False
    spec["inner"].remove("--strict-cuda")
    directory = tmp_path / "output/native"
    directory.mkdir(parents=True)
    scratch = tmp_path / "cache/f123"
    scratch.mkdir(parents=True)
    intent = {"container": spec, "source": {"sha256": "d" * 64},
              "resources": {"cores": 1, "gpus": 0, "architecture": "x86_64"}}
    monkeypatch.setattr(container, "validate_mounts", lambda *args: [])
    monkeypatch.setattr(container, "dependency_manifest", lambda *args: ({}, "b" * 64))
    monkeypatch.setattr(container, "image_identity", lambda *args: {"Id": "sha256:" + "a" * 64, "Architecture": "amd64", "Os": "linux"})
    monkeypatch.setattr(container, "podman_command", lambda *args, **kwargs: ["/usr/bin/podman", "create"])
    cid = "1" * 64
    launches = []
    def popen(argv, **kwargs):
        launches.append(argv)
        if len(launches) == 1:
            registered = json.loads((directory / "container.json").read_bytes())
            assert registered["cidfile"] == str(directory / "container.cid")
            assert registered["cid"] is None
            Path(registered["cidfile"]).write_text(cid)
            if boundary != "failed-start":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return SimpleNamespace(communicate=lambda: (cid, ""), returncode=0, poll=lambda: 0)
        assert argv == ["/usr/bin/podman", "start", "--attach", cid]
        return SimpleNamespace(wait=lambda: 125, poll=lambda: 125)
    monkeypatch.setattr(container.subprocess, "Popen", popen)
    inspection = [{"Id": cid, "Image": "a" * 64, "HostConfig": {"CpusetCpus": ",".join(map(str, sorted(os.sched_getaffinity(0))))}}]
    monkeypatch.setattr(container.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(inspection)))
    touched = []
    def cleanup(record):
        assert Path(record["cidfile"]).read_text() == cid
        touched.append(cid)
        record["cid"] = cid
        if boundary == "cleanup-fails":
            raise PermissionError("owned CID remove failed")
        record["removed"] = True
    monkeypatch.setattr(container, "cleanup_cid", cleanup)
    if boundary == "failed-start":
        with pytest.raises(FileNotFoundError):
            container.run(spec, tmp_path, {"FLUX_JOB_ID": "f123", "TMPDIR": str(scratch)}, directory, intent, "e" * 64)
    else:
        assert container.run(spec, tmp_path, {"FLUX_JOB_ID": "f123", "TMPDIR": str(scratch)}, directory, intent, "e" * 64) == 143
    record = json.loads((directory / "container.json").read_bytes())
    assert touched == [cid]
    assert record["cid"] == cid
    if boundary == "cleanup-fails":
        assert record["removed"] is False and "owned CID remove failed" in record["cleanup_errors"][-1]
    else:
        assert record["removed"] is True


def _mock_native_admission(monkeypatch, directory, intent, digest):
    import _suite_flux as flux
    original = {'tasks': [{'command': ['/usr/bin/python3', '-s', 'tools/_suite_flux.py', 'payload', str(directory), digest]}]}
    allocation = _allocation({"core-type": "perf"})
    state = {'original': original, 'allocation': allocation, 'events': [{'name': 'start'}],
             'qualification': {'profile': 'fixture', 'broker_starttime': 'generation-A'},
             'cgroup': '0::/user.slice/user-1000.slice/user@1000.service/flux.slice/flux-jobs.slice/shell-f123:sys.service\n'}
    def checked(argv):
        if argv == ['job', 'info', '--original', 'f123', 'jobspec']:
            return json.dumps(state['original'])
        if argv == ['job', 'info', 'f123', 'R']:
            return json.dumps(state['allocation'])
        assert argv == ['job', 'eventlog', '--format=json', 'f123']
        return '\n'.join(json.dumps(event) for event in state['events'])
    original_read = Path.read_text
    monkeypatch.setattr(flux, 'broker_starttime', lambda **kw: state['qualification']['broker_starttime'])
    monkeypatch.setattr(flux, 'checked', checked)
    monkeypatch.setattr(container.os, 'sched_getaffinity', lambda pid: {2, 42})
    monkeypatch.setattr(Path, 'read_text', lambda path, *args, **kwargs: state['cgroup'] if str(path) == '/proc/self/cgroup' else original_read(path, *args, **kwargs))
    return state


@pytest.mark.parametrize('fact', ['original', 'allocation', 'terminal', 'process-scope', 'instance'])
def test_direct_container_entry_authenticates_actual_native_context(tmp_path, monkeypatch, fact):
    import _suite_flux as flux
    checkout = tmp_path / 'source'
    checkout.mkdir()
    directory = tmp_path / 'receipt'
    directory.mkdir()
    monkeypatch.chdir(checkout)
    identity = {'schema': 'tessera.suite_source.v1', 'verification': 'verified', 'snapshot_commit': 'a' * 40, 'sha256': 'b' * 64}
    intent = {'directory': str(directory), 'checkout': str(checkout), 'source': identity,
              'qualification': {'profile': 'fixture', 'broker_starttime': 'generation-A'},
              'command': ['/usr/bin/python3', 'tools/suite_container.py'],
              'resources': {'architecture': 'x86_64', 'cores': 1, 'memory_gib': 1, 'gpus': 0}}
    flux.save(directory / 'launch.json', intent)
    digest = flux.hashlib.sha256((directory / 'launch.json').read_bytes()).hexdigest()
    flux.save(directory / 'submission.json', {'state': 'submitted', 'job_id': 'f123', 'launch_sha256': digest})
    monkeypatch.setattr(flux, 'source', lambda root: identity)
    state = _mock_native_admission(monkeypatch, directory, intent, digest)
    assert container.admitted_context(directory, digest, {'FLUX_JOB_ID': 'ƒ123'}) == intent
    if fact == 'original':
        state['original']['tasks'][0]['command'][-1] = '0' * 64
    elif fact == 'allocation':
        state['allocation']['scheduling']['graph']['nodes'][0]['metadata']['properties']['os-pus'] = '3,43'
    elif fact == 'terminal':
        state['events'].append({'name': 'clean'})
    elif fact == 'instance':
        state['qualification'] = {'broker_starttime': 'generation-B'}
    else:
        state['cgroup'] = state['cgroup'].replace('shell-f123:', 'shell-f124:')
    with pytest.raises(ValueError):
        container.admitted_context(directory, digest, {'FLUX_JOB_ID': 'f123'})
@pytest.mark.parametrize('handle', ['present', 'missing', 'conflicting'])
def test_captured_cid_remains_the_only_cleanup_authority(tmp_path, monkeypatch, handle):
    owned, foreign = '1' * 64, '2' * 64
    path = tmp_path / 'container.cid'
    path.write_text(owned)
    live = {owned, foreign}
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[-1] == owned, 'cleanup touched a foreign container'
        if argv[1] == 'rm':
            live.discard(owned)
        code = (0 if owned in live else 1) if argv[1:3] == ['container', 'exists'] else 0
        return SimpleNamespace(returncode=code, stderr='')
    monkeypatch.setattr(container.subprocess, 'run', run)
    positive = {'cidfile': str(path), 'cid': owned, 'removed': False, 'cleanup_errors': []}
    container.cleanup_cid(positive)
    assert positive['removed'] and live == {foreign} and len(calls) == 3
    live.add(owned); calls.clear()
    if handle == 'missing':
        path.unlink()
    elif handle == 'conflicting':
        path.write_text(foreign)
    record = {'cidfile': str(path), 'cid': owned, 'removed': False, 'cleanup_errors': []}
    container.cleanup_cid(record)
    if handle == 'conflicting':
        assert calls == [] and live == {owned, foreign}
        assert not record['removed'] and 'replaced' in record['cleanup_errors'][-1]
    else:
        assert record['removed'] and live == {foreign} and len(calls) == 3


@pytest.mark.parametrize('creation_code', [None, 125])
def test_uncaptured_partial_creation_stays_unknown(tmp_path, monkeypatch, creation_code):
    calls = []
    monkeypatch.setattr(container.subprocess, 'run', lambda argv, **kwargs: calls.append(argv))
    record = {'cidfile': str(tmp_path / 'missing.cid'), 'cid': None,
              'removed': False, 'cleanup_errors': [], 'creation_returncode': creation_code}
    container.cleanup_cid(record)
    assert calls == [] and record['removed'] is False
    assert record['cleanup_errors'] and 'unknown' in record['cleanup_errors'][-1]


def test_default_paired_cli_applies_gpu_options_only_to_gpu_arm(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(merge.suite_container, 'require_local_cache', lambda path: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must never submit')
    monkeypatch.setattr(merge.suite_flux, 'submit', forbidden)
    options = ['merge_suite', '--dry-run', '--checkout', str(Path(merge.__file__).resolve().parents[1]),
               '--out', str(tmp_path / 'output/receipt.json'), '--python', sys.executable,
               '--gpu-image', 'sha256:' + 'a' * 64, '--gpu-deps-site', str(tmp_path / 'deps'),
               '--gpu-deps-sha256', 'b' * 64, '--gpu-cache-dir', str(tmp_path / 'cache'),
               '--target', 'tests/test_grammar.py', '--timeout-s', '60', '--wait-s', '90']
    monkeypatch.setattr(sys, 'argv', [*options, '--arm', 'gpu'])
    assert merge.main() == 0
    single = json.loads(capsys.readouterr().out)
    assert [arm['arm'] for arm in single['arms']] == ['gpu']
    monkeypatch.setattr(sys, 'argv', options)
    assert merge.main() == 0
    paired = json.loads(capsys.readouterr().out)
    assert {arm['arm'] for arm in paired['arms']} == {'gpu', 'x86'}
    assert all('command' in arm for arm in paired['arms']), paired
    for arm in paired['arms']:
        command, refusal = merge._pytest_argv(arm['command'])
        assert refusal is None
        assert ('--strict-cuda' in command) == (arm['arm'] == 'gpu')
        assert merge._targets_of(command) == ['tests/test_grammar.py']
    assert not (tmp_path / 'output/receipt.json').exists()


@pytest.mark.parametrize('code', [0, 7])
@pytest.mark.parametrize('broken', ['absent', 'truncated', 'wrong-type', 'unreadable', 'stdout'])
def test_saved_native_failure_survives_unusable_publication(tmp_path, monkeypatch, code, broken):
    spec = importlib.util.spec_from_file_location('review_merge_helpers', Path(merge.__file__).resolve().parents[1] / 'tests/test_merge_suite.py')
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    jobs = helpers._native_jobs(merge, monkeypatch)
    world = jobs.add(tmp_path, 'x86', helpers._population('x86')['surface'])
    positive = merge._resume('x86', merge.ARMS['x86'], tmp_path)
    assert merge._verdict([positive]).startswith('green on')
    world.code = code
    if broken == 'absent':
        world.surface.unlink()
    elif broken == 'truncated':
        world.surface.write_text('{')
    elif broken == 'wrong-type':
        world.surface.write_text('[]')
    else:
        method = 'read_text' if broken == 'stdout' else 'read_bytes'
        target = world.directory / 'stdout' if broken == 'stdout' else world.surface
        original = getattr(Path, method)
        def refused(path, *args, **kwargs):
            if path == target:
                raise PermissionError('owned publication read denied')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, method, refused)
    record = merge._resume('x86', merge.ARMS['x86'], tmp_path)
    assert record['native_job']['exit_status_observed']
    assert record['native_job']['returncode'] == code
    if code:
        assert record['returncode'] == code
        assert merge._verdict([record]).startswith('red')
    else:
        assert merge._verdict([record]).startswith('incomplete')
    if broken != 'absent':
        assert record['publication_errors']
    assert all('submit' not in call for call in jobs.calls)
@pytest.mark.parametrize('extra,expected', [([], '10'), (['--durations=5'], '5'), (['-q'], '10')])
def test_final_submit_composition_owns_presentation_defaults(tmp_path, extra, expected):
    arm = {**merge.ARMS['gpu'], 'python': '/usr/bin/python3', 'fans_out': True,
           'targets': ['tests/test_grammar.py'], 'container': {
               'image': 'sha256:' + 'a' * 64, 'deps_site': str(tmp_path / 'deps'),
               'deps_sha256': 'b' * 64, 'cache_dir': str(tmp_path / 'cache')}}
    args = SimpleNamespace(cpus=1, mem_gb=1, timeout_s=60, wait_s=90,
                           artifact_root=[], gpu_image='sha256:' + 'a' * 64,
                           pytest_arg=extra, dry_run=True)
    result = merge._submit('gpu', arm, args, tmp_path / 'output')
    command, refusal = merge._pytest_argv(result['command'])
    assert refusal is None
    assert command.count('-q') == 1
    durations = [token.partition('=')[2] if '=' in token else command[i + 1]
                 for i, token in enumerate(command) if token.partition('=')[0] == '--durations']
    assert durations == [expected]
    assert merge._targets_of(command) == ['tests/test_grammar.py']


def test_generated_deadline_chain_excludes_private_user_startup(tmp_path):
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home'),
           'PYTHONUSERBASE': str(tmp_path / 'userbase'), 'PYTHONDONTWRITEBYTECODE': '1'}
    site = subprocess.check_output([sys.executable, '-s', '-c',
                                   'import site;print(site.getusersitepackages())'], env=env, text=True).strip()
    directory = Path(site); directory.mkdir(parents=True)
    marker = tmp_path / 'startup-marker'
    (directory / 'g01_startup_shadow.py').write_text('VALUE = 42\n')
    (directory / 'startup.pth').write_text('import pathlib;pathlib.Path(' + repr(str(marker)) + ').write_text("startup")\n')
    control = subprocess.check_output([sys.executable, '-s', '-c',
        'import site;site.addsitedir(' + repr(site) + ');import g01_startup_shadow;print(g01_startup_shadow.VALUE)'], env=env, text=True)
    assert control.strip() == '42' and marker.read_text() == 'startup'
    marker.unlink()
    probe = ('import importlib.util,json,sys;from pathlib import Path;'
             'print(json.dumps([sys.flags.no_user_site,importlib.util.find_spec("g01_startup_shadow") is None,'
             'not Path(' + repr(str(marker)) + ').exists()]))')
    command = merge._timed_command([sys.executable, '-s', '-c', probe], 30)
    assert command[:3] == [sys.executable, '-s', 'tools/suite_deadline.py']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [1, True, True]


def test_generated_native_wrapper_and_original_authentication_require_no_user_site(tmp_path, monkeypatch):
    command = merge._command({**merge.ARMS['x86'], 'python': sys.executable}, tmp_path / 'surface.json', [], 1)
    assert command[:4] == [sys.executable, '-s', '-m', 'pytest']
    monkeypatch.setattr(flux, 'source', lambda checkout: {'sha256': 'a' * 64})
    monkeypatch.setattr(flux, 'instance_identity', lambda **kw: {'broker_starttime': 'generation-A'})
    monkeypatch.setattr(flux, 'require_capacity', lambda *a, **kw: None)
    monkeypatch.setenv('FLUX_URI', 'local:///test-instance')
    monkeypatch.setenv('FLUX_CLIENT', sys.executable)
    monkeypatch.setattr(flux, 'instance_settings', lambda: tmp_path)
    monkeypatch.setattr(flux, 'runtime_identity', lambda *args, **kwargs: {'schema': 'tessera.suite_runtime.v1'})
    monkeypatch.setattr(flux, 'collect', lambda directory: {'state': 'running'})
    calls = []
    def native(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='{}' if argv[0] == 'submit' else 'f123', stderr='')
    monkeypatch.setattr(flux, 'native', native)
    directory = tmp_path / 'native'
    flux.submit(directory, tmp_path, command, 1, 1, 30, {})
    args = calls[0]
    start = args.index('tools/_suite_flux.py')
    assert args[start - 2:start] == [sys.executable, '-s']
    intent = json.loads((directory / 'launch.json').read_bytes())
    digest = json.loads((directory / 'submission.json').read_bytes())['launch_sha256']
    original = {'tasks': [{'command': args[start - 2:]}]}
    flux.require_original_launch(intent, digest, original)
    original['tasks'][0]['command'].remove('-s')
    with pytest.raises(ValueError, match='launch digest'):
        flux.require_original_launch(intent, digest, original)


def test_runtime_measurement_uses_the_same_isolated_startup(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"schema":"tessera.suite_runtime.v1"}', stderr='')
    monkeypatch.setattr(flux.subprocess, 'run', run)
    flux.runtime_identity('/usr/bin/python3', Path(merge.__file__).resolve().parents[1], timeout=30, wrapper=True)
    argv, options = calls[0]
    assert argv[1:3] == ['-s', '-B']
    assert options['env']['PYTHONNOUSERSITE'] == '1'
