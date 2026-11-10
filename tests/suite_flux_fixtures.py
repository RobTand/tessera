"""Saved native records with a controlled CLI, without contacting a broker."""
import copy
import hashlib
import importlib.util
import sys
import pytest
import json
from pathlib import Path
from types import SimpleNamespace

from tessera._dev.surface_publication import publication_line, POPULATION


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def suite():
    return load("native_merge_suite", ROOT / "tools/merge_suite.py")


def publication(suite, tmp_path):
    from tessera._dev.surface_publication import publication_line, POPULATION
    path = tmp_path / "surface.x86.json"
    identity = {"schema": "tessera.suite_source.v1", "verification": "verified", "snapshot_commit": "a" * 40,
                "sha256": "b" * 64}
    payload = {"schema": "tessera.test_surface.v3", "role": "population",
               "commit": "a" * 40, "source_identity": identity,
               "counts": {"passed": 3, "failed": 0, "skipped": 0}}
    path.write_text(json.dumps(payload))
    published = suite._read_publication(path)
    native = {"command": ["/usr/bin/python3", "-m", "pytest", "tests/test_a.py",
                          "--surface-json", str(path)],
              "source": identity, "returncode": 0, "exit_status_observed": True}
    stdout = publication_line(POPULATION, path, published.digest) + "\n3 passed in 0.10s\n"
    return published, native, stdout


def resource_set():
    nodes = []
    for rank, arch, cores, memory, gpus in [(4, 'x86_64', 4, 16, 0), (9, 'aarch64', 4, 16, 1)]:
        items = [{'type': 'node', 'name': f'worker-{rank}', 'properties': {'arch': arch}}]
        for kind, count in [('core', cores), ('memory', memory), ('gpu', gpus)]:
            items += [{'type': kind, 'id': i, 'size': 1, 'unit': 'GiB' if kind == 'memory' else '',
                       'properties': {'core-type': 'eff'}} for i in range(count)]
        nodes += [{'id': str(len(nodes) + i), 'metadata': {**item, 'rank': rank}} for i, item in enumerate(items)]
    return {'scheduling': {'graph': {'nodes': nodes}}, 'execution': {
        'R_lite': [{'rank': '4', 'children': {'core': '0-3'}},
                   {'rank': '9', 'children': {'core': '0-3', 'gpu': '0'}}],
        'properties': {'x86_64': '4', 'aarch64': '9'}}}


@pytest.fixture
def instance(suite, monkeypatch, tmp_path):
    flux = suite.suite_flux
    state = SimpleNamespace(config={'resource': {'path': '/operator/new-model.json'}},
                            model=resource_set(), generation='instance-A', online='4,9', calls=[])
    monkeypatch.setenv('FLUX_URI', 'local:///operator/selected-instance')
    monkeypatch.setenv('FLUX_CLIENT', sys.executable)
    monkeypatch.setattr(flux, 'instance_settings', lambda: tmp_path)
    monkeypatch.setattr(flux, 'source', lambda path: {'sha256': 'a' * 64})
    monkeypatch.setattr(flux, 'runtime_identity', lambda *a, **kw: {'schema': 'tessera.suite_runtime.v1'})
    monkeypatch.setattr(flux, 'collect', lambda directory: {'state': 'running'})
    # The old implementation reads deployment files before reaching the CLI.
    # Give it real changed files so its pre-fix refusal identifies the defect.
    for name, value in [('CONFIG', state.config), ('MODEL', state.model)]:
        if hasattr(flux, name):
            path = tmp_path / name
            path.write_text(json.dumps(value))
            monkeypatch.setattr(flux, name, path)
    def native(argv, **kwargs):
        state.calls.append(argv)
        if argv == ['config', 'get']:
            value = json.dumps(state.config)
        elif argv == ['kvs', 'get', 'resource.R']:
            value = json.dumps(state.model)
        elif argv[:2] == ['resource', 'status']:
            value = state.online
        elif argv == ['getattr', 'rank']:
            value = '0'
        elif argv == ['getattr', 'broker.starttime'] or argv[:1] == ['python']:
            value = state.generation
        elif argv == ['getattr', 'version']:
            value = 'test-flux-version'
        elif argv[:3] == ['submit', '--env=-*', '--dry-run']:
            value = '{}'
        elif argv[:2] == ['job', 'submit']:
            value = 'f123'
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=value, stderr='')
    monkeypatch.setattr(flux, 'native', native)
    state.submit = lambda directory, **kw: flux.submit(directory, tmp_path,
        [sys.executable, '-s', '-c', 'pass'], 2, 8, 30, {}, **kw)
    state.flux = flux
    return state


class NativeJobs:
    def __init__(self, module, monkeypatch):
        self.flux = module.suite_flux
        self.jobs = {}
        self.calls = []
        self.qualification = {"profile": "fixture", "broker_starttime": "generation-A"}
        monkeypatch.setattr(self.flux, "broker_starttime", lambda **kw: self.qualification["broker_starttime"])
        monkeypatch.setattr(self.flux, "native", self.native)

    def add(self, receipt_dir, name, population, command=None, code=0, application=None):
        directory = receipt_dir / f"flux.{name}"
        directory.mkdir()
        surface = receipt_dir / f"surface.{name}.json"
        surface.write_text(json.dumps(population))
        command = command or ["/usr/bin/python3", "-m", "pytest", "tests", "--surface-json", str(surface)]
        identity = population["source_identity"]
        intent = {"schema": "tessera.flux.launch.v1", "profile": "legacy-fixture-profile",
                  "uri": "local:///historical-instance", "directory": str(directory), "checkout": "/mnt/shared/fixture",
                  "source": identity, "command": command, "cores": 2, "memory_gib": 4,
                  "timeout_s": 60, "application": application, "runtime": {"sha256": "d" * 64},
                  "qualification": self.qualification, "broker_starttime": "generation-A"}
        self.flux.save(directory / "launch.json", intent)
        digest = hashlib.sha256(self.flux.encoded(intent)).hexdigest()
        job_id = str(100 + len(self.jobs))
        self.flux.save(directory / "submission.json", {"state": "submitted", "job_id": job_id,
                                                       "launch_sha256": digest})
        (directory / "jobid").write_text(job_id + "\n")
        spec = {"tasks": [{"command": ["/usr/bin/python3", "-s", "tools/_suite_flux.py", "payload", str(directory), digest]}], "attributes": {"system": {"environment": {}}}}
        self.flux.save(directory / "jobspec.json", spec)
        self.flux.save(directory / "payload.json", {"job_id": job_id, "launch_sha256": digest,
            "source": identity, "runtime": intent["runtime"], "runtime_verified": True,
            "scratch_removed": True})
        counts = population["counts"]
        summary = ", ".join(f"{value} {'errors' if key == 'error' else key}" for key, value in counts.items() if value)
        (directory / "stdout").write_text(publication_line(POPULATION, surface,
            hashlib.sha256(surface.read_bytes()).hexdigest()) + f"\n{summary} in 0.10s\n")
        (directory / "stderr").write_text("")
        world = SimpleNamespace(directory=directory, surface=surface, id=job_id,
            spec=copy.deepcopy(spec), code=code, exception=False, query_error=None,
            events=["alloc", "start", "finish", "clean"], application=application)
        self.jobs[job_id] = world
        return world

    def native(self, argv, timeout=30):
        self.calls.append(argv)
        if "submit" in argv:
            raise AssertionError("saved-ID collection submitted a duplicate")
        job_id = argv[3] if argv[:3] == ["job", "info", "--original"] else argv[-1] if argv[:2] != ["job", "info"] else argv[2]
        job = self.jobs[job_id]
        if job.query_error and argv[:2] == ["job", job.query_error]:
            return SimpleNamespace(returncode=1, stdout="not JSON", stderr="query failed")
        code = 0
        if argv[:3] == ["job", "info", "--original"]:
            value = job.spec
        elif argv[:2] == ["job", "status"]:
            code = 125 if job.exception else job.code
            value = {"id": int(job_id), "result": 2 if job.code else 1,
                     "exception_occurred": job.exception, "waitstatus": job.code * 256}
        elif argv[:2] == ["job", "eventlog"]:
            raw = "\n".join(json.dumps({"name": name, "context": {"status": job.code * 256}})
                            for name in job.events)
            return SimpleNamespace(returncode=0, stdout=raw, stderr="")
        elif argv[:2] == ["job", "info"] and argv[-1] == "R":
            value = {"version": 1, "execution": {"R_lite": [{"rank": "0", "children": {"core": "1-2"}}]}}
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=code, stdout=json.dumps(value), stderr="")
