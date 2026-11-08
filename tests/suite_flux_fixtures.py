"""Saved native records with a controlled CLI, without contacting a broker."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tessera._dev.surface_publication import publication_line, POPULATION


class NativeJobs:
    def __init__(self, module, monkeypatch):
        self.flux = module.suite_flux
        self.jobs = {}
        self.calls = []
        self.qualification = {"profile": "fixture", "broker_starttime": "generation-A"}
        monkeypatch.setattr(self.flux, "profile_identity", lambda: self.qualification)
        monkeypatch.setattr(self.flux, "native", self.native)

    def add(self, receipt_dir, name, population, command=None, code=0, application=None):
        directory = receipt_dir / f"flux.{name}"
        directory.mkdir()
        surface = receipt_dir / f"surface.{name}.json"
        surface.write_text(json.dumps(population))
        command = command or ["/usr/bin/python3", "-m", "pytest", "tests", "--surface-json", str(surface)]
        identity = population["source_identity"]
        intent = {"schema": "tessera.flux.launch.v1", "profile": self.flux.PROFILE,
                  "uri": self.flux.URI, "directory": str(directory), "checkout": "/mnt/shared/fixture",
                  "source": identity, "command": command, "cores": 2, "memory_gib": 4,
                  "timeout_s": 60, "application": application, "runtime": {"sha256": "d" * 64},
                  "qualification": self.qualification, "broker_starttime": "generation-A"}
        self.flux.save(directory / "launch.json", intent)
        digest = hashlib.sha256(self.flux.encoded(intent)).hexdigest()
        job_id = str(100 + len(self.jobs))
        self.flux.save(directory / "submission.json", {"state": "submitted", "job_id": job_id,
                                                       "launch_sha256": digest})
        (directory / "jobid").write_text(job_id + "\n")
        spec = {"tasks": [{"command": ["/usr/bin/python3", "tools/_suite_flux.py", "payload", str(directory), digest]}], "attributes": {"system": {"environment": {}}}}
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
