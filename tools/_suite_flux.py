"""Native Flux CLI transport for suite receipts, without result caching."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera._dev.suite_source import measured_source

PREFIX = "/home/rob/flux-vanilla/0.91.0-0.56.0"
URI = f"local://{PREFIX}/runs/fv1-model/run/local-0"
PROFILE = "fv1-explicit-perf-memory.v1"
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "TORCH_NUM_THREADS", "MAX_JOBS"), "1")


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def native(argv, timeout=30):
    # The client needs a broker connection; the payload gets its own allowlist.
    env = {"PATH": f"{PREFIX}/install/bin:{PREFIX}/deps/bin:/usr/bin:/bin",
           "LD_LIBRARY_PATH": f"{PREFIX}/install/lib:{PREFIX}/deps/lib",
           "FLUX_URI": URI, "HOME": "/home/rob"}
    return subprocess.run([f"{PREFIX}/install/bin/flux", *argv], env=env,
                          capture_output=True, text=True, timeout=timeout)


def checked(argv):
    result = native(argv)
    if result.returncode:
        raise ValueError(f"Flux {argv[0]} query failed")
    return result.stdout


def validate(cores, memory, timeout, environment):
    if (type(cores) is not int or not 1 <= cores <= 4
            or type(memory) is not int or not 1 <= memory <= 16
            or not 0 < timeout <= 1790):
        raise ValueError("native suite profile requires 1..4 cores, 1..16 GiB, 0 < timeout <= 1790s")
    if len(environment) > 32:
        raise ValueError("too many artifact roots")
    for key, value in environment.items():
        if (not re.fullmatch(r"TESSERA_[A-Z0-9_]+", key)
                or key in {"TESSERA_SOURCE_VERIFIER", "TESSERA_STRICT_CUDA"}
                or not Path(value).is_absolute() or not Path(value).is_dir()
                or "\0" in value):
            raise ValueError(f"invalid artifact root: {key}")


def source(checkout):
    checkout = Path(checkout).resolve()
    if not checkout.is_relative_to("/mnt/shared"):
        raise ValueError("native suite checkout must be under /mnt/shared")
    branch = subprocess.run(["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
                            capture_output=True)
    if branch.returncode != 1:
        raise ValueError("native suite source must be a detached checkout")
    identity = measured_source(checkout, verifier=None)
    if identity.get("verification") != "verified":
        raise ValueError("native suite source is not verified: " + str(identity.get("reason")))
    return identity


def submit(directory, checkout, command, cores, memory, timeout, environment):
    validate(cores, memory, timeout, environment)
    directory, checkout = Path(directory).resolve(), Path(checkout).resolve()
    identity = source(checkout)
    intent = {"schema": "tessera.flux.launch.v1", "profile": PROFILE, "uri": URI,
              "broker_starttime": checked(["getattr", "broker.starttime"]).strip(),
              "directory": str(directory), "checkout": str(checkout), "source": identity,
              "command": command, "cores": cores, "memory_gib": memory,
              "timeout_s": timeout}
    digest = hashlib.sha256(encoded(intent)).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "launch.json", intent)
    save(directory / "submission.json", {"state": "unknown", "launch_sha256": digest})
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/rob",
           "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1",
           "PRISMAQUANT_DEV_MODE": "0", **THREADS, **environment}
    wrapper = ["/usr/bin/python3", "tools/_suite_flux.py", "payload", str(directory), digest]
    args = ["submit", "--env=-*", *[f"--env={k}={v}" for k, v in sorted(env.items())],
            f"--resources-shape=node/slot/[core={cores}/perf=1;memory={memory}]",
            "--requires=x86_64", "-t", f"{timeout + 10}s", "--cwd", str(checkout),
            "--input=/dev/null", f"--output={directory}/stdout", f"--error={directory}/stderr",
            "--job-name=tessera-" + digest[:16], *wrapper]
    # Dry-run is native rendering only. It emits no whole-node request.
    jobspec = json.loads(checked([*args[:1], "--dry-run", *args[1:]]))
    save(directory / "jobspec.json", jobspec)
    result = native(["job", "submit", str(directory / "jobspec.json")])
    (directory / "submit.stdout").write_text(result.stdout)
    (directory / "submit.stderr").write_text(result.stderr)
    job_id = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"(?:ƒ[1-9A-HJ-NP-Za-km-z]+|[0-9]+)", job_id):
        raise ValueError(f"native submission unknown in {directory}; reconcile, never resubmit")
    save(directory / "submission.json", {"state": "submitted", "launch_sha256": digest,
                                         "job_id": job_id})
    (directory / "jobid").write_text(job_id + "\n")
    return collect(directory)


def collect(directory):
    """Reconnect to the saved ID. A query error never becomes a payload exit."""
    directory = Path(directory).resolve()
    record = {"returncode": None, "exit_status_observed": False, "query_errors": []}
    errors = record["query_errors"]
    try:
        intent = json.loads((directory / "launch.json").read_bytes())
        sent = json.loads((directory / "submission.json").read_bytes())
        digest = hashlib.sha256(encoded(intent)).hexdigest()
        if (sent.get("state") != "submitted" or sent.get("launch_sha256") != digest
                or intent["directory"] != str(directory) or intent["uri"] != URI
                or intent["broker_starttime"] != checked(["getattr", "broker.starttime"]).strip()):
            raise ValueError("unknown or changed submission identity")
        job_id = sent["job_id"]
        if not re.fullmatch(r"(?:ƒ[1-9A-HJ-NP-Za-km-z]+|[0-9]+)", job_id):
            raise ValueError("invalid saved native ID")
        record.update(job_id=job_id, command=intent["command"], source=intent["source"],
                      checkout=intent["checkout"], launch_sha256=digest)
        original = json.loads(checked(["job", "info", "--original", job_id, "jobspec"]))
        expected = json.loads((directory / "jobspec.json").read_bytes())
        if original != expected:
            raise ValueError("native jobspec differs from saved command/resources/environment")
        # No environments are copied into the public record.
        events = checked(["job", "eventlog", "--format=json", job_id])
        (directory / "eventlog.jsonl").write_text(events)
        events = [json.loads(line) for line in events.splitlines()]
        names = {event["name"] for event in events}
        if "clean" not in names:
            record["state"] = "running"
            return record
        status = native(["job", "status", "--json", "--exception-exit-code=125", job_id])
        (directory / "status.json").write_text(status.stdout)
        (directory / "status.stderr").write_text(status.stderr)
        (directory / "flux-exit-code.txt").write_text(str(status.returncode) + "\n")
        status_value = json.loads(status.stdout)
        finishes = [event["context"]["status"] for event in events if event["name"] == "finish"]
        if (type(status_value.get("id")) is not int
                or status_value.get("result") not in (1, 2, 4, 8)
                or type(status_value.get("exception_occurred")) is not bool
                or (finishes and status_value.get("waitstatus") != finishes[-1])
                or (status_value["result"] == 1) != (status.returncode == 0)):
            raise ValueError("invalid or inconsistent native status")
        allocation = native(["job", "info", job_id, "R"])
        (directory / "R.json").write_text(allocation.stdout)
        (directory / "R.stderr").write_text(allocation.stderr)
        if "alloc" in names:
            value = json.loads(allocation.stdout)
            if allocation.returncode or value.get("version") != 1 or not value["execution"]["R_lite"]:
                raise ValueError("native allocation missing")
        record["native_result"] = status_value
        record["state"] = "terminal"
        if finishes:
            record["returncode"] = os.waitstatus_to_exitcode(finishes[-1])
            record["exit_status_observed"] = True
        elif status_value["result"] != 1:
            record["returncode"] = status.returncode
            record["exit_status_observed"] = True
        if "start" in names:
            payload = json.loads((directory / "payload.json").read_bytes())
            if (payload["job_id"] != job_id or payload["launch_sha256"] != digest
                    or payload["source"] != intent["source"] or not payload["scratch_removed"]):
                raise ValueError("payload source or scratch binding missing")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        errors.append(str(exc))
        record.update(returncode=None, exit_status_observed=False, state="unknown")
    return record


def wait(directory, timeout):
    directory = Path(directory)
    sent = json.loads((directory / "submission.json").read_bytes())
    result = native(["job", "wait-event", sent["job_id"], "clean"], timeout=timeout)
    (directory / "wait.stdout").write_text(result.stdout)
    (directory / "wait.stderr").write_text(result.stderr)
    return collect(directory)


def payload(directory, digest):
    directory = Path(directory)
    raw = (directory / "launch.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("launch bytes changed")
    intent = json.loads(raw)
    if Path.cwd() != Path(intent["checkout"]) or source(Path.cwd()) != intent["source"]:
        raise ValueError("source changed before payload")
    job_id = os.environ["FLUX_JOB_ID"]
    scratch_root = (Path("/home/rob/tl-gates/flux-validation") if os.uname().machine == "x86_64"
                    else Path("/tmp/flux-vanilla"))
    scratch = scratch_root / job_id
    scratch.mkdir(exist_ok=False)
    record = {"job_id": job_id, "launch_sha256": digest, "source": intent["source"],
              "scratch": str(scratch), "scratch_removed": False,
              "affinity": sorted(os.sched_getaffinity(0))}
    try:
        env = {**os.environ, "TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch)}
        result = subprocess.run(intent["command"], env=env)
        if source(Path.cwd()) != intent["source"]:
            raise ValueError("source changed during payload")
        return result.returncode
    finally:
        shutil.rmtree(scratch)
        record["scratch_removed"] = not scratch.exists()
        save(directory / "payload.json", record)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "payload":
        raise SystemExit("only payload DIRECTORY SHA256 is supported")
    raise SystemExit(payload(sys.argv[2], sys.argv[3]))
