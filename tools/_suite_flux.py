"""Native Flux CLI transport for suite receipts, without result caching."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera._dev.suite_source import measured_source

PREFIX = "/home/rob/flux-vanilla/0.91.0-0.56.0"
URI = f"local://{PREFIX}/runs/fv1-model/run/local-0"
PROFILE = "fv1-explicit-perf-memory.v1"
CONFIG = Path(PREFIX) / "runs/fv1-model/config.json"
MODEL = Path(PREFIX) / "config/R-model.json"
CONFIG_SHA = "2b5618a591bbf545336ed9dfae6645b5dfca7516325ccbf9e0b2dbf4181607e1"
MODEL_SHA = "261a084b72cb3a0c5055816f642f51ccbf473d3373876f34e67be55d5b5221a8"
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
           "FLUX_URI": URI.replace("local-0", {"sparky": "local-1", "sparklina": "local-2"}.get(os.uname().nodename, "local-0")),
           "HOME": "/home/rob"}
    return subprocess.run([f"{PREFIX}/install/bin/flux", *argv], env=env,
                          capture_output=True, text=True, timeout=timeout)


def checked(argv):
    result = native(argv)
    if result.returncode:
        raise ValueError(f"Flux {argv[0]} query failed")
    return result.stdout


def profile_identity():
    """Require the qualified installed and live baseline, not just a URI."""
    config_raw, model_raw = CONFIG.read_bytes(), MODEL.read_bytes()
    if (hashlib.sha256(config_raw).hexdigest() != CONFIG_SHA
            or hashlib.sha256(model_raw).hexdigest() != MODEL_SHA):
        raise ValueError("unqualified Flux config/resource model; profile revision required")
    config, model = json.loads(config_raw), json.loads(model_raw)
    for section in ("resource", "exec", "sched-fluxion-resource", "sched-fluxion-qmanager"):
        if json.loads(checked(["config", "get", section])) != config[section]:
            raise ValueError("loaded Flux config differs from the qualified profile")
    if json.loads(checked(["kvs", "get", "resource.R"]))["scheduling"] != model["scheduling"]:
        raise ValueError("loaded Flux resource graph differs from the qualified profile")
    if checked(["resource", "status", "-s", "online", "-n", "-o", "{ranks}"]).strip() != "0-2":
        raise ValueError("all three Flux hosts must be online")
    if sum(line.endswith(": full") for line in checked(["overlay", "status"]).splitlines()) != 3:
        raise ValueError("all three Flux brokers must be healthy")
    return {"profile": PROFILE, "config_sha256": CONFIG_SHA, "model_sha256": MODEL_SHA,
            "broker_starttime": checked(["getattr", "broker.starttime"]).strip()}


def validate(cores, memory, timeout, environment):
    if (type(cores) is not int or not 1 <= cores <= 4
            or type(memory) is not int or not 1 <= memory <= 16
            or not 0 < timeout <= 1790):
        raise ValueError("native suite profile requires 1..4 cores, 1..16 GiB, 0 < timeout <= 1790s")
    if len(environment) > 32:
        raise ValueError("too many artifact roots")
    from _suite_container import artifact_variables
    allowed = artifact_variables()
    for key, value in environment.items():
        if (key not in allowed
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


def job_id(value):
    """Flux emits ASCII f58 in the empty locale and Unicode in job metadata."""
    if not isinstance(value, str) or not re.fullmatch(r"(?:[fƒ][1-9A-HJ-NP-Za-km-z]+|[0-9]+)", value):
        raise ValueError("invalid saved native ID")
    return "f" + value[1:] if value.startswith("ƒ") else value


def resources(cores, memory, timeout, *, gpus=0, architecture="x86_64"):
    validate(cores, memory, timeout, {})
    if type(gpus) is not int or gpus not in (0, 1) or architecture not in ("x86_64", "aarch64"):
        raise ValueError("finite architecture/whole-GPU resource profile required")
    if gpus and architecture != "aarch64":
        raise ValueError("qualified GPU architecture is aarch64")
    return {"schema": "tessera.suite_resources.v1", "architecture": architecture,
            "cores": cores, "memory_gib": memory, "gpus": gpus, "timeout_s": timeout}


def container_spec(command):
    from _suite_container import command_spec
    parts = command
    if len(parts) > 2 and parts[1] == "tools/suite_deadline.py" and "--" in parts[2:]:
        parts = parts[parts.index("--", 2) + 1:]
    return command_spec(parts)


def require_allocation(allocation, demand, affinity, mapped):
    """R's assigned hardware, not requested annotations, binds execution."""
    metadata = [node["metadata"] for node in allocation["scheduling"]["graph"]["nodes"]]
    kinds = {kind: [item for item in metadata if item["type"] == kind]
             for kind in ("core", "perf", "memory", "gpu", "node")}
    pus = sorted({int(pu) for core in kinds["core"] for pu in core["properties"]["os-pus"].split(",")})
    if (len(kinds["node"]) != 1 or kinds["node"][0]["properties"]["arch"] != demand["architecture"]
            or len(kinds["core"]) != demand["cores"] or len(kinds["perf"]) != demand["cores"]
            or len(kinds["memory"]) != demand["memory_gib"] or len(kinds["gpu"]) != demand["gpus"]
            or any(core["properties"]["core-type"] != "perf" for core in kinds["core"])
            or affinity != pus or mapped != pus):
        raise ValueError("actual native allocation/resources/affinity differ")


def require_original_launch(intent, digest, jobspec):
    """The original wrapper argv authenticates all saved launch bytes."""
    wrapper_python = intent["command"][0] if intent.get("resources") else "/usr/bin/python3"
    if jobspec["tasks"][0]["command"] != [wrapper_python, "tools/_suite_flux.py", "payload", intent["directory"], digest]:
        raise ValueError("original native command does not authenticate this launch digest")


def submit(directory, checkout, command, cores, memory, timeout, environment, *, application=None,
           gpus=0, architecture="x86_64"):
    validate(cores, memory, timeout, environment)
    demand = resources(cores, memory, timeout, gpus=gpus, architecture=architecture)
    container = container_spec(command)
    if gpus and container is None:
        raise ValueError("GPU suite requires the finite native container contract")
    if container is not None:
        from _suite_container import validate_mounts
        if container["workers"] != cores or container["strict_cuda"] != bool(gpus) or container["artifacts"] != environment:
            raise ValueError("container workers/GPU/artifact roots differ from native reservation")
        validate_mounts(container, Path(checkout))
    directory, checkout = Path(directory).resolve(), Path(checkout).resolve()
    identity = source(checkout)
    qualified = profile_identity()
    if application is not None:
        require_selection(application, identity)
    runtime = runtime_identity(command[0], checkout, timeout=timeout)
    intent = {"schema": "tessera.flux.launch.v1", "profile": PROFILE, "uri": URI,
              "qualification": qualified, "broker_starttime": qualified["broker_starttime"],
              "directory": str(directory), "checkout": str(checkout), "source": identity,
              "command": command, "cores": cores, "memory_gib": memory,
              "timeout_s": timeout, "runtime": runtime, "application": application,
              "resources": demand, "container": container}
    digest = hashlib.sha256(encoded(intent)).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "launch.json", intent)
    save(directory / "submission.json", {"state": "unknown", "launch_sha256": digest})
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/rob",
           "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1",
           "PRISMAQUANT_DEV_MODE": "0", **THREADS, **environment}
    if container is not None:
        env["XDG_RUNTIME_DIR"] = "/run/user/1000"
    wrapper = [command[0], "tools/_suite_flux.py", "payload", str(directory), digest]
    args = ["submit", "--env=-*", *[f"--env={k}={v}" for k, v in sorted(env.items())],
            f"--resources-shape=node/slot/[core={cores}/perf=1;memory={memory}" + (";gpu=1]" if gpus else "]"),
            "--requires=" + architecture, "-t", f"{timeout + 10}s", "--cwd", str(checkout),
            "--input=/dev/null", f"--output={directory}/stdout", f"--error={directory}/stderr",
            "--job-name=tessera-" + digest[:16], *wrapper]
    # Dry-run is native rendering only. It emits no whole-node request.
    jobspec = json.loads(checked([*args[:1], "--dry-run", *args[1:]]))
    save(directory / "jobspec.json", jobspec)
    result = native(["job", "submit", str(directory / "jobspec.json")])
    (directory / "submit.stdout").write_text(result.stdout)
    (directory / "submit.stderr").write_text(result.stderr)
    acknowledged = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"(?:[fƒ][1-9A-HJ-NP-Za-km-z]+|[0-9]+)", acknowledged):
        raise ValueError(f"native submission unknown in {directory}; reconcile, never resubmit")
    canonical_id = job_id(acknowledged)
    save(directory / "submission.json", {"state": "submitted", "launch_sha256": digest,
                                         "job_id": canonical_id})
    (directory / "jobid").write_text(canonical_id + "\n")
    return collect(directory)


def collect(directory):
    """Reconnect to the saved ID. A query error never becomes a payload exit."""
    directory = Path(directory).resolve()
    record = {"returncode": None, "exit_status_observed": False, "query_errors": [],
              "evidence_errors": []}
    terminal_confirmed = False
    errors = record["query_errors"]
    try:
        intent = json.loads((directory / "launch.json").read_bytes())
        sent = json.loads((directory / "submission.json").read_bytes())
        digest = hashlib.sha256(encoded(intent)).hexdigest()
        if (sent.get("state") != "submitted" or sent.get("launch_sha256") != digest
                or intent["directory"] != str(directory) or intent["uri"] != URI
                or intent["qualification"] != profile_identity()):
            raise ValueError("unknown or changed submission identity")
        saved_id = job_id(sent["job_id"])
        record.update(job_id=saved_id, command=intent["command"], source=intent["source"],
                      checkout=intent["checkout"], launch_sha256=digest,
                      application=intent["application"], runtime=intent["runtime"],
                      qualification=intent["qualification"])
        original = json.loads(checked(["job", "info", "--original", saved_id, "jobspec"]))
        expected = json.loads((directory / "jobspec.json").read_bytes())
        if original != expected:
            raise ValueError("native jobspec differs from saved command/resources/environment")
        require_original_launch(intent, digest, original)
        # No environments are copied into the public record.
        events = checked(["job", "eventlog", "--format=json", saved_id])
        (directory / "eventlog.jsonl").write_text(events)
        events = [json.loads(line) for line in events.splitlines()]
        names = {event["name"] for event in events}
        if "clean" not in names:
            record["state"] = "running"
            return record
        status = native(["job", "status", "--json", "--exception-exit-code=125", saved_id])
        (directory / "status.json").write_text(status.stdout)
        (directory / "status.stderr").write_text(status.stderr)
        (directory / "flux-exit-code.txt").write_text(str(status.returncode) + "\n")
        status_value = json.loads(status.stdout)
        finishes = [event["context"]["status"] for event in events if event["name"] == "finish"]
        if (type(status_value.get("id")) is not int
                or status_value.get("result") not in (1, 2, 4, 8)
                or type(status_value.get("exception_occurred")) is not bool
                or (finishes and status_value.get("waitstatus") != finishes[-1])
                or (status_value["result"] != 1 and status.returncode == 0)
                or (status_value["result"] == 1 and status.returncode != 0
                    and not status_value["exception_occurred"])):
            raise ValueError("invalid or inconsistent native status")
        record["native_result"] = status_value
        record["state"] = "terminal"
        if status_value["exception_occurred"] or status_value["result"] != 1:
            record["returncode"] = status.returncode
        elif finishes:
            record["returncode"] = os.waitstatus_to_exitcode(finishes[-1])
        else:
            raise ValueError("successful native status has no finish event")
        record["exit_status_observed"] = True
        terminal_confirmed = True
        allocation = native(["job", "info", saved_id, "R"])
        (directory / "R.json").write_text(allocation.stdout)
        (directory / "R.stderr").write_text(allocation.stderr)
        if "alloc" in names:
            value = json.loads(allocation.stdout)
            if allocation.returncode or value.get("version") != 1 or not value["execution"]["R_lite"]:
                raise ValueError("native allocation missing")
        if "start" in names:
            payload = json.loads((directory / "payload.json").read_bytes())
            if (job_id(payload["job_id"]) != saved_id or payload["launch_sha256"] != digest
                    or payload["source"] != intent["source"] or not payload["scratch_removed"]
                    or payload["runtime"] != intent["runtime"] or not payload["runtime_verified"]
                    or payload.get("process_cleanup_error") or payload.get("cleanup_error")):
                raise ValueError("payload source or scratch binding missing")
            if intent.get("resources") is not None:
                require_allocation(value, intent["resources"], payload["affinity"], payload["allocated_affinity"])
            if intent.get("container") is not None:
                from _suite_container import verify_record
                record["container"] = verify_record(directory, intent, digest, saved_id, payload["affinity"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if terminal_confirmed:
            record["evidence_errors"].append(str(exc))
        else:
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
    saved = json.loads((directory / "submission.json").read_bytes())
    original = json.loads(checked(["job", "info", "--original", os.environ["FLUX_JOB_ID"], "jobspec"]))
    if (saved["state"] not in ("submitted", "unknown") or saved["launch_sha256"] != digest
            or (saved["state"] == "submitted" and job_id(saved["job_id"]) != job_id(os.environ["FLUX_JOB_ID"]))
            or original != json.loads((directory / "jobspec.json").read_bytes())):
        raise ValueError("payload original native job/launch differs")
    require_original_launch(intent, digest, original)
    if Path.cwd() != Path(intent["checkout"]) or source(Path.cwd()) != intent["source"]:
        raise ValueError("source changed before payload")
    if intent["application"] is not None:
        require_selection(intent["application"], intent["source"])
    if runtime_identity(intent["command"][0], Path.cwd(), timeout=intent["timeout_s"]) != intent["runtime"]:
        raise ValueError("interpreter/dependencies changed before payload")
    actual_job_id = os.environ["FLUX_JOB_ID"]
    scratch_root = (Path("/home/rob/tl-gates/flux-validation") if os.uname().machine == "x86_64"
                    else Path("/tmp/flux-vanilla"))
    scratch = scratch_root / actual_job_id
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(scratch_root)], text=True).strip()
    if filesystem != ("xfs" if os.uname().machine == "x86_64" else "ext4"):
        raise ValueError("unqualified local scratch filesystem")
    record = {"job_id": actual_job_id, "launch_sha256": digest, "source": intent["source"],
              "runtime": intent["runtime"], "runtime_verified": False,
              "scratch": str(scratch), "scratch_removed": False,
              "affinity": sorted(os.sched_getaffinity(0))}
    if intent.get("resources") is not None:
        allocation = json.loads(checked(["job", "info", os.environ["FLUX_JOB_ID"], "R"]))
        record["allocated_affinity"] = sorted({int(pu) for node in allocation["scheduling"]["graph"]["nodes"]
            if node["metadata"]["type"] == "core" for pu in node["metadata"]["properties"]["os-pus"].split(",")})
        require_allocation(allocation, intent["resources"], record["affinity"], record["allocated_affinity"])
        if os.uname().machine != intent["resources"]["architecture"]:
            raise ValueError("payload architecture differs from native intention")
    scratch.mkdir(exist_ok=False)
    child = None
    interrupted = []
    class Interrupted(BaseException):
        pass
    def forward(signum, frame):
        interrupted.append(signum)
        if child is not None:
            if child.poll() is None:
                child.send_signal(signum)
            else:
                raise Interrupted
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD)}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, forward)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        save(directory / "payload.json", record)
        env = {**os.environ, "TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch),
               "TESSERA_SUITE_DIRECTORY": str(directory), "TESSERA_SUITE_LAUNCH_SHA256": digest}
        child = subprocess.Popen(intent["command"], env=env, start_new_session=True)
        for sig in interrupted:
            child.send_signal(sig)
        code = child.wait()
        if interrupted:
            # Flux bounds cancellation cleanup. A content seal cannot establish
            # success for an interrupted run and must not delay owned cleanup.
            record["interrupted_signal"] = interrupted[0]
            return 128 + interrupted[0]
        if source(Path.cwd()) != intent["source"]:
            raise ValueError("source changed during payload")
        if runtime_identity(intent["command"][0], Path.cwd(), timeout=intent["timeout_s"]) != intent["runtime"]:
            raise ValueError("interpreter/dependencies changed during payload")
        record["runtime_verified"] = True
        return code
    except Interrupted:
        record["interrupted_signal"] = interrupted[0]
        return 128 + interrupted[0]
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        try:
            # Keep the leader unreaped throughout grace so its PID anchors the
            # group. TERM lets the deadline supervisor stop its own inner group.
            if child is not None and child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    time.sleep(5)
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                except (OSError, subprocess.SubprocessError) as exc:
                    record["process_cleanup_error"] = str(exc)
            try:
                if intent.get("container") is not None and (directory / "container.json").exists():
                    from _suite_container import cleanup_cid
                    container = json.loads((directory / "container.json").read_bytes())
                    # A second cleanup is needed only when the child could not
                    # finish its own CID cleanup. Never rewrite its announced
                    # record after successful cleanup.
                    if not container["removed"]:
                        cleanup_cid(container)
                        save(directory / "container.cleanup.json", container)
                    if not container["removed"] or container["cleanup_errors"]:
                        record["cleanup_error"] = "owned container cleanup incomplete"
                shutil.rmtree(scratch)
                record["scratch_removed"] = not scratch.exists()
            except OSError as exc:
                record["cleanup_error"] = str(exc)
            save(directory / "payload.json", record)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def require_selection(binding, identity):
    raw = Path(binding["selection_path"]).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != binding["selection_sha256"]
            or binding["source"] != identity):
        raise ValueError("selection bytes or selection-time source changed")


def runtime_identity(python, checkout, *, timeout):
    result = subprocess.run([python, "-I", "-B", str(Path(checkout) / "tools/_suite_runtime.py")],
                            capture_output=True, text=True, timeout=timeout,
                            env={"PATH": "/usr/bin:/bin", "HOME": "/home/rob"})
    if result.returncode:
        raise ValueError("native interpreter/dependency seal refused: " + result.stderr[-300:])
    identity = json.loads(result.stdout)
    if identity.get("schema") != "tessera.suite_runtime.v1":
        raise ValueError("unrecognized native runtime identity")
    return identity


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "payload":
        raise SystemExit("only payload DIRECTORY SHA256 is supported")
    raise SystemExit(payload(sys.argv[2], sys.argv[3]))
