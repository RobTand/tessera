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
from _flux_shape import THREADS, duration_seconds, resource_shape


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def instance_settings():
    path = Path("/mnt/shared/flux-migration/INSTANCE")
    try:
        fields = path.read_text().split()
    except OSError as exc:
        raise ValueError(f"Flux INSTANCE cannot be read: {exc}") from exc
    if len(fields) != 2 or not Path(fields[0]).is_absolute():
        raise ValueError("Flux INSTANCE must contain one absolute PREFIX and RUN")
    return Path(fields[0])


def connection(*, client=None):
    """Resolve SHAPE section 2's operator-owned instance and client settings."""
    prefix = instance_settings()
    env = {"PATH": f"{prefix}/install/bin:{prefix}/deps/bin:/usr/bin:/bin",
           "LD_LIBRARY_PATH": f"{prefix}/install/lib:{prefix}/deps/lib",
           "HOME": os.environ.get("HOME", "/home/rob"), "PYTHONNOUSERSITE": "1"}
    uri = os.environ.get("FLUX_URI")
    if not uri:
        resolver = "/mnt/shared/flux-migration/flux-uri.sh"
        try:
            result = subprocess.run([resolver], env=env, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"Flux resolver {resolver} failed: {exc}") from exc
        if result.returncode:
            raise ValueError(f"Flux resolver {resolver} failed: {result.stderr.strip()!r}")
        uri = result.stdout.strip()
    if not uri or any(character.isspace() or character == "\0" for character in uri):
        raise ValueError("FLUX_URI or flux-uri.sh must name the Flux instance")
    env["FLUX_URI"] = uri
    requested = client or os.environ.get("FLUX_CLIENT") or str(prefix / "install/bin/flux")
    executable = shutil.which(requested, path=env["PATH"])
    if executable is None:
        raise ValueError("FLUX_CLIENT or INSTANCE client on PATH must name an executable Flux client")
    return str(Path(executable).absolute()), env


def native(argv, timeout=30, *, client=None):
    # The client needs a broker connection; the payload gets its own allowlist.
    executable, env = connection(client=client)
    return subprocess.run([executable, *argv], env=env,
                          capture_output=True, text=True, timeout=timeout)


def checked(argv, **options):
    result = native(argv, **options)
    if result.returncode:
        raise ValueError(f"Flux {argv[0]} query failed")
    return result.stdout


def broker_starttime(**options):
    # attr.get is broker-local. Address rank zero directly over the selected
    # connection, without guessing its socket from this client's hostname.
    generation = checked(["python", "-c",
        'import flux; print(flux.Flux().rpc("attr.get", '
        '{"name": "broker.starttime"}, nodeid=0).get()["value"])'], **options).strip()
    if not generation:
        raise ValueError("Flux broker_starttime is missing")
    return generation


def instance_identity(**options):
    """Observe the live deployment. Digests are evidence, never admission pins."""
    generation = broker_starttime(**options)
    config_raw = checked(["config", "get"], **options)
    model_raw = checked(["kvs", "get", "resource.R"], **options)
    json.loads(config_raw)
    model = json.loads(model_raw)
    version = checked(["getattr", "version"], **options).strip()
    if broker_starttime(**options) != generation:
        raise ValueError("Flux broker_starttime changed during observation")
    return {"config_sha256": hashlib.sha256(config_raw.encode()).hexdigest(),
            "model_sha256": hashlib.sha256(encoded(model["scheduling"])).hexdigest(),
            "r_sha256": hashlib.sha256(model_raw.encode()).hexdigest(),
            "broker_starttime": generation, "flux_version": version}


def require_instance(intent, **options):
    """Job IDs are instance-local; historical profile names remain evidence."""
    saved = intent["qualification"].get("broker_starttime", intent.get("broker_starttime"))
    current = broker_starttime(**options)
    if not saved or saved != current:
        raise ValueError("Flux broker_starttime differs from saved submission; instance restarted or FLUX_URI selects another instance")


def idset(value):
    result = set()
    for part in value.strip().strip("[]").split(","):
        if not part:
            continue
        low, separator, high = part.partition("-")
        result.update(range(int(low), int(high) + 1) if separator else [int(low)])
    return result


def require_capacity(demand, **options):
    """Require one online eligible host with the requested stock slot shape."""
    model = json.loads(checked(["kvs", "get", "resource.R"], **options))
    online = idset(checked(["resource", "status", "-s", "online", "-n", "-o", "{ranks}"], **options))
    metadata = [node["metadata"] for node in model["scheduling"]["graph"]["nodes"]]
    hosts = [item for item in metadata if item["type"] == "node"]
    eligible = idset(model["execution"].get("properties", {}).get(demand["architecture"], ""))
    hosts = [item for item in hosts if item.get("properties", {}).get("arch") == demand["architecture"]
             and item["rank"] in eligible]
    if not hosts:
        raise ValueError("Flux architecture cannot satisfy request: " + demand["architecture"])
    if demand.get("host") is not None:
        hosts = [item for item in hosts if item["name"] == demand["host"]]
        if not hosts:
            raise ValueError("Flux host cannot satisfy local image dependency: " + demand["host"])
    hosts = [item for item in hosts if item["rank"] in online]
    if not hosts:
        raise ValueError("Flux online resources cannot satisfy request")
    execution = {}
    for entry in model["execution"]["R_lite"]:
        for rank in idset(entry["rank"]):
            execution[rank] = entry["children"]
    for field, kind in (("cores", "core"), ("memory_gib", "memory"), ("gpus", "gpu")):
        remaining = []
        for host in hosts:
            children = execution.get(host["rank"], {})
            leaves = [item for item in metadata if item["type"] == kind and item["rank"] == host["rank"]
                      and item.get("size", 1) == 1
                      and (item.get("unit") == "GiB" if kind == "memory"
                           else item["id"] in idset(children.get(kind, "")))]
            if len(leaves) >= demand[field]:
                remaining.append(host)
        if not remaining:
            raise ValueError(f"Flux {field} cannot satisfy request on one eligible online host: {demand[field]}")
        hosts = remaining


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


def resources(cores, memory, timeout, *, gpus=0, architecture="x86_64", host=None):
    validate(cores, memory, timeout, {})
    if type(gpus) is not int or gpus not in (0, 1) or architecture not in ("x86_64", "aarch64"):
        raise ValueError("finite architecture/whole-GPU resource profile required")
    if gpus and architecture != "aarch64":
        raise ValueError("qualified GPU architecture is aarch64")
    if host is not None and (not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host)):
        raise ValueError("invalid local image host")
    return {"schema": "tessera.suite_resources.v1", "architecture": architecture,
            "cores": cores, "memory_gib": memory, "gpus": gpus, "timeout_s": timeout,
            **({"host": host} if host is not None else {})}


def container_spec(command):
    from _suite_container import command_spec
    parts = command
    if len(parts) > 3 and parts[1:3] == ["-s", "tools/suite_deadline.py"] and "--" in parts[3:]:
        parts = parts[parts.index("--", 3) + 1:]
    return command_spec(parts)


def require_allocation(allocation, demand, affinity, mapped):
    """R's assigned hardware, not requested annotations, binds execution.

    The core class is not read: the shape asks for cores, and which class R
    assigned is the topology's decision (Rob, 2026-11-08).
    """
    metadata = [node["metadata"] for node in allocation["scheduling"]["graph"]["nodes"]]
    kinds = {kind: [item for item in metadata if item["type"] == kind]
             for kind in ("core", "memory", "gpu", "node")}
    rows = allocation["execution"]["R_lite"]
    nodes = kinds["node"]
    if len(rows) != 1 or len(nodes) != 1:
        raise ValueError("actual native allocation must contain one rank and host")
    ranks = idset(rows[0]["rank"])
    core_ids = idset(rows[0]["children"]["core"])
    gpu_ids = idset(rows[0]["children"].get("gpu", ""))
    if (ranks != {nodes[0]["rank"]}
            or allocation["execution"]["nodelist"] != [nodes[0]["name"]]
            or len(core_ids) != demand["cores"]
            or core_ids != {item["id"] for item in kinds["core"]}
            or gpu_ids != ({0} if demand["gpus"] else set())
            or any(item["rank"] != nodes[0]["rank"]
                   for kind in ("core", "memory", "gpu") for item in kinds[kind])):
        raise ValueError("actual native allocation core IDs/GPU/rank/nodelist differ")
    pus = sorted({int(pu) for core in kinds["core"] for pu in core["properties"]["os-pus"].split(",")})
    if (len(kinds["node"]) != 1 or kinds["node"][0]["properties"]["arch"] != demand["architecture"]
            or len(kinds["core"]) != demand["cores"]
            or len(kinds["memory"]) != demand["memory_gib"] or len(kinds["gpu"]) != demand["gpus"]
            or (demand.get("host") is not None and kinds["node"][0]["name"] != demand["host"])
            or affinity != pus or mapped != pus):
        raise ValueError("actual native allocation/resources/affinity differ")


def require_original_launch(intent, digest, jobspec):
    """The original wrapper argv authenticates all saved launch bytes."""
    wrapper_python = intent["command"][0] if intent.get("resources") else "/usr/bin/python3"
    if jobspec["tasks"][0]["command"] != [wrapper_python, "-s", "tools/_suite_flux.py", "payload", intent["directory"], digest]:
        raise ValueError("original native command does not authenticate this launch digest")


def submit(directory, checkout, command, cores, memory, timeout, environment, *, application=None,
           gpus=0, architecture="x86_64", host=None, client=None):
    validate(cores, memory, timeout, environment)
    demand = resources(cores, memory, timeout, gpus=gpus, architecture=architecture, host=host)
    container = container_spec(command)
    if gpus and container is None:
        raise ValueError("GPU suite requires the finite native container contract")
    if host is not None and container is None:
        raise ValueError("a native host constraint requires an explicit local container image dependency")
    if command[1:2] != ["-s"]:
        raise ValueError("native suite Python requires explicit -s startup isolation")
    if container is not None:
        from _suite_container import validate_mounts
        if container["workers"] != cores or container["strict_cuda"] != bool(gpus) or container["artifacts"] != environment:
            raise ValueError("container workers/GPU/artifact roots differ from native reservation")
        validate_mounts(container, Path(checkout))
    directory, checkout = Path(directory).resolve(), Path(checkout).resolve()
    identity = source(checkout)
    executable, client_env = connection(client=client)
    options = {"client": executable}
    require_capacity(demand, **options)
    qualified = instance_identity(**options)
    if application is not None:
        require_selection(application, identity)
    runtime = runtime_identity(command[0], checkout, timeout=timeout,
                               **({"wrapper": True} if container is not None else {}))
    intent = {"schema": "tessera.flux.launch.v1", "uri": client_env["FLUX_URI"],
              "client": executable, "library_path": client_env.get("LD_LIBRARY_PATH", ""),
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
           "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PRISMAQUANT_DEV_MODE": "0", "FLUX_F58_FORCE_ASCII": "1", **THREADS, **environment}
    if container is not None:
        env["XDG_RUNTIME_DIR"] = "/run/user/1000"
    wrapper = [command[0], "-s", "tools/_suite_flux.py", "payload", str(directory), digest]
    args = ["submit", "--env=-*", *[f"--env={k}={v}" for k, v in sorted(env.items())],
            f"--resources-shape={resource_shape(cores, memory, gpus)}",
            *(["--requires=" + architecture + (" and host:" + host if host is not None else "")] if not gpus
              else ["--requires=host:" + host] if host is not None else []),
            "-t", f"{timeout + 10}s", "--cwd", str(checkout),
            "--input=/dev/null", f"--output={directory}/stdout", f"--error={directory}/stderr",
            "--job-name=tessera-" + digest[:16], *wrapper]
    # Dry-run is native rendering only. It emits no whole-node request.
    jobspec = json.loads(checked([*args[:2], "--dry-run", *args[2:]], **options))
    save(directory / "jobspec.json", jobspec)
    result = native(["job", "submit", str(directory / "jobspec.json")], **options)
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


def require_saved_submission(directory, intent, sent, digest):
    """Authenticate saved launch bytes before querying an instance-local ID."""
    if (sent.get("state") != "submitted" or sent.get("launch_sha256") != digest
            or intent["directory"] != str(directory)):
        raise ValueError("unknown or changed submission identity")
    require_instance(intent)


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
        require_saved_submission(directory, intent, sent, digest)
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
                    or payload.get("process_cleanup_error") or payload.get("cleanup_error")
                    or payload.get("container_cleanup_error")):
                raise ValueError("payload source or scratch binding missing")
            if intent.get("resources") is not None:
                require_allocation(value, intent["resources"], payload["affinity"], payload["allocated_affinity"])
            if intent.get("container") is not None:
                from _suite_container import verify_record
                record["container"] = verify_record(directory, intent, digest, saved_id,
                                                    payload["affinity"], job_id=job_id)
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
    intent = json.loads((directory / "launch.json").read_bytes())
    digest = hashlib.sha256(encoded(intent)).hexdigest()
    require_saved_submission(directory.resolve(), intent, sent, digest)
    result = native(["job", "wait-event", sent["job_id"], "clean"], timeout=timeout)
    (directory / "wait.stdout").write_text(result.stdout)
    (directory / "wait.stderr").write_text(result.stderr)
    return collect(directory)


def payload_scratch_root():
    """The local scratch filesystem for the allocated host architecture."""
    return (Path("/home/rob/tl-gates/flux-validation") if os.uname().machine == "x86_64"
            else Path("/tmp/flux-vanilla"))


def payload(directory, digest):
    directory = Path(directory)
    raw = (directory / "launch.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("launch bytes changed")
    intent = json.loads(raw)
    saved = json.loads((directory / "submission.json").read_bytes())
    require_instance(intent)
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
    runtime_options = {"wrapper": True} if intent.get("container") is not None else {}
    if runtime_identity(intent["command"][0], Path.cwd(), timeout=intent["timeout_s"], **runtime_options) != intent["runtime"]:
        raise ValueError("interpreter/dependencies changed before payload")
    actual_job_id = os.environ["FLUX_JOB_ID"]
    scratch_root = payload_scratch_root()
    plain_job_id = checked(["job", "id", "--to=f58plain", actual_job_id]).strip()
    scratch = scratch_root / plain_job_id
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
        env = {**os.environ, "FLUX_JOB_ID": plain_job_id, "TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch),
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
        if runtime_identity(intent["command"][0], Path.cwd(), timeout=intent["timeout_s"], **runtime_options) != intent["runtime"]:
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
            if intent.get("container") is not None:
                try:
                    from _suite_container import cleanup_cid
                    container = json.loads((directory / "container.json").read_bytes())
                    # A second cleanup is needed only when the child could not
                    # finish its own CID cleanup. Never rewrite its announced
                    # record after successful cleanup.
                    if not container["removed"]:
                        cleanup_cid(container)
                        save(directory / "container.cleanup.json", container)
                    if not container["removed"] or container["cleanup_errors"]:
                        record["container_cleanup_error"] = "owned container cleanup incomplete"
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    record["container_cleanup_error"] = str(exc)
            try:
                shutil.rmtree(scratch)
                record["scratch_removed"] = not scratch.exists()
            except OSError as exc:
                record["cleanup_error"] = str(exc)
            record['declared_memory_gib'] = intent.get('memory_gib')
            peak = Path('/sys/fs/cgroup' + Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]) / 'memory.peak'
            record['memory_peak_bytes'] = None
            try:
                record['memory_peak_bytes'] = int(peak.read_text())
            except OSError as exc:
                record['memory_peak_read_error'] = str(exc)
            save(directory / "payload.json", record)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def require_selection(binding, identity):
    raw = Path(binding["selection_path"]).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != binding["selection_sha256"]
            or binding["source"] != identity):
        raise ValueError("selection bytes or selection-time source changed")


def runtime_identity(python, checkout, *, timeout, wrapper=False):
    result = subprocess.run([python, "-s", "-B", str(Path(checkout) / "tools/_suite_runtime.py"),
                             *(["--wrapper"] if wrapper else [])],
                            capture_output=True, text=True, timeout=timeout,
                            env={"PATH": "/usr/bin:/bin", "HOME": "/home/rob",
                                 "PYTHONPATH": "src", "PYTHONNOUSERSITE": "1"})
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
