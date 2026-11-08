"""Finite suite command, rootless container lifetime and execution evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RUNNER = "tools/suite_container.py"
PODMAN = "/usr/bin/podman"
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}")
_INTERPRETERS = {"/usr/bin/python3", "/usr/local/bin/python3", "/opt/tl3/bin/python3"}
_SHA = re.compile(r"[0-9a-f]{64}")
_SINGLE = {"--image", "--deps-site", "--deps-sha256", "--surface-dir", "--cache-dir"}
_REPEAT = {"--data-root", "--artifact-root"}
THREAD_LIMITS = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "NUMEXPR_NUM_THREADS"), "1")


def _require(condition, reason):
    if not condition:
        raise ValueError("suite container: " + reason)


_NONLOCAL_CACHE_FILESYSTEMS = {
    "autofs", "9p", "afs", "ceph", "cifs", "glusterfs", "lustre", "ncpfs", "nfs", "nfs4",
    "smb3", "smbfs", "davfs", "fuse.ceph", "fuse.curlftpfs", "fuse.davfs",
    "fuse.gcsfuse", "fuse.glusterfs", "fuse.rclone", "fuse.s3fs", "fuse.smbnetfs",
    "fuse.sshfs",
}


def require_local_cache(path):
    """One cache admission rule for the coordinator and container launcher.

    The mount owner has no exhaustive local filesystem classification, so
    refuse known network types and unresolved automounts, not a local roster.
    """
    from tessera._dev.native_identity import native_cache_mount

    option = "--cache-dir (merge-suite --gpu-cache-dir)"
    try:
        cache, mountpoint, filesystem = native_cache_mount(path)
    except (OSError, RuntimeError) as exc:
        _require(False, f"{option}: {exc}")
    _require(filesystem not in _NONLOCAL_CACHE_FILESYSTEMS,
             f"{option}: cache path {cache} is on mount point {mountpoint} "
             f"with filesystem type {filesystem}; use local disk for native build "
             "locks and mapped inode identity")


def absolute(value):
    path = Path(value)
    _require(path.is_absolute() and str(path) == value and ".." not in path.parts
             and "," not in value and "\x00" not in value, "noncanonical absolute mount path")
    return path


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def artifact_specs():
    repo = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(repo / "tests/box_artifacts.py"))["ROOTS"]


def artifact_variables():
    return {spec.env for spec in artifact_specs().values()}


def validate_targets(targets):
    """One canonical, nonoverlapping target roster at composition and parse."""
    _require(targets and len(targets) <= 1024 and len(set(targets)) == len(targets),
             "suite targets require unique canonical paths under tests/")
    for target in targets:
        path = target.split("::", 1)[0]
        _require(isinstance(target, str) and "\0" not in target
                 and (path == "tests" or path.startswith("tests/"))
                 and str(Path(path)) == path and ".." not in Path(path).parts,
                 "suite targets require unique canonical paths under tests/")
    ordered = sorted(targets)
    _require(not any(right.startswith(left + separator)
                     for index, left in enumerate(ordered) for right in ordered[index + 1:]
                     for separator in ("/", "::", "[")), "suite targets must not overlap")


def parse(argv):
    """Reject option shadows instead of applying last-option-wins semantics."""
    _require("--" in argv, "runner has no command separator")
    options, inner = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    values, repeated = {}, {key: [] for key in _REPEAT}
    _require(len(options) % 2 == 0, "runner options require explicit values")
    for index in range(0, len(options), 2):
        key, value = options[index:index + 2]
        _require(key in _SINGLE | _REPEAT, "unknown runner option " + key)
        if key in _SINGLE:
            _require(key not in values, "duplicate runner option " + key)
            values[key] = value
        else:
            _require(value not in repeated[key], "duplicate runner option " + key)
            repeated[key].append(value)
    _require(set(values) == _SINGLE, "missing runner option")
    _require(_IMAGE.fullmatch(values["--image"]), "image must be immutable")
    _require(_SHA.fullmatch(values["--deps-sha256"]), "dependency content identity is absent")
    for key in ("--deps-site", "--surface-dir", "--cache-dir"):
        absolute(values[key])
    for path in repeated["--data-root"]:
        absolute(path)
    artifacts = {}
    for item in repeated["--artifact-root"]:
        key, separator, value = item.partition("=")
        _require(separator and key in artifact_variables(), "unknown artifact environment")
        _require(key not in artifacts, "duplicate artifact environment")
        artifacts[key] = str(absolute(value))
    _require(len(inner) >= 4 and inner[0] in _INTERPRETERS and inner[1:3] == ["-m", "pytest"],
             "runner only executes a declared interpreter -m pytest")
    outputs, junits, seen, plugins, workers, targets = [], [], set(), set(), 1, []
    flags = {"-q", "-v", "-s", "-x", "--quiet", "--verbose", "--strict-cuda", "--no-header", "--disable-warnings", "--full-trace", "--showlocals"}
    valued = {"-n", "--dist", "--durations", "--tb", "--maxfail", "-k", "-m", "--surface-json", "--junitxml", "-p", "-o"}
    options_started = False
    index = 3
    while index < len(inner):
        option = inner[index]
        if option in flags:
            _require(option not in seen, "duplicate pytest option " + option)
            seen.add(option)
            options_started = True
            index += 1
            continue
        key, equal, attached = option.partition("=")
        if key in valued:
            options_started = True
            _require(key == "-p" or key not in seen, "duplicate pytest option " + key)
            seen.add(key)
            if equal:
                value = attached
            else:
                index += 1
                _require(index < len(inner), "pytest option has no value")
                value = inner[index]
            if key == "--surface-json":
                outputs.append(value)
            elif key == "--junitxml":
                junits.append(value)
            elif key == "-o":
                _require(value == "addopts=", "pytest config shadow")
            elif key == "-p":
                _require(value in ("no:cacheprovider", "xdist.plugin") and value not in plugins, "pytest plugin shadow")
                plugins.add(value)
            elif key == "-n":
                _require(value.isdigit() and int(value) > 0, "worker count must be an explicit positive integer")
                workers = int(value)
            elif key == "--dist":
                _require(value == "worksteal", "container distribution must be worksteal")
            elif key in ("--durations", "--maxfail"):
                _require(value.isdigit(), "pytest count must be finite")
            elif key == "--tb":
                _require(value in ("auto", "long", "short", "line", "native", "no"), "unknown traceback mode")
        else:
            _require(not options_started, "pytest target or control appended after owned options: " + option)
            targets.append(option)
        index += 1
    _require(len(outputs) == 1, "population must name exactly one surface output")
    validate_targets(targets)
    _require("-o" in seen and "no:cacheprovider" in plugins, "tool-owned config clearing is absent")
    _require((workers > 1) == ("xdist.plugin" in plugins) and (workers > 1) == ("--dist" in seen),
             "worker/plugin/distribution controls differ")
    output = absolute(outputs[0])
    _require(output.parent == absolute(values["--surface-dir"]), "surface output escapes its mount")
    _require(len(junits) == 1, "population must name exactly one JUnit output")
    junit = absolute(junits[0])
    _require(junit.parent == output.parent and junit != output, "JUnit output escapes or shadows population")
    _require(not _overlap(absolute(values["--surface-dir"]), absolute(values["--cache-dir"])), "surface/cache mount overlap")
    _require(not _overlap(absolute(values["--deps-site"]), absolute(values["--cache-dir"]))
             and not _overlap(absolute(values["--deps-site"]), absolute(values["--surface-dir"])), "dependency writable mount shadow")
    scratch_env = artifact_specs()["scratch"].env
    _require(scratch_env not in artifacts or artifacts[scratch_env] == values["--cache-dir"] + "/tmp",
             "scratch must use the action-owned cache")
    return {"schema": "tessera.suite_container.v1", **values, **repeated,
            "artifacts": artifacts, "inner": inner, "surface": str(output),
            "junit": str(junit), "workers": workers, "strict_cuda": "--strict-cuda" in seen}


def command_spec(command):
    """Only the canonical relative runner path can declare this contract."""
    if (len(command) >= 2 and re.fullmatch(r"python(3(\.\d+)?)?", Path(command[0]).name)
            and command[1] == RUNNER):
        return parse(command[2:])
    return None


def dependency_manifest(site):
    """Version declarations and all actual dependency bytes form one seal."""
    from importlib.metadata import distributions
    root = absolute(str(site))
    _require(root.resolve() == root and root.is_dir(), "dependency site is missing or a symlink")
    files = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "dependency site contains a symlink")
        if path.is_file():
            raw = path.read_bytes()
            files.append([str(path.relative_to(root)), path.stat().st_mode & 0o777,
                          len(raw), hashlib.sha256(raw).hexdigest()])
    versions = sorted((d.metadata["Name"], d.version) for d in distributions(path=[str(root)]))
    _require({"pytest", "pytest-xdist"} <= {name.lower() for name, _ in versions}, "scoped pytest and xdist versions are missing")
    body = {"schema": "tessera.suite_dependencies.v1", "versions": versions, "files": files}
    seal = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body, seal



def validate_mounts(spec, checkout):
    """Readonly and writable ownership cannot shadow one another."""
    require_local_cache(spec["--cache-dir"])
    checkout = absolute(str(checkout))
    deps, output, cache = [absolute(spec[key]) for key in ("--deps-site", "--surface-dir", "--cache-dir")]
    paths = [checkout, deps, output, cache]
    _require(all(path.resolve() == path for path in paths), "source/dependency/output/cache symlink refused")
    _require(all(not _overlap(left, right) for index, left in enumerate(paths)
                 for right in paths[index + 1:]), "mount shadows source/dependency/output/cache")
    scratch_env = artifact_specs()["scratch"].env
    readonly = [checkout, deps, *[absolute(path) for path in spec["--data-root"]],
                *[absolute(path) for key, path in spec["artifacts"].items() if key != scratch_env]]
    for path in readonly:
        _require(path.resolve() == path and path.is_dir(), "readonly mount missing or symlink: " + str(path))
        _require(not any(_overlap(path, writable) for writable in (output, cache)),
                 "readonly root overlaps writable output/cache")
        _require(path in (checkout, deps) or not any(_overlap(path, root) for root in (checkout, deps)),
                 "data/artifact mount shadows source/dependencies")
    # Linked worktrees need the existing Git metadata, never the interactive
    # worktree files. Mount only the two explicit Git metadata directories.
    for option in ("--absolute-git-dir", "--git-common-dir"):
        result = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", option], text=True, timeout=30).strip()
        path = Path(result)
        path = (checkout / path).resolve() if not path.is_absolute() else path.resolve()
        _require(not any(_overlap(path, writable) for writable in (output, cache)), "Git metadata overlaps writable roots")
        readonly.append(path)
    return sorted(set(readonly))


def podman_command(spec, checkout, environment, *, cidfile, affinity, gpus):
    readonly = validate_mounts(spec, checkout)
    _require(type(gpus) is int and gpus in (0, 1), "GPU count must be zero or one whole GPU")
    _require(spec["strict_cuda"] == bool(gpus), "strict CUDA differs from native GPU intention")
    _require(affinity and all(type(cpu) is int and cpu >= 0 for cpu in affinity), "assigned cpuset missing")
    for key in artifact_variables():
        if environment.get(key):
            _require(spec["artifacts"].get(key) == environment[key], "artifact environment shadow: " + key)
    _require(not environment.get("TESSERA_SOURCE_VERIFIER"), "native source verifier shadow")
    _require(environment.get("FLUX_JOB_ID") and environment.get("TMPDIR"), "native scratch/job ID absent")
    scratch = Path(environment["TMPDIR"])
    _require(scratch.parent == Path(spec["--cache-dir"])
             and scratch.name == environment["FLUX_JOB_ID"], "cache is not native job-ID-owned scratch")
    cache = scratch / "container-cache"
    env = {**THREAD_LIMITS, "TORCH_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PRISMAQUANT_DEV_MODE": "0",
           "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(cache),
           "TMPDIR": str(cache / "tmp"), "XDG_CACHE_HOME": str(cache / "xdg"),
           "TORCH_EXTENSIONS_DIR": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
           "NINJAFLAGS": "-j1", "MAKEFLAGS": "-j1",
           "PYTHONPATH": str(checkout / "src") + ":" + spec["--deps-site"],
           "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": str(checkout),
           "FLUX_JOB_ID": environment["FLUX_JOB_ID"]}
    env.update(spec["artifacts"])
    env[artifact_specs()["scratch"].env] = env["TMPDIR"]
    for key in ("TESSERA_PRISMAQUANT_DIR", "TESSERA_PRISMAQUANT_WORKTREE"):
        if key in spec["artifacts"]:
            env["PYTHONPATH"] += ":" + spec["artifacts"][key]
    command = [PODMAN, "create", "--pull=never", "--cidfile", str(cidfile),
               "--cgroups=disabled", "--network=none", "--env-host=false", "--unsetenv-all",
               "--cpuset-cpus", ",".join(map(str, affinity)),
               "--entrypoint", spec["inner"][0], "--workdir", str(checkout)]
    if os.uname().machine == "aarch64":
        command += ["--userns=keep-id", "--user=1000:1000"]
    else:
        command += ["--user=0:0"]
    if gpus:
        command += ["--device", "nvidia.com/gpu=all"]
    else:
        command += ["--read-only"]
    for path in readonly:
        command += ["--mount", f"type=bind,src={path},dst={path},readonly"]
    for path in (Path(spec["--surface-dir"]), cache):
        command += ["--mount", f"type=bind,src={path},dst={path}"]
    for key, value in sorted(env.items()):
        command += ["--env", f"{key}={value}"]
    return [*command, spec["--image"]]


def image_identity(image):
    result = subprocess.run([PODMAN, "image", "inspect", image], capture_output=True, text=True, timeout=30)
    _require(result.returncode == 0, "immutable local image unavailable; pulling is forbidden")
    value, = json.loads(result.stdout)
    identity = {key: value[key] for key in ("Id", "Architecture", "Os")}
    identity["Id"] = "sha256:" + identity["Id"].removeprefix("sha256:")
    _require(identity["Id"] == image and identity["Os"] == "linux", "local image content differs")
    return identity


def driver_identity(gpus):
    if not gpus:
        return None
    result = subprocess.run(["/usr/bin/nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30)
    _require(result.returncode == 0 and result.stdout.strip(), "host/device driver identity unavailable")
    return result.stdout.strip().splitlines()


def cleanup_cid(record):
    """Reap the captured handle only, preserving errors even on failed remove."""
    path = Path(record["cidfile"])
    try:
        if path.is_symlink():
            raise ValueError("owned cidfile replaced by symlink")
        cid = path.read_text().strip() if path.exists() else None
        if cid is None:
            record["removed"] = record.get("creation_returncode") not in (None, 0)
            return
        _require(re.fullmatch(r"[0-9a-f]{64}", cid), "invalid captured CID")
        _require(record.get("cid") in (None, cid), "owned cidfile replaced")
        record["cid"] = cid
        for args in (["stop", "--time", "2", cid], ["rm", "--force", cid]):
            result = subprocess.run([PODMAN, *args], capture_output=True, text=True, timeout=15)
            if result.returncode:
                record["cleanup_errors"].append(" ".join(args[:1]) + ": " + result.stderr.strip())
        result = subprocess.run([PODMAN, "container", "exists", cid], capture_output=True, timeout=15)
        record["removed"] = result.returncode == 1
        _require(result.returncode in (0, 1), "container absence query failed")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["cleanup_errors"].append(str(exc))


def admitted_context(directory, digest, environment):
    import _suite_flux as flux
    directory = Path(directory)
    raw = (directory / "launch.json").read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == digest, "launch bytes changed")
    intent = json.loads(raw)
    sent = json.loads((directory / "submission.json").read_bytes())
    _require(sent["launch_sha256"] == digest and sent["state"] == "submitted"
             and flux.job_id(sent["job_id"]) == flux.job_id(environment["FLUX_JOB_ID"]), "native job ID differs")
    _require(intent["directory"] == str(directory) and Path(intent["checkout"]) == Path.cwd(), "native source/root differs")
    _require(flux.source(Path.cwd()) == intent["source"], "native source authentication failed")
    native_id = flux.job_id(environment["FLUX_JOB_ID"])
    original = json.loads(flux.checked(["job", "info", "--original", native_id, "jobspec"]))
    flux.require_original_launch(intent, digest, original)
    events = [json.loads(line) for line in flux.checked(["job", "eventlog", "--format=json", native_id]).splitlines()]
    names = {event["name"] for event in events}
    _require("start" in names and not names.intersection(("finish", "clean")), "native job is not executing")
    allocation = json.loads(flux.checked(["job", "info", native_id, "R"]))
    affinity = sorted(os.sched_getaffinity(0))
    flux.require_allocation(allocation, intent["resources"], affinity, affinity)
    unit = "/shell-" + native_id + ":sys.service"
    _require(any(line.split(":", 2)[-1].endswith(unit)
                 for line in Path("/proc/self/cgroup").read_text().splitlines()), "native process scope differs")
    return intent


def run(spec, checkout, environment, directory, intent, digest):
    import _suite_flux as flux
    directory, checkout = Path(directory), Path(checkout)
    resources = intent["resources"]
    _require(intent["container"] == spec and spec["workers"] == resources["cores"], "container/native intent differs")
    _require(spec["strict_cuda"] == bool(resources["gpus"]), "strict CUDA/GPU reservation differs")
    validate_mounts(spec, checkout)
    body, deps = dependency_manifest(spec["--deps-site"])
    _require(deps == spec["--deps-sha256"], "dependency content identity differs")
    image = image_identity(spec["--image"])
    _require(image["Architecture"] == {"x86_64": "amd64", "aarch64": "arm64"}[resources["architecture"]], "image architecture differs")
    affinity = sorted(os.sched_getaffinity(0))
    record = {"schema": "tessera.suite_container_execution.v1", "job_id": environment["FLUX_JOB_ID"],
              "launch_sha256": digest, "source": intent["source"], "image": image,
              "cidfile": str(directory / "container.cid"), "cid": None, "affinity": affinity,
              "gpus": resources["gpus"], "dependencies": deps, "entry": None, "exit": None,
              "returncode": None, "cleanup_errors": [], "removed": False,
              "driver": driver_identity(resources["gpus"])}
    path = directory / "container.json"
    _require(not path.exists() and not Path(record["cidfile"]).exists(), "container record already exists")
    flux.save(path, record)  # Register the cidfile before the first create.
    process = None
    class Interrupted(BaseException):
        pass
    def interrupt(signum, frame):
        record["interrupted_signal"] = signum
        raise Interrupted
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        cache = Path(environment["TMPDIR"]) / "container-cache"
        cache.mkdir(exist_ok=False)
        for name in ("tmp", "xdg", "torch", "triton"):
            (cache / name).mkdir()
        command = podman_command(spec, checkout, environment, cidfile=record["cidfile"], affinity=affinity, gpus=resources["gpus"])
        command += [RUNNER, "inside", str(directory), digest]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        record["creation_returncode"] = process.returncode
        (directory / "container.create.stdout").write_text(stdout)
        (directory / "container.create.stderr").write_text(stderr)
        _require(process.returncode == 0, "rootless container create failed: " + stderr.strip())
        cid = Path(record["cidfile"]).read_text().strip()
        _require(re.fullmatch(r"[0-9a-f]{64}", cid), "create returned no captured CID")
        record["cid"] = cid
        inspection = subprocess.run([PODMAN, "inspect", cid], capture_output=True, text=True, timeout=30)
        _require(inspection.returncode == 0, "created container inspect failed")
        actual, = json.loads(inspection.stdout)
        record["inspection"] = {"Id": actual["Id"], "Image": actual["Image"],
                                "CpusetCpus": actual["HostConfig"]["CpusetCpus"]}
        _require(actual["Id"] == cid and actual["Image"].removeprefix("sha256:") == image["Id"].removeprefix("sha256:"), "actual container image differs")
        flux.save(path, record)
        process = subprocess.Popen([PODMAN, "start", "--attach", cid])
        record["returncode"] = process.wait()
        for phase in ("entry", "exit"):
            raw = (directory / f"container.{phase}.json").read_bytes()
            record[phase] = json.loads(raw)
            record[phase + "_sha256"] = hashlib.sha256(raw).hexdigest()
        _require(record["entry"] == record["exit"], "container source/runtime/dependency identity changed")
        _require(dependency_manifest(spec["--deps-site"])[1] == deps, "dependency content changed during execution")
        _require(image_identity(spec["--image"]) == image, "image changed during execution")
        return record["returncode"]
    except Interrupted:
        return 128 + record["interrupted_signal"]
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            cleanup_cid(record)
        except (OSError, subprocess.SubprocessError) as exc:
            record["cleanup_errors"].append(str(exc))
        finally:
            flux.save(path, record)
            print("Tessera container evidence " + str(path) + " sha256=" + hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def inside(directory, digest):
    """Measure the numerical runtime inside the actual immutable container."""
    import _suite_flux as flux
    from _suite_runtime import identity
    directory = Path(directory)
    raw = (directory / "launch.json").read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == digest, "inside launch changed")
    intent = json.loads(raw)
    spec = intent["container"]
    def snapshot():
        return {"schema": "tessera.suite_container_runtime.v1", "job_id": os.environ["FLUX_JOB_ID"],
                "launch_sha256": digest, "source": flux.source(Path.cwd()),
                "runtime": identity(extra_sites=(spec["--deps-site"],)),
                "dependencies": dependency_manifest(spec["--deps-site"])[1],
                "affinity": sorted(os.sched_getaffinity(0)), "driver": driver_identity(intent["resources"]["gpus"])}
    entry = snapshot()
    _require(entry["source"] == intent["source"] and entry["dependencies"] == spec["--deps-sha256"], "inside source/dependencies differ")
    flux.save(directory / "container.entry.json", entry)
    command = [*spec["inner"], "--basetemp", str(Path(os.environ["TMPDIR"]) / "pytest")]
    result = subprocess.call(command)
    exit_record = snapshot()
    flux.save(directory / "container.exit.json", exit_record)
    _require(entry == exit_record, "inside source/runtime/dependencies changed")
    return result


def verify_record(directory, intent, digest, native_id, affinity):
    import _suite_flux as flux
    path = Path(directory) / "container.json"
    _require(not path.is_symlink(), "container evidence symlink")
    raw = path.read_bytes()
    record = json.loads(raw)
    announcement = "Tessera container evidence " + str(path) + " sha256=" + hashlib.sha256(raw).hexdigest()
    _require(announcement in (Path(directory) / "stdout").read_text().splitlines(), "container evidence missing or replaced")
    spec, resources = intent["container"], intent["resources"]
    _require(record["schema"] == "tessera.suite_container_execution.v1"
             and flux.job_id(record["job_id"]) == flux.job_id(native_id)
             and record["launch_sha256"] == digest and record["source"] == intent["source"]
             and record["image"]["Id"] == spec["--image"]
             and record["image"]["Os"] == "linux"
             and record["image"]["Architecture"] == {"x86_64": "amd64", "aarch64": "arm64"}[resources["architecture"]]
             and record["gpus"] == resources["gpus"] and record["affinity"] == affinity
             and record["dependencies"] == spec["--deps-sha256"]
             and record["removed"] and not record["cleanup_errors"], "container execution/cleanup binding differs")
    _require(record["entry"] is not None and record["entry"] == record["exit"], "container runtime span missing or changed")
    for phase in ("entry", "exit"):
        phase_path = Path(directory) / f"container.{phase}.json"
        _require(not phase_path.is_symlink(), "container phase evidence symlink")
        raw = phase_path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == record[phase + "_sha256"]
                 and json.loads(raw) == record[phase], "container phase evidence replaced")
    entry = record["entry"]
    _require(entry["schema"] == "tessera.suite_container_runtime.v1"
             and entry["source"] == intent["source"] and entry["launch_sha256"] == digest
             and flux.job_id(entry["job_id"]) == flux.job_id(native_id)
             and entry["dependencies"] == spec["--deps-sha256"] and entry["affinity"] == affinity
             and entry["driver"] == record["driver"] and bool(entry["driver"]) == bool(resources["gpus"])
             and entry["runtime"]["schema"] == "tessera.suite_runtime.v1", "inside runtime/source/allocation binding differs")
    inspection = record["inspection"]
    _require(re.fullmatch(r"[0-9a-f]{64}", record["cid"])
             and inspection["Id"] == record["cid"]
             and inspection["Image"].removeprefix("sha256:") == spec["--image"].removeprefix("sha256:")
             and inspection["CpusetCpus"] == ",".join(map(str, affinity)), "actual CID/image/cpuset differs")
    return record


def main(argv=None):
    try:
        argv = sys.argv[1:] if argv is None else argv
        if len(argv) == 3 and argv[0] == "inside":
            return inside(argv[1], argv[2])
        spec = parse(argv)
        directory, digest = os.environ["TESSERA_SUITE_DIRECTORY"], os.environ["TESSERA_SUITE_LAUNCH_SHA256"]
        intent = admitted_context(directory, digest, os.environ)
        return run(spec, Path.cwd(), os.environ, directory, intent, digest)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2
