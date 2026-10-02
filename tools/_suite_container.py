"""The one finite command contract for a PB-contained pytest population.

This is an argv owner, not a scheduler. The launcher uses PB's PATH Docker
shim; the receipt reader parses this exact source-sealed launcher contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
import tempfile

RUNNER = "tools/suite_container.py"
OWNER = "tools/_suite_container.py"
SOURCE_FILES = (RUNNER, OWNER)
PB_VERIFIER = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbsnapshot.py")
_IMAGE = re.compile(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}")
_SHA = re.compile(r"[0-9a-f]{64}")
_SINGLE = {"--image", "--deps-site", "--deps-sha256", "--surface-dir", "--cache-dir"}
_REPEAT = {"--data-root", "--artifact-root"}
THREAD_LIMITS = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "NUMEXPR_NUM_THREADS"), "1")


def _require(condition, reason):
    if not condition:
        raise ValueError("suite container: " + reason)


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
    _require(inner[:3] == ["/usr/bin/python3", "-m", "pytest"], "runner only executes /usr/bin/python3 -m pytest")
    _require("--strict-cuda" in inner[3:], "container population requires strict CUDA")
    outputs, seen, workers = [], set(), 1
    flags = {"-q", "-v", "-s", "--quiet", "--verbose", "--strict-cuda", "--no-header", "--disable-warnings"}
    valued = {"-n", "--dist", "--durations", "--tb", "--maxfail", "-k", "-m", "--surface-json", "-p"}
    index = 3
    while index < len(inner):
        option = inner[index]
        if option in flags:
            index += 1
            continue
        key, equal, attached = option.partition("=")
        if key in valued:
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
            elif key == "-p":
                _require(value in ("no:cacheprovider", "xdist.plugin"), "pytest plugin shadow")
            elif key == "-n":
                _require(value.isdigit() and int(value) > 0, "worker count must be an explicit positive integer")
                workers = int(value)
            elif key == "--dist":
                _require(value == "worksteal", "container distribution must be worksteal")
        else:
            test_path = option.split("::", 1)[0]
            _require((test_path == "tests" or test_path.startswith("tests/"))
                     and str(Path(test_path)) == test_path and ".." not in Path(test_path).parts,
                     "unknown pytest option or foreign test source: " + option)
        index += 1
    _require(len(outputs) == 1, "population must name exactly one surface output")
    output = absolute(outputs[0])
    _require(output.parent == absolute(values["--surface-dir"]), "surface output escapes its mount")
    _require(not _overlap(absolute(values["--surface-dir"]), absolute(values["--cache-dir"])), "surface/cache mount overlap")
    _require(not _overlap(absolute(values["--deps-site"]), absolute(values["--cache-dir"]))
             and not _overlap(absolute(values["--deps-site"]), absolute(values["--surface-dir"])), "dependency writable mount shadow")
    scratch_env = artifact_specs()["scratch"].env
    _require(scratch_env not in artifacts or artifacts[scratch_env] == values["--cache-dir"] + "/tmp",
             "scratch must use the action-owned cache")
    return {**values, **repeated, "artifacts": artifacts, "inner": inner, "surface": str(output), "workers": workers}


def command_spec(command):
    """Only the canonical relative runner path can declare this contract."""
    if (len(command) >= 2 and re.fullmatch(r"python(3(\.\d+)?)?", Path(command[0]).name)
            and command[1] == RUNNER):
        return parse(command[2:])
    return None


def admission(spec, payload):
    """The runner's image and process count must be PB's declared demand."""
    params = payload.get("params") or {}
    _require(spec["--image"] in params.get("container_images", []), "container image is not PB-declared")
    demand = params.get("demand") or {}
    _require(demand.get("cpu") == spec["workers"] and demand.get("gpu", 0) > 0,
             "container workers/GPU differ from PB reservation")
    variables = (payload.get("environment") or {}).get("variables") or {}
    for key, value in spec["artifacts"].items():
        _require(variables.get(key) == value, "artifact root is not sealed: " + key)


def admitted_request(checkout, environment):
    """Reuse PB's own closure verifier rather than inventing action ownership."""
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    verified = runpy.run_path(str(PB_VERIFIER))["verify"](checkout, commit)
    key = environment["PRISMABUILD_ACTION_KEY"]
    stamps = [stamp for stamp in verified["generated"] if stamp.get("action_key") == key]
    _require(len(stamps) == 1, "source verifier did not bind this executing action")
    request = Path("/mnt/shared/prismabuild-fleet/cas/requests", key[:2], key + ".json").read_bytes()
    _require(hashlib.sha256(request).hexdigest() == stamps[0]["request_sha256"], "executing request digest differs")
    return json.loads(request)


def source_bound(payload, cas_root, verified_bundles=None):
    """Authenticate runner blobs from the request's sealed snapshot bundle."""
    snapshot = payload["params"]["checkout_snapshot"]
    entry = snapshot["input"]
    _require(entry in payload["inputs"], "snapshot input is not sealed")
    digest = entry["sha256"]
    _require(_SHA.fullmatch(digest) and 0 < entry["bytes"] <= 256 * 1024 * 1024,
             "snapshot bundle identity/size is invalid")
    cache = {} if verified_bundles is None else verified_bundles
    if digest not in cache:
        blob = Path(cas_root, "blobs", digest[:2], digest)
        _require(blob.stat().st_size == entry["bytes"], "snapshot bundle length differs")
        raw = blob.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == digest, "snapshot bundle digest differs")
        cache[digest] = {"bytes": raw, "commits": set()}
    raw = cache[digest]["bytes"]
    _require(len(raw) == entry["bytes"], "snapshot bundle length differs")
    if snapshot["commit"] in cache[digest]["commits"]:
        return
    parent = Path(os.environ.get("TMPDIR", "/home/rob/tmp"))
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="suite-runner-source-", dir=parent) as directory:
        blob = Path(directory, "source.bundle")
        blob.write_bytes(raw)
        def git(*args):
            return subprocess.check_output(["git", "-C", directory, *args], stderr=subprocess.DEVNULL, timeout=30)
        git("init", "--bare", "--quiet")
        git("fetch", "--quiet", "--no-tags", str(blob), snapshot["commit"])
        repo = Path(__file__).resolve().parents[1]
        for name in SOURCE_FILES:
            mode = git("ls-tree", snapshot["commit"], "--", name).split()[0]
            _require(mode in (b"100644", b"100755"), "runner source is not a regular blob")
            _require(git("show", f"{snapshot['commit']}:{name}") == (repo / name).read_bytes(),
                     "authenticated snapshot runner source differs: " + name)
    cache[digest]["commits"].add(snapshot["commit"])


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


def docker_command(spec, checkout, environment):
    """Construct only this contract; PB's Docker shim owns scope and affinity."""
    checkout = absolute(str(checkout))
    paths = [checkout, absolute(spec["--deps-site"]), absolute(spec["--surface-dir"]), absolute(spec["--cache-dir"])]
    _require(all(path.resolve() == path for path in paths), "source/dependency/output/cache symlink refused")
    _require(all(not _overlap(checkout, path) for path in paths[1:]), "mount shadows source checkout")
    readonly = [absolute(path) for path in spec["--data-root"]]
    scratch_env = artifact_specs()["scratch"].env
    readonly += [absolute(path) for key, path in spec["artifacts"].items() if key != scratch_env]
    readonly += paths[:2]
    for path in readonly:
        _require(path.resolve() == path and path.exists(), "readonly mount missing or symlink: " + str(path))
        _require(not any(path == writable or writable in path.parents for writable in paths[2:]), "data mount shadows writable output/cache")
    readonly = sorted(set(readonly), key=lambda p: (len(p.parts), str(p)))
    # A readonly parent already supplies its readonly descendants. Writable
    # output/cache are the only intentional children, emitted last below.
    readonly = [path for path in readonly if not any(parent in path.parents for parent in readonly)]
    env = {**THREAD_LIMITS, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "HOME": spec["--cache-dir"],
           "TMPDIR": spec["--cache-dir"] + "/tmp", "XDG_CACHE_HOME": spec["--cache-dir"] + "/xdg",
           "TORCH_EXTENSIONS_DIR": spec["--cache-dir"] + "/torch", "TRITON_CACHE_DIR": spec["--cache-dir"] + "/triton",
           "CMAKE_BUILD_PARALLEL_LEVEL": "1", "NINJAFLAGS": "-j1", "MAKEFLAGS": "-j1",
           "PYTHONPATH": str(checkout / "src") + ":" + spec["--deps-site"],
           "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": str(checkout)}
    for name in ("TESSERA_SOURCE_VERIFIER", "PRISMABUILD_CONTAINER_OWNER"):
        _require(environment.get(name), "admitted action lacks " + name)
        env[name] = environment[name]
    if "CUDA_VISIBLE_DEVICES" in environment:
        _require(environment["CUDA_VISIBLE_DEVICES"], "CUDA device visibility is empty")
        env["CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"]
    verifier = shlex.split(env["TESSERA_SOURCE_VERIFIER"])
    _require(len(verifier) == 3 and verifier[0] == "/usr/bin/python3" and verifier[2] == "verify"
             and Path(verifier[1]).name == "pbsnapshot.py", "source verifier shape refused")
    _require(Path(verifier[1]).resolve(strict=True) == PB_VERIFIER.resolve(strict=True), "source verifier is not the published PB helper")
    _require(any(absolute(verifier[1]).is_relative_to(path) for path in readonly), "source verifier is not mounted")
    for key in artifact_variables():
        if environment.get(key):
            _require(spec["artifacts"].get(key) == environment[key], "artifact environment shadow: " + key)
    env.update(spec["artifacts"])
    env[scratch_env] = env["TMPDIR"]
    for name in ("TESSERA_PRISMAQUANT_DIR", "TESSERA_PRISMAQUANT_WORKTREE"):
        if name in spec["artifacts"]:
            env["PYTHONPATH"] += ":" + spec["artifacts"][name]
    command = ["docker", "run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
               "--entrypoint", "/usr/bin/python3", "--workdir", str(checkout)]
    for path in readonly:
        command += ["--mount", f"type=bind,src={path},dst={path},readonly"]
    for path in paths[2:]:
        command += ["--mount", f"type=bind,src={path},dst={path}"]
    for key, value in sorted(env.items()):
        command += ["--env", f"{key}={value}"]
    return [*command, spec["--image"], *spec["inner"][1:], "--basetemp", spec["--cache-dir"] + "/pytest"]


def main(argv=None):
    try:
        spec = parse(sys.argv[1:] if argv is None else argv)
        _require(os.environ.get("PRISMABUILD_ACTION_KEY") and os.environ.get("PRISMABUILD_ACTION_SCOPE"), "container launch requires an admitted PB action")
        admission(spec, admitted_request(Path.cwd(), os.environ))
        _, digest = dependency_manifest(spec["--deps-site"])
        _require(digest == spec["--deps-sha256"], "dependency content identity differs")
        command = docker_command(spec, Path.cwd(), os.environ)
        cache = absolute(spec["--cache-dir"])
        for name in ("", "tmp", "xdg", "torch", "triton"):
            (cache / name).mkdir(parents=True, exist_ok=True)
        owner = cache / ".suite-action"
        key = os.environ["PRISMABUILD_ACTION_KEY"]
        try:
            with owner.open("x") as handle:
                handle.write(key)
        except FileExistsError:
            _require(not owner.is_symlink() and owner.read_text() == key, "cache belongs to another action")
        result = subprocess.call(command)
        # Bind dependency bytes across execution, just as suite_source binds
        # tracked source at entry and publication. A host-side mutation must
        # not let a clean pytest summary establish a successful arm.
        _require(dependency_manifest(spec["--deps-site"])[1] == digest,
                 "dependency content changed during execution")
        return result
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2
