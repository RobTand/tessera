"""The paired receipt can bind the finite, source-sealed container runner."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from tessera._dev import suite_container as owner, surface_publication


def merge_module():
    tool = Path(__file__).resolve().parents[1] / "tools" / "merge_suite.py"
    spec = importlib.util.spec_from_file_location("_merge_container", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_container_runner_names_its_effective_population():
    command = ["/usr/bin/python3", "tools/suite_container.py",
               "--image", "example/suite@sha256:" + "a" * 64,
               "--deps-site", "/scoped/site-packages",
               "--deps-sha256", "b" * 64,
               "--surface-dir", "/receipts/run",
               "--cache-dir", "/cache/run",
               "--", "/usr/bin/python3", "-m", "pytest", "tests",
               "--strict-cuda", "--surface-json", "/receipts/run/surface.gpu.json"]
    output, refusal = merge_module()._effective_surface_json(command)
    assert refusal is None, refusal
    assert output == "/receipts/run/surface.gpu.json"


def command(surface="/receipts/run/surface.gpu.json", cache="/cache/run", site="/scoped/site-packages"):
    return ["/usr/bin/python3", owner.RUNNER, "--image", "example/suite@sha256:" + "a" * 64,
            "--deps-site", site, "--deps-sha256", "b" * 64,
            "--surface-dir", str(Path(surface).parent), "--cache-dir", cache,
            "--", "/usr/bin/python3", "-m", "pytest", "tests", "--strict-cuda",
            "--surface-json", surface]


@pytest.mark.parametrize("mutation", ["entrypoint", "environment", "duplicate", "escape", "overlap", "second-output", "foreign-command", "rootdir", "plugin", "short-shadow", "foreign-test", "workers"])
def test_container_command_shadows_are_refused(mutation):
    argv = command()
    cut = argv.index("--")
    if mutation == "entrypoint":
        argv[cut:cut] = ["--entrypoint", "/bin/sh"]
    elif mutation == "environment":
        argv[cut:cut] = ["--env", "PYTHONPATH=/foreign"]
    elif mutation == "duplicate":
        argv[cut:cut] = ["--image", "example/suite@sha256:" + "c" * 64]
    elif mutation == "escape":
        argv[-1] = "/receipts/run/../foreign.json"
    elif mutation == "overlap":
        argv[argv.index("--cache-dir") + 1] = "/receipts/run/cache"
    elif mutation == "second-output":
        argv += ["--surface-json", "/receipts/run/other.json"]
    elif mutation == "foreign-command":
        argv[cut + 1:] = ["/bin/sh", "-c", "echo pytest --surface-json /receipts/run/surface.gpu.json"]
    elif mutation == "rootdir":
        argv += ["--rootdir", "/foreign"]
    elif mutation == "plugin":
        argv += ["-p", "foreign.plugin"]
    elif mutation == "short-shadow":
        argv += ["-opythonpath=/foreign"]
    elif mutation == "foreign-test":
        argv += ["/foreign/test.py"]
    elif mutation == "workers":
        argv += ["-n", "auto"]
    output, refusal = merge_module()._effective_surface_json(argv)
    assert output is None and refusal


@pytest.mark.parametrize("path", ["other/suite_container.py", "./tools/suite_container.py", "/foreign/tools/suite_container.py", "tools/../tools/suite_container.py"])
def test_a_runner_basename_establishes_no_authority(path):
    argv = command()
    argv[1] = path
    assert merge_module()._effective_surface_json(argv)[0] is None


def mounts(tmp_path):
    source, site, surface, cache, data = [tmp_path / p for p in ("source", "site", "surface", "cache", "data")]
    for path in (source, site, surface, data):
        path.mkdir()
    (data / "pbsnapshot.py").write_text("# verifier fixture")
    argv = command(str(surface / "surface.gpu.json"), str(cache), str(site))
    cut = argv.index("--")
    argv[cut:cut] = ["--data-root", str(data)]
    spec = owner.command_spec(argv)
    env = {"TESSERA_SOURCE_VERIFIER": f"/usr/bin/python3 {data}/pbsnapshot.py verify",
           "PRISMABUILD_CONTAINER_OWNER": "sealed-owner", "CUDA_VISIBLE_DEVICES": "0"}
    return source, spec, env


def test_docker_builder_preserves_scope_route_readonly_source_and_owned_cache(tmp_path, monkeypatch):
    source, spec, env = mounts(tmp_path)
    monkeypatch.setattr(owner, "PB_VERIFIER", tmp_path / "data/pbsnapshot.py")
    built = owner.docker_command(spec, source, env)
    assert built[:2] == ["docker", "run"]  # PATH shim, never the host binary
    assert "--cpuset-cpus" not in built  # PB's ownership shim preserves affinity
    assert built[built.index("--entrypoint") + 1] == "/usr/bin/python3"
    assert built[built.index("--workdir") + 1] == str(source)
    assert f"type=bind,src={source},dst={source},readonly" in built
    assert "PYTHONNOUSERSITE=1" in built
    assert "PYTHONDONTWRITEBYTECODE=1" in built
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in built
    assert built[-2:] == ["--basetemp", spec["--cache-dir"] + "/pytest"]
    for name in owner.THREAD_LIMITS:
        assert f"{name}=1" in built


@pytest.mark.parametrize("problem", ["source-cache", "symlink-source", "artifact-environment", "missing-verifier"])
def test_mount_and_environment_controls_fail_closed(tmp_path, problem, monkeypatch):
    source, spec, env = mounts(tmp_path)
    monkeypatch.setattr(owner, "PB_VERIFIER", tmp_path / "data/pbsnapshot.py")
    if problem == "source-cache":
        spec["--cache-dir"] = str(source / "cache")
    elif problem == "symlink-source":
        link = tmp_path / "source-link"
        link.symlink_to(source, target_is_directory=True)
        source = link
    elif problem == "artifact-environment":
        env["TESSERA_RUNS_DIR"] = "/unsealed-runs"
    else:
        env.pop("TESSERA_SOURCE_VERIFIER")
    with pytest.raises(ValueError):
        owner.docker_command(spec, source, env)


def snapshot(tmp_path, altered=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    original = Path(__file__).resolve().parents[1]
    for name in owner.SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    if altered:
        (repo / owner.RUNNER).write_text("print('forged runner')\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).strip()
    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@local", "commit", "--quiet", "-m", "source")
    commit = git("rev-parse", "HEAD").decode()
    bundle = tmp_path / "source.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    raw = bundle.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    cas = tmp_path / "cas"
    blob = cas / "blobs" / digest[:2] / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(raw)
    entry = {"id": "pbrun.checkout-snapshot", "bytes": len(raw), "sha256": digest}
    return cas, {"inputs": [entry], "params": {"checkout_snapshot": {"commit": commit, "input": entry}}}


def test_authenticated_runner_snapshot_is_required(tmp_path):
    cas, payload = snapshot(tmp_path, altered=True)
    with pytest.raises(ValueError, match="runner source differs"):
        owner.source_bound(payload, cas)


def test_an_unchanged_authenticated_snapshot_is_read(tmp_path):
    cas, payload = snapshot(tmp_path)
    owner.source_bound(payload, cas)
    entry = payload["inputs"][0]
    blob = cas / "blobs" / entry["sha256"][:2] / entry["sha256"]
    blob.write_bytes(blob.read_bytes()[:-1])
    with pytest.raises(ValueError, match="length differs"):
        owner.source_bound(payload, cas)


def test_resume_keeps_existing_binding_and_verifies_runner_once(tmp_path, monkeypatch):
    module = merge_module()
    cas, payload = snapshot(tmp_path)
    surface = tmp_path / "surface" / "surface.gpu.json"
    surface.parent.mkdir()
    payload["params"]["command"] = command(str(surface), str(tmp_path / "cache"))
    payload["params"]["demand"] = {"cpu": 1, "gpu": 1}
    payload["params"]["container_images"] = ["example/suite@sha256:" + "a" * 64]
    raw_request = json.dumps(payload).encode()
    key = "a" * 64
    counts = {"passed": 1, "failed": 0, "error": 0, "skipped": 0}
    population = {"schema": "tessera.test_surface.v3", "role": "population",
                  "commit": payload["params"]["checkout_snapshot"]["commit"], "counts": counts,
                  "source_identity": {"excluded_metadata": [{"action_key": key, "request_sha256": hashlib.sha256(raw_request).hexdigest()}]}}
    surface.write_text(json.dumps(population))
    publication = module._read_publication(surface)
    outcome = {"status": "executed", "attempts": 1, "detail": {"stdout":
        surface_publication.publication_line("population", surface, publication.digest) + "\n1 passed in 0.01s\n"}}
    module.POOL_CAS_REQUESTS = cas / "requests"
    calls = []
    actual = Path.read_bytes
    bundle = payload["inputs"][0]["sha256"]
    def observed(path):
        if path.name == bundle:
            calls.append(path)
        return actual(path)
    monkeypatch.setattr(Path, "read_bytes", observed)
    seen = {}
    assert module._binding_refusal(key, payload, raw_request, outcome, publication, seen) is None
    assert module._binding_refusal(key, payload, raw_request, outcome, publication, seen) is None
    assert len(calls) == 1
    outcome["detail"]["stdout"] = "1 passed in 0.01s\n"
    assert "never said" in module._binding_refusal(key, payload, raw_request, outcome, publication, seen)


def test_container_submission_spends_its_parallel_reservation(tmp_path):
    module = merge_module()
    args = SimpleNamespace(cpus=4, pytest_arg=[], gpu_tag="gb10", mem_gb=24,
        checkout=tmp_path, timeout_s=300, wait_s=300, dry_run=True,
        gpu_image="example/suite@sha256:" + "a" * 64,
        gpu_deps_site="/deps/site", gpu_deps_sha256="b" * 64,
        gpu_cache_dir="/cache/run", gpu_data_root=[], artifact_root=[])
    record = module._submit("gpu", module.ARMS["gpu"], args, tmp_path / "receipt")
    import shlex
    argv = shlex.split(record["pbrun"])
    assert record["cpus_used"] == 4
    assert argv[argv.index("--cpus") + 1] == "4"
    assert "--container-image" in argv
    inner, refusal = module._pytest_argv(argv[argv.index("--") + 1:])
    assert refusal is None
    assert inner[inner.index("-n") + 1] == "4"
    assert inner[inner.index("--dist") + 1] == "worksteal"


def test_dependency_versions_and_bytes_are_both_sealed(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    for name, version in (("pytest", "9.0.3"), ("pytest-xdist", "3.8.0")):
        metadata = site / (name + ".dist-info")
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    code = site / "pytest.py"
    code.write_text("old bytes")
    body, before = owner.dependency_manifest(site)
    assert ["pytest", "9.0.3"] in [list(row) for row in body["versions"]]
    code.write_text("changed bytes")
    assert owner.dependency_manifest(site)[1] != before
    code.unlink()
    code.symlink_to(site / "pytest.dist-info/METADATA")
    with pytest.raises(ValueError, match="symlink"):
        owner.dependency_manifest(site)


@pytest.mark.parametrize("mismatch", ["image", "cpu", "gpu", "artifact"])
def test_runner_and_receipt_require_the_sealed_admission(mismatch):
    spec = owner.command_spec(command())
    payload = {"params": {"container_images": [spec["--image"]], "demand": {"cpu": 1, "gpu": 1}},
               "environment": {"variables": {}}}
    if mismatch == "image":
        payload["params"]["container_images"] = []
    elif mismatch == "cpu":
        payload["params"]["demand"]["cpu"] = 2
    elif mismatch == "gpu":
        payload["params"]["demand"]["gpu"] = 0
    else:
        spec["artifacts"]["TESSERA_RUNS_DIR"] = "/runs"
    with pytest.raises(ValueError):
        owner.admission(spec, payload)


def test_gpu_resources_can_be_sized_independently_from_cpu(tmp_path):
    module = merge_module()
    args = SimpleNamespace(cpus=16, mem_gb=64, gpu_cpus=2, gpu_mem_gb=24,
        pytest_arg=[], gpu_tag="gb10", checkout=tmp_path, timeout_s=300,
        wait_s=300, dry_run=True, gpu_image="example/suite@sha256:" + "a" * 64,
        gpu_deps_site="/deps/site", gpu_deps_sha256="b" * 64,
        gpu_cache_dir="/cache/run", gpu_data_root=[], artifact_root=[])
    record = module._submit("gpu", module.ARMS["gpu"], args, tmp_path / "receipt")
    assert record["cpus_used"] == 2 and record["mem_gb"] == 24
    assert "mem_gb=24" in record["pbrun"]
