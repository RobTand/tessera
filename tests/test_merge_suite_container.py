"""The paired receipt can bind the finite, source-sealed container runner."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from tessera._dev import surface_publication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import _suite_container as owner


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
    assert f"{owner.artifact_specs()['scratch'].env}={spec['--cache-dir']}/tmp" in built
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


def test_authenticated_runner_snapshot_is_required(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cas, payload = snapshot(tmp_path, altered=True)
    with pytest.raises(ValueError, match="runner source differs"):
        owner.source_bound(payload, cas)


def test_an_unchanged_authenticated_snapshot_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cas, payload = snapshot(tmp_path)
    owner.source_bound(payload, cas)
    entry = payload["inputs"][0]
    blob = cas / "blobs" / entry["sha256"][:2] / entry["sha256"]
    blob.write_bytes(blob.read_bytes()[:-1])
    with pytest.raises(ValueError, match="length differs"):
        owner.source_bound(payload, cas)


def test_resume_keeps_existing_binding_and_verifies_runner_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
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


class SubmissionReached(Exception):
    pass


def _mount(mount_id, mountpoint, filesystem, *, parent_id=1):
    escaped = str(mountpoint).replace("\\", r"\134").replace(" ", r"\040")
    return f"{mount_id} {parent_id} 0:{mount_id} / {escaped} rw - {filesystem} source rw\n"


@pytest.fixture
def admission(tmp_path, monkeypatch):
    module = merge_module()
    root = Path(__file__).resolve().parents[1]
    tool = root / "tools" / "merge_suite.py"
    monkeypatch.setattr(module, "DEFAULT_RECEIPT_ROOT", tmp_path / "receipts")
    original_read = Path.read_text
    mount_table = []

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return "".join(mount_table)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def submit(*args, **kwargs):
        raise SubmissionReached("suite submission was reached")

    monkeypatch.setattr(module, "_submit", submit)

    def run(cache, mounts):
        mount_table[:] = mounts
        monkeypatch.setattr(sys, "argv", [str(tool), "--arm", "gpu",
            "--gpu-tag", "fake-worker", "--checkout", str(root),
            "--gpu-image", "sha256:" + "a" * 64,
            "--gpu-deps-site", str(tmp_path / "dependencies"),
            "--gpu-deps-sha256", "b" * 64, "--gpu-cache-dir", str(cache)])
        return module.main()

    return run


@pytest.mark.parametrize("filesystem", ["nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"])
def test_network_cache_refuses_before_submission(tmp_path, admission, capsys, filesystem):
    with pytest.raises(SystemExit) as error:
        admission(tmp_path, [_mount(1, "/", filesystem)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "--gpu-cache-dir" in message
    assert str(tmp_path) in message
    assert "mount point /" in message
    assert f"filesystem type {filesystem}" in message
    assert "local disk" in message
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("filesystem", ["ext4", "xfs", "btrfs", "tmpfs"])
def test_local_cache_is_admitted(tmp_path, admission, filesystem):
    with pytest.raises(SubmissionReached):
        admission(tmp_path, [_mount(1, "/", filesystem)])


def test_nested_network_mount_wins_over_local_root(tmp_path, admission, capsys):
    mountpoint = tmp_path / "network cache"
    mountpoint.mkdir()
    cache = mountpoint / "cache"
    cache.mkdir()
    # Deliberately put the parent last: matching must follow depth, not order.
    with pytest.raises(SystemExit):
        admission(cache, [_mount(2, mountpoint, "nfs"), _mount(1, "/", "ext4")])
    assert f"mount point {mountpoint}" in capsys.readouterr().err


@pytest.mark.parametrize("filesystem,refused", [("nfs4", True), ("ext4", False)])
def test_missing_cache_uses_nearest_existing_parent(tmp_path, admission, capsys, filesystem, refused):
    cache = tmp_path / "not yet created" / "cache"
    assert not cache.exists()
    # The deeper fake mount does not exist; it must not determine admission.
    mounts = [_mount(1, "/", "ext4"), _mount(2, tmp_path, filesystem),
              _mount(3, cache.parent, "ext4" if refused else "nfs")]
    if refused:
        with pytest.raises(SystemExit):
            admission(cache, mounts)
        message = capsys.readouterr().err
        assert str(cache) in message
        assert f"mount point {tmp_path}" in message
        assert f"filesystem type {filesystem}" in message
    else:
        with pytest.raises(SubmissionReached):
            admission(cache, mounts)
    assert not cache.exists()


def test_network_mount_prefix_does_not_match_sibling(tmp_path, admission):
    mountpoint = tmp_path / "network"
    mountpoint.mkdir()
    cache = tmp_path / "network-local"
    cache.mkdir()
    with pytest.raises(SubmissionReached):
        admission(cache, [_mount(1, "/", "ext4"), _mount(2, mountpoint, "nfs")])


def test_cache_symlink_is_judged_by_its_target(tmp_path, admission, capsys):
    mountpoint = tmp_path / "network"
    mountpoint.mkdir()
    alias = tmp_path / "local-alias"
    alias.symlink_to(mountpoint, target_is_directory=True)
    with pytest.raises(SystemExit):
        admission(alias / "cache", [_mount(1, "/", "ext4"), _mount(2, mountpoint, "nfs")])
    assert f"mount point {mountpoint}" in capsys.readouterr().err


@pytest.mark.parametrize("mounts", [[], ["not a mount table\n"]])
def test_unknown_mount_provenance_refuses(tmp_path, admission, capsys, mounts):
    with pytest.raises(SystemExit):
        admission(tmp_path, mounts)
    message = capsys.readouterr().err
    assert "--gpu-cache-dir" in message
    assert str(tmp_path) in message
    assert "mount" in message


@pytest.fixture
def cache_mount_table(monkeypatch):
    original_read = Path.read_text
    table = []

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return "".join(table)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return table


@pytest.mark.parametrize("filesystem", ["nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"])
def test_container_network_cache_refuses_before_command(tmp_path, monkeypatch, cache_mount_table, filesystem):
    source, spec, env = mounts(tmp_path)
    monkeypatch.setattr(owner, "PB_VERIFIER", tmp_path / "data/pbsnapshot.py")
    cache_mount_table[:] = [_mount(1, "/", "ext4"), _mount(2, tmp_path, filesystem)]
    with pytest.raises(ValueError) as error:
        owner.docker_command(spec, source, env)
    message = str(error.value)
    assert "--cache-dir" in message
    assert spec["--cache-dir"] in message
    assert f"mount point {tmp_path}" in message
    assert f"filesystem type {filesystem}" in message
    assert "local disk" in message
    assert not Path(spec["--cache-dir"]).exists()


@pytest.mark.parametrize("filesystem", ["ext4", "xfs", "btrfs", "tmpfs"])
def test_container_local_cache_is_admitted(tmp_path, monkeypatch, cache_mount_table, filesystem):
    source, spec, env = mounts(tmp_path)
    monkeypatch.setattr(owner, "PB_VERIFIER", tmp_path / "data/pbsnapshot.py")
    cache_mount_table[:] = [_mount(1, "/", filesystem)]
    assert owner.docker_command(spec, source, env)[:2] == ["docker", "run"]
    assert not Path(spec["--cache-dir"]).exists()


@pytest.mark.parametrize("filesystem,refused", [("nfs", True), ("ext4", False)])
def test_container_missing_cache_uses_existing_mount(tmp_path, monkeypatch, cache_mount_table, filesystem, refused):
    source, spec, env = mounts(tmp_path)
    monkeypatch.setattr(owner, "PB_VERIFIER", tmp_path / "data/pbsnapshot.py")
    cache = tmp_path / "missing" / "cache"
    spec["--cache-dir"] = str(cache)
    cache_mount_table[:] = [_mount(1, "/", "ext4"), _mount(2, tmp_path, filesystem),
                           _mount(3, cache.parent, "ext4" if refused else "nfs")]
    if refused:
        with pytest.raises(ValueError, match=f"mount point {tmp_path}"):
            owner.docker_command(spec, source, env)
    else:
        assert owner.docker_command(spec, source, env)[:2] == ["docker", "run"]
    assert not cache.parent.exists()


def test_container_cache_refuses_before_creating_or_launching(tmp_path, monkeypatch, cache_mount_table, capsys):
    source, spec, env = mounts(tmp_path)
    monkeypatch.setattr(owner, "PB_VERIFIER", tmp_path / "data/pbsnapshot.py")
    cache_mount_table[:] = [_mount(1, "/", "ext4"), _mount(2, tmp_path, "nfs4")]
    monkeypatch.chdir(source)
    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", "fake-action")
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", "fake-scope")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(owner, "admitted_request", lambda *args: {})
    monkeypatch.setattr(owner, "admission", lambda *args: None)
    monkeypatch.setattr(owner, "dependency_manifest", lambda *args: ({}, "b" * 64))
    launches = []
    monkeypatch.setattr(owner.subprocess, "call", lambda argv: launches.append(argv) or 0)
    argv = command(spec["surface"], spec["--cache-dir"], spec["--deps-site"])
    cut = argv.index("--")
    argv[cut:cut] = ["--data-root", str(tmp_path / "data")]
    assert owner.main(argv[2:]) == 2
    assert "filesystem type nfs4" in capsys.readouterr().err
    assert launches == []
    assert not Path(spec["--cache-dir"]).exists()


# Exact /proc/self/mountinfo lines read on 2026-10-05. No test accesses /mnt/shared.
_FLEET_SHARED_MOUNTS = {
    "celestia": (
        "8498 44 0:106 / /mnt/shared rw,relatime shared:783 - autofs systemd-1 rw,fd=101,pgrp=1,timeout=0,minproto=5,maxproto=5,direct,pipe_ino=626733\n",
        "8533 8498 0:109 / /mnt/shared rw,noatime shared:802 - nfs4 192.168.1.107:/storage_pool/shared rw,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,hard,fatal_neterrors=none,proto=tcp,nconnect=8,timeo=600,retrans=2,sec=sys,clientaddr=192.168.1.68,local_lock=none,addr=192.168.1.107\n",
    ),
    "sparky": (
        "52 36 0:40 / /mnt/shared rw,relatime shared:30 - autofs systemd-1 rw,fd=62,pgrp=1,timeout=0,minproto=5,maxproto=5,direct,pipe_ino=20555\n",
        "211 52 0:66 / /mnt/shared rw,noatime shared:729 - nfs4 10.100.98.3:/storage_pool/shared rw,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,hard,fatal_neterrors=none,proto=rdma,nconnect=16,port=20049,timeo=600,retrans=2,sec=sys,clientaddr=0.0.0.0,local_lock=none,addr=10.100.98.3\n",
    ),
    "sparklina": (
        "52 37 0:41 / /mnt/shared rw,relatime shared:30 - autofs systemd-1 rw,fd=64,pgrp=1,timeout=0,minproto=5,maxproto=5,direct,pipe_ino=21625\n",
        "772 52 0:76 / /mnt/shared rw,noatime shared:751 - nfs4 10.100.99.3:/storage_pool/shared rw,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,hard,fatal_neterrors=none,proto=rdma,nconnect=16,port=20049,timeo=600,retrans=2,sec=sys,clientaddr=10.100.99.2,local_lock=none,addr=10.100.99.3\n",
    ),
}
_CELESTIA_LOCAL_MOUNT = "44 1 259:14 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"


@pytest.mark.parametrize("host", _FLEET_SHARED_MOUNTS)
@pytest.mark.parametrize("mounted,filesystem", [(True, "nfs4"), (False, "autofs")])
def test_real_fleet_shared_mounts_refuse_by_type_and_admit_local_cache(
        tmp_path, monkeypatch, cache_mount_table, host, mounted, filesystem):
    cache = Path("/mnt/shared")
    directory_stat = tmp_path.stat()
    original_resolve, original_stat = Path.resolve, Path.stat

    def resolve(path, *args, **kwargs):
        return cache if path == cache else original_resolve(path, *args, **kwargs)

    def stat(path, *args, **kwargs):
        return directory_stat if path == cache else original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "stat", stat)
    automount, network = _FLEET_SHARED_MOUNTS[host]
    # Reverse the entries: visibility follows parent identity, not table order.
    cache_mount_table[:] = ([network, automount] if mounted else [automount]) + [_CELESTIA_LOCAL_MOUNT]
    expected = (
        "suite container: --cache-dir (merge-suite --gpu-cache-dir): "
        f"cache path /mnt/shared is on mount point /mnt/shared with filesystem type {filesystem}; "
        "use local disk for native build locks and mapped inode identity"
    )
    with pytest.raises(ValueError) as error:
        owner.require_local_cache(cache)
    assert str(error.value) == expected
    # The same real mount table must still admit an unrelated local directory.
    owner.require_local_cache(tmp_path)



@pytest.mark.parametrize("edges", [
    pytest.param([(101, 44), (102, 44)], id="siblings"),
    pytest.param([(101, 44), (101, 44), (102, 101)], id="duplicate-id"),
    pytest.param([(101, 102), (102, 101)], id="pure-cycle"),
    pytest.param([(101, 44), (102, 101), (201, 202), (202, 201)],
                 id="chain-and-disconnected-cycle"),
    pytest.param([(101, 44), (102, 101), (201, 44), (202, 201)],
                 id="two-disconnected-stacks"),
])
def test_all_local_mount_ambiguity_refuses(tmp_path, cache_mount_table, edges):
    """A pure cycle has no top; a separate cycle escapes the top's parent walk."""
    cache_mount_table[:] = [_mount(44, "/", "ext4")] + [
        _mount(mount_id, tmp_path, "ext4", parent_id=parent_id)
        for mount_id, parent_id in edges
    ]
    with pytest.raises(ValueError, match="mount provenance is ambiguous"):
        owner.require_local_cache(tmp_path)


def test_hidden_submount_is_judged_by_the_mount_the_descriptor_serves(
        tmp_path, monkeypatch, cache_mount_table):
    """A later, shallower mount can hide an earlier submount entirely; the
    opened descriptor's mount id, not the deepest mount point, decides (#949)."""
    from tessera._dev import native_identity

    data = tmp_path / "data"
    hidden = data / "cache"
    data.mkdir()
    hidden.mkdir()
    cache = hidden / "run"
    cache_mount_table[:] = [_mount(1, "/", "ext4"),
                            _mount(30, hidden, "nfs", parent_id=20),
                            _mount(40, data, "ext4", parent_id=1)]
    monkeypatch.setattr(native_identity, "_fd_mount_id", lambda fd: "40")
    assert native_identity.native_cache_mount(cache)[1:] == (data, "ext4")
    owner.require_local_cache(cache)


def test_visible_refused_submount_is_still_refused(tmp_path, monkeypatch, cache_mount_table):
    """Without a later covering mount, the refused submount itself decides."""
    from tessera._dev import native_identity

    data = tmp_path / "data"
    sub = data / "cache"
    data.mkdir()
    sub.mkdir()
    cache = sub / "run"
    cache_mount_table[:] = [_mount(1, "/", "ext4"),
                            _mount(30, sub, "nfs", parent_id=1)]
    monkeypatch.setattr(native_identity, "_fd_mount_id", lambda fd: "30")
    assert native_identity.native_cache_mount(cache)[1:] == (sub, "nfs")
    with pytest.raises(ValueError, match="filesystem type nfs"):
        owner.require_local_cache(cache)

