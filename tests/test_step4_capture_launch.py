"""The step-4 capture launcher carries its CPU reservation into the container.

WHY THIS TEST EXISTS.  ``step4_capture_launch`` runs the full-engine capture as
a ``docker run`` child of the launcher process.  When PrismaBuild admits that
launcher it grants a CPU mask, and the fleet's execution policy requires the
mask to be preserved "including inside containers": a container started without
one is placed on every host CPU regardless of what the pool reserved, so the
capture competes with whatever else the pool admitted beside it.  That matters
here beyond politeness -- this launcher's whole output is a resource and timing
observation, and an observation taken while the box is oversubscribed is not
the observation the configuration describes.

``experiments/owned_container.sh`` measured why the mask is spelled
``--cpuset-cpus`` and not ``--cpus`` (a CFS quota changes no CPU count a
library can read, so a quota-limited container still sizes its pools for the
whole box), and ``experiments/run_glm_native_construction.py`` is the existing
docker-under-PrismaBuild launcher that already spells it that way.  This test
holds the step-4 launcher to the same rule.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _launcher():
    spec = importlib.util.spec_from_file_location(
        "step4_capture_launch_under_test", ROOT / "experiments" / "step4_capture_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(**overrides):
    module = _launcher()
    arguments = dict(name="step4-capture-test", image_id="sha256:" + "0" * 64,
                     mounts=[(Path("/tree"), "/tessera", "ro")],
                     environment={"PYTHONPATH": "/tessera"},
                     entry=["/tessera/experiments/full_engine_plugin_install.py"],
                     out_host=Path("/out"), jit_host=Path("/jit"), ext_host=Path("/jit/ext"),
                     ext_readonly=False)
    arguments.update(overrides)
    return module.docker_command(**arguments)


def test_cpu_affinity_becomes_a_cpuset():
    """The granted mask is spelled into the container, sorted and as a list."""
    command = _command(affinity=[5, 7, 6])
    assert "--cpuset-cpus" in command
    assert command[command.index("--cpuset-cpus") + 1] == "5,6,7"
    # A CFS quota is the wrong instrument (owned_container.sh's measurement).
    assert "--cpus" not in command


def test_no_affinity_pins_nothing():
    """An unconstrained launcher still launches; the flag is absent, not empty."""
    command = _command(affinity=None)
    assert "--cpuset-cpus" not in command


def test_tp2_peer_uses_host_network_and_its_own_installer_mount():
    command = _command(tp2=True, extra_mounts=[(Path("/shared/rank-1"), "/peer-out", "rw")])
    assert command[command.index("--network") + 1] == "host"
    assert "/shared/rank-1:/peer-out:rw" in command
    assert "--cpuset-cpus" not in command


def test_scoped_worker_cache_roots_are_owned_before_launch(tmp_path, monkeypatch):
    import os
    launcher = _launcher()
    monkeypatch.setattr(launcher, "WORKER_UID", os.getuid())
    launcher.prepare_worker_jit_cache(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} >= {
        "home", "xdg", "tmp", "triton", "torch-extensions", "inductor", "cuda-cache"}
    monkeypatch.setattr(launcher, "WORKER_UID", os.getuid() + 1)
    with pytest.raises(RuntimeError, match="not writable by pinned worker UID"):
        launcher.prepare_worker_jit_cache(tmp_path)


def test_timeout_stops_the_exact_owned_container_and_records_absence(tmp_path, monkeypatch):
    launcher = _launcher()
    cidfile = tmp_path / "container.cid"
    cidfile.write_text("a" * 64)
    stopped = []
    monkeypatch.setattr(launcher, "stop_owned_container", lambda path, name: (
        stopped.append((path, name)) or {"container_id": "a" * 64, "absent": True}))

    class FinishedTimeout:
        def wait(self, timeout=None):
            return 124

        def poll(self):
            return 124

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: FinishedTimeout())
    phase = launcher.run_phase("tp2-head", ["docker", "run", "--name", "owned-head",
                                             "--cidfile", str(cidfile)], tmp_path / "phase.log", 1)
    assert phase["returncode"] == 124
    assert phase["owned_container_cleanup"]["absent"] is True
    assert stopped == [(cidfile, "owned-head")]


def test_missing_cid_recovers_only_a_matching_owned_container_after_stop_timeout(tmp_path, monkeypatch):
    launcher = _launcher()
    container_id = "a" * 64
    state = {"removed": False, "operations": []}

    def docker(command, **_kwargs):
        operation = command[1]
        state["operations"].append(operation)
        if operation == "inspect":
            if state["removed"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="error: no such object: " + container_id)
            return SimpleNamespace(returncode=0, stdout=json.dumps([{
                "Id": container_id,
                "Config": {"Labels": {"org.prismaquant.pact-observer": "owned-head"}}}]), stderr="")
        if operation == "stop":
            raise launcher.subprocess.TimeoutExpired(command, 30)
        state["removed"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", docker)
    result = launcher.stop_owned_container(tmp_path / "unwritten.cid", "owned-head")
    assert result["absent"] is True
    assert result["identity_source"] == "owned_name_and_label"
    assert [entry["operation"] for entry in result["actions"]] == ["stop", "rm"]
    assert state["operations"] == ["inspect", "stop", "inspect", "rm", "inspect"]


def test_missing_cid_never_stops_another_label(tmp_path, monkeypatch):
    launcher = _launcher()
    commands = []

    def docker(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps([{
            "Id": "a" * 64, "Config": {"Labels": {"org.prismaquant.pact-observer": "other"}}}]),
            stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", docker)
    with pytest.raises(RuntimeError, match="owner label"):
        launcher.stop_owned_container(tmp_path / "unwritten.cid", "owned-head")
    assert len(commands) == 1 and commands[0][1] == "inspect"


def test_head_failure_before_plan_notifies_only_its_session_peer(tmp_path):
    launcher = _launcher()
    from experiments.step4_tp2_peer import wait_for_plan
    session = "fixture-session"
    launcher.write_head_finished(tmp_path, session, returncode=3)
    record = json.loads((tmp_path / "head-finished.json").read_text())
    assert record["session_id"] == session and record["plan_sha256"] is None
    with pytest.raises(RuntimeError, match="before a sealed shared plan"):
        wait_for_plan(tmp_path / "capture" / "observer-plan.json", 1,
                      finished=tmp_path / "head-finished.json", session_id=session)
    with pytest.raises(TimeoutError):
        wait_for_plan(tmp_path / "capture" / "observer-plan.json", 0,
                      finished=tmp_path / "head-finished.json", session_id="foreign-session")


def test_empty_affinity_is_refused_not_ignored():
    """An empty mask is a read that failed; pinning a container to nothing would hang."""
    with pytest.raises(ValueError):
        _command(affinity=[])


def test_unset_allocator_policy_is_absent_from_the_container():
    """``"unset"`` names the variable's ABSENCE, and torch aborts on the literal.

    tessera#558 / PR #565 made the configuration bind
    ``PYTORCH_CUDA_ALLOC_CONF`` because the reserved-extent witness only
    transfers under an equal allocator segment policy, and it spelled "no
    policy" as the explicit string ``"unset"``.
    ``capture_full_engine_resources.prepare`` pops the key when the binding is
    that sentinel, so the WORKER never sees it. The launcher one level out
    copied the configuration's environment block into ``docker run --env``
    verbatim, so the container's own interpreter did see it -- and ``"unset"``
    is not a token c10 can parse. Measured on sparklina 2026-09-21, the first
    capture attempt of this run: ``terminate called after throwing an instance
    of 'c10::Error' ... Index out of bounds in ConfigTokenizer`` from
    ``libc10_cuda.so``'s load-time parse, phase returncode 133, before any
    engine existed.
    """
    module = _launcher()
    config = {"environment": {"PYTORCH_CUDA_ALLOC_CONF": "unset", "TESSERA_SERVE_MODE": "resident"}}
    environment = module.bound_container_environment(config)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in environment
    assert environment == {"TESSERA_SERVE_MODE": "resident"}


def test_tp2_source_commit_and_packaged_bytes_are_both_bound(tmp_path):
    launcher = _launcher()
    (tmp_path / "src" / "tessera").mkdir(parents=True)
    (tmp_path / "src" / "tessera" / "__init__.py").write_text("# fixture\n")
    (tmp_path / "pyproject.toml").write_text("[build-system]\n")
    source_sha, _count = launcher.source_tree_identity(tmp_path)
    config = {"runtime_identity": {"plugin_source_commit": "a" * 40,
                                   "plugin_source_sha256": source_sha}}
    assert launcher.require_tp2_source(config, tmp_path, "a" * 40) == source_sha
    with pytest.raises(ValueError, match="different source commits"):
        launcher.require_tp2_source(config, tmp_path, "b" * 40)
    (tmp_path / "src" / "tessera" / "__init__.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source bytes differ"):
        launcher.require_tp2_source(config, tmp_path, "a" * 40)


def test_a_real_allocator_policy_reaches_the_container():
    """A policy that is a policy is passed through unchanged."""
    module = _launcher()
    config = {"environment": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                              "TESSERA_SERVE_MODE": "resident"}}
    environment = module.bound_container_environment(config)
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_an_unbound_allocator_policy_is_refused():
    """The launcher refuses where the capture CLI refuses, not later and not silently."""
    module = _launcher()
    with pytest.raises(ValueError):
        module.bound_container_environment({"environment": {"TESSERA_SERVE_MODE": "resident"}})


def _driver():
    spec = importlib.util.spec_from_file_location(
        "step4_capture_driver_under_test", ROOT / "experiments" / "step4_capture_driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_modules_reads_every_family_the_manifest_assigns(tmp_path):
    """The driver qualifies each family, so the launcher hands it every family's count and names."""
    manifest = {"modules": {
        "model.layers.1.mlp.down_proj": {"family": "TESSERA_FP8"},
        "model.layers.0.mlp.down_proj": {"family": "TESSERA_FP8"},
        "model.layers.0.self_attn.qkv_proj": {"family": "TESSERA_BF16"},
        "model.layers.0.mlp.gate_up_proj": {"family": "TESSERA_NVFP4"}}}
    (tmp_path / "tessera_serving_manifest.json").write_text(json.dumps(manifest))
    assert _launcher().family_modules(tmp_path) == {
        "TESSERA_FP8": {"count": 2, "names": ["model.layers.0.mlp.down_proj",
                                              "model.layers.1.mlp.down_proj"]},
        "TESSERA_BF16": {"count": 1, "names": ["model.layers.0.self_attn.qkv_proj"]},
        "TESSERA_NVFP4": {"count": 1, "names": ["model.layers.0.mlp.gate_up_proj"]}}


def test_family_modules_refuses_a_module_without_a_family(tmp_path):
    (tmp_path / "tessera_serving_manifest.json").write_text(
        json.dumps({"modules": {"model.layers.0.mlp.down_proj": {}}}))
    with pytest.raises(ValueError, match="names no family"):
        _launcher().family_modules(tmp_path)


def test_the_driver_parses_the_launchers_module_map_and_refuses_unknown_families():
    driver = _driver()
    parsed = driver._expected_modules(json.dumps(
        {"TESSERA_FP8": {"count": 110, "names": []}, "TESSERA_NVFP4": {"count": 1, "names": ["x"]}}))
    assert parsed["TESSERA_FP8"]["count"] == 110
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="unknown families"):
        driver._expected_modules(json.dumps({"TESSERA_INT4": 1}))
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty JSON object"):
        driver._expected_modules("110")  # the old --expected-fp4-modules integer is not a map
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty JSON object"):
        driver._expected_modules("{}")
    with pytest.raises(argparse.ArgumentTypeError, match="not JSON"):
        driver._expected_modules("TESSERA_FP8=110")


def test_the_driver_smokes_compile_as_python():
    """Both child-interpreter programs are strings; a syntax error would surface only in-container."""
    driver = _driver()
    compile(driver.NATIVE_SMOKE, "NATIVE_SMOKE", "exec")
    compile(driver.OBSERVER_SMOKE, "OBSERVER_SMOKE", "exec")
    assert "tessera_nvfp4" not in driver.NATIVE_SMOKE
    assert "require_tessera_ext" not in driver.NATIVE_SMOKE


def test_tp2_peer_requires_the_sealed_rank_one_installer_and_stock_mp_world(tmp_path):
    from experiments.step4_tp2_peer import digest, peer_engine_args
    evidence = tmp_path / "per-job-runtime.json"
    evidence.write_text('{"rank": 1}')
    plan = {"rank_runtime_evidence": {"1": {"sha256": digest(evidence)}},
            "selected_configuration": {
                "engine_args": {"nnodes": 2, "node_rank": 0, "tensor_parallel_size": 2,
                                "distributed_executor_backend": "mp", "master_addr": "10.0.0.1"},
                "environment": {"VLLM_HOST_IP": "10.0.0.1"}},
            "observer_engine_args": {"worker_cls": "experiments.full_engine_worker.ResourceCaptureWorker"},
            "model": "/shared/stub"}
    actual = peer_engine_args(plan, evidence, host_ip="10.0.0.2")
    assert actual["node_rank"] == 1 and actual["nnodes"] == 2
    assert actual["worker_cls"] == plan["observer_engine_args"]["worker_cls"]
    with pytest.raises(ValueError, match="equals the head"):
        peer_engine_args(plan, evidence, host_ip="10.0.0.1")
    evidence.write_text('{"rank": 0}')
    with pytest.raises(ValueError, match="installer evidence differs"):
        peer_engine_args(plan, evidence, host_ip="10.0.0.2")
