"""The run's own inventories, as the report driver assembles them (tessera#399).

The ownership rules read package rosters and filesystem roots. Nothing here
may be inferred from a path that looks right: the plugin package comes from
the worker's loaded-package record, the vLLM root from the attested core
manifest, and the observer roots and JIT prefixes from the container
environment the launcher itself recorded in its docker argv. These tests pin
that assembly on hand-written records; they establish nothing about a run.
"""
import pytest

from experiments.report_full_engine_resources import _container_environment, ownership_evidence

SITE = "/img/site-packages"


def _launch(argv=None, phases=None):
    capture = {"phase": "capture", "command": argv if argv is not None else [
        "docker", "run", "--rm",
        "-e", "PYTHONPATH=/observer:/src",
        "--env", "TRITON_CACHE_DIR=/cache/triton",
        "--env", "TORCH_EXTENSIONS_DIR=/cache/torch-ext",
        "-e", "TESSERA_EXT_DIR=/ext",
        "image@sha256:abc"]}
    return {"schema": "tessera.step4_launch_summary.v1", "image_id": "sha256:abc",
            "phases": [{"phase": "census", "command": ["docker", "run", "-e", "IGNORED=yes"]},
                       capture] if phases is None else phases}


def _plan():
    return {"collector_library": "/observer/libcollector.so",
            "collector_library_sha256": "c" * 64,
            "blas_workspace_observer": {"path": "/observer/libblas.so", "sha256": "b" * 64},
            "canonical_roster": [{"unit_id": "l:m", "module": "m", "family": "TESSERA_FP8",
                                  "members": ["m.weight"]}]}


def _runtime_observation():
    return {"loaded_package": {"package_path": SITE + "/tessera",
                               "installer_evidence_sha256": "e" * 64}}


def _per_job():
    return {"plugin_files": ["decode.py", "serving/nvfp4_route.py"],
            "plugin_source_tree": "/src", "plugin_source_sha256": "p" * 64,
            "core_manifest_sha256": "m" * 64}


def _core_manifest():
    return {"root": SITE + "/vllm",
            "files": {"v1/worker/gpu_model_runner.py": {"sha256": "1" * 64, "bytes": 10}}}


def _evidence(**overrides):
    arguments = {"plan": _plan(), "runtime_observation": _runtime_observation(),
                 "per_job": _per_job(), "core_manifest": _core_manifest(),
                 "launch": _launch(), "jit_preflight": {"ext_dir": "/ext",
                                                        "library_sha256": "j" * 64},
                 "dense_startup": None}
    arguments.update(overrides)
    return ownership_evidence(**arguments)


# --- the container environment -----------------------------------------------


def test_the_environment_is_read_from_the_capture_phases_own_docker_argv():
    environment = _container_environment(_launch())
    assert environment == {"PYTHONPATH": "/observer:/src",
                           "TRITON_CACHE_DIR": "/cache/triton",
                           "TORCH_EXTENSIONS_DIR": "/cache/torch-ext",
                           "TESSERA_EXT_DIR": "/ext"}
    # The census phase runs in another container; its environment is not this
    # capture's.
    assert "IGNORED" not in environment


def test_an_env_argument_that_names_no_value_is_not_an_environment_entry():
    argv = ["docker", "run", "-e", "PYTHONPATH", "--env", "A=1"]
    assert _container_environment(_launch(argv=argv)) == {"A": "1"}


def test_a_launch_summary_with_no_capture_phase_yields_no_environment():
    assert _container_environment(_launch(phases=[])) == {}
    assert _container_environment({}) == {}


# --- the assembled evidence --------------------------------------------------


def test_the_plugin_and_core_inventories_come_from_the_records_that_attest_them():
    evidence = _evidence()
    assert evidence["plugin_package_path"] == SITE + "/tessera"
    assert evidence["plugin_files"] == {"decode.py", "serving/nvfp4_route.py"}
    assert evidence["vllm_root"] == SITE + "/vllm"
    assert evidence["vllm_files"] == {"v1/worker/gpu_model_runner.py"}
    assert evidence["roster"] == _plan()["canonical_roster"]


def test_the_observer_roots_are_the_recorded_pythonpath_plus_the_plugin_source_tree():
    assert _evidence()["observer_roots"] == ["/observer", "/src"]


def test_a_plugin_source_tree_already_on_the_pythonpath_is_not_repeated():
    per_job = dict(_per_job(), plugin_source_tree="/observer")
    assert _evidence(per_job=per_job)["observer_roots"] == ["/observer", "/src"]


def test_the_jit_prefixes_come_from_the_preflight_and_the_recorded_cache_variables():
    evidence = _evidence()
    assert evidence["plugin_jit_prefix"] == "/ext/tessera_nvfp4/"
    assert evidence["jit_cache_prefixes"] == ["/cache/triton", "/cache/torch-ext"]


def test_the_ext_dir_falls_back_to_the_recorded_container_environment():
    evidence = _evidence(jit_preflight={"library_sha256": "j" * 64})
    assert evidence["plugin_jit_prefix"] == "/ext/tessera_nvfp4/"


def test_a_run_that_declared_no_extension_directory_carries_no_plugin_jit_prefix():
    argv = ["docker", "run", "-e", "PYTHONPATH=/observer"]
    evidence = _evidence(jit_preflight=None, launch=_launch(argv=argv))
    assert evidence["plugin_jit_prefix"] is None
    assert evidence["jit_cache_prefixes"] == []


def test_the_observer_libraries_are_the_ones_the_plan_loaded():
    assert _evidence()["observer_libraries"] == ["/observer/libcollector.so",
                                                 "/observer/libblas.so"]
    plan = dict(_plan())
    plan.pop("blas_workspace_observer")
    assert _evidence(plan=plan)["observer_libraries"] == ["/observer/libcollector.so"]


def test_every_inventory_digest_the_views_record_travels_with_is_carried():
    digests = _evidence()["inventory_digests"]
    assert digests == {"plugin_source_sha256": "p" * 64,
                       "plugin_installer_evidence_sha256": "e" * 64,
                       "core_manifest_sha256": "m" * 64,
                       "collector_library_sha256": "c" * 64,
                       "blas_workspace_observer_sha256": "b" * 64,
                       "plugin_jit_library_sha256": "j" * 64}


def test_a_dense_startup_observation_travels_to_the_rules_unchanged():
    dense = {"schema": "tessera.full_engine_dense_startup_observation.v1",
             "memory_allocated_bytes": 4096, "units": {}}
    assert _evidence(dense_startup=dense)["dense_startup"] is dense
    assert _evidence()["dense_startup"] is None
