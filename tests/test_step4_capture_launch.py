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
from pathlib import Path

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
