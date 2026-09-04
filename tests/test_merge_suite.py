"""The merge suite composes submissions; these pin what it composes.

``tools/merge_suite.py`` never runs a suite itself -- it builds ``pbrun``
command lines and reads back what they publish.  So the things worth pinning
are the properties of those command lines and of the verdict it derives, not a
placement, which belongs to the pool.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "merge_suite.py"


def _module():
    spec = importlib.util.spec_from_file_location("_merge_suite", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gpu_arm_carries_the_gate_and_the_cpu_arm_does_not():
    """``--strict-cuda`` on the arm that exists to cover the CUDA surface.

    Without it the GPU arm placed on a device-less box returns a green tick for
    a population it never touched -- the exact shape of tessera#112.  The x86
    arm must NOT carry it: that box has no torch by design, and refusing there
    would make an honest population unreportable.
    """

    merge_suite = _module()
    surface = Path("/dev/null")
    gpu = merge_suite._command(merge_suite.ARMS["gpu"], surface, [])
    x86 = merge_suite._command(merge_suite.ARMS["x86"], surface, [])
    assert "--strict-cuda" in gpu
    assert "--strict-cuda" not in x86
    for command in (gpu, x86):
        # Both arms must publish a population, or the receipt has nothing to
        # put side by side.
        assert "--surface-json" in command
        # The interpreter is named, never inherited: a pool action runs in a
        # sealed environment on a box that is not this one.
        assert command[0].startswith("/") and command[0] != sys.executable


def test_a_missing_surface_is_reported_as_absent_not_as_a_pass():
    """An arm that was never placed is not a green arm."""

    merge_suite = _module()
    assert merge_suite._verdict([{"surface": None, "returncode": 0}]) == (
        "incomplete: an arm published no population")
    assert merge_suite._verdict(
        [{"surface": {}, "returncode": 0}, {"surface": {}, "returncode": 1}]
    ) == "red"
    assert merge_suite._verdict(
        [{"surface": {}, "returncode": 0}]) == "green on both populations"


def test_the_x86_arm_refuses_a_checkout_only_one_box_can_see():
    """pbrun pins a local checkout to the submitting box; say so, do not route around it."""

    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "x86", "--dry-run",
         "--checkout", str(ROOT)],
        capture_output=True, text=True, timeout=120,
    )
    if str(ROOT).startswith("/mnt/shared"):
        pytest.skip("this checkout IS shared, so the refusal cannot fire here")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "/mnt/shared" in result.stderr


def test_the_receipt_states_which_tree_it_is_about(tmp_path):
    """A branch receipt is not a merge receipt, and must not read as one."""

    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--dry-run",
         "--checkout", str(ROOT), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, "a dry run has covered no population"
    receipt = json.loads(out.read_text())
    population = receipt["population"]
    assert population["commit"]
    assert "is_master_head" in population
    # A clone made for a pool run carries only ``origin/master``; the ref that
    # actually answered is recorded so an unresolved comparison reads as
    # "not established" rather than as "not master".
    assert population["master_ref_used"] != "none resolved"
    assert population["is_master_head"] is not None
    assert receipt["verdict"] == "not run"
    # Both arms' numbers live under one key, so quoting one without its device
    # means quoting it out of this object rather than out of a scrollback.
    assert "reading_note" in receipt
