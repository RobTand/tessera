"""Native container authority, starting with independently known commands."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import _suite_container as container
import merge_suite as merge


def composed(tmp_path, targets=None, cpus=2):
    arm = {**merge.ARMS["gpu"], "python": "/usr/bin/python3", "container": True,
           "fans_out": True, "targets": targets or ["tests/test_grammar.py"]}
    surface = tmp_path / "output/surface.gpu.json"
    inner = merge._command(arm, surface, [], cpus)
    inner += ["--junitxml", str(surface.parent / "junit.gpu.xml"), "--durations=10"]
    return ["/usr/bin/python3", container.RUNNER,
            "--image", "sha256:" + "a" * 64,
            "--deps-site", str(tmp_path / "dependencies"), "--deps-sha256", "b" * 64,
            "--surface-dir", str(surface.parent), "--cache-dir", str(tmp_path / "cache"),
            "--", *inner]


def test_actual_generated_command_passes_finite_parser(tmp_path):
    command = composed(tmp_path)
    spec = container.command_spec(command)
    assert spec["workers"] == 2
    assert spec["surface"] == str(tmp_path / "output/surface.gpu.json")
    assert spec["junit"] == str(tmp_path / "output/junit.gpu.xml")
    assert merge._effective_surface_json(merge._timed_command(command, 60)) == (spec["surface"], None)


@pytest.mark.parametrize("shadow", [
    ["-o", "pythonpath=/foreign"], ["-o", "addopts="], ["-c", "/foreign"],
    ["--junitxml", "/foreign.xml"], ["--surface-json", "/foreign.json"],
    ["-n", "1"], ["-p", "xdist.plugin"], ["-p", "foreign.plugin"],
    ["tests/test_other.py"], ["tests/test_grammar.py::test_x"],
])
def test_composed_controls_cannot_be_shadowed(tmp_path, shadow):
    command = composed(tmp_path)
    assert container.command_spec(command)["workers"] == 2
    with pytest.raises(ValueError):
        container.command_spec(command + shadow)


@pytest.mark.parametrize("targets", [
    ["tests/a.py", "tests/a.py"], ["tests/./a.py"],
    ["tests/a", "tests/a-else.py", "tests/a/b.py"],
    ["tests/a.py::test_a", "tests/a.py::test_a-other", "tests/a.py::test_a[param]"],
])
def test_parser_refuses_target_aliases_and_nonadjacent_overlap(tmp_path, targets):
    command = composed(tmp_path)
    assert container.command_spec(command)["workers"] == 2
    start = command.index("pytest") + 1
    command[start:start + 1] = targets
    with pytest.raises(ValueError, match="target"):
        container.command_spec(command)
