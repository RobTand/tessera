"""A failed prelaunch gate cannot confer ownership of somebody's container."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("driver", ["ts5_lfm_teacher_bound.py", "ts5_lfm_served_bound.py"])
def test_prelaunch_name_collision_never_removes_existing_container(driver):
    tree = ast.parse((ROOT / "experiments" / driver).read_text())
    finalizer = next(node.finalbody for node in tree.body
                     if isinstance(node, ast.Try) and node.finalbody)
    commands = []

    def capture(command):
        return "pre-existing-container" if command[:2] == ["docker", "ps"] else ""

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=capture(command), returncode=0)

    helper_path = ROOT / "experiments" / "ts5_stage_cleanup.py"
    helper = None
    if helper_path.exists():
        spec = importlib.util.spec_from_file_location("stage_cleanup_test", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

    def cleanup_stage(name, **kwargs):
        return helper.cleanup_stage(name, **kwargs, run=run)

    namespace = {
        "NAME": "conflicting-stage", "completed": False, "launched": False,
        "capture": capture, "subprocess": SimpleNamespace(run=run),
        "gpu_processes": lambda: "", "cleanup_stage": cleanup_stage,
        "write": lambda *args: None,
        "stop": SimpleNamespace(set=lambda: None),
        "monitor": SimpleNamespace(join=lambda **kwargs: None),
    }
    try:
        exec(compile(ast.Module(body=finalizer, type_ignores=[]), driver, "exec"), namespace)
    except AssertionError:
        # The original prelaunch/unsafe-state failure still must propagate.
        pass
    assert not any(command[:3] == ["docker", "rm", "-f"] for command in commands), commands
