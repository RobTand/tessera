"""Campaign retries receive explicit fresh namespaces without overwriting evidence."""
import ast
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "experiments/ts5_lfm_served_bound.py"


def _paths():
    tree = ast.parse(DRIVER.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "campaign_stage_paths"]
    assert functions, "campaign driver has no explicit attempt namespace"
    namespace = {"Path": Path, "CAMPAIGN": Path("/campaign")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(DRIVER), "exec"), namespace)
    return namespace["campaign_stage_paths"]


@pytest.mark.parametrize("stage", ["census", "student"])
def test_each_attempt_has_fresh_output_container_and_local_paths(stage):
    paths = _paths()
    first, second = paths(stage, 1), paths(stage, 2)
    assert all(a != b for a, b in zip(first, second))
    assert first[0] == Path(f"/campaign/{stage}-bound-r1")
    assert first[1] == f"ts5-lfm-r2-{stage}-bound-r1"


@pytest.mark.parametrize("attempt", [0, -1])
def test_nonpositive_attempts_are_refused(attempt):
    with pytest.raises(ValueError, match="positive"):
        _paths()("census", attempt)
