"""Metadata correction preserves explicit declarations unrelated to its defect."""
import ast
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "experiments/ts5_lfm_correct_passthrough.py"


def _correct():
    functions = [node for node in ast.parse(DRIVER.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "corrected_ignore"]
    assert functions, "correction has no bounded ignore rewrite"
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(DRIVER), "exec"), namespace)
    return namespace["corrected_ignore"]


def test_tied_head_and_other_unrelated_ignores_survive():
    prefix = "model.layers.0.feed_forward."
    old = {"lm_head", "model.embed_tokens", prefix + "w1", prefix + "w3"}
    derived = {"model.embed_tokens", prefix + "w13"}
    assert _correct()(old, derived) == {"lm_head", "model.embed_tokens", prefix + "w13"}


@pytest.mark.parametrize("derived", [{"unrelated"}, {"m.feed_forward.w13"}])
def test_unrelated_or_incomplete_rewrites_refuse(derived):
    with pytest.raises(AssertionError):
        _correct()({"m.feed_forward.w1"}, derived)
