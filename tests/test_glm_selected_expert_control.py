"""A research control cannot pass without running an operator case."""
import pytest

from experiments.glm_selected_expert_control import run


@pytest.mark.parametrize('cases', [[], None])
def test_empty_selected_control_is_refused_before_runtime_construction(cases):
    with pytest.raises(ValueError, match='nonempty.*cases'):
        run(None, {'cases': cases}, None, None, None, None, None, None, None, None)
