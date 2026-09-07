"""Pure dispatch controls; these are not engine timing qualification."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from experiments.full_engine_timing_boundaries import resolve_apply_boundaries, observe_apply_boundaries


class Method:
    is_monolithic = False

    def apply(self, layer, x, topk_ids=None):
        return (layer, x, topk_ids)


def model(**modules):
    return SimpleNamespace(named_modules=lambda: modules.items())


def test_routed_boundary_excludes_routing_and_restores_shared_method():
    method = Method()
    dense, routed, outside = [SimpleNamespace(quant_method=method) for _ in range(3)]
    wrapper = SimpleNamespace(routed_experts=routed)
    roster = [{"unit_id": "l:dense", "module": "dense"}, {"unit_id": "s:moe", "module": "moe"}]
    boundaries = resolve_apply_boundaries(model(dense=dense, moe=wrapper), roster)
    assert boundaries[1]["boundary"] == "moe.routed_experts.quant_method.apply"
    assert boundaries[1]["includes_router"] is False
    seen = []

    @contextmanager
    def observe(row, call):
        seen.append(("begin", row["unit_id"]))
        yield
        assert call["result"][0] is call["arguments"]["layer"]
        seen.append(("end", row["unit_id"]))

    with observe_apply_boundaries(boundaries, observe):
        assert method.apply(dense, 3) == (dense, 3, None)
        assert method.apply(layer=routed, x=4, topk_ids=5) == (routed, 4, 5)
        assert method.apply(outside, 6) == (outside, 6, None)
    assert seen == [("begin", "l:dense"), ("end", "l:dense"), ("begin", "s:moe"), ("end", "s:moe")]
    assert "apply" not in vars(method)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "alias", "missing_routed", "monolithic"])
def test_canonical_census_ambiguity_refuses(defect):
    owner = SimpleNamespace(quant_method=Method())
    roster = [{"unit_id": "l:dense", "module": "dense"}]
    modules = {"dense": owner}
    if defect == "missing":
        modules = {}
    elif defect == "duplicate":
        roster *= 2
    elif defect == "alias":
        roster.append({"unit_id": "l:other", "module": "other"})
        modules["other"] = owner
    else:
        roster = [{"unit_id": "s:moe", "module": "moe"}]
        modules = {"moe": SimpleNamespace(routed_experts=owner) if defect == "monolithic" else owner}
        if defect == "monolithic":
            owner.quant_method.is_monolithic = True
    with pytest.raises(ValueError):
        resolve_apply_boundaries(model(**modules), roster)


def test_observer_failure_restores_original_method():
    owner = SimpleNamespace(quant_method=Method())
    boundaries = resolve_apply_boundaries(model(dense=owner), [{"unit_id": "l:dense", "module": "dense"}])

    @contextmanager
    def fail(row, call):
        raise RuntimeError("observation failed")
        yield

    with pytest.raises(RuntimeError, match="observation failed"):
        with observe_apply_boundaries(boundaries, fail):
            owner.quant_method.apply(owner, 1)
    assert "apply" not in vars(owner.quant_method)
