"""Joined TCQ needs paths only; diagnostic SSE must not drain its CUDA stream."""
import pytest
import torch

import tessera.encode as encoder
from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.manifest import BodyKind
from tessera.trellis import ConvCode


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_joined_tcq_does_not_read_unused_sse(monkeypatch, graph, count):
    if not torch.cuda.is_available():
        pytest.skip("joined TCQ scalar-read regression needs CUDA")
    monkeypatch.setenv("TESSERA_TCQ_GRAPH", "1" if graph else "0")
    encoder.tcq_plan_cache_clear()
    grid = next(g for g in SERIALISABLE_GRIDS.values() if g.name == "E2M1x2")
    forest = encoder.build_forest(7, grid=grid)
    code = ConvCode()
    torch.manual_seed(385)
    calls = [encoder._TrellisCall(
        body=BodyKind.TCQ, rate=7,
        targets=torch.randn(16, 5, device="cuda"),
        weights=torch.rand(16, 5, device="cuda") + 0.5,
        forest=forest, code=code, level=0, span=2,
    ) for _ in range(count)]
    expected = [encoder.viterbi_columns(
        c.targets, forest, code, 0, span=2, weights=c.weights, impl="reference",
    ) for c in calls]
    assert all(isinstance(answer[2], float) for answer in expected)

    def unused_scalar_read(self):
        pytest.fail("joined TCQ extracted discarded SSE and drained the CUDA stream")

    monkeypatch.setattr(encoder._TCQPlan, "sse", unused_scalar_read)
    first = encoder._run_joined(calls)
    # Repeat after changing the input so cache reuse cannot return stale paths.
    changed = [encoder._TrellisCall(
        body=c.body, rate=c.rate, targets=-c.targets, weights=c.weights,
        forest=forest, code=code, level=c.level, span=c.span,
    ) for c in calls]
    encoder._run_joined(changed)
    again = encoder._run_joined(calls)
    for want, got, repeated in zip(expected, first, again):
        assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])
        assert torch.equal(want[0], repeated[0]) and torch.equal(want[1], repeated[1])
