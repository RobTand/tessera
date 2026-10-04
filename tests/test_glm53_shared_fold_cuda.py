"""GPU receipts for folding the MoE shared add into the fused routed token sum.

``token_sum_shared`` must store, bit for bit, what the stock path stores:
``token_sum`` and then ATen's bf16 ``shared_output + fused_output``
(``MoERunner.forward``; ``tessera.serving.glm53_shared_fold``). Both round
twice, in the same order, with the same intrinsic, so the comparison is
``torch.equal`` on int16 views, not a tolerance.

The token-sum cases run on each of the three libraries the routed lane builds,
at the GLM-5.3 shape (``H`` 4096, top-k 8) and T in {1, 4, 2048}, on the data
``shared_fold_data.adversarial`` builds to bite (an overflow only the final
rounding makes, exact cancellation, a double-rounding tie, subnormals, NaN and
infinities). The adapter cases fold through ``FusedRoutedWindowMoE.__call__``
eagerly and under CUDA-graph replay.

The lane is a CUDA kernel JIT-built on first use. These tests run through
PrismaBuild inside the pinned serving image (``experiments/routed_fused_tests.sh``).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import routed_fused as rf                    # noqa: E402
from tessera.errors import GrammarError                   # noqa: E402

from shared_fold_data import adversarial, targeted_failures  # noqa: E402
from test_routed_fused_window import (HIDDEN, LIBRARY_IDS, TOP_K, _bundles,  # noqa: E402,F401
                                      _fused, _routes, _stacks, cuda, family)

GLM_HIDDEN, GLM_TOP_K = 4096, 8


def _stock(lib, routed, shared, top_k):
    """token_sum, then the runner's add."""
    fused = torch.empty(shared.shape, dtype=torch.bfloat16, device=shared.device)
    rf._ext(lib).token_sum(routed, fused, top_k)
    return shared + fused


def _fold(lib, routed, shared, top_k):
    out = torch.empty(shared.shape, dtype=torch.bfloat16, device=shared.device)
    rf._ext(lib).token_sum_shared(routed, shared, out, top_k)
    return out


@cuda
@pytest.mark.parametrize("lib", LIBRARY_IDS)
@pytest.mark.parametrize("tokens", [1, 4, 2048])
def test_token_sum_shared_is_token_sum_then_the_bf16_add(lib, tokens):
    routed, shared = adversarial(tokens, GLM_HIDDEN, GLM_TOP_K, 1000 + tokens,
                                 rf._ext(lib).token_sum)
    stock = _stock(lib, routed, shared, GLM_TOP_K)
    out = _fold(lib, routed, shared, GLM_TOP_K)
    torch.cuda.synchronize()
    a, b = out.view(torch.int16), stock.view(torch.int16)
    assert torch.equal(a, b), f"{lib} T={tokens}: {int((a != b).sum())} of {a.numel()} differ"
    # The targeted groups stored their known answers, so the equality bites.
    assert targeted_failures(out, GLM_TOP_K) == []


@cuda
@pytest.mark.parametrize("lib", LIBRARY_IDS)
def test_token_sum_shared_refuses_a_shared_output_it_cannot_read_in_place(lib):
    routed, shared = adversarial(4, GLM_HIDDEN, GLM_TOP_K, 7, rf._ext(lib).token_sum)
    out = torch.empty_like(shared)
    misaligned = torch.empty(4 * GLM_HIDDEN + 1, dtype=torch.bfloat16, device="cuda")[1:]
    for bad in (shared.float(), shared[:2], torch.cat([shared, shared], 1)[:, ::2],
                misaligned.view(4, GLM_HIDDEN), shared.cpu()):
        with pytest.raises(RuntimeError, match="shared must be"):
            rf._ext(lib).token_sum_shared(routed, bad, out, GLM_TOP_K)


@cuda
@pytest.mark.parametrize("lib", LIBRARY_IDS)
def test_token_sum_shared_refuses_a_cpu_output_even_for_an_empty_workload(lib):
    """Zero-token views keep the pre-fix device-hole control free of launches."""
    routed = torch.empty((1, GLM_HIDDEN), dtype=torch.bfloat16, device="cuda")[:0]
    shared = torch.empty((1, GLM_HIDDEN), dtype=torch.bfloat16, device="cuda")[:0]
    out = torch.empty((0, GLM_HIDDEN), dtype=torch.bfloat16, device="cpu")
    with pytest.raises(RuntimeError, match="out must be.*routed's CUDA device"):
        rf._ext(lib).token_sum_shared(routed, shared, out, GLM_TOP_K)


@cuda
@pytest.mark.parametrize("lib", (*rf.LIBRARIES, "e2m1"))
def test_original_token_sum_refuses_a_cpu_output_even_for_an_empty_workload(lib):
    """#859: retain real device allocations while safely exposing the hole."""
    routed = torch.empty((1, GLM_HIDDEN), dtype=torch.bfloat16, device="cuda")[:0]
    out = torch.empty((0, GLM_HIDDEN), dtype=torch.bfloat16, device="cpu")
    if lib == "e2m1":
        from tessera import routed_fused_e2m1
        extension = routed_fused_e2m1._ext()
    else:
        extension = rf._ext(lib)
    with pytest.raises(RuntimeError, match="out must be.*routed's CUDA device"):
        extension.token_sum(routed, out, GLM_TOP_K)


@cuda
@pytest.mark.parametrize("lib", (*rf.LIBRARIES, "e2m1"))
def test_token_sum_bindings_refuse_an_output_on_another_cuda_device(lib):
    """A real second device is required; empty views never launch the old code."""
    if torch.cuda.device_count() < 2:
        pytest.skip("output-device refusal requires two CUDA devices")
    routed = torch.empty((1, GLM_HIDDEN), dtype=torch.bfloat16, device="cuda:0")[:0]
    shared = torch.empty((1, GLM_HIDDEN), dtype=torch.bfloat16, device="cuda:0")[:0]
    out = torch.empty((1, GLM_HIDDEN), dtype=torch.bfloat16, device="cuda:1")[:0]
    if lib == "e2m1":
        from tessera import routed_fused_e2m1
        extension = routed_fused_e2m1._ext()
    else:
        extension = rf._ext(lib)
    with pytest.raises(RuntimeError, match="out must be.*routed's CUDA device"):
        extension.token_sum(routed, out, GLM_TOP_K)
    with pytest.raises(RuntimeError, match="out must be.*routed's CUDA device"):
        extension.token_sum_shared(routed, shared, out, GLM_TOP_K)


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("t", [1, 7, 71])
def test_the_fused_forward_folds_the_shared_add_bitwise(family, t):
    """``fused(x, ids, rw, shared=s)`` stores ``s + fused(x, ids, rw)``, bit for bit."""
    fused = _fused(_bundles(family, _stacks(family)))
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 700 + t)
    shared = (torch.randn(t, HIDDEN, device="cuda") * 4).bfloat16()
    stock = shared + fused(x, ids, rw)
    folded = fused(x, ids, rw, shared=shared)
    assert torch.equal(folded.view(torch.int16), stock.view(torch.int16))
    assert torch.equal(fused(x, ids, rw, shared=None), fused(x, ids, rw))


@cuda
@pytest.mark.parametrize("family", ["value"], indirect=True)
def test_the_fused_forward_refuses_a_bad_shared_output(family):
    fused = _fused(_bundles(family, _stacks(family)))
    x = torch.randn(7, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(7, TOP_K, 5)
    shared = torch.randn(7, HIDDEN, device="cuda").bfloat16()
    for bad in (shared.float(), shared[:3], torch.cat([shared, shared], 1)[:, ::2], shared.cpu()):
        with pytest.raises(GrammarError, match="shared must be contiguous bf16"):
            fused(x, ids, rw, shared=bad)


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_the_folded_forward_captures_and_replays_twice_against_eager(family):
    """Decode at NO_OVERLAP folds under FULL graphs; replay must equal eager."""
    fused = _fused(_bundles(family, _stacks(family)))
    t = 4
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 41)
    shared = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    eager = fused(x, ids, rw, shared=shared)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fused(x, ids, rw, shared=shared)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused(x, ids, rw, shared=shared)
    for _ in range(2):
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)
    shared.copy_(torch.randn(t, HIDDEN, device="cuda").bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, shared + fused(x, ids, rw))
