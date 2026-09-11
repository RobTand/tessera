"""The encoder's pass never waits on the device between Viterbi chunks.

Measured on the GLM census (PrismaQuant boundary-feed, 2026-09-11): each
joined Viterbi chunk ended in ``float(final.sum())``, a host round trip the
encoder's batch driver then discarded, and every unit's next LDLQ block began
with a pageable host-to-device index upload, which is a second stream sync.
Between them the host sat in ``cudaStreamSynchronize`` for 80-85 % of a pass
while the GPU ran 87-90 % busy, and every batch started from an empty queue.

The contract these pin: a Viterbi call that discards its cost makes no host
sync; one that reads it makes exactly one; and an LDLQ encode makes the same
number of syncs however many blocks its schedule has -- so the per-block cost
is launches only, and the host runs ahead of the device through the pass.
Counted with ``torch.profiler``'s CUDA runtime rows, which is the instrument
the census measurement used.
"""
import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.encode import viterbi_window
from tessera.export import (ActivationSource, DEFAULT_SCALE_REFIT, encode_linear,
                            encode_linears, wire_recipe)
from tessera.window_viterbi import fused_available, window_plan_cache_clear

pytestmark = pytest.mark.skipif(
    not fused_available(),
    reason="the fused window Viterbi is a CUDA path and needs triton")

SYNCS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
         "cudaMemcpy", "cudaMemcpy2D")


def host_syncs(fn):
    """Run ``fn`` under the profiler; return (its result, its host sync count,
    the CUDA runtime calls it made in order)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = fn()
    events = sorted(prof.events(), key=lambda e: e.time_range.start)
    runtime = [e.name for e in events if e.name.startswith(("cuda", "cu"))]
    return out, sum(1 for name in runtime if name in SYNCS), runtime


def _case(L, R, rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(rows, cols, generator=g).cuda()
    vectors = torch.randn(1 << L, 1, generator=g).cuda()
    weights = (torch.rand(rows, cols, generator=g).cuda() + 0.5)
    return targets, vectors, weights


def test_a_viterbi_call_syncs_once_for_its_float_and_never_without_it():
    targets, vectors, weights = _case(14, 4, 128, 1100, seed=3)   # three chunks
    ref, sse_ref = viterbi_window(targets, vectors, 14, 4, weights=weights, impl="reference")
    window_plan_cache_clear()
    for _ in range(2):                    # eager, then captured: the plan is warm
        viterbi_window(targets, vectors, 14, 4, weights=weights, impl="fused")
    # What the instrument itself costs: one small launch under the profiler.
    _, baseline, _ = host_syncs(lambda: targets.add(1.0))
    (got, sse), syncs, calls = host_syncs(
        lambda: viterbi_window(targets, vectors, 14, 4, weights=weights,
                               impl="fused", want_sse=False))
    torch.cuda.synchronize()
    assert torch.equal(got, ref) and sse is None
    assert syncs - baseline == 0, (
        f"a call that discards its cost waited on the device {syncs - baseline} "
        f"times (baseline {baseline}); runtime calls: {calls}")
    (got, sse), syncs, calls = host_syncs(
        lambda: viterbi_window(targets, vectors, 14, 4, weights=weights, impl="fused"))
    assert torch.equal(got, ref) and sse == sse_ref
    assert syncs - baseline == 1, (
        f"a call that reads its cost should wait once, not {syncs - baseline} "
        f"times (baseline {baseline}); runtime calls: {calls}")


ROWS, COLS = 64, 256


def _weights(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(ROWS, COLS, generator=g) * 0.02
    w[3, 7] = 0.6
    return w.to(device="cuda", dtype=torch.bfloat16).contiguous()


def _hessian(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(4 * COLS, COLS, generator=g)
    mix = torch.eye(COLS) + torch.randn(COLS, COLS, generator=g) / COLS ** 0.5
    x = x @ mix
    x[:, ::29] *= 5.0
    return (x.T @ x).to(device="cuda", dtype=torch.float32)


def _encode(count, block):
    """An E4M3 K1 @1042 LDLQ encode (two rates in every block) at ``block``."""
    grid, q256 = E4M3_GRID, 1042
    recipe = wire_recipe(grid, q256)
    weights = [_weights(10 + i) for i in range(count)]
    source = ActivationSource(
        hessians={f"u{i}": _hessian(20 + i) for i in range(count)},
        provenance={"text_sha256": "0" * 64, "fit_tokens": 4 * COLS,
                    "fit_ids_sha256": "1" * 64},
        ldlq_block=block)
    per_unit = [source.for_unit(f"u{i}", COLS, "cuda", scale_plane=recipe.scale_plane,
                                weight=w) for i, w in enumerate(weights)]
    assert all(kw["ldl_block"] == block for kw in per_unit)
    if count == 1:
        return lambda: [encode_linear(weights[0], grid=grid, q256=q256, name="u0",
                                      scale_refit=DEFAULT_SCALE_REFIT, **per_unit[0])]
    return lambda: encode_linears(
        weights, grid=grid, q256=q256, names=[f"u{i}" for i in range(count)],
        per_unit=per_unit, scale_refit=DEFAULT_SCALE_REFIT)


@pytest.mark.parametrize("count", [1, 2])
def test_an_ldlq_encode_syncs_the_same_number_of_times_at_every_block_size(count):
    """Eight blocks or two: the same host syncs, so no block waits.  The
    blobs are compared across the two schedules only for inequality -- a
    different block IS a different artifact -- and each against itself
    under the profiler, which must not change the bytes."""
    window_plan_cache_clear()
    counts = {}
    blobs = {}
    for block in (COLS // 2, COLS // 8):
        run = _encode(count, block)
        plain = run()
        run()                                  # every shape seen twice: plans captured
        out, syncs, _ = host_syncs(run)
        torch.cuda.synchronize()
        assert [u.blob for u in out] == [u.blob for u in plain]
        counts[block] = syncs
        blobs[block] = [u.blob for u in out]
    assert counts[COLS // 2] == counts[COLS // 8], (
        f"host syncs grew with the block count: {counts}")
    assert blobs[COLS // 2] != blobs[COLS // 8]
