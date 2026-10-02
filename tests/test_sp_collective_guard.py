"""Pinned no-op writer reproduction and CPU SP collective safety controls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from test_glm53_prefill import TwoRanks, _NoCuda, _ops, _run, _stack, stock_forward
from tessera.serving import glm53_prefill as gp


@pytest.mark.parametrize("disabled_rank", [0, 1])
def test_disabled_stock_noop_writer_is_declined_before_any_rank_shards(disabled_rank):
    ranks = TwoRanks()
    forwarded, writer_calls, availability = [], [0, 0], [0, 0]
    for rank in range(2):
        ops = _ops(ranks)
        def available(_rank=rank):
            availability[_rank] += 1
            # Both ranks decide before changing their activation shape.
            return not ranks.max_across_tp(float(_rank == disabled_rank))
        def rs(x, _rank=rank):
            writer_calls[_rank] += 1
            # The pinned PyNccl method returns without writing when disabled.
            # Poison makes its containing CudaCommunicator's torch.empty output
            # deterministic in this CPU reproduction, rather than lucky memory.
            if _rank == disabled_rank:
                # Still exchange input so the stand-in peer's control can finish.
                ranks.sp_reduce_scatter(x)
                return torch.full_like(x[:len(x)//2], float("nan"))
            return ranks.sp_reduce_scatter(x)
        ops.sp_available, ops.sp_reduce_scatter = available, rs
        forwarded.append(gp.make_forward(stock_forward, ops, gp.SpState("force", 8, 2), _NoCuda))
    with pytest.raises(RuntimeError, match="SP collective unavailable before activation"):
        _run(ranks, forwarded, 8, passes=2)
    assert writer_calls == [0, 0]
    assert availability == [1, 1]


def _dc():
    nccl = NS(available=True, disabled=False, _suspended=False, world_size=2,
              reduce_scatter=lambda *a: None, all_gather=lambda *a: None)
    return NS(world_size=2, ca_comm=None, pynccl_comm=nccl)


@pytest.mark.parametrize("fault", ["none", "disabled", "unavailable", "suspended", "custom",
                                   "symm", "missing", "unknown", "world", "method"])
def test_only_an_active_inspected_sp_writer_is_available(fault):
    dc = _dc()
    symm = False
    if fault == "disabled": dc.pynccl_comm.disabled = True
    if fault == "unavailable": dc.pynccl_comm.available = False
    if fault == "suspended": dc.pynccl_comm._suspended = True
    if fault == "custom": dc.ca_comm = object()
    if fault == "symm": symm = True
    if fault == "missing": dc.pynccl_comm = None
    if fault == "unknown": del dc.pynccl_comm.disabled
    if fault == "world": dc.pynccl_comm.world_size = 1
    if fault == "method": del dc.pynccl_comm.reduce_scatter
    why = gp.sp_collective_decline(dc, symmetric_ag_rs=symm)
    assert (why is None) == (fault == "none")


def _guard(dc, *, capturing=False, agree=None):
    return gp.SpCollectiveGuard(route=lambda: gp.sp_collective_decline(dc, symmetric_ag_rs=False),
                               agree=agree or (lambda value: value),
                               capturing=lambda: capturing)


def test_disabled_writer_cannot_reach_stock_noop_and_resume_restores_availability():
    dc = _dc()
    guard = _guard(dc)
    assert guard.available()
    called = []
    dc.pynccl_comm.disabled = True
    assert not guard.available()
    with pytest.raises(RuntimeError, match="SP collective unavailable"):
        guard.call(lambda x: called.append(x), "input")
    assert called == []
    dc.pynccl_comm.disabled = False
    assert guard.available()
    assert guard.call(lambda x: x, "input") == "input"


def test_suspend_is_unavailable_even_with_disabled_false_and_resume_restores_it():
    dc = _dc()
    guard = _guard(dc)
    dc.pynccl_comm._suspended = True
    assert not guard.available()
    dc.pynccl_comm._suspended = False
    assert guard.available()


def test_capture_makes_no_control_collective_or_sp_writer_call():
    calls = []
    guard = _guard(_dc(), capturing=True, agree=lambda value: calls.append(value))
    assert not guard.available()
    with pytest.raises(RuntimeError, match="capture"):
        guard.call(lambda x: calls.append(x), "input")
    assert calls == []


def test_peer_transient_disable_refuses_both_ranks_before_either_writer():
    ranks = TwoRanks()
    written = []
    dcs = [_dc(), _dc()]
    dcs[1].pynccl_comm.disabled = True
    def worker(rank):
        ranks.local.rank = rank
        guard = _guard(dcs[rank], agree=ranks.max_across_tp)
        with pytest.raises(RuntimeError, match="SP collective unavailable"):
            guard.call(lambda x: written.append(rank), "input")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))
    assert written == []


def test_active_writer_arguments_results_and_operation_order_are_unchanged():
    calls = []
    guard = _guard(_dc())
    for name in ("AG", "RS", "AG"):
        assert guard.call(lambda x, _name=name: (calls.append((_name, x)), x)[1], name) == name
    assert calls == [("AG", "AG"), ("RS", "RS"), ("AG", "AG")]
