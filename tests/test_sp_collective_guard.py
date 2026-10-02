"""Pinned no-op writer reproduction and CPU SP collective safety controls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
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
                               capturing=lambda: capturing, owner=lambda: dc.pynccl_comm)


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
        assert not guard.available()  # one agreement at activation, before any writer
        with pytest.raises(RuntimeError, match="SP collective unavailable"):
            guard.call(lambda x: written.append(rank), "input")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))
    assert written == []


def test_active_writer_arguments_results_and_operation_order_are_unchanged():
    calls = []
    agreements = []
    guard = _guard(_dc(), agree=lambda value: (agreements.append(value), value)[1])
    assert guard.available()
    for name in ("AG", "RS", "AG"):
        assert guard.call(lambda x, _name=name: (calls.append((_name, x)), x)[1], name) == name
    assert calls == [("AG", "AG"), ("RS", "RS"), ("AG", "AG")]
    assert agreements == [0.0]  # no CPU control exchange at individual RS/AG boundaries


def test_availability_control_uses_explicit_cpu_device_and_existing_tp_cpu_group():
    calls = []
    group = NS(cpu_group=object())
    control = NS(item=lambda: 1)
    def tensor(values, **kwargs):
        calls.append(("tensor", values, kwargs))
        return control
    def all_reduce(value, **kwargs):
        calls.append(("all_reduce", value, kwargs))
    fake = NS(tensor=tensor, int32=object(), distributed=NS(
        all_reduce=all_reduce, ReduceOp=NS(MAX=object())))
    assert gp.agree_sp_collective(fake, group, 0.0) == 1.0
    assert calls == [("tensor", [0], {"dtype": fake.int32, "device": "cpu"}),
                     ("all_reduce", control, {"op": fake.distributed.ReduceOp.MAX,
                                              "group": group.cpu_group})]


def test_unqualified_midpass_owner_replacement_refuses_before_enqueue_without_exchange():
    dc, agreements, written = _dc(), [], []
    guard = _guard(dc, agree=lambda value: (agreements.append(value), value)[1])
    assert guard.available()
    dc.pynccl_comm = NS(**dc.pynccl_comm.__dict__)
    with pytest.raises(RuntimeError, match="SP collective unavailable"):
        guard.call(lambda x: written.append(x), "input")
    assert agreements == [0.0] and written == []


@pytest.mark.parametrize("case", ["profile", "small", "below", "declined", "capture"])
def test_non_sp_pass_preserves_stock_fallback_without_querying_writer(case):
    reference_ranks, ranks = TwoRanks(), TwoRanks()
    passes = 1 if case == "profile" else 2
    reference, _ = _run(reference_ranks, stock_forward, 8, passes=passes)
    forwards = []
    for rank in range(2):
        state = gp.SpState("force", 8, 2)
        if case == "below": state.t_star = 1024
        if case == "declined": state.declined = "unsupported layer"
        ops = _ops(ranks, exact=lambda *a: case != "small")
        ops.sp_available = lambda: pytest.fail("non-SP pass queried unavailable writer")
        forwards.append(gp.make_forward(stock_forward, ops, state,
            NS(cuda=NS(is_current_stream_capturing=lambda: case == "capture"))))
    result, _ = _run(ranks, forwards, 8, passes=passes)
    assert all(torch.equal(a, b) for a, b in zip(reference, result))
    assert ranks.calls == {"all_reduce": 6 * passes, "all_gather": 0, "reduce_scatter": 0}


def test_auto_measurement_skips_unsupported_writer_then_preserves_profile_fallback():
    ranks = TwoRanks()
    checked = [0, 0]
    forwards = []
    for rank in range(2):
        ops = _ops(ranks)
        def available(_rank=rank):
            checked[_rank] += 1
            return False
        ops.sp_available = available
        forwards.append(gp.make_forward(stock_forward, ops, gp.SpState("auto", 8, 2), _NoCuda))
    outputs, _ = _run(ranks, forwards, 8)
    assert checked == [1, 1]
    assert all(torch.isfinite(x).all() for x in outputs)
    assert ranks.calls == {"all_reduce": 6, "all_gather": 0, "reduce_scatter": 0}


@pytest.mark.parametrize("disabled,has_suspend", [(False, True), (False, False), (True, True)])
def test_pinned_suspend_resume_preserve_disabled_and_change_only_supported_suspension(disabled, has_suspend):
    # Exact method slices from the pinned image, attributed by full module SHA.
    # Executed only inside the admitted CPU test, with NCCL calls as stand-ins.
    fixture = json.loads((Path(__file__).parent / "fixtures/pynccl_lifecycle_20260929.json").read_text())
    assert fixture["source_sha256"] == gp._INTERFACES[0].digests[gp.SP_MODULES.index(
        "vllm.distributed.device_communicators.pynccl")]
    calls, flag = [], object()
    namespace = {"logger": NS(warning_once=lambda *a: calls.append("warning")),
                 "_NCCL_SUSPEND_MEM": flag}
    for method in fixture["methods"].values():
        exec(compile(method["source"], "<pinned PyNccl lifecycle CPU fixture>", "exec"), namespace)
    obj = NS(disabled=disabled, _suspended=False, comm=object(), nccl=NS(
        has_symbol=lambda name: has_suspend,
        ncclCommSuspend=lambda comm, flags: calls.append(("suspend", comm, flags)),
        ncclCommResume=lambda comm: calls.append(("resume", comm))))
    namespace["suspend"](obj)
    assert obj.disabled is disabled
    assert obj._suspended == (not disabled and has_suspend)
    namespace["resume"](obj)
    assert obj.disabled is disabled and obj._suspended is False
    expected = [] if disabled else [("suspend", obj.comm, flag), ("resume", obj.comm)] if has_suspend else ["warning"]
    assert calls == expected
