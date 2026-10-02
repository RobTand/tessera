"""CPU token/stream fixtures; these do not qualify CUDA or NCCL arithmetic."""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from test_glm53_prefill import TwoRanks
from tessera.serving.glm53_prefill import SplitForcer
from tessera.serving.sp_tile_overlap import (SiteOutputs, SpTileOverlap, SpTilePlan,
                                           make_pynccl_runner, pynccl_sp_decline, stock_post_pre_into)


def _ids(plan, rank):
    # Independent construction from global token positions, not plan.tiles/shard.
    ids = []
    for a in range(0, plan.tokens, plan.tile):
        length = min(plan.tile, plan.tokens - a)
        half = (length + 1) // 2
        ids.extend(range(a + rank * half, a + (rank + 1) * half))
    return ids


@pytest.mark.parametrize("tokens,tile", [(2048, 1024), (2049, 1024), (9, 4), (1, 2)])
@pytest.mark.parametrize("rank", [0, 1])
def test_initial_shard_owns_each_rank_half_of_every_tile(tokens, tile, rank):
    plan = SpTilePlan(tokens, tile)
    full = torch.arange(1, tokens + 1).reshape(tokens, 1)
    got = plan.shard(full, rank, torch).flatten().tolist()
    expected = [i + 1 if i < tokens else 0 for i in _ids(plan, rank)]
    assert got == expected


@pytest.mark.parametrize("args", [(0, 4), (9, 3), (9, 4, 4), (True, 4), (9, 2.0)])
def test_plan_refuses_unsupported_geometry(args):
    with pytest.raises(ValueError):
        SpTilePlan(*args)


class Stream:
    """Delayed tasks plus immutable event-record snapshots, like CUDA waits."""

    def __init__(self, name, log, *, ignore_waits=False):
        self.name, self.log, self.ignore_waits = name, log, ignore_waits
        self.tasks, self.cursor, self.synchronizations = [], 0, 0

    def enqueue(self, name, fn):
        self.log.append(("enqueue", self.name, name))
        self.tasks.append((name, fn))

    def wait_event(self, event):
        self.log.append(("wait", self.name, event.label))
        mark = event.mark
        assert mark is not None, "wait on an unrecorded event"
        if not self.ignore_waits:
            self.enqueue("wait:" + event.label, lambda: mark[0].flush(mark[1]))

    def flush(self, end=None):
        end = len(self.tasks) if end is None else end
        while self.cursor < end:
            name, fn = self.tasks[self.cursor]
            self.cursor += 1
            self.log.append(("execute", self.name, name))
            fn()

    def synchronize(self):
        self.synchronizations += 1
        self.flush()


class Event:
    def __init__(self, label, log):
        self.label, self.log, self.mark = label, log, None

    def record(self, stream):
        self.log.append(("record", stream.name, self.label))
        self.mark = (stream, len(stream.tasks))


def _storage(tensor):
    return tensor.untyped_storage().data_ptr()


def _runner(ranks=None, *, ignore_compute_waits=False, ignore_side_waits=False,
            capture=False, fail=None):
    log, kept, reduced_ready, mhc_ready = [], set(), set(), set()
    compute = Stream("compute", log, ignore_waits=ignore_compute_waits)
    side = Stream("side", log, ignore_waits=ignore_side_waits)
    count = {"rs": 0, "ag": 0, "events": 0}

    def event():
        count["events"] += 1
        return Event(str(count["events"]), log)

    def keep(tensor, stream):
        assert stream is side
        kept.add(_storage(tensor))
        log.append(("keep", _storage(tensor)))

    def rs(out, x, stream):
        assert _storage(out) in kept and _storage(x) in kept, "early owner release"
        i = count["rs"]
        count["rs"] += 1

        def action():
            value = ranks.sp_reduce_scatter(x) if ranks else x[:len(out)]
            out.copy_(value)
            reduced_ready.add(out.data_ptr())
        stream.enqueue("RS" + str(i), action)
        if fail == "rs" and i == 0:
            raise RuntimeError("partial RS enqueue")

    def ag(out, x, stream):
        assert _storage(out) in kept and _storage(x) in kept, "early owner release"
        i = count["ag"]
        count["ag"] += 1

        def action():
            if count["rs"]:
                assert x.data_ptr() in mhc_ready, "AG read before mHC"
            value = ranks.sp_all_gather(x) if ranks else torch.cat((x, x))
            out.copy_(value)
        stream.enqueue("AG" + str(i), action)
        if fail == "ag" and i == 0:
            raise RuntimeError("partial AG enqueue")

    def mhc(x, res, post, comb, out):
        def action():
            assert x.data_ptr() in reduced_ready, "mHC read before RS"
            _site(x, res, post, comb, out)
            mhc_ready.add(out.layer_input.data_ptr())
        compute.enqueue("MHC", action)
        if fail == "mhc":
            raise RuntimeError("partial mHC enqueue")

    split_calls = []
    @contextlib.contextmanager
    def full_split(tokens):
        split_calls.append(tokens)
        yield

    runner = SpTileOverlap(torch=torch, reduce_scatter_into=rs, all_gather_into=ag,
                           side=side, compute=lambda: compute, new_event=event, keep=keep,
                           full_split=full_split, capturing=lambda: capture)
    return NS(runner=runner, mhc=mhc, compute=compute, side=side, log=log,
              kept=kept, count=count, split_calls=split_calls)


def _site(x, res, post, comb, out):
    # Per-token fixture with explicit BF16 rounding and token-dependent state.
    # No stock mHC numerical or exact-reduction claim comes from this toy.
    out.residual.copy_((res.float() + x.float().unsqueeze(1) * post).to(torch.bfloat16))
    out.post.copy_(post + res[:, :, :1].float() / 128)
    out.comb.copy_(comb + post.transpose(1, 2) / 64)
    out.layer_input.copy_(out.residual.float().mean(1).to(torch.bfloat16))


def _inputs(tokens):
    gen = torch.Generator().manual_seed(tokens)
    parts = [torch.randn(tokens, 8, generator=gen).to(torch.bfloat16) for _ in range(2)]
    residual = torch.randn(tokens, 4, 8, generator=gen).to(torch.bfloat16)
    post = torch.randn(tokens, 4, 1, generator=gen)
    comb = torch.randn(tokens, 4, 4, generator=gen)
    return parts, residual, post, comb


def _expected(parts, residual, post, comb):
    out = SiteOutputs(torch.empty_like(residual), torch.empty_like(post),
                      torch.empty_like(comb), torch.empty_like(parts[0]))
    _site(parts[0] + parts[1], residual, post, comb, out)
    return out


@pytest.mark.parametrize("tokens,tile", [(8, 4), (9, 4), (17, 8), (1, 2)])
def test_two_ranks_reconstruct_global_order_and_preserve_all_site_outputs(tokens, tile):
    plan, ranks = SpTilePlan(tokens, tile), TwoRanks()
    parts, residual, post, comb = _inputs(tokens)
    expected = _expected(parts, residual, post, comb)

    def rank_run(rank):
        ranks.local.rank = rank
        ctx = _runner(ranks)
        initial = plan.shard(residual, rank, torch)
        local_post, local_comb = plan.shard(post, rank, torch), plan.shard(comb, rank, torch)
        initial_gather = ctx.runner.gather(plan, initial)
        ctx.compute.flush()
        assert torch.equal(initial_gather, residual)
        result = ctx.runner.run(plan, parts[rank], initial, local_post, local_comb, ctx.mhc)
        # Success has only enqueued waits; it never synchronizes the host.
        assert ctx.side.synchronizations == 0
        ctx.compute.flush()
        assert torch.equal(result.layer_input, expected.layer_input)
        for field in ("residual", "post", "comb"):
            # Exclude the padded dummy token, whose toy site inputs intentionally differ.
            ids = _ids(plan, rank)
            live = [j for j, i in enumerate(ids) if i < tokens]
            want = [i for i in ids if i < tokens]
            assert torch.equal(getattr(result, field)[live], getattr(expected, field)[want])
        assert ctx.split_calls == [tokens]
        return ctx

    with ThreadPoolExecutor(max_workers=2) as executor:
        contexts = list(executor.map(rank_run, range(2)))
    assert ranks.calls == {"all_reduce": 0, "all_gather": 2 * len(plan.tiles),
                           "reduce_scatter": len(plan.tiles)}
    for ctx in contexts:
        collectives = [item[2] for item in ctx.log
                       if item[0] == "enqueue" and item[1] == "side"
                       and item[2][:2] in ("RS", "AG")]
        # Initial gather, then one site's all RS before all AG.
        assert collectives[len(plan.tiles):] == (
            ["RS" + str(i) for i in range(len(plan.tiles))]
            + ["AG" + str(i + len(plan.tiles)) for i in range(len(plan.tiles))])


@pytest.mark.parametrize("mutation,needle", [("compute", "mHC read before RS"),
                                               ("side", "AG read before mHC")])
def test_missing_dependency_mutants_are_observed(mutation, needle):
    plan = SpTilePlan(9, 4)
    parts, residual, post, comb = _inputs(plan.tokens)
    ctx = _runner(ignore_compute_waits=mutation == "compute", ignore_side_waits=mutation == "side")
    ctx.runner.run(plan, parts[0], plan.shard(residual, 0, torch),
                   plan.shard(post, 0, torch), plan.shard(comb, 0, torch), ctx.mhc)
    # Execute the dependent stream first to expose the deliberately deleted wait.
    with pytest.raises(AssertionError, match=needle):
        (ctx.compute if mutation == "compute" else ctx.side).flush()


@pytest.mark.parametrize("fail", ["rs", "mhc", "ag"])
def test_partial_enqueue_or_consumer_failure_retires_side_work(fail):
    plan = SpTilePlan(9, 4)
    parts, residual, post, comb = _inputs(plan.tokens)
    ctx = _runner(fail=fail)
    with pytest.raises(RuntimeError, match="partial"):
        ctx.runner.run(plan, parts[0], plan.shard(residual, 0, torch),
                       plan.shard(post, 0, torch), plan.shard(comb, 0, torch), ctx.mhc)
    assert ctx.side.synchronizations == 1
    assert ctx.side.cursor == len(ctx.side.tasks)
    assert any(item == ("execute", "side", "RS0") for item in ctx.log)
    if fail == "ag":
        assert ("execute", "side", "AG0") in ctx.log
    # The non-reentrant guard is always released on failure.
    assert ctx.runner._lock.acquire(blocking=False)
    ctx.runner._lock.release()


def test_owner_registration_precedes_collectives_and_deleted_registration_is_observed(monkeypatch):
    plan = SpTilePlan(9, 4)
    parts, residual, post, comb = _inputs(plan.tokens)
    ctx = _runner()
    monkeypatch.setattr(ctx.runner, "keep", lambda tensor, stream: None)
    with pytest.raises(AssertionError, match="early owner release"):
        ctx.runner.run(plan, parts[0], plan.shard(residual, 0, torch),
                       plan.shard(post, 0, torch), plan.shard(comb, 0, torch), ctx.mhc)
    assert ctx.count["rs"] == 0


def test_capture_refuses_before_allocations_or_collectives():
    ctx = _runner(capture=True)
    with pytest.raises(RuntimeError, match="graph capture"):
        ctx.runner.gather(SpTilePlan(8, 4), torch.ones(4, 8))
    assert ctx.log == []


def test_overlapping_calls_refuse_before_collectives():
    ctx = _runner()
    ctx.runner._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already in flight"):
            ctx.runner.gather(SpTilePlan(8, 4), torch.ones(4, 8))
    finally:
        ctx.runner._lock.release()
    assert ctx.log == []


@pytest.mark.parametrize("bad", ["shape", "dtype", "stride"])
def test_invalid_site_inputs_refuse_before_any_enqueue(bad):
    plan = SpTilePlan(9, 4)
    parts, residual, post, comb = _inputs(plan.tokens)
    x = parts[0]
    local_res, local_post, local_comb = [plan.shard(t, 0, torch) for t in (residual, post, comb)]
    if bad == "shape":
        local_res = local_res[:-1]
    elif bad == "dtype":
        x = x.float()
    else:
        x = torch.zeros(9, 16, dtype=torch.bfloat16)[:, ::2]
    ctx = _runner()
    with pytest.raises(ValueError):
        ctx.runner.run(plan, x, local_res, local_post, local_comb, ctx.mhc)
    assert ctx.log == []


def test_stock_adapter_uses_explicit_kernels_and_full_batch_split_for_one_row_tail():
    split = SplitForcer(lambda block_k, k, grid: grid)
    calls = []
    def post(comb, res, post, x, out, hc, hidden):
        calls.append(("post", len(x)))
        out.copy_(res)
    def gemm(res, fn, **kwargs):
        calls.append(("gemm", len(res), split(64, res.shape[1], (len(res) + 63) // 64)))
        return torch.zeros(1, len(res), 24), torch.zeros(1, len(res))
    def pre(mul, sqr, scale, base, res, post, comb, x, *args, **kwargs):
        calls.append(("pre", len(res)))
        post.zero_(); comb.zero_(); x.zero_()
    k = NS(torch=torch, post=post, gemm=gemm, pre=pre)
    adapter = stock_post_pre_into(k, fn=torch.zeros(24, 32), scale=torch.zeros(3),
                                  base=torch.zeros(24), rms_eps=1e-5, hc_pre_eps=1e-6,
                                  hc_sinkhorn_eps=1e-6, hc_post_mult_value=2, sinkhorn_repeat=20)
    with split.full_batch(2049):
        for tokens in (512, 1):
            res = torch.ones(tokens, 4, 8, dtype=torch.bfloat16)
            out = SiteOutputs(torch.empty_like(res), torch.empty(tokens, 4, 1),
                              torch.empty(tokens, 4, 4), torch.empty(tokens, 8, dtype=torch.bfloat16))
            adapter(out.layer_input, res, out.post, out.comb, out)
    assert calls == [("post", 512), ("gemm", 512, 33), ("pre", 512),
                     ("post", 1), ("gemm", 1, 33), ("pre", 1)]


@pytest.mark.parametrize("case", ["ok", "custom", "missing_custom", "symm", "unknown_symm",
                                  "rocm", "missing_nccl", "disabled", "tp4", "missing_method"])
def test_sp_route_matches_pinned_pynccl_branch_or_declines(case):
    nccl = NS(world_size=2, available=True, disabled=False, _suspended=False,
              reduce_scatter=lambda *a: None, all_gather=lambda *a: None)
    dc = NS(world_size=2, ca_comm=None, pynccl_comm=nccl)
    symm, rocm = False, False
    if case == "custom": dc.ca_comm = object()
    if case == "missing_custom": del dc.ca_comm
    if case == "symm": symm = True
    if case == "unknown_symm": symm = None
    if case == "rocm": rocm = True
    if case == "missing_nccl": del dc.pynccl_comm
    if case == "disabled": nccl.disabled = True
    if case == "tp4": nccl.world_size = 4
    if case == "missing_method": del nccl.all_gather
    assert (pynccl_sp_decline(dc, symmetric_ag_rs=symm, rocm=rocm) is None) == (case == "ok")


@pytest.mark.parametrize("extra", [["--tokens", "8192"], ["--tile", "512"],
                                    ["--iterations", "65"], ["--pairs", "3"],
                                    ["--tokens", "2048", "2048"],
                                    ["--timeout-seconds", "181"],
                                    ["--control-iterations", "0"],
                                    ["--control-iterations", "129"]])
def test_gpu_probe_refuses_protocol_expansion(extra):
    from experiments.mhc.sp_tile_probe import parse_args
    base = ["--rank", "0", "--init-method", "tcp://127.0.0.1:39951",
            "--model", "/model", "--out", "/output"]
    with pytest.raises(SystemExit) as exc:
        parse_args(base + extra)
    assert exc.value.code == 2


def test_gpu_probe_retains_explicit_pair_and_finite_work():
    from experiments.mhc.sp_tile_probe import parse_args
    args = parse_args(["--rank", "1", "--init-method", "tcp://10.0.0.1:39951",
                       "--model", "/model", "--out", "/output"])
    assert (args.rank, args.tokens, args.tile, args.iterations, args.pairs,
            args.control_iterations) == (1, [512, 2048, 2049], 1024, 32, 2, 64)


def test_pynccl_binding_keeps_reduce_op_default_and_passes_stream_by_name():
    side, compute, calls = object(), object(), []
    def rs(out, x, op="SUM", stream=None):
        calls.append(("rs", out, x, op, stream))
    def ag(out, x, stream=None):
        calls.append(("ag", out, x, stream))
    cuda = NS(Stream=lambda: side, current_stream=lambda: compute, Event=lambda: None,
              is_current_stream_capturing=lambda: False)
    guard = NS(call=lambda fn, x: fn(x))
    runner = make_pynccl_runner(NS(cuda=cuda), NS(reduce_scatter=rs, all_gather=ag),
                               lambda t: None, guard=guard)
    runner.reduce_scatter_into("out", "in", side)
    runner.all_gather_into("out", "in", side)
    assert calls == [("rs", "out", "in", "SUM", side), ("ag", "out", "in", side)]


def test_unreadable_route_facts_decline():
    class Unreadable:
        @property
        def world_size(self):
            raise OSError("unreadable")
    assert pynccl_sp_decline(Unreadable(), symmetric_ag_rs=False) == "no TP2 device communicator"


def test_concrete_runner_uses_the_admitted_guard_before_each_pynccl_enqueue():
    calls = []
    cuda = NS(Stream=lambda: object(), current_stream=lambda: object(), Event=lambda: None,
              is_current_stream_capturing=lambda: False)
    nccl = NS(reduce_scatter=lambda *a, **k: calls.append("RS"),
              all_gather=lambda *a, **k: calls.append("AG"))
    def refuse(fn, x):
        raise RuntimeError("invalid writer scope")
    runner = make_pynccl_runner(NS(cuda=cuda), nccl, lambda t: None, guard=NS(call=refuse))
    for fn in (runner.reduce_scatter_into, runner.all_gather_into):
        with pytest.raises(RuntimeError, match="invalid writer scope"):
            fn("out", "input", runner.side)
    assert calls == []


def test_gpu_probe_disabled_writer_control_uses_both_rank_refusal_and_restores_scope():
    from experiments.mhc.sp_tile_probe import disabled_writer_refusal
    from tessera.serving.glm53_prefill import SpCollectiveGuard
    ranks = TwoRanks()
    owners = [NS(disabled=False), NS(disabled=False)]
    calls = []
    for rank, owner in enumerate(owners):
        owner.reduce_scatter = lambda *a, rank=rank, **k: calls.append(rank)
    # GPU allocation/stream names are translated only in this CPU fixture.
    # The actual initialized PyNccl writer remains a separate GPU acceptance.
    def allocate(fn):
        return lambda *a, **k: fn(*a, **{key: val for key, val in k.items() if key != "device"})
    fixture_torch = NS(full=allocate(torch.full), zeros=allocate(torch.zeros),
                       equal=torch.equal, uint8=torch.uint8, bfloat16=torch.bfloat16,
                       cuda=NS(synchronize=lambda: None, current_stream=lambda: None))
    def work(rank):
        ranks.local.rank = rank
        owner = owners[rank]
        guard = SpCollectiveGuard(route=lambda: "disabled" if owner.disabled else None,
                                  owner=lambda: owner, agree=ranks.max_across_tp,
                                  capturing=lambda: False)
        assert guard.available()
        return disabled_writer_refusal(fixture_torch, owner, guard, rank)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, rank) for rank in (0, 1)]
        results = [future.result(timeout=10) for future in futures]
    assert calls == []
    assert all(row["writer_invocations"] == 0 and row["poison_bytes_unchanged"]
               and not row["admitted"] and row["restored_and_readmitted"] for row in results)
    assert [row["this_rank_disabled"] for row in results] == [False, True]
    assert [owner.disabled for owner in owners] == [False, False]


def test_gpu_probe_refusal_control_never_invokes_writer_if_admission_is_broken():
    from experiments.mhc.sp_tile_probe import disabled_writer_refusal
    calls = []
    owner = NS(disabled=False, reduce_scatter=lambda *a, **k: calls.append("writer"))
    guard = NS(available=lambda: True, call=lambda fn, value: fn(value))
    def allocate(fn):
        return lambda *a, **k: fn(*a, **{key: val for key, val in k.items() if key != "device"})
    fixture_torch = NS(full=allocate(torch.full), zeros=allocate(torch.zeros),
                       uint8=torch.uint8, bfloat16=torch.bfloat16,
                       cuda=NS(synchronize=lambda: None))
    with pytest.raises(RuntimeError, match="disabled-peer writer was admitted"):
        disabled_writer_refusal(fixture_torch, owner, guard, 1)
    assert calls == [] and owner.disabled is False


def test_gpu_probe_activation_timing_counts_only_actual_agreements():
    from experiments.mhc.sp_tile_probe import activation_timings
    calls = []
    guard = NS(available=lambda: calls.append("agree") or True)
    row = activation_timings(guard, 4, lambda: calls.append("barrier"))
    assert calls == ["barrier"] + ["agree"] * 4
    assert row["iterations"] == len(row["host_ms_samples"]) == 4
    assert 0 <= row["host_ms_min"] <= row["host_ms_median"] <= row["host_ms_max"]
