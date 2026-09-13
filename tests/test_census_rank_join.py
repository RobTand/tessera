"""One record per rank, and a joined record that is their sum.

THE DEFECT THIS PINS.  A route record says what a module executed.  It does not
say which rank's shard executed it, which box that rank ran on, or how many
ranks there were -- and ``tools/tessera_route_census.py`` read
``llm.apply_model(...)[0]``, so every census described rank 0 and a receipt of
rank 0 alone reads identically at world size 1 and at world size 8.

Two values close that.  ``rank_census_record`` is the missing half of a route
record: who observed it and where.  ``join_rank_histograms`` is what the
attestation reads, and it SUMS: every rank builds a module for every declared
target and names it identically, so a union would report one served module as
one module no matter how many ranks served it.

THE ONE THING THE SUM MUST NOT CHANGE.  At a single rank the join is the
identity, because a single-rank receipt is quoted, diffed and re-read, and the
same observation must be the same bytes.

THE FAIL-BEFORE.  On the pre-change tree ``tessera.serving.census`` exports
neither function, so every test here fails at import.
"""
from __future__ import annotations

import pytest

from tessera.serving.census import (
    RANK_CENSUS_SCHEMA, join_rank_histograms, phase_histogram, rank_census_record)


def _records(names, *, decoder="window_gemv", shape="M1:N1024:K1024",
             policy="TESSERA_FP8:streamed", state="served"):
    """One phase's Tessera records, in the shape ``read_route`` returns."""
    return {name: {"kind": "dense", "policy": policy, "symbol": "torch._scaled_mm",
                   "tile_m": 0, "shape": shape, "contract": "fp8_per_token_dynamic",
                   "state": state, "reason": None, "decoder": decoder} for name in names}


def _shard(rank, n=4, **kwargs):
    """One rank's ``{phase: histogram}``; every rank names its modules alike."""
    names = [f"model.layers.{i}.mlp.down_proj" for i in range(n)]
    return {"prefill": phase_histogram(_records(names, shape="M8:N512:K1024", **kwargs),
                                       regime="batch", other_route_modules=1),
            "decode": phase_histogram(_records(names, shape="M1:N512:K1024", **kwargs),
                                      regime="decode", other_route_modules=1)}


# --- the record shape -------------------------------------------------------

def test_a_rank_record_carries_who_observed_the_routes_and_where():
    histogram = _shard(0)["decode"]
    record = rank_census_record(
        rank=1, world_size=2, node="sparklina", platform_token="sm_121",
        runtime_image="eugr/spark-vllm@sha256:" + "0" * 64, device="cuda:0", local_rank=0,
        histogram=histogram, lane_refusals={"m.0": "no toolchain"},
        records={"decode": _records(["m.0"])})
    assert record["schema"] == RANK_CENSUS_SCHEMA
    assert record["rank"] == 1 and record["world_size"] == 2 and record["local_rank"] == 0
    assert record["node"] == "sparklina" and record["device"] == "cuda:0"
    assert record["platform_token"] == "sm_121"
    assert record["runtime_image"].startswith("eugr/spark-vllm@sha256:")
    assert record["histogram"] == histogram
    assert record["lane_refusals"] == {"m.0": "no toolchain"}
    assert set(record["records"]) == {"decode"}


def test_a_rank_record_is_complete_without_the_optional_halves():
    record = rank_census_record(rank=0, world_size=1, node="sparky", platform_token="sm_121",
                                runtime_image=None, histogram={})
    assert record["runtime_image"] is None and record["device"] is None
    assert record["local_rank"] is None
    assert record["lane_refusals"] == {} and record["records"] == {}


@pytest.mark.parametrize("kwargs, reason", [
    ({"rank": -1, "world_size": 2}, "non-negative"),
    ({"rank": 0, "world_size": 0}, "positive"),
    ({"rank": 2, "world_size": 2}, "outside a world"),
    ({"rank": 0, "world_size": 1, "node": "  "}, "names the node"),
    ({"rank": 0, "world_size": 1, "platform_token": ""}, "platform token"),
])
def test_a_rank_record_that_names_no_world_is_refused(kwargs, reason):
    fields = {"rank": 0, "world_size": 1, "node": "sparky", "platform_token": "sm_121",
              "runtime_image": None, "histogram": {}}
    fields.update(kwargs)
    with pytest.raises(ValueError) as excinfo:
        rank_census_record(**fields)
    assert reason in str(excinfo.value)


def test_a_bool_is_not_a_rank():
    """``True == 1`` in Python; a rank read off a flag is not a rank."""
    with pytest.raises(ValueError):
        rank_census_record(rank=True, world_size=2, node="sparky", platform_token="sm_121",
                           runtime_image=None, histogram={})


# --- the join ---------------------------------------------------------------

def test_one_rank_joins_to_itself_byte_for_byte():
    """The single-rank receipt must be exactly what it was before the join."""
    shard = _shard(0)
    assert join_rank_histograms([shard]) == shard


def test_the_joined_histogram_is_the_sum_over_ranks():
    joined = join_rank_histograms([_shard(0), _shard(1)])
    assert set(joined) == {"prefill", "decode"}
    for phase, regime in (("prefill", "batch"), ("decode", "decode")):
        assert joined[phase]["regime"] == regime
        assert joined[phase]["tessera_modules"] == 8
        assert joined[phase]["tessera_modules_by_family"] == {"TESSERA_FP8": 8}
        assert joined[phase]["other_route_modules"] == 2
        assert sum(r["modules"] for r in joined[phase]["routes"]) == 8
        assert [r["modules"] for r in joined[phase]["routes"]] == [8]


def test_ranks_that_took_different_routes_are_counted_separately():
    """A rank that fell back is visible in the sum; a union would hide it."""
    joined = join_rank_histograms([_shard(0), _shard(1, decoder="torch_window")])
    routes = {r["decoder"]: r["modules"] for r in joined["decode"]["routes"]}
    assert routes == {"window_gemv": 4, "torch_window": 4}


def test_a_column_parallel_shards_own_shape_survives_the_join():
    """Shapes are a set union: rank 1's N is its own, and a receipt says so."""
    joined = join_rank_histograms([
        _shard(0), {phase: phase_histogram(
            _records([f"model.layers.{i}.mlp.down_proj" for i in range(4)],
                     shape=f"{'M8' if phase == 'prefill' else 'M1'}:N256:K1024"),
            regime="batch" if phase == "prefill" else "decode", other_route_modules=1)
            for phase in ("prefill", "decode")}])
    assert joined["decode"]["shapes"] == ["M1:N256:K1024", "M1:N512:K1024"]


def test_the_join_refuses_ranks_that_drove_different_phases():
    one = _shard(0)
    other = {"decode": _shard(1)["decode"]}
    with pytest.raises(ValueError) as excinfo:
        join_rank_histograms([one, other])
    assert "not one observation" in str(excinfo.value)


def test_the_join_refuses_one_phase_driven_at_two_regimes():
    one = _shard(0)
    other = _shard(1)
    other["decode"] = dict(other["decode"], regime="batch")
    with pytest.raises(ValueError) as excinfo:
        join_rank_histograms([one, other])
    assert "one phase is one regime" in str(excinfo.value)


def test_the_join_refuses_a_histogram_whose_routes_do_not_add_up():
    """The sum is the attestation; a miscounted rank must not pass silently."""
    broken = _shard(0)
    broken["decode"] = dict(broken["decode"], tessera_modules=9)
    with pytest.raises(ValueError) as excinfo:
        join_rank_histograms([broken])
    assert "sum over ranks" in str(excinfo.value)


def test_zero_ranks_is_not_a_census():
    with pytest.raises(ValueError) as excinfo:
        join_rank_histograms([])
    assert "zero ranks" in str(excinfo.value)


def test_the_joined_route_order_does_not_depend_on_the_order_ranks_answered():
    ranks = [_shard(0), _shard(1, decoder="torch_window"), _shard(2, decoder="cuda_window")]
    forward = join_rank_histograms(ranks)
    backward = join_rank_histograms(list(reversed(ranks)))
    assert forward == backward
