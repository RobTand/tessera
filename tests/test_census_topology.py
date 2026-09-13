"""The topology a census is taken at, checked before the first model load.

THE DEFECT THIS PINS.  ``tools/tessera_route_census.py`` built ``LLM(...)``
with no topology argument, so every census ran at world size 1 and no receipt
could name a larger world.  Adding the arguments is only half the fix: a census
is two model loads of 85-160 s each, so a topology mistake caught after the
load costs the run.  ``topology_kwargs`` is therefore the whole rule set, it
runs off the parsed arguments alone, and the parser turns a refusal into a
usage error.

THE FAIL-BEFORE.  On the pre-change tree ``tessera.serving.topology`` does not
exist, so every test here fails at import.
"""
from __future__ import annotations

import argparse

import pytest

from tessera.serving.topology import (
    DISTRIBUTED_EXECUTOR_BACKENDS, add_topology_arguments, topology_kwargs, topology_record,
    validate_topology_arguments)


def _parser():
    parser = argparse.ArgumentParser(prog="census")
    add_topology_arguments(parser)
    return parser


def _args(*argv):
    return _parser().parse_args(list(argv))


def test_default_topology_is_todays_single_process_census():
    """No topology argument must reach the engine with the kwargs it always did."""
    assert topology_kwargs(_args()) == {"tensor_parallel_size": 1}


def test_every_argument_the_issue_names_is_accepted_and_passed_through():
    kwargs = topology_kwargs(_args(
        "--tensor-parallel-size", "2", "--distributed-executor-backend", "mp",
        "--nnodes", "2", "--node-rank", "0",
        "--master-addr", "10.100.96.1", "--master-port", "29500"))
    assert kwargs == {"tensor_parallel_size": 2, "distributed_executor_backend": "mp",
                      "nnodes": 2, "node_rank": 0,
                      "master_addr": "10.100.96.1", "master_port": 29500}


def test_ray_takes_its_rendezvous_from_the_cluster():
    """A ray census names no master address: vLLM never reads one there."""
    kwargs = topology_kwargs(_args("--tensor-parallel-size", "2",
                                   "--nnodes", "2", "--distributed-executor-backend", "ray"))
    assert kwargs == {"tensor_parallel_size": 2, "nnodes": 2,
                      "distributed_executor_backend": "ray"}


def test_single_node_needs_no_backend():
    assert topology_kwargs(_args("--tensor-parallel-size", "2")) == {"tensor_parallel_size": 2}


@pytest.mark.parametrize("argv, reason", [
    (("--tensor-parallel-size", "0"), "positive"),
    (("--tensor-parallel-size", "-2"), "positive"),
    (("--tensor-parallel-size", "2", "--nnodes", "0"), "positive"),
    (("--tensor-parallel-size", "3", "--nnodes", "2",
      "--distributed-executor-backend", "ray"), "divide"),
    (("--tensor-parallel-size", "2", "--node-rank", "1"), "node_rank 0"),
    (("--tensor-parallel-size", "2", "--nnodes", "2"), "distributed_executor_backend"),
    (("--tensor-parallel-size", "2", "--nnodes", "2",
      "--distributed-executor-backend", "mp"), "master_addr"),
    (("--tensor-parallel-size", "2", "--nnodes", "2", "--distributed-executor-backend", "mp",
      "--master-addr", "10.100.96.1"), "master_port"),
    (("--tensor-parallel-size", "2", "--master-addr", "   "), "host address"),
    (("--tensor-parallel-size", "2", "--master-port", "70000"), "1..65535"),
    (("--tensor-parallel-size", "2", "--master-port", "0"), "1..65535"),
])
def test_a_topology_that_cannot_be_served_is_refused(argv, reason):
    with pytest.raises(ValueError) as excinfo:
        topology_kwargs(_args(*argv))
    assert reason in str(excinfo.value)


def test_the_parser_refuses_before_the_engine_exists():
    """The refusal is a usage error, not a traceback two model loads in."""
    parser = _parser()
    args = parser.parse_args(["--tensor-parallel-size", "3", "--nnodes", "2",
                              "--distributed-executor-backend", "ray"])
    with pytest.raises(SystemExit) as excinfo:
        validate_topology_arguments(parser, args)
    assert excinfo.value.code == 2


def test_an_unknown_backend_never_reaches_the_engine():
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--distributed-executor-backend", "external_launcher"])
    assert DISTRIBUTED_EXECUTOR_BACKENDS == ("mp", "ray")


def test_the_record_publishes_the_request_including_what_was_not_asked_for():
    """The receipt carries the REQUEST; the world size is counted from ranks."""
    record = topology_record(_args("--tensor-parallel-size", "2",
                                   "--nnodes", "2", "--distributed-executor-backend", "ray"))
    assert record == {"requested_tensor_parallel_size": 2, "nnodes": 2, "node_rank": None,
                      "master_addr": None, "master_port": None,
                      "distributed_executor_backend": "ray"}
    assert "observed_world_size" not in record


def test_the_group_does_not_collide_with_the_census_tool_arguments():
    """Adding the group to the real parser must not shadow an existing flag."""
    parser = argparse.ArgumentParser(prog="census")
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--expect-modules", type=int, default=None)
    add_topology_arguments(parser)
    args = parser.parse_args(["/ckpt", "/out.json", "--expect-modules", "112",
                              "--tensor-parallel-size", "2"])
    assert args.expect_modules == 112
    assert topology_kwargs(args) == {"tensor_parallel_size": 2}
