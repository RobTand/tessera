"""The topology a census is taken at, as arguments and as engine kwargs.

A route census used to be able to observe one rank, because
``tools/tessera_route_census.py`` built ``LLM(...)`` with no topology argument
at all.  Everything above one rank -- a column-parallel shard's route, a rank
whose lane refused to prepare while its twin's prepared, a world size a
contract cell wants to name -- was unobservable, and an unobservable fact is
one no receipt can carry.  So the census takes the topology the same way
PrismaQuant's gold coordinator does (``tools/gold_engine_options.py``): the
arguments are declared, validated before the first model load, and passed
through to the engine unchanged.

Three rules the validation keeps, each because the alternative is a receipt
about a world nobody served:

* **The census runs at node rank 0.**  It is the process that holds the model
  and reads every worker's record back through ``collective_rpc``; the other
  nodes are stock headless workers.  A census launched at rank 1 would load a
  second engine beside the one under test.
* **``nnodes`` must divide ``tensor_parallel_size``.**  Pipeline, data and
  context parallel stay 1 here -- this tool drives one forward per regime, and
  a split it does not drive is a rank whose records it would report as missing.
* **A multi-node census names its backend explicitly**, and an ``mp`` one also
  names the master address and port.  Ray takes those from the cluster the
  driver already joined, so requiring them there would require values vLLM
  never reads.

Torch-free and vLLM-free on purpose: the tool imports this before it imports a
serving runtime, and the tests exercise it with no GPU at all.
"""
from __future__ import annotations

import argparse

__all__ = [
    "DISTRIBUTED_EXECUTOR_BACKENDS",
    "TOPOLOGY_ARGUMENT_NAMES",
    "add_topology_arguments",
    "topology_kwargs",
    "topology_record",
    "validate_topology_arguments",
]

#: The backends this census can be driven under.  ``mp`` is vLLM's own
#: multi-node launcher (``--nnodes/--node-rank/--master-addr/--master-port``),
#: ``ray`` the cluster the driver joined before it built the engine.
DISTRIBUTED_EXECUTOR_BACKENDS = ("mp", "ray")

#: The optional arguments, in the order they reach the engine.  Named once so
#: the parser, the kwargs and the receipt cannot drift apart.
TOPOLOGY_ARGUMENT_NAMES = (
    "nnodes", "node_rank", "master_addr", "master_port",
    "distributed_executor_backend",
)


def add_topology_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the topology group to a census parser.

    The defaults are today's single-process census exactly: tensor parallel 1
    and every other value unset, so a command line written before this group
    existed reaches the engine with the kwargs it always did.
    """
    group = parser.add_argument_group("census topology")
    group.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="ranks this census observes; the receipt carries one record per "
                            "rank and a joined record whose histogram is their sum")
    group.add_argument("--distributed-executor-backend",
                       choices=DISTRIBUTED_EXECUTOR_BACKENDS, default=None,
                       help="required above one node: mp launches vLLM's own multi-node "
                            "workers, ray uses the cluster this driver already joined")
    group.add_argument("--nnodes", type=int, default=None,
                       help="nodes the world spans; must divide --tensor-parallel-size")
    group.add_argument("--node-rank", type=int, default=None,
                       help="the census runs on rank 0, which holds the model; launch the "
                            "other nodes with stock `vllm serve --headless`")
    group.add_argument("--master-addr", default=None,
                       help="rendezvous address for the mp backend (ray takes its own)")
    group.add_argument("--master-port", type=int, default=None,
                       help="rendezvous port for the mp backend (ray takes its own)")


def topology_kwargs(args: argparse.Namespace) -> dict:
    """The engine kwargs this topology asks for, validated before any load.

    Omission preserves the original single-process kwargs: an argument left
    unset is absent from the result rather than defaulted here, so the engine
    keeps whatever default it publishes.

    Raises ``ValueError`` with the reason, which the parser turns into a usage
    error -- before the first model load, because a topology mistake found
    after two 85-160 s loads is a mistake found at the cost of the run.
    """
    result = {"tensor_parallel_size": getattr(args, "tensor_parallel_size", 1)}
    for name in TOPOLOGY_ARGUMENT_NAMES:
        value = getattr(args, name, None)
        if value is not None:
            result[name] = value
    tp, nodes = result["tensor_parallel_size"], result.get("nnodes", 1)
    for name, value in (("tensor_parallel_size", tp), ("nnodes", nodes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if tp % nodes:
        raise ValueError("nnodes must evenly divide tensor_parallel_size "
                         "(pipeline, data and context parallel stay 1)")
    rank = result.get("node_rank", 0)
    if type(rank) is not int or rank != 0:
        raise ValueError("the census runs on node_rank 0, which holds the model; launch the "
                         "other nodes with stock `vllm serve --headless`")
    backend = result.get("distributed_executor_backend")
    if backend is not None and backend not in DISTRIBUTED_EXECUTOR_BACKENDS:
        raise ValueError(f"distributed_executor_backend must be one of "
                         f"{list(DISTRIBUTED_EXECUTOR_BACKENDS)}")
    if "master_addr" in result and (
            not isinstance(result["master_addr"], str) or not result["master_addr"].strip()):
        raise ValueError("master_addr must be a nonempty host address")
    if "master_port" in result and (
            type(result["master_port"]) is not int or not 1 <= result["master_port"] <= 65535):
        raise ValueError("master_port must be an integer in 1..65535")
    if nodes > 1:
        required = ["distributed_executor_backend"]
        if backend != "ray":
            # Ray reads the rendezvous from the cluster this driver joined.
            required += ["master_addr", "master_port"]
        missing = [name for name in required if name not in result]
        if missing:
            raise ValueError(f"a census above one node requires explicit {', '.join(missing)}")
    return result


def validate_topology_arguments(parser: argparse.ArgumentParser,
                                args: argparse.Namespace) -> None:
    """Turn an invalid topology into a usage error, before the engine exists."""
    try:
        topology_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))


def topology_record(args: argparse.Namespace) -> dict:
    """What the operator ASKED for, for the receipt.

    This is the request, never the observation: the world size a receipt
    attests is counted from the ranks that answered, not from the number typed
    here.  Both are published so a disagreement between them is readable.
    """
    kwargs = topology_kwargs(args)
    record = {"requested_tensor_parallel_size": kwargs["tensor_parallel_size"]}
    for name in TOPOLOGY_ARGUMENT_NAMES:
        record[name] = kwargs.get(name)
    return record
