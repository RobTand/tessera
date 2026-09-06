"""Source tensor membership of a dense serving owner, shared by export and replay.

Two rules live here and they answer different questions.

``fused_module`` is the NAME rule: which checkpoint tensors vLLM merges into
one Linear, from the tensor names alone.  It requires no tensor runtime.

``partition_members`` is the GEOMETRY rule: how that Linear's rows are cut into
the output partitions the runtime builds -- ``ColumnParallelLinear.output_sizes``,
the list ``create_weights`` later receives as ``output_partition_sizes`` and
the list ``sharding.plan_shard`` pairs with the declared roles by position.
The two lists are the same list only when the exporter derives its roles from
the runtime's, so the partition list is read off the construction census
(``contract.output_partitions``), never guessed from the names: LFM2's
``ShortConv`` builds ``in_proj`` as ``MergedColumnParallelLinear(output_sizes=
[dim] * 3)`` from ONE checkpoint tensor, and a one-role container for it is a
checkpoint the runtime refuses at load (tessera#377).  vLLM itself cuts that
tensor by row in ``output_sizes`` order (``MergedColumnParallelLinear.
weight_loader`` with ``loaded_shard_id=None``), which is the order used here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj)\.weight$"), "qkv_proj", ("q_proj", "k_proj", "v_proj")),
    # NOT scoped to ``.mlp.``: a shared expert is its own MLP module
    # (``...mlp.shared_experts.{gate,up,down}_proj`` in the checkpoint) and vLLM
    # merges ITS gate/up too -- ``Glm5NextMLP`` takes ``prefix=f"{prefix}.gate_up_proj"``
    # for whatever prefix it is built at (glm5next/nvidia/model.py:124, :216).
    # Naming the unmerged leaves there declares two modules vLLM never builds
    # and leaves the one it does build undeclared, which the plugin refuses at
    # load.  The lookahead keeps ROUTED experts out: their gate/up merge into
    # ``w13`` inside the FusedMoE, which is a different mechanism entirely.
    (re.compile(r"^(?!.*\.experts\.\d+\.)(.*\.)(gate_proj|up_proj)\.weight$"), "gate_up_proj", ("gate_proj", "up_proj")),
    # Lfm2MoeMlp uses w1/w3 source leaves for its dense gate/up pair and
    # constructs one w13 Linear. Routed experts have an intervening
    # .experts.INDEX segment and remain owned by the expert stack rule.
    (re.compile(r"^(.*\.feed_forward\.)(w1|w3)\.weight$"), "w13", ("w1", "w3")),
)


def fused_module(tensor_name: str):
    """``(fused module name, ordered member tensor names)`` or ``None``."""
    for pattern, fused, members in FUSED:
        match = pattern.match(tensor_name)
        if match:
            return match.group(1) + fused, tuple(match.group(1) + m + ".weight" for m in members)
    return None


def role_name(tensor_name: str) -> str:
    """The scheme role a whole checkpoint tensor is declared under: its leaf."""
    module = tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else tensor_name
    return module.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class Member:
    """One scheme role of a dense owner: a row window of one checkpoint tensor."""

    role: str
    tensor: str
    row_offset: int
    rows: int


def partition_members(module: str, members: Sequence[str], tensor_rows: Mapping[str, int],
                      output_sizes: "Sequence[int] | None") -> tuple[Member, ...]:
    """The roles of ``module`` in the runtime's stacking order.

    ``members`` are the checkpoint tensors ``fused_module`` (or the tensor
    itself) says the module is built from, in stacking order; ``tensor_rows``
    their row counts; ``output_sizes`` the runtime's attested output partition
    list for this module, or ``None`` when no census recorded one.

    * Unattested: one role per tensor, whole rows -- what every checkpoint
      before tessera#377 declared.  The caller records that the geometry was
      not checked.
    * As many partitions as tensors: the roles are the tensors, and each
      tensor's rows must be its partition's size.  A tensor that is not is a
      checkpoint/serve disagreement caught here rather than at load.
    * One tensor, several partitions (``MergedColumnParallelLinear`` over one
      source tensor): the tensor is cut by row in partition order, and the
      roles are ``<leaf>.<index>`` -- an index, because an index is what the
      runtime hands the loader.
    * Anything else is refused: several tensors cannot be re-paired with a
      different number of partitions without a rule this module does not have.
    """
    members = tuple(str(m) for m in members)
    if not members:
        raise ValueError(f"{module}: a dense owner has at least one member tensor")
    for tensor in members:
        if tensor not in tensor_rows:
            raise ValueError(f"{module}: no row count for member {tensor}")
    if output_sizes is None:
        return tuple(Member(role_name(t), t, 0, int(tensor_rows[t])) for t in members)
    sizes = tuple(int(s) for s in output_sizes)
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"{module}: attested output partitions must be positive, got {list(sizes)}")
    if len(sizes) == len(members):
        out = []
        for tensor, size in zip(members, sizes):
            rows = int(tensor_rows[tensor])
            if rows != size:
                raise ValueError(
                    f"{module}: member {tensor} has {rows} rows but the runtime builds its output "
                    f"partition with {size}; the checkpoint and the serve disagree about this "
                    f"module's geometry (attested partitions {list(sizes)})")
            out.append(Member(role_name(tensor), tensor, 0, rows))
        return tuple(out)
    if len(members) == 1:
        tensor = members[0]
        rows = int(tensor_rows[tensor])
        if sum(sizes) != rows:
            raise ValueError(
                f"{module}: {tensor} has {rows} rows but the runtime's {len(sizes)} output "
                f"partitions {list(sizes)} sum to {sum(sizes)}; the checkpoint and the serve "
                "disagree about this module's geometry")
        leaf = role_name(tensor)
        out, offset = [], 0
        for index, size in enumerate(sizes):
            out.append(Member(f"{leaf}.{index}", tensor, offset, size))
            offset += size
        return tuple(out)
    raise ValueError(
        f"{module}: {len(members)} member tensors {list(members)} cannot be paired with the "
        f"runtime's {len(sizes)} output partitions {list(sizes)}; there is no rule that "
        "re-cuts several source tensors into a different number of partitions")
