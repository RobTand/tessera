"""Source tensor geometry of a dense serving owner, shared by export and replay.

``tessera.metadata.fused_module`` owns the name rule for checkpoint tensors
merged into one Linear.

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

from dataclasses import dataclass
from typing import Mapping, Sequence

from .. import metadata


def source_padding_rows(module: str, members: Sequence[str], config: Mapping) -> dict[str, int]:
    """Apply the current GLM NoPE loader rule to its exact source member."""
    if not module.endswith((".self_attn.fused_qkv_a_proj", ".self_attn.kv_a_proj_with_mqa")):
        return {}
    text = config.get("text_config", config)
    if not text.get("mla_use_nope", text.get("mla_nope", False)):
        return {}
    pad = int(text.get("qk_rope_head_dim", 0))
    return {member: pad for member in members
            if member.endswith(".kv_a_proj_with_mqa.weight") and pad > 0}


@dataclass(frozen=True)
class Member:
    """One scheme role of a dense owner: a row window of one checkpoint tensor."""

    role: str
    tensor: str
    row_offset: int
    rows: int
    padding_rows: int = 0

    @property
    def source_rows(self) -> int:
        return self.rows - self.padding_rows


def partition_members(module: str, members: Sequence[str], tensor_rows: Mapping[str, int],
                      output_sizes: "Sequence[int] | None", *,
                      padding_rows: Mapping[str, int] | None = None) -> tuple[Member, ...]:
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
    padding_rows = dict(padding_rows or {})
    if set(padding_rows) - set(members) or any(int(n) < 0 for n in padding_rows.values()):
        raise ValueError(f"{module}: padding rows must name members with nonnegative counts")
    if not members:
        raise ValueError(f"{module}: a dense owner has at least one member tensor")
    for tensor in members:
        if tensor not in tensor_rows:
            raise ValueError(f"{module}: no row count for member {tensor}")
    if output_sizes is None:
        if any(padding_rows.values()):
            raise ValueError(f"{module}: padding rows require runtime output partitions")
        return tuple(Member(metadata.role_name(t), t, 0, int(tensor_rows[t])) for t in members)
    sizes = tuple(int(s) for s in output_sizes)
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"{module}: attested output partitions must be positive, got {list(sizes)}")
    if len(sizes) == len(members):
        out = []
        for tensor, size in zip(members, sizes):
            rows = int(tensor_rows[tensor])
            pad = int(padding_rows.get(tensor, 0))
            if rows + pad != size:
                raise ValueError(
                    f"{module}: member {tensor} has {rows} rows but the runtime builds its output "
                    f"partition with {size}; the checkpoint and the serve disagree about this "
                    f"module's geometry (attested partitions {list(sizes)})")
            out.append(Member(metadata.role_name(tensor), tensor, 0, size, pad))
        return tuple(out)
    if len(members) == 1:
        if any(padding_rows.values()):
            raise ValueError(f"{module}: padding rows cannot alter a sliced source tensor")
        tensor = members[0]
        rows = int(tensor_rows[tensor])
        if sum(sizes) != rows:
            raise ValueError(
                f"{module}: {tensor} has {rows} rows but the runtime's {len(sizes)} output "
                f"partitions {list(sizes)} sum to {sum(sizes)}; the checkpoint and the serve "
                "disagree about this module's geometry")
        leaf = metadata.role_name(tensor)
        out, offset = [], 0
        for index, size in enumerate(sizes):
            out.append(Member(f"{leaf}.{index}", tensor, offset, size))
            offset += size
        return tuple(out)
    raise ValueError(
        f"{module}: {len(members)} member tensors {list(members)} cannot be paired with the "
        f"runtime's {len(sizes)} output partitions {list(sizes)}; there is no rule that "
        "re-cuts several source tensors into a different number of partitions")
