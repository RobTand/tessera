"""The fused-MoE expert parameter layout: variable-length wires in dense rows.

Step 3 of issue #5, and deliberately only the layout: no ``ROUTES`` entry, no
``apply``, no kernel.  vLLM's ``RoutedExperts.build_expert_params_mapping`` is
suffix-agnostic, so a ``w13_wire`` / ``w2_wire`` pair routes through the same
mapping the stock weights do; what blocked the route is that the wire blob's
length is not a function of ``(shape, grid, q256)``.  The manifest writes the
``global_scale`` as an exact varint ratio whose width follows the value, which
follows the data -- 4215545 against 4215544 bytes on the issue's pair, while
``exact_bytes`` is flat -- so ``[E, 2, nbytes]`` with one stride is bytes that
fit all but one row.  Each row is therefore padded to a declared stride and
its true length rides beside it, because ``fused.parse_fused`` refuses
trailing bytes and a padded blob handed back whole is a refusal there rather
than a shorter read.

Each cell holds one projection's wire as a fused container -- gate and up
under ``w13``, down under ``w2`` -- so the future loader reuses
``parse_fused`` and the per-role scheme checks unchanged.  This module is
agnostic to what the cells contain; it only promises that what comes out is
byte-for-byte what went in, sliced so ``parse_fused`` sees no padding.

The strides are the maxima over the blobs being packed, derived per pack, so
a stride is never a constant with slack in it.  Unpacking refuses, each by
name: a length past its row's end, a length tensor whose shape disagrees with
the expert count or the projection count, and a stride that is not what the
lengths imply.  Padding is zero -- the dtype's own zero, the only fill that
needs no value -- and is never read back.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .errors import GrammarError
from .serving.scheme import MOE_GROUP_ROLES

__all__ = [
    "W13_PROJECTIONS",
    "MoePacked",
    "pack_moe_wires",
    "unpack_moe_wires",
]

#: Projections fused into the w13 group: gate and up.  A mechanism count, not
#: a round number -- vLLM merges an expert's gate/up into one ``w13`` matrix
#: inside the FusedMoE (the exporter's fused rule keeps routed experts out of
#: the dense gate/up merge for exactly this reason), while down rides ``w2``
#: alone.  A group that is not a pair is a different mechanism, not a wider one.
#: DERIVED from the runtime's own shard table rather than restated beside it.
W13_PROJECTIONS = MOE_GROUP_ROLES["w13"]


@dataclass(frozen=True)
class MoePacked:
    """One MoE layer's wires as dense tensors plus their true lengths.

    ``w13_wire`` is ``uint8 [E, 2, S13]`` holding gate (``[:, 0, :]``) and up
    (``[:, 1, :]``), with ``w13_wire_len [E, 2]`` saying how many of each
    row's bytes are real.  ``w2_wire`` is ``uint8 [E, S2]`` holding down, with
    ``w2_wire_len [E]``.  The strides ``S13``/``S2`` are the maxima over the
    blobs packed -- see ``pack_moe_wires`` -- so unpacking refuses a stride
    that is not what the lengths imply rather than serving slack as room.
    """

    w13_wire: torch.Tensor
    w13_wire_len: torch.Tensor
    w2_wire: torch.Tensor
    w2_wire_len: torch.Tensor

    @property
    def experts(self) -> int:
        """The expert count every tensor of the packing agrees on."""
        return int(self.w13_wire.shape[0])


def _require_blob(blob, expert: int, projection: int, group: str) -> bytes:
    if isinstance(blob, (bytes, bytearray)):
        blob = bytes(blob)
    else:
        raise GrammarError(
            f"moe {group} expert {expert} projection {projection}: a wire is bytes, "
            f"got {type(blob).__name__}")
    if not blob:
        raise GrammarError(
            f"moe {group} expert {expert} projection {projection}: an empty wire packs "
            "nothing -- fused.parse_fused refuses an empty member, so it would never "
            "unpack to anything a loader accepts")
    return blob


def pack_moe_wires(
    w13_blobs: "list[list[bytes]]",
    w2_blobs: "list[bytes]",
) -> MoePacked:
    """Pack one layer's expert wires into dense rows at a derived stride.

    ``w13_blobs[e]`` is the expert's ``[gate, up]`` pair, ``w2_blobs[e]`` its
    down wire.  Returns CPU ``uint8`` rows padded with zero to the max over
    the blobs packed -- ``S13`` over every gate/up blob, ``S2`` over every
    down blob -- with companion ``long`` length tensors.  The max is the whole
    rule: no constant, no slack, no second argument to disagree with the data.
    """
    experts = len(w13_blobs)
    if experts == 0:
        raise GrammarError("moe pack: no experts -- an [E, ...] tensor with E=0 names no layer")
    if len(w2_blobs) != experts:
        raise GrammarError(
            f"moe pack: {experts} w13 expert(s) but {len(w2_blobs)} w2 wire(s); the two "
            "groups describe the same experts and must agree")
    gate_up: "list[list[bytes]]" = []
    for expert, pair in enumerate(w13_blobs):
        if not isinstance(pair, (list, tuple)) or len(pair) != W13_PROJECTIONS:
            got = len(pair) if isinstance(pair, (list, tuple)) else type(pair).__name__
            raise GrammarError(
                f"moe w13 expert {expert}: a w13 group is a [gate, up] pair "
                f"({W13_PROJECTIONS} wires), got {got}")
        gate_up.append([_require_blob(blob, expert, proj, "w13")
                        for proj, blob in enumerate(pair)])
    down = [_require_blob(blob, expert, 0, "w2") for expert, blob in enumerate(w2_blobs)]

    stride13 = max(len(blob) for expert in gate_up for blob in expert)
    stride2 = max(len(blob) for blob in down)
    w13_wire = torch.zeros(experts, W13_PROJECTIONS, stride13, dtype=torch.uint8)
    w13_wire_len = torch.zeros(experts, W13_PROJECTIONS, dtype=torch.long)
    for expert, pair in enumerate(gate_up):
        for proj, blob in enumerate(pair):
            w13_wire[expert, proj, :len(blob)] = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
            w13_wire_len[expert, proj] = len(blob)
    w2_wire = torch.zeros(experts, stride2, dtype=torch.uint8)
    w2_wire_len = torch.zeros(experts, dtype=torch.long)
    for expert, blob in enumerate(down):
        w2_wire[expert, :len(blob)] = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        w2_wire_len[expert] = len(blob)
    return MoePacked(w13_wire=w13_wire, w13_wire_len=w13_wire_len,
                     w2_wire=w2_wire, w2_wire_len=w2_wire_len)


def _check_lengths_shape(packed: MoePacked) -> int:
    """The expert count, or a refusal naming which tensor disagrees with what."""
    wire, lengths = packed.w13_wire, packed.w13_wire_len
    w2_wire, w2_lengths = packed.w2_wire, packed.w2_wire_len
    for name, tensor, rank in (("w13_wire", wire, 3), ("w13_wire_len", lengths, 2),
                               ("w2_wire", w2_wire, 2), ("w2_wire_len", w2_lengths, 1)):
        if not isinstance(tensor, torch.Tensor):
            raise GrammarError(f"moe {name}: a packed tensor, got {type(tensor).__name__}")
        if tensor.dim() != rank:
            raise GrammarError(
                f"moe {name}: rank {tensor.dim()}, expected {rank} "
                f"(shape {tuple(tensor.shape)})")
    if not isinstance(wire, torch.Tensor) or wire.dtype != torch.uint8:
        raise GrammarError(f"moe w13_wire: uint8 wire bytes, got {wire.dtype}")
    if not isinstance(w2_wire, torch.Tensor) or w2_wire.dtype != torch.uint8:
        raise GrammarError(f"moe w2_wire: uint8 wire bytes, got {w2_wire.dtype}")
    experts = wire.shape[0]
    if experts <= 0:
        raise GrammarError("moe w13_wire: no experts -- an [E, ...] tensor with E=0 names no layer")
    if wire.shape[1] != W13_PROJECTIONS:
        raise GrammarError(
            f"moe w13_wire: {wire.shape[1]} projections, expected {W13_PROJECTIONS} "
            "(gate and up)")
    if tuple(lengths.shape) != (experts, W13_PROJECTIONS):
        raise GrammarError(
            f"moe w13_wire_len: shape {tuple(lengths.shape)} for {experts} experts and "
            f"{W13_PROJECTIONS} projections, expected ({experts}, {W13_PROJECTIONS})")
    if w2_wire.shape[0] != experts or tuple(w2_lengths.shape) != (experts,):
        raise GrammarError(
            f"moe w2 companions: w2_wire {tuple(w2_wire.shape)} and w2_wire_len "
            f"{tuple(w2_lengths.shape)} for {experts} experts, expected ([{experts}, S2], "
            f"[{experts}])")
    return experts


def unpack_moe_wires(packed: MoePacked) -> "tuple[list[list[bytes]], list[bytes]]":
    """Recover one layer's expert wires byte-for-byte, sliced so ``parse_fused`` fits.

    Each ``[e, p, :length]`` prefix is returned exactly -- no padding -- so the
    slice handed to ``fused.parse_fused`` downstream ends where the packed blob
    ended.  Refuses, each by name: a length past its row's declared stride, a
    length tensor whose shape disagrees with the expert count or the projection
    count, and a stride that is not the max its lengths imply.
    """
    experts = _check_lengths_shape(packed)
    stride13, stride2 = packed.w13_wire.shape[2], packed.w2_wire.shape[1]

    def length(group: str, expert: int, proj: "int | None", value) -> int:
        where = f"expert {expert} projection {proj}" if proj is not None else f"expert {expert}"
        number = int(value)
        if number <= 0:
            raise GrammarError(
                f"moe {group} {where}: wire length {number} -- an empty wire unpacks to "
                "nothing fused.parse_fused accepts, so it was never packable")
        return number

    w13_lengths = [[length("w13", e, p, packed.w13_wire_len[e, p])
                    for p in range(W13_PROJECTIONS)] for e in range(experts)]
    w2_lengths = [length("w2", e, None, packed.w2_wire_len[e]) for e in range(experts)]
    for expert in range(experts):
        for proj in range(W13_PROJECTIONS):
            if w13_lengths[expert][proj] > stride13:
                raise GrammarError(
                    f"moe w13 expert {expert} projection {proj}: wire length "
                    f"{w13_lengths[expert][proj]} is longer than the declared stride "
                    f"{stride13} -- the row ends before the blob does, so the blob is "
                    "truncated data, not a shorter read")
    for expert in range(experts):
        if w2_lengths[expert] > stride2:
            raise GrammarError(
                f"moe w2 expert {expert}: wire length {w2_lengths[expert]} is longer than "
                f"the declared stride {stride2} -- the row ends before the blob does, so "
                "the blob is truncated data, not a shorter read")
    implied13, implied2 = max(n for row in w13_lengths for n in row), max(w2_lengths)
    if stride13 != implied13:
        raise GrammarError(
            f"moe w13 stride {stride13} is not what its lengths imply ({implied13}): "
            "the stride is the max over the packed blobs, so a declared stride beside it "
            "is a wrong tensor, not room")
    if stride2 != implied2:
        raise GrammarError(
            f"moe w2 stride {stride2} is not what its lengths imply ({implied2}): the stride "
            "is the max over the packed blobs, so a declared stride beside it is a wrong "
            "tensor, not room")

    back13 = [[_blob_bytes(packed.w13_wire[e, p, :w13_lengths[e][p]]) for p in range(W13_PROJECTIONS)]
              for e in range(experts)]
    back2 = [_blob_bytes(packed.w2_wire[e, :w2_lengths[e]]) for e in range(experts)]
    return back13, back2


def _blob_bytes(wire: torch.Tensor) -> bytes:
    """One uint8 row as the container bytes it holds, copied once.

    ``bytes(t.tolist())`` boxed every byte as a Python int first -- tens of
    millions of them on a 22-stack x 32-expert x 3-projection load -- for the
    same bytes a single contiguous copy yields.
    """
    return wire.detach().cpu().contiguous().numpy().tobytes()
