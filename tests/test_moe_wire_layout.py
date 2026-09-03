"""The fused-MoE expert parameter layout: variable-length wires in dense rows.

Step 3 of issue #5.  vLLM's ``RoutedExperts.build_expert_params_mapping`` is
suffix-agnostic, so custom suffixes route fine; the obstruction is the wire.
Two units at the same ``(shape, grid, q256)`` do not serialise to the same
byte count -- the ``global_scale`` rides the manifest as a varint pair whose
length is a function of its value, which is a function of the data
(``tests/test_fused.py`` pins the mechanism) -- while ``exact_bytes`` is flat.
So an ``[E, 2, nbytes]`` expert parameter cannot be one stride: each row is
padded to a declared stride and its true length rides beside it, because
``fused.parse_fused`` refuses trailing bytes and a padded blob handed back
whole is a refusal, not a shorter read.

The layout is ``tessera.moe_layout`` and nothing else: no ``ROUTES`` entry,
no ``apply``, no kernel.  Each cell holds one projection's wire as a fused
container (one member here, so the loader reuses ``parse_fused`` and the
per-role scheme checks unchanged), and the companions say how many of each
row's bytes are real: ``w13_wire [E, 2, S13]`` with ``w13_wire_len [E, 2]``
for gate/up, ``w2_wire [E, S2]`` with ``w2_wire_len [E]`` for down.  The
strides are the maxima over the blobs packed -- derived, never a constant.

The round-trip test encodes real units on CPU rather than synthesising bytes:
at 32x64, E4M3 q1024, scaling the source by powers of two moves the blob
across six distinct lengths, so the variable-length premise and the 1-byte
difference the issue measured are both in the fixture by construction.
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import E4M3_GRID                                  # noqa: E402
from tessera.errors import GrammarError                                 # noqa: E402
from tessera.export import encode_linear_planes                         # noqa: E402
from tessera.fused import pack_fused, parse_fused                       # noqa: E402
from tessera.moe_layout import (                                        # noqa: E402
    MoePacked, pack_moe_wires, unpack_moe_wires)

EXPERTS = 4
GATE_ROWS, HIDDEN = 32, 64
DOWN_ROWS, INTER = 64, 32
Q256 = 1024


def _unit_blob(rows: int, cols: int, scale: float, seed: int) -> bytes:
    """One real unit artifact: a CPU encode whose global scale follows ``scale``."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator) * scale
    exported, _unit, _forests = encode_linear_planes(
        weight.contiguous(), grid=E4M3_GRID, q256=Q256,
        name=f"expert-{seed}", verify=False)
    return exported.blob


def _expert_fixture():
    """``(w13_blobs, w2_blobs)``: E experts of real unit wires in 1-member containers.

    Each cell is ``pack_fused`` of the projection's own unit artifact, so an
    unpacked slice is something ``parse_fused`` must accept.  The per-blob
    data scale walks powers of two, which is what moves the manifest's varint
    global scale across byte lengths at fixed shape, grid and rung.
    """
    w13, w2, scales = [], [], []
    for expert in range(EXPERTS):
        gate = _unit_blob(GATE_ROWS, HIDDEN, 2.0 ** (-9 + expert), seed=100 + expert)
        up = _unit_blob(GATE_ROWS, HIDDEN, 2.0 ** (3 - expert), seed=200 + expert)
        down = _unit_blob(DOWN_ROWS, INTER, 2.0 ** (-5 + expert), seed=300 + expert)
        scales.append((-9 + expert, 3 - expert, -5 + expert))
        w13.append([pack_fused([("gate_proj", GATE_ROWS, gate)]),
                    pack_fused([("up_proj", GATE_ROWS, up)])])
        w2.append(pack_fused([("down_proj", DOWN_ROWS, down)]))
    return w13, w2


def test_variable_length_wires_pack_and_unpack_byte_exact():
    """The layout's premise and its promise, both on real wires.

    Premise: the eight w13 blobs genuinely differ in length, including a pair
    exactly one byte apart -- the difference the issue measured on real data.
    Promise: every blob comes back byte-for-byte, and ``parse_fused`` accepts
    every unpacked slice, which is what fails if a slice carries padding.
    """
    w13, w2 = _expert_fixture()
    lengths = sorted(len(blob) for expert in w13 for blob in expert)
    assert len(set(lengths)) > 1, (
        f"the premise moved: {len(lengths)} real wires at one (shape, grid, q256) "
        f"serialised to one length {lengths[0]}; the layout exists because they do not")
    gaps = [b - a for a, b in zip(lengths, lengths[1:]) if b != a]
    assert gaps and min(gaps) == 1, (
        f"no two wires differ by the issue's 1 byte: lengths {lengths}")

    packed = pack_moe_wires(w13, w2)
    assert isinstance(packed, MoePacked)
    assert packed.w13_wire.shape == (EXPERTS, 2, max(lengths))
    assert packed.w13_wire_len.shape == (EXPERTS, 2)
    assert packed.w2_wire.shape[0] == EXPERTS and packed.w2_wire_len.shape == (EXPERTS,)

    back13, back2 = unpack_moe_wires(packed)
    assert len(back13) == EXPERTS and len(back2) == EXPERTS
    for expert in range(EXPERTS):
        for proj in range(2):
            assert back13[expert][proj] == w13[expert][proj], (expert, proj)
            members = parse_fused(back13[expert][proj])
            assert [(m.name, m.rows) for m in members] == (
                [("gate_proj", GATE_ROWS)] if proj == 0 else [("up_proj", GATE_ROWS)])
        assert back2[expert] == w2[expert]
        assert [(m.name, m.rows) for m in parse_fused(back2[expert])] == [
            ("down_proj", DOWN_ROWS)]


def _tiny_pack():
    """A small valid packing the refusal tests mutate one field at a time."""
    w13 = [[b"gate-bytes-e0", b"up-bytes-e0!"],
           [b"gate-e1", b"up-e1-longer"]]
    w2 = [b"down-e0", b"down-e1-longest"]
    return pack_moe_wires(w13, w2), w13, w2


def test_unpack_refuses_a_blob_longer_than_the_declared_stride():
    """A length past the row's end is truncated data, and says so by name."""
    packed, _w13, _w2 = _tiny_pack()
    stride = packed.w13_wire.shape[2]
    bad_len = packed.w13_wire_len.clone()
    bad_len[1, 0] = stride + 1
    with pytest.raises(GrammarError, match=r"expert 1 projection 0.*longer than.*stride"):
        unpack_moe_wires(dataclasses.replace(packed, w13_wire_len=bad_len))


def test_unpack_refuses_a_length_shape_that_disagrees_with_the_expert_count():
    """The companions index experts, so a row more or less is a refusal, not a zip."""
    packed, _w13, _w2 = _tiny_pack()
    bad_len = torch.zeros(3, 2, dtype=torch.long)
    with pytest.raises(GrammarError, match=r"w13_wire_len.*\(3, 2\).*2 experts"):
        unpack_moe_wires(dataclasses.replace(packed, w13_wire_len=bad_len))


def test_unpack_refuses_a_length_shape_that_disagrees_with_the_projection_count():
    """w13 carries gate AND up: a length row that is not a pair names the wrong fault."""
    packed, _w13, _w2 = _tiny_pack()
    bad_len = packed.w13_wire_len[:, :1].clone()
    with pytest.raises(GrammarError, match=r"w13_wire_len.*projection"):
        unpack_moe_wires(dataclasses.replace(packed, w13_wire_len=bad_len))


def test_unpack_refuses_a_stride_beyond_what_the_lengths_imply():
    """The stride is the max over the packed blobs; slack is a wrong tensor, not room."""
    packed, _w13, _w2 = _tiny_pack()
    implied = int(packed.w13_wire_len.max())
    padded = torch.zeros(packed.w13_wire.shape[0], 2, implied + 4, dtype=torch.uint8)
    padded[:, :, :packed.w13_wire.shape[2]] = packed.w13_wire
    with pytest.raises(GrammarError, match=r"w13.*stride.*is not what.*lengths imply"):
        unpack_moe_wires(dataclasses.replace(packed, w13_wire=padded))
