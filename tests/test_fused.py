"""``tessera.fused``: one blob per vLLM module, one global per module."""
import pytest
import box_artifacts
import torch

from tessera.errors import GrammarError
from tessera.fused import (
    _HEADER,
    _MEMBER,
    FusedMember,
    pack_fused,
    parse_fused,
    shared_input_global_scale,
    shared_lut_global,
)


def test_container_round_trips_and_refuses_framing_errors():
    members = [("q_proj", 2048, b"\x01\x02\x03"), ("k_proj", 1024, b"\x04"), ("v_proj", 1024, b"\x05\x06")]
    blob = pack_fused(members)
    assert parse_fused(blob) == [FusedMember(*m) for m in members]
    with pytest.raises(GrammarError):
        parse_fused(blob[:-1])
    with pytest.raises(GrammarError):
        parse_fused(blob + b"\x00")
    with pytest.raises(GrammarError):
        parse_fused(b"GBTCQ1\0\0" + blob[8:])
    with pytest.raises(GrammarError):
        pack_fused([("a", 1, b"x"), ("a", 1, b"y")])


def _dequant(table_bytes, global_):
    return table_bytes.view(torch.float8_e4m3fn).float() * global_


def test_shared_global_is_exact_or_refused():
    # every byte an E4M3 normal (0x08..0x7E): 2^-4, 1.0, 2.0, 448.  A moved table
    # that lands outside that range is refused outright -- the fused decoder reads
    # a scale byte by field arithmetic, so a zero exponent field decodes wrong.
    t = torch.tensor([0x18, 0x38, 0x40, 0x7E], dtype=torch.uint8)
    # three roles two binades apart: the smallest multiplier wins and the others shift up
    tables, globals_ = [t[:3], t[:3], t[:3]], [2.0**-10, 2.0**-12, 2.0**-11]
    shared, moved = shared_lut_global(tables, globals_, ["q", "k", "v"])
    assert shared == 2.0**-12
    for tb, g, mv in zip(tables, globals_, moved):
        assert torch.equal(_dequant(mv, shared), _dequant(tb, g))
    # q holds 448 (cannot move up), k holds the smallest normal 2^-6 (cannot move
    # down -- 2^-8 is a subnormal byte): no candidate carries both, so the group
    # is refused with the failing roles named.  Moving DOWN is exact when the
    # entries stay on the E4M3 normal lattice, which is why a 448 alone is not
    # a refusal.
    q = torch.tensor([0x7E], dtype=torch.uint8)
    k = torch.tensor([0x08], dtype=torch.uint8)
    with pytest.raises(GrammarError, match="fails on q"):
        shared_lut_global([q, k], [2.0**-10, 2.0**-12], ["q", "k"])
    s2, m2 = shared_lut_global([t, t], [2.0**-10, 2.0**-12], ["q", "k"])
    assert s2 == 2.0**-10 and torch.equal(m2[0], t)          # q stays, k moves down exactly
    assert torch.equal(_dequant(m2[1], s2), _dequant(t, 2.0**-12))
    # already shared: untouched
    s, m = shared_lut_global([t, t], [0.5, 0.5])
    assert s == 0.5 and all(torch.equal(a, t) for a in m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_shared_global_agrees_with_stock_share_global_on_qwen():
    """The table rule and the byte rule pick the same global and the same tiles."""
    from safetensors import safe_open
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import DEFAULT_CODE, encode_linear_planes
    from tessera.stock import materialize_stock, share_global, stock_dequant

    K2 = tuple_grid(E2M1_GRID, 2)
    f = safe_open(str(box_artifacts.skip_now(
        "models", "Qwen3-0.6B", "model.safetensors")), "pt")
    roles = ["q_proj", "k_proj", "v_proj"]
    units, tensors = {}, {}
    for r in roles:
        w = f.get_tensor(f"model.layers.13.self_attn.{r}.weight").to("cuda", torch.float32).contiguous()
        _e, unit, forests = encode_linear_planes(w, grid=K2, q256=896, name=r, verify=False)
        units[r] = unit
        tensors[r] = materialize_stock(unit, forests, DEFAULT_CODE)
    rewritten, divisor = share_global(tensors)
    shared, moved = shared_lut_global([units[r].scale_lut for r in roles], [float(units[r].scale_global) for r in roles], roles)
    assert shared == pytest.approx(1.0 / divisor)
    for r, mv in zip(roles, moved):
        before = stock_dequant(tensors[r])
        table_values = mv.view(torch.float8_e4m3fn).float().to("cuda")
        nib = tensors[r]["weight_packed"]
        # rebuild the dequant from the moved table: the stock scale bytes are table[index]
        idx = units[r].scale_refine.reshape(before.shape[0], -1).long().to("cuda")
        scale = table_values[idx]
        codes = torch.stack([nib & 0xF, nib >> 4], dim=2).reshape(before.shape[0], -1).long()
        from tessera.stock import _nvfp4_values
        after = _nvfp4_values(codes).float() * scale.repeat_interleave(16, dim=1) * shared
        assert torch.equal(after, before), r


def test_a_units_blob_length_is_not_a_function_of_its_shape_grid_and_rate():
    """The MoE expert parameter cannot be ``[E, 2, nbytes]`` with one stride.

    Issue #5 planned a fused routed-expert parameter around the assumption that
    two units of the same ``(shape, grid, q256)`` serialise to the same number
    of bytes.  They do not, and the mechanism is here rather than in the body:
    ``ScalePlane.encode`` writes ``global_scale`` as an exact ``Fraction``
    through ``canonical.Writer.ratio`` (``manifest.py:312,314``), a varint pair
    -- and a varint's LENGTH is a function of its VALUE, while the global scale
    is a function of the DATA.

    MEASURED at 32x256, one seed, weights scaled by 2^k for k in [-14, +14]:
    the E4M3/window q1024 blob spans 21143..21146 bytes and the E2M1x2/TCQ q896
    blob spans 5222..5225, while ``exact_bytes`` -- which counts the plane
    region -- is 20544 and 4608 flat.  Real artifacts on this box carry
    ``global_scale`` 2^-10 and 2^-12, two exponents apart, so a body of experts
    at one shape genuinely lands on more than one length.  The encode is a
    minute of CPU per unit, so what is asserted here is the mechanism, at the
    manifest, in milliseconds; ``tests/test_wire.py`` owns the blob itself.

    The consequence for the expert stack: pad each member to a declared stride
    and carry its true length beside it.  ``parse_fused`` refuses trailing
    bytes -- deliberately, a container is exactly its members -- so a padded
    blob handed back without its length is a refusal, not a shorter read.
    """
    from fractions import Fraction

    from tessera.canonical import Writer
    from tessera.manifest import ScalePlane

    def encoded(global_scale):
        writer = Writer()
        ScalePlane.channel(global_scale).encode(writer)
        return writer.bytes

    lengths = {g: len(encoded(g)) for g in (2.0 ** -10, 2.0 ** -12, 2.0 ** -24)}
    assert len(set(lengths.values())) > 1, lengths
    # And it is the denominator's width that moves, not the kind byte.
    assert Fraction(2.0 ** -24).denominator.bit_length() > \
        Fraction(2.0 ** -10).denominator.bit_length()

    # The framing half: a container is exactly its members.
    blob = pack_fused([("gate_proj", 8, b"\x01\x02"), ("up_proj", 8, b"\x03")])
    with pytest.raises(GrammarError):
        parse_fused(blob + b"\x00" * 4)


def test_the_reader_refuses_a_member_its_writer_could_not_have_written():
    """`parse_fused` bounds-checked the member table and the blobs but not
    `name_len`, and never re-checked `rows > 0` the way `pack_fused` does on
    write.  A truncated name ran the cursor past the end and the framing check
    then reported the wrong thing one step later:

        GrammarError: fused member 'q_proj\x00...\x00': truncated blob

    -- a name that swallowed the payload, reported as a blob problem.
    """
    good = pack_fused([("q_proj", 8, b"\x00" * 12)])
    assert [m.name for m in parse_fused(good)] == ["q_proj"]

    # A name longer than the bytes that follow it.
    truncated = good[: _HEADER.size] + _MEMBER.pack(64, 8, 12) + good[_HEADER.size + _MEMBER.size :]
    with pytest.raises(GrammarError, match="name"):
        parse_fused(truncated)

    # rows = 0, which the writer refuses.
    zero_rows = good[: _HEADER.size] + _MEMBER.pack(6, 0, 12) + good[_HEADER.size + _MEMBER.size :]
    with pytest.raises(GrammarError, match="rows"):
        parse_fused(zero_rows)

    # A name that is not UTF-8 escaped the taxonomy as UnicodeDecodeError.
    bad_utf8 = bytearray(good)
    bad_utf8[_HEADER.size + _MEMBER.size] = 0xFF
    with pytest.raises(GrammarError, match="UTF-8"):
        parse_fused(bytes(bad_utf8))


def test_the_fused_a_side_scale_is_the_min_member_scale():
    """``input_global_scale`` is capacity/amax -- inverse in the activation
    range -- and the fused GEMM's one input tensor spans every member's range,
    so the join is the MIN member scale (the largest calibrated amax).  A max
    would pick the smallest range and clip the rest."""
    assert shared_input_global_scale([4.0], ["q"]) == 4.0
    ulp = 2.0 ** -7
    assert shared_input_global_scale(
        [4.0, 4.0 * (1.0 + ulp), 4.0], ["q", "k", "v"]) == 4.0


def test_a_member_scale_the_route_would_refuse_is_refused_here_by_name():
    """The same predicate the NVFP4 route's load gate applies (finite and
    positive -- ``not (nan > 0)`` catches the unloaded sentinel), applied
    where the bytes are decided, naming the member."""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(GrammarError, match="k_proj"):
            shared_input_global_scale([4.0, bad], ["q_proj", "k_proj"])
    with pytest.raises(GrammarError, match="at least one"):
        shared_input_global_scale([], [])
    with pytest.raises(GrammarError, match="one name per scale"):
        shared_input_global_scale([4.0, 4.0], ["q_proj"])


def test_a_spread_beyond_one_bf16_ulp_is_two_calibrations_and_refused():
    """The bound is declared policy (#283), pinned here so it cannot drift
    by hand: one bf16 ULP (2^-7, the dtype's eps), the lattice the route
    casts the A tensor to before the quantiser sees it.  Exactly one ULP
    passes; anything wider is two calibrations and refuses by name."""
    from tessera.fused import FUSED_INPUT_SCALE_ULP
    assert FUSED_INPUT_SCALE_ULP == 2.0 ** -7 == torch.finfo(torch.bfloat16).eps
    ulp = 2.0 ** -7
    assert shared_input_global_scale(
        [4.0, 4.0 * (1.0 + ulp)], ["q", "k"]) == 4.0
    with pytest.raises(GrammarError, match="bf16") as caught:
        shared_input_global_scale(
            [4.0, 4.0 * (1.0 + 2.0 ** -6)], ["q_proj", "k_proj"])
    message = str(caught.value)
    assert "q_proj=4" in message and "k_proj=4.0625" in message
    with pytest.raises(GrammarError, match="two calibrations"):
        shared_input_global_scale([6.0 / 8.0, 6.0 / 4.0], ["gate", "up"])
