"""``tessera.fused``: one blob per vLLM module, one global per module."""
import pytest
import torch

from tessera.errors import GrammarError
from tessera.fused import FusedMember, pack_fused, parse_fused, shared_lut_global


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
    f = safe_open("/home/rob/models/Qwen3-0.6B/model.safetensors", "pt")
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
