"""CPU stand-in for skipping the zero-width RoPE query cat in stock sparse MLA.

The bitwise claim on the real kernel is a GPU receipt
(``experiments/t8r_speed/empty_rope_cat_check.py``). These tests pin the
host-side contract: which queries skip the cat, that everything else reaches
the stock code unchanged, and that installation runs from the production
quant-config entry, only when opted in, and only on an inspected source.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import types

import pytest
import torch

import test_serving_dispatch as dispatch
from test_serving_dispatch import TARGET, TESSERA_MODE_ENV, _config, _layer
from test_serving_dispatch import runtime_modules as _runtime_modules
from test_serving_dispatch import _the_platform_these_tests_are_about  # noqa: F401  (autouse)

runtime_modules = _runtime_modules  # pytest discovers the isolated runtime fixture

STOCK = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120"


def _aligned(shape, dtype=torch.bfloat16, misalign=0):
    """A contiguous tensor starting on a 512-byte boundary, plus ``misalign`` elements."""
    numel = 1
    for size in shape:
        numel *= size
    item = torch.empty((), dtype=dtype).element_size()
    buf = torch.empty(numel + 1024 // item, dtype=dtype)
    start = ((-buf.data_ptr()) % 512) // item + misalign
    out = buf[start:start + numel].view(shape)
    out.copy_(torch.randn(shape).to(dtype))
    return out


def _served_query(tokens=8, heads=4, latent=16):
    nope = _aligned((tokens, heads, latent))
    return nope, nope.new_empty((tokens, heads, 0))


@pytest.fixture
def empty_rope(monkeypatch, tmp_path):
    """The module under test with a fresh flag latch and a stand-in stock backend."""
    from tessera.serving import flags, glm53_empty_rope as mod

    flags.reset_for_tests(mod.FLAG)
    monkeypatch.setattr(mod, "_DECLINED", [])
    calls = []

    class FlashInferMLASparseSM120Impl:
        def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
            if isinstance(q, tuple):  # the stock cat
                q = torch.cat(q, dim=-1)
            calls.append(q)
            return q, None

    source = tmp_path / "flashinfer_mla_sparse_sm120.py"
    source.write_text("# stand-in for the inspected stock source\n")
    stock = types.ModuleType(STOCK)
    stock.__file__ = str(source)
    stock.FlashInferMLASparseSM120Impl = FlashInferMLASparseSM120Impl
    parts = STOCK.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, STOCK, stock)
    monkeypatch.setattr(mod, "_INSPECTED_SHA256",
                        frozenset({hashlib.sha256(source.read_bytes()).hexdigest()}))
    yield types.SimpleNamespace(mod=mod, impl=FlashInferMLASparseSM120Impl,
                                original=FlashInferMLASparseSM120Impl.forward_mqa,
                                calls=calls, source=source)
    flags.reset_for_tests(mod.FLAG)


def test_the_served_query_form_skips_the_cat_with_identical_bytes():
    from tessera.serving.glm53_empty_rope import query_without_empty_rope

    nope, pe = _served_query()
    joined = torch.cat((nope, pe), dim=-1)
    passed = query_without_empty_rope((nope, pe))
    assert passed is nope
    assert passed.shape == joined.shape and passed.stride() == joined.stride()
    assert torch.equal(passed.view(torch.int16), joined.view(torch.int16))


@pytest.mark.parametrize("case", [
    "rope_not_empty", "dtype_mismatch", "leading_shape_mismatch", "not_contiguous",
    "misaligned", "three_parts", "list", "tensor"])
def test_every_other_query_reaches_stock_unchanged(case):
    from tessera.serving.glm53_empty_rope import query_without_empty_rope

    nope, pe = _served_query()
    q = {
        "rope_not_empty": lambda: (nope, nope[..., :1].clone()),
        "dtype_mismatch": lambda: (nope, pe.float()),
        "leading_shape_mismatch": lambda: (nope, nope.new_empty((7, 4, 0))),
        "not_contiguous": lambda: (_aligned((4, 8, 16)).transpose(0, 1), pe),
        "misaligned": lambda: (_aligned((8, 4, 16), misalign=1), pe),
        "three_parts": lambda: (nope, pe, pe),
        "list": lambda: [nope, pe],
        "tensor": lambda: nope,
    }[case]()
    assert query_without_empty_rope(q) is q


def test_flag_unset_leaves_the_stock_method(empty_rope, monkeypatch):
    monkeypatch.delenv(empty_rope.mod.FLAG, raising=False)
    assert empty_rope.mod.install_for_current_config() is False
    assert empty_rope.impl.forward_mqa is empty_rope.original


def test_the_production_entry_installs_it_and_q_nope_reaches_stock(
        runtime_modules, empty_rope, monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    config = dispatch.TesseraConfig.from_config(_config())
    config.get_quant_method(_layer(), TARGET)
    wrapped = empty_rope.impl.forward_mqa
    assert wrapped is not empty_rope.original
    assert wrapped.__wrapped_stock__ is empty_rope.original

    nope, pe = _served_query()
    out, _ = empty_rope.impl().forward_mqa((nope, pe), None, None, None)
    assert out is nope  # no cat ran
    rope = nope[..., :2].clone()
    out, _ = empty_rope.impl().forward_mqa((nope, rope), None, None, None)
    assert torch.equal(out, torch.cat((nope, rope), dim=-1))  # stock cat ran
    tensor = _aligned((8, 4, 16))
    out, _ = empty_rope.impl().forward_mqa(tensor, None, None, None)
    assert out is tensor


def test_install_is_idempotent(empty_rope, monkeypatch):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    assert empty_rope.mod.install_for_current_config() is True
    first = empty_rope.impl.forward_mqa
    assert empty_rope.mod.install_for_current_config() is True
    assert empty_rope.impl.forward_mqa is first
    assert first.__wrapped_stock__ is empty_rope.original


def test_an_uninspected_source_declines_with_one_warning(empty_rope, monkeypatch, caplog):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    empty_rope.source.write_text("# a different stock source\n")
    with caplog.at_level(logging.WARNING, logger=empty_rope.mod.__name__):
        assert empty_rope.mod.install_for_current_config() is False
        assert empty_rope.mod.install_for_current_config() is False
    assert empty_rope.impl.forward_mqa is empty_rope.original
    declined = [r for r in caplog.records if "declined" in r.getMessage()]
    assert len(declined) == 1 and "not an inspected source" in declined[0].getMessage()


def test_a_changed_signature_declines(empty_rope, monkeypatch):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")

    def forward_mqa(self, q, kv_cache, attn_metadata, layer):
        return q, None

    monkeypatch.setattr(empty_rope.impl, "forward_mqa", forward_mqa)
    assert empty_rope.mod.install_for_current_config() is False
    assert empty_rope.impl.forward_mqa is forward_mqa


def test_an_absent_backend_declines(empty_rope, monkeypatch):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    monkeypatch.setitem(sys.modules, STOCK, None)
    assert empty_rope.mod.install_for_current_config() is False
