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
    monkeypatch.setattr(mod, "_REPORTED", [])
    monkeypatch.setattr(mod, "_FIRST_QUERY", [])
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
    from tessera.serving.glm53_empty_rope import query_without_empty_rope, skip_reason

    nope, pe = _served_query()
    joined = torch.cat((nope, pe), dim=-1)
    passed = query_without_empty_rope((nope, pe))
    assert skip_reason((nope, pe)) is None
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
    from tessera.serving.glm53_empty_rope import skip_reason
    assert query_without_empty_rope(q) is q
    assert skip_reason(q)


def _lines(caplog, mod):
    return [r.getMessage() for r in caplog.records if r.name == mod.__name__]


def test_flag_unset_leaves_the_stock_method_and_says_off_once(empty_rope, monkeypatch, caplog):
    monkeypatch.delenv(empty_rope.mod.FLAG, raising=False)
    with caplog.at_level(logging.WARNING, logger=empty_rope.mod.__name__):
        assert empty_rope.mod.install_for_current_config() is False
        assert empty_rope.mod.install_for_current_config() is False
    assert empty_rope.impl.forward_mqa is empty_rope.original
    assert _lines(caplog, empty_rope.mod) == [
        "tessera.glm53_empty_rope: query cat skip off "
        "(TESSERA_GLM53_SKIP_EMPTY_ROPE_CAT unset or 0)"]


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


def test_install_is_idempotent_and_says_installed_once(empty_rope, monkeypatch, caplog):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    monkeypatch.setenv("TESSERA_CENSUS_RUNTIME_IMAGE", "localhost/x@sha256:5be13705acaecc7b4aaf")
    with caplog.at_level(logging.WARNING, logger=empty_rope.mod.__name__):
        assert empty_rope.mod.install_for_current_config() is True
        first = empty_rope.impl.forward_mqa
        assert empty_rope.mod.install_for_current_config() is True
    assert empty_rope.impl.forward_mqa is first
    assert first.__wrapped_stock__ is empty_rope.original
    digest = hashlib.sha256(empty_rope.source.read_bytes()).hexdigest()[:12]
    assert _lines(caplog, empty_rope.mod) == [
        f"tessera.glm53_empty_rope: query cat skip installed (stock source sha256 {digest}, "
        "image sha 5be13705acae): FlashInferMLASparseSM120Impl.forward_mqa passes a "
        "zero-width-RoPE query without torch.cat"]


def test_the_first_tuple_query_is_reported_once(empty_rope, monkeypatch, caplog):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    assert empty_rope.mod.install_for_current_config() is True
    nope, pe = _served_query()
    caplog.clear()  # the install line is the other test's subject
    with caplog.at_level(logging.WARNING, logger=empty_rope.mod.__name__):
        empty_rope.impl().forward_mqa(_aligned((8, 4, 16)), None, None, None)  # not a tuple
        empty_rope.impl().forward_mqa((nope, pe), None, None, None)
        empty_rope.impl().forward_mqa((nope, nope[..., :2].clone()), None, None, None)
    assert _lines(caplog, empty_rope.mod) == [
        "tessera.glm53_empty_rope: query cat skip: first tuple query "
        "[(8, 4, 16), (8, 4, 0)]: cat skipped"]


def test_an_uninspected_source_declines_with_one_warning(empty_rope, monkeypatch, caplog):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    empty_rope.source.write_text("# a different stock source\n")
    with caplog.at_level(logging.WARNING, logger=empty_rope.mod.__name__):
        assert empty_rope.mod.install_for_current_config() is False
        assert empty_rope.mod.install_for_current_config() is False
    assert empty_rope.impl.forward_mqa is empty_rope.original
    lines = _lines(caplog, empty_rope.mod)
    assert len(lines) == 1
    assert lines[0].startswith("tessera.glm53_empty_rope: query cat skip declined, stock "
                               "FlashInferMLASparseSM120Impl.forward_mqa: ")
    assert "is not an inspected source" in lines[0]


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


@pytest.mark.parametrize('fault',['missing_source','directory_source','invalid_path','missing_method','bad_signature','import_failure'])
def test_uninspectable_stock_declines_without_rebinding(empty_rope,monkeypatch,fault):
    mod=empty_rope.mod;monkeypatch.setenv(mod.FLAG,'1')
    stock=sys.modules[STOCK]
    if fault=='missing_source':empty_rope.source.unlink()
    elif fault=='directory_source':monkeypatch.setattr(stock,'__file__',str(empty_rope.source.parent))
    elif fault=='invalid_path':monkeypatch.setattr(stock,'__file__',object())
    elif fault=='missing_method':monkeypatch.delattr(empty_rope.impl,'forward_mqa')
    elif fault=='bad_signature':
        class BadSignature:
            def __call__(self,*args):pass
            @property
            def __signature__(self):raise ValueError('uninspectable stock method')
        monkeypatch.setattr(empty_rope.impl,'forward_mqa',BadSignature())
    else:
        def unavailable(_):raise RuntimeError('stock dependency failed during import')
        monkeypatch.setattr(mod.importlib,'import_module',unavailable)
    before=getattr(empty_rope.impl,'forward_mqa',None)
    assert mod.install_for_current_config() is False
    assert getattr(empty_rope.impl,'forward_mqa',None) is before


@pytest.mark.parametrize('exception',[OSError,ValueError,TypeError])
def test_unavailable_stock_class_declines_without_rebinding(empty_rope,monkeypatch,exception):
    mod=empty_rope.mod;monkeypatch.setenv(mod.FLAG,'1')
    stock=sys.modules[STOCK]
    class UnavailableClassModule(types.ModuleType):
        def __getattribute__(self,name):
            if name=='FlashInferMLASparseSM120Impl':raise exception('stock class unavailable')
            return super().__getattribute__(name)
    broken=UnavailableClassModule(stock.__name__);broken.__file__=stock.__file__
    monkeypatch.setitem(sys.modules,STOCK,broken)
    assert mod.install_for_current_config() is False


# CPU stand-ins establish probe refusal, not actual CUDA cache-writer arithmetic.
@pytest.mark.parametrize("failure", ["import", "writer", "completion"])
def test_qualification_cache_producer_failures_are_not_synthetic_passes(monkeypatch, failure):
    from experiments.t8r_speed import empty_rope_cat_check as probe

    monkeypatch.setattr(probe, "CONTEXT", probe.PAGE)
    error = RuntimeError("stock cache production failed")
    def writer(*args):
        if failure == "writer":
            raise error
    def synchronize():
        if failure == "completion":
            raise error
    runtime = types.ModuleType("vllm")
    runtime._custom_ops = types.SimpleNamespace(concat_and_cache_mla=writer)
    monkeypatch.setitem(sys.modules, "vllm", None if failure == "import" else runtime)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    log = {}
    expected = ModuleNotFoundError if failure == "import" else RuntimeError
    with pytest.raises(expected) as caught:
        probe.build_cache(torch.device("cpu"), log)
    if failure != "import":
        assert caught.value is error
    assert "cache_writer" not in log and "cache_sha256" not in log


def test_qualification_cache_keeps_stock_writer_bytes_and_completion_order(monkeypatch):
    from experiments.t8r_speed import empty_rope_cat_check as probe

    monkeypatch.setattr(probe, "CONTEXT", probe.PAGE)
    calls = []
    def writer(latent, rope, cache, slots, dtype, scale):
        assert latent.shape == (probe.CONTEXT, probe.LATENT)
        assert latent.dtype == torch.bfloat16
        assert rope.shape == (probe.CONTEXT, probe.ROPE_PAD)
        assert torch.count_nonzero(rope).item() == 0
        assert torch.equal(slots, torch.arange(probe.CONTEXT, dtype=torch.int64))
        assert dtype == "fp8_ds_mla" and torch.equal(scale, torch.ones(1))
        cache.fill_(17)
        calls.append("write")
    runtime = types.ModuleType("vllm")
    runtime._custom_ops = types.SimpleNamespace(concat_and_cache_mla=writer)
    monkeypatch.setitem(sys.modules, "vllm", runtime)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("complete"))
    log = {}
    cache = probe.build_cache(torch.device("cpu"), log)
    assert calls == ["write", "complete"]
    assert torch.equal(cache, torch.full_like(cache, 17))
    assert log["cache_writer"] == "vllm._custom_ops.concat_and_cache_mla(fp8_ds_mla)"
    assert log["cache_sha256"] == hashlib.sha256(cache.numpy().tobytes()).hexdigest()

