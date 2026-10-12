"""CPU stand-in for skipping the zero-width RoPE query cat in stock sparse MLA.

The bitwise claim on the real kernel is a GPU receipt
(``experiments/t8r_speed/empty_rope_cat_check.py``). These tests pin the
host-side contract: which queries skip the cat, that everything else reaches
the stock code unchanged, and that installation runs from the production
quant-config entry, only when opted in, and only on an inspected source.
"""
from __future__ import annotations

import hashlib
import json
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


@pytest.mark.parametrize('exception',[OSError,ValueError,TypeError,RuntimeError])
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


@pytest.mark.parametrize("fault", ["source", "method", "signature"])
def test_stock_inspection_runtime_errors_decline_without_rebinding(empty_rope, monkeypatch, caplog, fault):
    mod = empty_rope.mod
    monkeypatch.setenv(mod.FLAG, "1")
    stock = sys.modules[STOCK]
    if fault == "source":
        class UnreadableSource(types.ModuleType):
            def __getattribute__(self, name):
                if name == "__file__":
                    raise RuntimeError("unavailable source")
                return super().__getattribute__(name)
        broken = UnreadableSource(STOCK)
        broken.__dict__.update(stock.__dict__)
        monkeypatch.setitem(sys.modules, STOCK, broken)
        stock = broken
    elif fault == "method":
        class UnavailableMethod(type):
            def __getattribute__(self, name):
                if name == "forward_mqa":
                    raise RuntimeError("unavailable method")
                return super().__getattribute__(name)
        impl = UnavailableMethod("FlashInferMLASparseSM120Impl", (),
                                 {"forward_mqa": empty_rope.original})
        monkeypatch.setattr(stock, "FlashInferMLASparseSM120Impl", impl)
    else:
        class UninspectableMethod:
            def __call__(self, *args):
                pass

            @property
            def __signature__(self):
                raise RuntimeError("unavailable signature")
        monkeypatch.setattr(empty_rope.impl, "forward_mqa", UninspectableMethod())
    impl = stock.__dict__["FlashInferMLASparseSM120Impl"]
    before = dict(impl.__dict__)
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        assert mod.install_for_current_config() is False
        assert mod.install_for_current_config() is False
    assert dict(impl.__dict__) == before
    lines = _lines(caplog, mod)
    assert len(lines) == 1 and "declined" in lines[0] and "RuntimeError" in lines[0]


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


@pytest.mark.parametrize("case", [
    "first_nontensor", "second_nontensor", "both_scalars", "first_scalar",
    "second_scalar", "rank_mismatch", "device_mismatch", "empty_tuple", "one_part",
])
def test_query_eligibility_rejects_nontensors_scalars_rank_and_device_mismatches(case):
    from tessera.serving.glm53_empty_rope import query_without_empty_rope, skip_reason

    nope, pe = _served_query()
    q = {
        "first_nontensor": lambda: (object(), pe),
        "second_nontensor": lambda: (nope, object()),
        "both_scalars": lambda: (torch.tensor(1), torch.tensor(0)),
        "first_scalar": lambda: (torch.tensor(1), pe),
        "second_scalar": lambda: (nope, torch.tensor(0)),
        "rank_mismatch": lambda: (nope, nope.new_empty((8, 0))),
        "device_mismatch": lambda: (nope, torch.empty(pe.shape, dtype=pe.dtype, device="meta")),
        "empty_tuple": lambda: (),
        "one_part": lambda: (nope,),
    }[case]()
    assert isinstance(skip_reason(q), str) and skip_reason(q)
    assert query_without_empty_rope(q) is q


def test_query_eligibility_accepts_tuple_and_tensor_subclasses_and_rank_one():
    from tessera.serving.glm53_empty_rope import query_without_empty_rope, skip_reason

    class Query(tuple):
        pass

    class Tensor(torch.Tensor):
        pass

    nope = _aligned((16,)).as_subclass(Tensor)
    pe = torch.empty(0, dtype=nope.dtype).as_subclass(Tensor)
    q = Query((nope, pe))
    assert skip_reason(q) is None
    assert query_without_empty_rope(q) is nope


def test_installation_target_and_inspected_revision_are_the_pinned_contract():
    from tessera.serving import glm53_empty_rope as mod

    assert mod._MODULE == STOCK
    assert mod._CLASS == "FlashInferMLASparseSM120Impl"
    assert mod._INSPECTED_SHA256 == {
        "102ca08793d567f95598eefeb34c3f6ec50b3b9d704f162d1402b97b12b5777b"}


@pytest.mark.parametrize("value, enabled", [
    (None, False), ("", False), ("0", False), ("1", True),
    (" \t\n", False), (" 0\n", False), ("\t1 ", True),
])
def test_empty_rope_flag_spellings_and_latch(empty_rope, monkeypatch, value, enabled):
    mod = empty_rope.mod
    if value is None:
        monkeypatch.delenv(mod.FLAG, raising=False)
    else:
        monkeypatch.setenv(mod.FLAG, value)
    assert mod.install_for_current_config() is enabled
    before = empty_rope.impl.forward_mqa
    normalized = "" if value is None else value.strip()
    monkeypatch.setenv(mod.FLAG, f" \t{normalized}\n")
    assert mod.install_for_current_config() is enabled
    monkeypatch.setenv(mod.FLAG, "0" if normalized != "0" else "")
    with pytest.raises(RuntimeError, match=mod.FLAG):
        mod.install_for_current_config()
    assert empty_rope.impl.forward_mqa is before


@pytest.mark.parametrize("value", ["true", "false", "on", "off", "yes", "no", "2", "-1", "01", " 1 0 "])
def test_empty_rope_flag_invalid_spellings_name_the_flag(empty_rope, monkeypatch, value):
    monkeypatch.setenv(empty_rope.mod.FLAG, value)
    with pytest.raises(ValueError, match=empty_rope.mod.FLAG):
        empty_rope.mod.install_for_current_config()
    assert empty_rope.impl.forward_mqa is empty_rope.original


@pytest.mark.parametrize("form", ["eligible", "ineligible", "nontuple"])
@pytest.mark.parametrize("raises", [False, True])
def test_wrapper_preserves_arguments_results_and_exceptions(empty_rope, monkeypatch, form, raises):
    mod = empty_rope.mod
    nope, pe = _served_query()
    q = {"eligible": (nope, pe), "ineligible": (nope, nope[..., :1]), "nontuple": nope}[form]
    cache, metadata, layer, result = object(), object(), object(), object()
    error = RuntimeError("stock forward failed")
    calls = []

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        calls.append((self, q, kv_c_and_k_pe_cache, attn_metadata, layer))
        if raises:
            raise error
        return result

    monkeypatch.setattr(empty_rope.impl, "forward_mqa", forward_mqa)
    monkeypatch.setenv(mod.FLAG, "1")
    assert mod.install_for_current_config() is True
    assert empty_rope.impl.forward_mqa.__wrapped_stock__ is forward_mqa
    instance = empty_rope.impl()
    if raises:
        with pytest.raises(RuntimeError) as caught:
            instance.forward_mqa(q, cache, metadata, layer)
        assert caught.value is error
    else:
        assert instance.forward_mqa(q, cache, metadata, layer) is result
    assert len(calls) == 1
    expected = (instance, nope if form == "eligible" else q, cache, metadata, layer)
    assert all(actual is wanted for actual, wanted in zip(calls[0], expected))


def test_first_ineligible_tuple_reports_kept_reason_once(empty_rope, monkeypatch, caplog):
    monkeypatch.setenv(empty_rope.mod.FLAG, "1")
    assert empty_rope.mod.install_for_current_config() is True
    nope, pe = _served_query()
    q = (nope, nope[..., :2].clone())
    reason = empty_rope.mod.skip_reason(q)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=empty_rope.mod.__name__):
        instance = empty_rope.impl()
        instance.forward_mqa(nope, None, None, None)
        assert _lines(caplog, empty_rope.mod) == []
        instance.forward_mqa(q, None, None, None)
        instance.forward_mqa((nope, pe), None, None, None)
        instance.forward_mqa(q, None, None, None)
    assert _lines(caplog, empty_rope.mod) == [
        "tessera.glm53_empty_rope: query cat skip: first tuple query "
        f"[(8, 4, 16), (8, 4, 2)]: cat kept ({reason})"]


@pytest.mark.parametrize("fault", [
    None, "changed_first_direct", "changed_second_cat", "changed_second_direct",
    "helper_clone", "helper_second_call_clone", "helper_third_call_clone",
])
def test_qualification_receipt_refuses_changed_output_or_nonidentity_helper(monkeypatch, fault):
    from experiments.t8r_speed import empty_rope_cat_check as probe

    cache, workspace, mla = object(), object(), object()
    positions = torch.tensor([8191, 8191])
    indices = torch.tensor([[0, 1, -1], [0, 1, -1]], dtype=torch.int32)
    calls, queries = [], []

    def build_indices(actual_positions, dev, seed):
        assert actual_positions is positions and dev == torch.device("cpu") and seed == 4
        return indices

    def helper(q):
        queries.append(q)
        clone = (fault == "helper_clone"
                 or fault == "helper_second_call_clone" and len(queries) == 2
                 or fault == "helper_third_call_clone" and len(queries) == 3)
        return q[0].clone() if clone else q[0]

    def run_kernel(q, actual_cache, actual_indices, actual_workspace, actual_mla):
        assert actual_cache is cache and actual_indices is indices
        assert actual_workspace is workspace and actual_mla is mla
        calls.append(q)
        out = q.clone()
        changed_run = {"changed_first_direct": 2, "changed_second_cat": 3,
                       "changed_second_direct": 4}.get(fault)
        if len(calls) == changed_run:
            out.view(torch.int16)[0, 0, 0] ^= 1
        return out

    monkeypatch.setattr(probe, "build_indices", build_indices)
    monkeypatch.setattr(probe, "run_kernel", run_kernel)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    receipt = probe.check_case("cpu_control", positions, cache, workspace, mla,
                               helper, torch.device("cpu"), 3)
    assert len(calls) == 4 and len(queries) == 3
    assert all(q is queries[0] for q in queries)
    assert calls[0] is not queries[0][0] and calls[2] is not queries[0][0]
    assert queries[0][0].shape == (2, 32, 512) and queries[0][1].shape == (2, 32, 0)
    assert receipt["case"] == "cpu_control" and receipt["positions"] == [8191, 8191]
    assert [row["form"] for row in receipt["runs"]] == ["cat", "direct", "cat", "direct"]
    assert [row["order"] for row in receipt["runs"]] == [0, 1, 2, 3]
    helper_identity = fault not in {
        "helper_clone", "helper_second_call_clone", "helper_third_call_clone"}
    assert receipt["helper_returned_q_nope"] is helper_identity
    assert receipt["bitwise"] is (fault is None)
    if fault is None:
        assert calls[1] is queries[0][0] and calls[3] is queries[0][0]
        assert all(row["bitwise_equal_to_first_cat"] for row in receipt["runs"])
        assert len({row["sha256"] for row in receipt["runs"]}) == 1
    elif fault.startswith("changed"):
        expected = {"changed_first_direct": 1, "changed_second_cat": 2,
                    "changed_second_direct": 3}[fault]
        assert [row["bitwise_equal_to_first_cat"] for row in receipt["runs"]] == [
            index != expected for index in range(4)]


@pytest.mark.parametrize("fault", [None, "mismatch", "case_exception"])
def test_qualification_main_reports_all_five_cases_and_fails_on_mismatch(monkeypatch, tmp_path, capsys, fault):
    from experiments.t8r_speed import empty_rope_cat_check as probe
    from tessera.serving.glm53_empty_rope import query_without_empty_rope

    mla, cache, workspace = object(), object(), object()
    runtime = types.ModuleType("vllm")
    runtime.__version__ = "cpu-control-vllm"
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__version__ = "cpu-control-flashinfer"
    utils = types.ModuleType("vllm.utils.flashinfer")
    utils.flashinfer_trtllm_batch_decode_with_kv_cache_mla = mla
    sparse = types.ModuleType("vllm.v1.attention.backends.mla.flashinfer_mla_sparse")

    def get_workspace(dev):
        assert dev == torch.device("cuda")
        return workspace

    sparse._get_workspace_buffer = get_workspace
    for name, module in [("vllm", runtime), ("flashinfer", flashinfer),
                         (utils.__name__, utils), (sparse.__name__, sparse)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["empty_rope_cat_check", "--out", str(tmp_path), "--reps", "7"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU control")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (0, 0))
    monkeypatch.setenv("ORACLE_IMAGE", "cpu-control-image")
    monkeypatch.setenv("TESSERA_HEAD", "cpu-control-source")
    monkeypatch.setenv("TESSERA_STATE", "cpu-control-state")
    monkeypatch.setenv("PB_ACTION_KEY", "cpu-control-action")
    arange, full = torch.arange, torch.full

    def cpu_positions(factory):
        def make(*args, **kwargs):
            assert kwargs.pop("device") == torch.device("cuda")
            return factory(*args, **kwargs)
        return make

    monkeypatch.setattr(torch, "arange", cpu_positions(arange))
    monkeypatch.setattr(torch, "full", cpu_positions(full))
    events, results = [], []

    def build_cache(dev, log):
        assert dev == torch.device("cuda")
        log.update(cache_writer="cpu-control-writer", cache_sha256="cpu-control-cache")
        events.append("cache")
        return cache

    def check_case(name, positions, actual_cache, actual_workspace, actual_mla, helper, dev, seed):
        assert events[0] == "cache"
        assert actual_cache is cache and actual_workspace is workspace and actual_mla is mla
        assert helper is query_without_empty_rope and dev == torch.device("cuda")
        events.append((name, positions.tolist(), seed))
        result = dict(case=name, bitwise=not (fault == "mismatch" and name == "decode_b2"))
        if fault == "case_exception" and name == "decode_b2":
            result = dict(case=name, bitwise=False, error="RuntimeError: controlled case failure")
            results.append(result)
            raise RuntimeError("controlled case failure")
        results.append(result)
        return result

    def screen(actual_cache, actual_workspace, actual_mla, helper, dev, reps):
        assert actual_cache is cache and actual_workspace is workspace and actual_mla is mla
        assert helper is query_without_empty_rope and dev == torch.device("cuda") and reps == 7
        events.append("screen")
        return {}

    monkeypatch.setattr(probe, "build_cache", build_cache)
    monkeypatch.setattr(probe, "check_case", check_case)
    monkeypatch.setattr(probe, "screen", screen)
    assert probe.main() == (0 if fault is None else 1)
    assert events == ["cache", ("prefill_chunk1", list(range(2048)), 3),
                      ("prefill_chunk4", list(range(6144, 8192)), 13),
                      ("decode_b1", [8191], 23), ("decode_b2", [8191, 8191], 33),
                      ("decode_b4", [8191] * 4, 43), "screen"]
    receipt = json.loads((tmp_path / "empty_rope_cat_check.json").read_text())
    assert receipt["cases"] == results and len(receipt["cases"]) == 5
    assert receipt["all_bitwise"] is (fault is None)
    assert receipt["cache_writer"] == "cpu-control-writer"
    assert receipt["cache_sha256"] == "cpu-control-cache"
    assert receipt["vllm"] == "cpu-control-vllm" and receipt["flashinfer"] == "cpu-control-flashinfer"
    assert receipt["image"] == "cpu-control-image" and receipt["tessera_head"] == "cpu-control-source"
    assert receipt["tessera_state"] == "cpu-control-state" and receipt["pb_action"] == "cpu-control-action"
    assert receipt["device"] == "CPU control" and receipt["capability"] == [0, 0]
    output = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in output[:5]] == results
    assert output[5:] == ["ALL_BITWISE" if fault is None else "NOT_BITWISE"]

