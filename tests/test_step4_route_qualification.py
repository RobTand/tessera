"""The step-4 capture's native-route qualification refuses what it must.

The mechanism under test is the per-family dispatch leg: every dispatch on
each family's activation contract must be the ONE native ``(symbol, decoder)``
its dense route stamps (``fp8_route.DENSE_LAUNCH``, ``bf16_route.DENSE_LAUNCH``,
``nvfp4_route``'s ``native_span2_gemm``), over exactly the manifest's modules.
Each test drives the qualifier with evidence a real run could produce and
asserts it refuses, or passes only when every family holds.  The library leg
has no subject on this tree and is recorded only; the tests pin that too.
"""
from __future__ import annotations

import json

import pytest

from experiments.step4_route_qualification import (
    A4_DENSE_GEMM_SYMBOL, BF16_ACTIVATION_CONTRACT, DENSE_LAUNCHES, FP8_ACTIVATION_CONTRACT,
    NATIVE_SPAN2_GEMM_DECODER, NATIVE_WINDOW_GEMM_DECODER, NATIVE_WINDOW_GEMM_FOLDED_DECODER,
    NVFP4_ACTIVATION_CONTRACT,
    QUALIFICATION_SCHEMA, QualificationRefused, WINDOW_GEMM_SYMBOL, WINDOW_GEMV_LIBRARY_GLOB,
    mapped_native_libraries, qualify_dispatch, qualify_native_route, refusal_record,
    trace_launches_by_contract)

SHA = "a" * 64
GEMV_LIBRARY = "/jit/torch-extensions/tessera_window_gemv_sm_121/tessera_window_gemv.so"

#: The artifact under #399: 110 FP8, one BF16 (layer-0 qkv), one NVFP4 (layer-0 gate_up).
EXPECTED = {"TESSERA_FP8": {"count": 2, "names": ["model.layers.0.mlp.down_proj",
                                                  "model.layers.1.mlp.down_proj"]},
            "TESSERA_BF16": {"count": 1, "names": ["model.layers.0.self_attn.qkv_proj"]},
            "TESSERA_NVFP4": {"count": 1, "names": ["model.layers.0.mlp.gate_up_proj"]}}


def observation(libraries):
    return {"base": {"native_libraries": libraries}}


def entry(family, *, symbol=None, decoder=None, launches=1, modules=1, shape="M512:N6144:K1024",
          names=None, unnamed=0, mode="resident"):
    contract, (native_symbol, native_decoder) = DENSE_LAUNCHES[family]
    record = {"policy": f"{family}:{mode}", "shape": shape,
              "symbol": native_symbol if symbol is None else symbol,
              "decoder": native_decoder if decoder is None else decoder,
              "contract": contract, "kind": "dense", "launches": launches, "modules": modules}
    if names is not None:
        record["module_names"] = list(names)
        record["unnamed_modules"] = unnamed
        record["dispatches_without_prefix"] = 0
    return record


def trace(*entries, identity=False):
    payload = {"schema": "tessera.route_trace/1", "entries": list(entries)}
    if identity:
        payload.update(identity_version=1, rank=0, world_size=1, rank_source="torch.distributed",
                       rank_conflict=None, platform="sm_121")
    return payload


def good_trace(identity=False):
    fp8 = EXPECTED["TESSERA_FP8"]["names"]
    return trace(
        entry("TESSERA_FP8", shape="M512:N1024:K3072", modules=2, launches=2,
              names=fp8 if identity else None),
        entry("TESSERA_FP8", shape="M1:N1024:K3072", modules=2, launches=2,
              names=fp8 if identity else None),
        entry("TESSERA_BF16", shape="M512:N4096:K1024",
              names=EXPECTED["TESSERA_BF16"]["names"] if identity else None),
        entry("TESSERA_BF16", shape="M1:N4096:K1024",
              names=EXPECTED["TESSERA_BF16"]["names"] if identity else None),
        entry("TESSERA_NVFP4", shape="M512:N6144:K1024",
              names=EXPECTED["TESSERA_NVFP4"]["names"] if identity else None),
        entry("TESSERA_NVFP4", shape="M1:N6144:K1024",
              names=EXPECTED["TESSERA_NVFP4"]["names"] if identity else None),
        identity=identity)


# -- the launch table is the routes' -------------------------------------------

def test_the_dense_launch_table_names_the_routes_one_launch_each():
    assert DENSE_LAUNCHES["TESSERA_FP8"] == (FP8_ACTIVATION_CONTRACT,
                                             (WINDOW_GEMM_SYMBOL, NATIVE_WINDOW_GEMM_DECODER))
    assert DENSE_LAUNCHES["TESSERA_BF16"] == (BF16_ACTIVATION_CONTRACT,
                                              (WINDOW_GEMM_SYMBOL,
                                               NATIVE_WINDOW_GEMM_FOLDED_DECODER))
    assert DENSE_LAUNCHES["TESSERA_NVFP4"] == (NVFP4_ACTIVATION_CONTRACT,
                                               (A4_DENSE_GEMM_SYMBOL, NATIVE_SPAN2_GEMM_DECODER))
    assert WINDOW_GEMM_SYMBOL == "tessera::window_gemm_dense"
    assert A4_DENSE_GEMM_SYMBOL == "tessera.kernel_a4.a4_span2_gemm"


# -- the library census is recorded, never required --------------------------

def test_mapped_libraries_read_both_recorded_shapes():
    assert mapped_native_libraries(observation({GEMV_LIBRARY: SHA})) == {GEMV_LIBRARY: SHA}
    assert mapped_native_libraries(
        observation({GEMV_LIBRARY: {"sha256": SHA, "bytes": 12}})) == {GEMV_LIBRARY: SHA}


def test_unreadable_library_digest_is_refused_not_ignored():
    with pytest.raises(QualificationRefused, match="no readable sha256"):
        mapped_native_libraries(observation({GEMV_LIBRARY: 17}))


def test_other_libraries_do_not_match_the_glob():
    assert WINDOW_GEMV_LIBRARY_GLOB == "tessera_window_gemv*.so"
    assert mapped_native_libraries(observation({
        "/usr/lib/aarch64-linux-gnu/libcuda.so.595.84": SHA,
        "/jit/triton/ZUTMPTWYP/cuda_utils.cpython-312-aarch64-linux-gnu.so": SHA,
        "/jit-ext/tessera_nvfp4/abc/tessera_nvfp4_abc.so": SHA})) == {}


def test_no_mapped_library_qualifies_and_the_record_says_why():
    record = qualify_native_route(observation({"/usr/lib/libc.so.6": SHA}), good_trace(),
                                  mode="resident", expected_modules=EXPECTED)
    assert record["qualified"] is True
    assert record["schema"] == QUALIFICATION_SCHEMA
    assert record["mapped_extension_libraries"] == {}
    assert "no library leg" in record["scope"]
    assert "no timing, fixed-resource or release admission" in record["scope"]


def test_a_mapped_gemv_library_is_recorded_not_a_pass():
    record = qualify_native_route(observation({GEMV_LIBRARY: SHA}), good_trace(),
                                  mode="resident", expected_modules=EXPECTED)
    assert record["mapped_extension_libraries"] == {GEMV_LIBRARY: SHA}
    # ...and it does not excuse a foreign dispatch.
    with pytest.raises(QualificationRefused, match="torch_window"):
        qualify_native_route(observation({GEMV_LIBRARY: SHA}),
                             trace(entry("TESSERA_FP8", symbol="torch._scaled_mm", decoder="torch_window"),
                                   entry("TESSERA_BF16"), entry("TESSERA_NVFP4")),
                             mode="resident", expected_modules={"TESSERA_FP8": 1, "TESSERA_BF16": 1,
                                                                "TESSERA_NVFP4": 1})


# -- the dispatch leg ---------------------------------------------------------

def test_the_pre_retirement_trace_shape_is_refused():
    # The real 2026-09-18 capture (frontier-qwen3-0.6b-20260918-399/capture-resources/
    # route-trace.json) served FP8 and BF16 through the torch window decoder and NVFP4
    # through the retired cpp-extension span-2 decode.  On this tree none of those is a
    # launch a dense route can make, so that trace must NOT qualify.
    old = trace(
        {"policy": "TESSERA_FP8:resident", "shape": "M512:N1024:K3072", "symbol": "torch._scaled_mm",
         "decoder": "torch_window", "contract": FP8_ACTIVATION_CONTRACT, "kind": "dense",
         "launches": 110, "modules": 110},
        {"policy": "TESSERA_BF16:resident", "shape": "M512:N4096:K1024", "symbol": "torch.mm",
         "decoder": "torch_window", "contract": BF16_ACTIVATION_CONTRACT, "kind": "dense",
         "launches": 1, "modules": 1},
        {"policy": "TESSERA_NVFP4:resident", "shape": "M512:N6144:K1024", "symbol": "torch._scaled_mm",
         "decoder": "native_span2", "contract": NVFP4_ACTIVATION_CONTRACT, "kind": "dense",
         "launches": 1, "modules": 1})
    expected = {"TESSERA_FP8": 110, "TESSERA_BF16": 1, "TESSERA_NVFP4": 1}
    with pytest.raises(QualificationRefused, match="torch_window"):
        qualify_dispatch(old, mode="resident", expected_modules=expected)
    # Each family alone is refused too, so a qualifier cannot pass on the first family it sees.
    for family, first in (("TESSERA_FP8", "torch._scaled_mm / torch_window"),
                          ("TESSERA_BF16", "torch.mm / torch_window"),
                          ("TESSERA_NVFP4", "torch._scaled_mm / native_span2")):
        with pytest.raises(QualificationRefused, match=first):
            qualify_dispatch(old, mode="resident", expected_modules={family: expected[family]})


def test_the_retired_stock_substitute_is_refused_on_nvfp4():
    with pytest.raises(QualificationRefused, match="torch_materialize_stock"):
        qualify_dispatch(trace(entry("TESSERA_NVFP4", symbol="torch._scaled_mm",
                                     decoder="torch_materialize_stock")),
                         mode="resident", expected_modules={"TESSERA_NVFP4": 1})


def test_a_native_pair_on_the_wrong_contract_is_refused():
    # The window GEMM pair stamped on the NVFP4 contract is not the A4 launch.
    with pytest.raises(QualificationRefused, match="not tessera.kernel_a4.a4_span2_gemm / native_span2_gemm"):
        qualify_dispatch(trace(entry("TESSERA_NVFP4", symbol=WINDOW_GEMM_SYMBOL,
                                     decoder=NATIVE_WINDOW_GEMM_DECODER)),
                         mode="resident", expected_modules={"TESSERA_NVFP4": 1})


def test_a_family_the_artifact_carries_but_the_trace_never_served_is_refused():
    with pytest.raises(QualificationRefused, match="never served the route it prices"):
        qualify_dispatch(trace(entry("TESSERA_FP8", modules=2), entry("TESSERA_BF16")),
                         mode="resident", expected_modules=EXPECTED)


def test_a_family_with_no_module_is_skipped_not_required():
    record = qualify_dispatch(trace(entry("TESSERA_FP8")), mode="resident",
                              expected_modules={"TESSERA_FP8": 1, "TESSERA_NVFP4": 0})
    assert sorted(record) == ["TESSERA_FP8"]


def test_an_artifact_with_no_dense_module_is_refused():
    with pytest.raises(QualificationRefused, match="nothing to qualify"):
        qualify_dispatch(trace(entry("TESSERA_FP8")), mode="resident",
                         expected_modules={"TESSERA_FP8": 0})


def test_an_unknown_family_or_mode_is_refused():
    with pytest.raises(QualificationRefused, match="no dense launch for"):
        qualify_dispatch(trace(entry("TESSERA_FP8")), mode="resident",
                         expected_modules={"TESSERA_FP8": 1, "TESSERA_INT4": 1})
    with pytest.raises(QualificationRefused, match="unknown residency mode"):
        qualify_dispatch(trace(entry("TESSERA_FP8")), mode="hybrid", expected_modules={"TESSERA_FP8": 1})


def test_a_dispatch_under_another_residency_is_refused():
    with pytest.raises(QualificationRefused, match="policy 'TESSERA_FP8:streamed'"):
        qualify_dispatch(trace(entry("TESSERA_FP8", mode="streamed")), mode="resident",
                         expected_modules={"TESSERA_FP8": 1})


def test_an_entryless_trace_is_not_verified():
    with pytest.raises(QualificationRefused, match="not verified"):
        trace_launches_by_contract({"schema": "tessera.route_trace/1"}, FP8_ACTIVATION_CONTRACT)


def test_an_entry_missing_a_key_field_refuses():
    base = {"contract": FP8_ACTIVATION_CONTRACT, "symbol": WINDOW_GEMM_SYMBOL,
            "decoder": NATIVE_WINDOW_GEMM_DECODER, "launches": 1, "shape": "M1:N1:K1"}
    for missing, message in (("symbol", "names no symbol"), ("decoder", "names no decoder"),
                             ("launches", "counts no launches"), ("shape", "names no shape")):
        broken = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(QualificationRefused, match=message):
            trace_launches_by_contract(trace(broken), FP8_ACTIVATION_CONTRACT)


def test_a_record_written_under_compile_tracing_is_refused_not_counted():
    with pytest.raises(QualificationRefused, match="torch.compile tracing"):
        trace_launches_by_contract(trace(entry("TESSERA_FP8", shape="M*:N6144:K1024")),
                                   FP8_ACTIVATION_CONTRACT)


def test_one_module_served_at_two_token_counts_is_counted_once():
    totals = trace_launches_by_contract(
        trace(entry("TESSERA_NVFP4", shape="M512:N6144:K1024"),
              entry("TESSERA_NVFP4", shape="M1:N6144:K1024")), NVFP4_ACTIVATION_CONTRACT)
    (native,) = totals.values()
    assert (native["launches"], native["modules"], native["entries"]) == (2, 1, 2)
    assert native["symbol"] == A4_DENSE_GEMM_SYMBOL and native["decoder"] == NATIVE_SPAN2_GEMM_DECODER


def test_modules_with_different_geometries_are_added_within_one_token_count():
    # Three modules of different N:K served in the same forward are three
    # modules, not one (the 2026-09-18 capture's fp8 contract had 110 where a
    # max over keys returned 28).
    totals = trace_launches_by_contract(
        trace(entry("TESSERA_FP8", shape="M512:N6144:K1024"),
              entry("TESSERA_FP8", shape="M512:N1024:K3072"),
              entry("TESSERA_FP8", shape="M512:N2048:K1024"),
              entry("TESSERA_FP8", shape="M1:N6144:K1024")), FP8_ACTIVATION_CONTRACT)
    (native,) = totals.values()
    assert native["modules"] == 3


def test_fewer_modules_than_the_artifact_assigns_refuses():
    with pytest.raises(QualificationRefused, match="modules dispatched on"):
        qualify_dispatch(trace(entry("TESSERA_FP8", modules=1)), mode="resident",
                         expected_modules={"TESSERA_FP8": 2})


# -- module identity (route trace identity_version 1) -------------------------

def test_an_unnamed_module_refuses_the_per_module_claim():
    with pytest.raises(QualificationRefused, match="unnamed_modules == 0"):
        qualify_dispatch(trace(entry("TESSERA_BF16", names=[], unnamed=1), identity=True),
                         mode="resident", expected_modules={"TESSERA_BF16": 1})


def test_module_names_must_be_the_manifests_when_both_sides_name_them():
    with pytest.raises(QualificationRefused, match="not the manifest's"):
        qualify_dispatch(trace(entry("TESSERA_BF16", names=["model.layers.3.self_attn.qkv_proj"]),
                               identity=True),
                         mode="resident", expected_modules={"TESSERA_BF16": EXPECTED["TESSERA_BF16"]})


def test_a_trace_that_names_no_modules_is_qualified_by_count_alone():
    # A pre-identity_version trace carries counts only; the names cannot be
    # checked and the record says so.
    record = qualify_dispatch(good_trace(identity=False), mode="resident", expected_modules=EXPECTED)
    assert all(value["expected"]["names_checked"] is False for value in record.values())


def test_every_family_holding_qualifies_and_says_what_it_does_not_claim():
    record = qualify_native_route(observation({}), good_trace(identity=True),
                                  mode="resident", expected_modules=EXPECTED)
    assert record["qualified"] is True
    assert sorted(record["families"]) == ["TESSERA_BF16", "TESSERA_FP8", "TESSERA_NVFP4"]
    fp8 = record["families"]["TESSERA_FP8"]
    assert fp8["observed"]["launches"] == 4 and fp8["observed"]["modules"] == 2
    assert fp8["observed"]["module_names"] == EXPECTED["TESSERA_FP8"]["names"]
    assert fp8["expected"] == {"symbol": WINDOW_GEMM_SYMBOL, "decoder": NATIVE_WINDOW_GEMM_DECODER,
                               "modules": 2, "names_checked": True}
    assert record["trace_identity"]["identity_version"] == 1
    assert record["trace_identity"]["platform"] == "sm_121"
    json.dumps(record)  # the record is JSON, sets and all


def test_a_refusal_record_is_never_a_price():
    record = refusal_record("native_preflight", "no Triton", triton_cache_dir="/jit/triton")
    assert record["qualified"] is False and record["phase"] == "native_preflight"
    assert json.loads(json.dumps(record))["triton_cache_dir"] == "/jit/triton"
