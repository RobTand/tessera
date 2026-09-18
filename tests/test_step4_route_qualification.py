"""The step-4 capture's native-route qualification refuses what it must.

The mechanism under test is the one that makes a silent substitution loud: in
``resident`` residency the NVFP4 route serves through ``torch_materialize_stock``
when the CUDA extension cannot build, so a capture taken in a JIT-incapable
container looks exactly like a qualified one.  Each test drives the qualifier
with evidence a real run could produce and asserts it refuses, or passes only
when BOTH legs hold.
"""
from __future__ import annotations

import json

import pytest

from experiments.step4_route_qualification import (
    NATIVE_DECODER, SUBSTITUTED_DECODER, QualificationRefused, mapped_native_libraries,
    qualify_native_route, refusal_record, trace_decoders_by_contract)

CONTRACT = "e2m1_group16_ue4m3_static"
LIBRARY = "/jit-ext/tessera_nvfp4/abc123/tessera_nvfp4_abc123.so"
SHA = "a" * 64


def observation(libraries):
    return {"base": {"native_libraries": libraries}}


def entry(decoder, *, launches=1, modules=1, shape="512", contract=CONTRACT):
    return {"policy": "TESSERA_NVFP4:resident", "shape": shape,
            "symbol": "torch._scaled_mm", "decoder": decoder, "contract": contract,
            "kind": "dense", "launches": launches, "modules": modules}


def trace(*entries):
    return {"schema": "tessera.route_trace.v1", "entries": list(entries)}


def test_mapped_libraries_read_both_recorded_shapes():
    # ``base.native_libraries`` maps path -> sha256; the worker's own
    # ``native_library_observation`` keeps {sha256, bytes}.
    assert mapped_native_libraries(observation({LIBRARY: SHA})) == {LIBRARY: SHA}
    assert mapped_native_libraries(
        observation({LIBRARY: {"sha256": SHA, "bytes": 12}})) == {LIBRARY: SHA}


def test_unreadable_library_digest_is_refused_not_ignored():
    with pytest.raises(QualificationRefused, match="no readable sha256"):
        mapped_native_libraries(observation({LIBRARY: 17}))


def test_other_libraries_do_not_match_the_glob():
    assert mapped_native_libraries(observation({
        "/usr/lib/aarch64-linux-gnu/libcuda.so.595.84": SHA,
        "/opt/observer/cupti-memory-arguments.so": SHA})) == {}


def test_absent_library_refuses_and_names_the_substitution():
    with pytest.raises(QualificationRefused, match=SUBSTITUTED_DECODER):
        qualify_native_route(observation({"/usr/lib/libc.so.6": SHA}),
                             trace(entry(NATIVE_DECODER)), expected_library_sha256=SHA,
                             activation_contract=CONTRACT)


def test_a_different_library_than_the_preflight_built_refuses():
    with pytest.raises(QualificationRefused, match="did not run the proven library"):
        qualify_native_route(observation({LIBRARY: "b" * 64}), trace(entry(NATIVE_DECODER)),
                             expected_library_sha256=SHA, activation_contract=CONTRACT)


def test_two_decode_libraries_refuse():
    with pytest.raises(QualificationRefused, match="several NVFP4 decode libraries"):
        qualify_native_route(observation({LIBRARY: SHA, LIBRARY + ".2.so": SHA}),
                             trace(entry(NATIVE_DECODER)), expected_library_sha256=SHA,
                             activation_contract=CONTRACT)


def test_mapped_library_does_not_excuse_a_substituted_dispatch():
    # The whole point of the second leg: the extension can be mapped by some
    # other importer while the route still decoded through stock Torch.
    with pytest.raises(QualificationRefused, match=SUBSTITUTED_DECODER):
        qualify_native_route(observation({LIBRARY: SHA}), trace(entry(SUBSTITUTED_DECODER)),
                             expected_library_sha256=SHA, activation_contract=CONTRACT)


def test_a_trace_without_the_contract_is_not_verified():
    with pytest.raises(QualificationRefused, match="never served the route it prices"):
        qualify_native_route(observation({LIBRARY: SHA}),
                             trace(entry(NATIVE_DECODER, contract="fp8_per_token_dynamic")),
                             expected_library_sha256=SHA, activation_contract=CONTRACT)


def test_an_entryless_trace_is_not_verified():
    with pytest.raises(QualificationRefused, match="not verified"):
        trace_decoders_by_contract({"schema": "tessera.route_trace.v1"}, CONTRACT)


def test_an_entry_without_a_decoder_or_launches_refuses():
    with pytest.raises(QualificationRefused, match="names no decoder"):
        trace_decoders_by_contract(trace({"contract": CONTRACT, "launches": 1}), CONTRACT)
    with pytest.raises(QualificationRefused, match="counts no launches"):
        trace_decoders_by_contract(trace({"contract": CONTRACT, "decoder": NATIVE_DECODER}), CONTRACT)


def test_modules_are_the_widest_key_not_a_sum_across_shapes():
    # One module serving a prefill and a decode shape appears under two keys;
    # adding them would claim twice as many modules as the artifact has.
    totals = trace_decoders_by_contract(
        trace(entry(NATIVE_DECODER, shape="512", modules=1),
              entry(NATIVE_DECODER, shape="1", modules=1)), CONTRACT)
    assert totals[NATIVE_DECODER] == {"launches": 2, "modules": 1, "entries": 2}


def test_fewer_modules_than_the_artifact_assigns_refuses():
    with pytest.raises(QualificationRefused, match="modules dispatched on"):
        qualify_native_route(observation({LIBRARY: SHA}), trace(entry(NATIVE_DECODER, modules=1)),
                             expected_library_sha256=SHA, activation_contract=CONTRACT,
                             expected_modules=2)


def test_both_legs_holding_qualifies_and_says_what_it_does_not_claim():
    record = qualify_native_route(observation({LIBRARY: SHA}),
                                  trace(entry(NATIVE_DECODER, launches=2, modules=1)),
                                  expected_library_sha256=SHA, activation_contract=CONTRACT,
                                  expected_modules=1)
    assert record["qualified"] is True
    assert record["library"] == {"path": LIBRARY, "sha256": SHA}
    assert record["decoders"][NATIVE_DECODER]["launches"] == 2
    assert "no timing, fixed-resource or release admission" in record["scope"]


def test_a_refusal_record_is_never_a_price():
    record = refusal_record("jit_preflight", "nvcc missing", ext_dir="/jit-ext")
    assert record["qualified"] is False and record["phase"] == "jit_preflight"
    assert json.loads(json.dumps(record))["ext_dir"] == "/jit-ext"
