"""Synthetic feature evidence through the real vLLM WINDOW plugin.

Run this file with pytest on CPU first. The CPU leg reads actual framed files,
validates the storage declarations and all tensor parallel cuts, and imports the
pure reference entry. It does not substitute for the skipped CUDA plugin cases.
Run the same file in the serving image for the real loader and changed graph
replay. Tensor parallel degree two checks each rank on one device, not a gang.
No artifact, quality, serving pin or allowable-rung claim follows from this test.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest
import torch

from tessera import routed_fused as rf
from tessera.compact_prep import require_compact_cut
from tessera.errors import GrammarError
from tessera.fused_frame import parse_fused
from tessera.native_window_moe import PackedWindowMoeBundles
from tessera.serving import moe_route
from tessera.serving.native_window import _role_cut
from tessera.serving.scheme import (expert_role_declarations,
                                    parse_compact_tessera_expert_blob,
                                    validate_tessera_moe_scheme)
from tessera.window_gemm_grouped import prepare_grouped_window_gemm
from _routed_classes_plugin_fixture import (EXPERTS, HIDDEN, INTERMEDIATE,
                                            ROLES, TOP_K, route_cases, wire_fixture)
from test_routed_window_classes_cuda import _pure_launch

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="CPU fixture control only; real vLLM loader/CUDA graph cases did not execute")
TP_CUTS = [(1, 0), (2, 0), (2, 1)]


@pytest.mark.parametrize("family", ["e4m3", "value"])
@pytest.mark.parametrize("layout", ["mixed", "three", "uniform"])
def test_cpu_same_entry_reads_actual_window_files_and_all_tp_cuts(family, layout, tmp_path):
    scheme, wires = wire_fixture(family, layout)
    declared = validate_tessera_moe_scheme(scheme, "fixture")
    assert declared["expert_ids"] == scheme["expert_ids"]
    assert (scheme["expert_ids"] == list(range(EXPERTS))) == (layout == "uniform")
    for storage, original in enumerate(scheme["expert_ids"]):
        for group, index, shard, name, rows, columns in ROLES:
            # These are actual on-disk inputs, not an echo of a mocked parser.
            path = tmp_path / f"experts.{storage}.{name}.wire"
            path.write_bytes(wires[(storage, shard)])
            blob = path.read_bytes()
            member, = parse_fused(blob)
            assert (member.name, member.rows) == (name, rows)
            role = expert_role_declarations(declared["groups"][group], expert=storage)[index]
            parsed, = parse_compact_tessera_expert_blob(blob, role, str(path), device="cpu")
            parsed_name, wire = parsed
            assert parsed_name == name
            assert (wire.metadata.rows, wire.metadata.columns) == (rows, columns)
            assert wire.metadata.manifest.window_bits == 14
            assert len(wire.metadata.rates) == columns
            assert wire.metadata.chunks, "read the actual payload planes"
            for degree, rank in TP_CUTS:
                plan = moe_route._packed_group_shard_plan(declared, group, "fixture", rank, degree)
                cut = require_compact_cut(wire, **_role_cut(plan, name))
                (r0, r1), (c0, c1) = cut
                assert (r1 - r0, c1 - c0) == (
                    (INTERMEDIATE // degree, HIDDEN) if group == "w13"
                    else (HIDDEN, INTERMEDIATE // degree))
                if degree == 2:
                    assert (r0 if group == "w13" else c0) == rank * (INTERMEDIATE // degree)
    cases = route_cases(scheme, 1)
    inverse = torch.tensor([scheme["expert_ids"].index(e) for e in range(EXPERTS)])
    for (ids, weights), counts in zip(cases, ([8, 0], [0, 8], [4, 4], [6, 2], [7, 1])):
        stored = inverse[ids.long()]
        assert [int((stored == 0).sum()), int((stored == EXPERTS - 1).sum())] == counts
        assert ids.shape == weights.shape == (1, TOP_K)
        assert len(set(weights.reshape(-1).tolist())) == TOP_K
    print(f"CPU fixture only: {family}/{layout}, 24 storage wires, TP1 and both TP2 cuts; no vLLM or CUDA execution")


@pytest.mark.parametrize("field", ["expert_ids", "expert_classes"])
def test_cpu_missing_storage_metadata_is_a_load_time_error(field):
    # The old consumer silently accepted this complete uniform declaration.
    scheme, _ = wire_fixture("e4m3", "uniform")
    bad = copy.deepcopy(scheme)
    del bad[field]
    with pytest.raises(ValueError, match=field):
        validate_tessera_moe_scheme(bad, "missing storage metadata")


def _new_method(family, layout, degree, rank):
    # Reuse the established real-vLLM layer fixture, not its CPU stub_runtime.
    from test_native_window_moe_method import _native_layer
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

    scheme, wires = wire_fixture(family, layout)
    layer = _native_layer(tp_rank=rank, tp_size=degree, hidden=HIDDEN,
                          inter=INTERMEDIATE, experts=EXPERTS)
    layer.global_num_experts = EXPERTS
    layer.moe_config.experts_per_token = TOP_K
    layer.swiglu_limit = 2.0
    # EP owns this object. Even a tensor with deliberately unrelated values
    # must remain untouched; it must not become the storage inverse.
    layer.expert_map = torch.arange(EXPERTS - 1, -1, -1, dtype=torch.int32, device="cuda")
    method = moe_route.build_tessera_moe_method(scheme, "synthetic.experts", "resident", layer)
    assert isinstance(method, FusedMoEMethodBase)
    method.create_weights(layer, EXPERTS, HIDDEN, INTERMEDIATE // degree, torch.bfloat16)
    assert set(dict(layer.named_parameters())) == {"w13_wire", "w2_wire"}
    return method, layer, scheme, wires


def _callback(layer, storage, shard, blob):
    param = layer.w2_wire if shard == "w2" else layer.w13_wire
    name = next(name for group, index, tag, name, rows, cols in ROLES if tag == shard)
    assert param.weight_loader(param, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
                               f"synthetic.experts.{storage}.{name}.wire", shard, storage,
                               return_success=True)


def _finish(method, layer, wires):
    # Reverse load order rules out a fixture dependent on callback order.
    for (storage, shard), blob in reversed(list(wires.items())):
        _callback(layer, storage, shard, blob)
    method.process_weights_after_loading(layer)
    assert not dict(layer.named_parameters())
    assert method._native.counters.dtype == torch.int32
    assert method._native.counters.shape == (len(method._native.classes), 2)


def _independent_packed(scheme, wires, degree, rank):
    """Prepare definition-side wires without the plugin intake or inverse."""
    declared = validate_tessera_moe_scheme(scheme, "pure control")
    stacks = {name: [] for group, index, shard, name, rows, cols in ROLES}
    for storage in range(EXPERTS):
        for group, index, shard, name, rows, cols in ROLES:
            role = expert_role_declarations(declared["groups"][group], expert=storage)[index]
            plan = moe_route._packed_group_shard_plan(declared, group, "pure control", rank, degree)
            parsed_name, unit = moe_route._compact_expert_units(
                wires[(storage, shard)], role, plan, "pure control", device="cuda", family=(
                    "e4m3" if scheme["family"] == "TESSERA_FP8" else "value"))
            assert parsed_name == name
            stacks[name].append(unit)
    family = "e4m3" if scheme["family"] == "TESSERA_FP8" else "value"
    bundles = [prepare_grouped_window_gemm(stacks[name])
               for name in ("gate_proj", "up_proj", "down_proj")]
    return PackedWindowMoeBundles(*bundles, family=family, expert_classes=scheme["expert_classes"])


def _assert_loaded_coordinates(actual, expected, degree, rank):
    library = actual.adapter().library
    for role in ("gate", "up", "down"):
        got, want = getattr(actual, role), getattr(expected, role)
        assert (got.rows, got.cols, got.experts) == (want.rows, want.cols, EXPERTS)
        for field in ("words_all", "scale_all", "init_all", "has_init"):
            a, b = getattr(got, field), getattr(want, field)
            if field == "words_all" and got.word_layout == "piece_major":
                from tessera.kernel_window_gemv import PIECES_PER_TILE

                b = b.reshape(EXPERTS, -1, want.cols, PIECES_PER_TILE, 8).transpose(2, 3).contiguous()
            assert torch.equal(a.view(torch.uint8).reshape(-1), b.view(torch.uint8).reshape(-1)), (role, field, degree, rank)
        # The native table is composed from definition-side codes and alphabet.
        table = rf.compose_table(want, library)
        assert torch.equal(got.table_all.view(torch.uint8), table.view(torch.uint8)), (role, "table")
        # Finalization discards these preparation planes. Do not access their bytes.
        for field in ("codes_all", "native_all", "runs_all", "word_off", "tile_words",
                      "total_words", "run_off", "perm_all"):
            assert getattr(got, field) is None, (role, "unretired", field)
        if role != "down" and rank == 1:
            assert torch.all(got.has_init == 1), "TP2 rank one must carry row-cut WINDOW history"

@pytest.mark.parametrize("family", ["e4m3", "value"])
def test_cpu_native_owner_coordinate_helper_respects_retired_planes(family, monkeypatch):
    from test_routed_window_classes_cuda import _packed

    # Only device admission, extension loading and stream creation are absent.
    # Lookup composition, class descriptors and native-owner retirement are real.
    monkeypatch.setattr(rf, "fused_routed_window_supported", lambda *args: None)
    monkeypatch.setattr(rf, "_ext", lambda library: object())
    monkeypatch.setattr(rf, "_make_dispatch_resources", lambda device, kernel: rf._DispatchResources(
        (object(), object()), object(), (object(), object()),
        torch.empty(0, dtype=torch.float32, device=device), kernel))
    prepared = _packed(family, three=True, device="cpu")
    native = prepared.native_owner()
    definition = _packed(family, three=True, device="cpu")
    _assert_loaded_coordinates(native, definition, 1, 0)
    assert native is not prepared
    print(f"CPU owner lifetime only: {family}, real composed tables and retired metadata; no forward operation")

def _bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _saved_reference(packed, x, ids, weights):
    """Use the shared pure native entry at this fixture's actual rank geometry."""
    routed = torch.empty(ids.numel(), packed.down.rows, device=x.device, dtype=torch.bfloat16)
    flat_ids = ids.reshape(-1)
    for desc in packed.expert_classes:
        start, end = desc["start"], desc["end"]
        take = torch.where((flat_ids >= start) & (flat_ids < end))[0]
        if take.numel() == 0:
            continue
        roles = [rf.grouped_class_view(role, start, end)
                 for role in (packed.gate, packed.up, packed.down)]
        pure = rf.FusedRoutedWindowMoE.from_bundles(*roles, expert_classes=[
            {"start": 0, "end": end - start, "q256": desc["q256"]}])
        local_ids = (flat_ids[take] - start).reshape(-1, 1)
        rw = weights.reshape(-1)[take].reshape(-1, 1)
        xin = x[take // ids.shape[1]]
        routing = pure._routing(local_ids, rw)
        xq, a1 = pure._quantized(xin, None, len(take))
        act = torch.empty(len(take), packed.gate.rows, device=x.device, dtype=torch.bfloat16)
        _pure_launch(pure, 0, xq, a1, routing, act, weight=False)
        aq, a2 = pure._quantized(act, None, len(take))
        output = torch.empty(len(take), packed.down.rows, device=x.device, dtype=torch.bfloat16)
        _pure_launch(pure, 2, aq, a2, routing, output, weight=True)
        routed[take] = output
    out = torch.empty_like(x)
    rf._ext(rf.library_for(packed.family)).token_sum(routed, out, ids.shape[1])
    return out.clone()



def _save_references(tmp_path, name, scheme, cases, expected, degree, rank, library):
    root = Path(os.environ.get("TERMINAL_NATIVE_IDENTITY_DIR", str(tmp_path)))
    root = root.parent / "plugin-proof" if "TERMINAL_NATIVE_IDENTITY_DIR" in os.environ else root
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    payload = {"scope": "synthetic feature only; rank-local tensor parallel control",
               "head": os.environ.get("TESSERA_HEAD", "unreported"), "library": library,
               "device": torch.cuda.get_device_name(), "tensor_parallel_degree": degree,
               "tensor_parallel_rank": rank, "expert_ids": scheme["expert_ids"],
               "expert_classes": scheme["expert_classes"],
               "references": [{"ids": ids.cpu().tolist(), "weights": weights.cpu().tolist(),
                               "bf16_bits": ref.view(torch.int16).cpu().tolist()}
                              for (ids, weights), ref in zip(cases, expected)]}
    path.write_text(json.dumps(payload) + "\n")
    print(f"Saved stitched pure-schedule references: {path}")
    saved = json.loads(path.read_text())["references"]
    return [torch.tensor(row["bf16_bits"], dtype=torch.int16, device=expected[0].device)
            .view(torch.bfloat16) for row in saved]


@cuda
@pytest.mark.parametrize("family", ["e4m3", "value"])
@pytest.mark.parametrize("degree,rank", TP_CUTS)
@pytest.mark.parametrize("layout", ["mixed", "three"])
@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("reverse", [False, True])
def test_real_plugin_maps_storage_and_replays_changed_global_routes(
        family, degree, rank, layout, tokens, reverse, tmp_path, monkeypatch):
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3")
    method, layer, scheme, wires = _new_method(family, layout, degree, rank)
    ep_map, ep_bytes = layer.expert_map, layer.expert_map.clone()
    _finish(method, layer, wires)
    definition = _independent_packed(scheme, wires, degree, rank)
    _assert_loaded_coordinates(method._packed, definition, degree, rank)
    inverse = method._expert_inverse
    want_inverse = [scheme["expert_ids"].index(e) for e in range(EXPERTS)]
    assert inverse.dtype == torch.int32 and inverse.shape == (EXPERTS,)
    assert inverse.cpu().tolist() == want_inverse
    inverse_ptr = inverse.untyped_storage().data_ptr()
    inverse_storage = [(name, t) for name, t in method.resident_tensors(layer)
                       if t.untyped_storage().data_ptr() == inverse_ptr]
    assert len(inverse_storage) == 1 and inverse_storage[0][0] == "tessera_expert_inverse"
    assert inverse.untyped_storage().nbytes() == EXPERTS * 4
    if reverse:
        method._native = dataclasses.replace(method._native,
            class_issue_order=tuple(reversed(range(len(method._native.classes)))))
    x = (torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(4107)) * 0.125).bfloat16().cuda()
    cases = route_cases(scheme, tokens, device="cuda")
    expected = [_saved_reference(definition, x,
                torch.tensor(want_inverse, dtype=torch.int32, device="cuda").index_select(
                    0, ids.reshape(-1)).reshape_as(ids), weights)
                for ids, weights in cases]
    assert all(torch.isfinite(ref).all() for ref in expected)
    assert any(torch.count_nonzero(ref) for ref in expected), "zero output cannot prove expert identity"
    # A missing inverse must produce different bytes, not merely a missing symbol.
    assert any(not torch.equal(ref, _saved_reference(definition, x, ids, weights))
               for (ids, weights), ref in zip(cases, expected))
    expected = _save_references(tmp_path, f"{family}-{layout}-tp{degree}-rank{rank}-m{tokens}-reverse{int(reverse)}",
                     scheme, cases, expected, degree, rank, method._native.library)
    for (ids, weights), ref in zip(cases, expected):
        saved_ids, saved_weights = ids.clone(), weights.clone()
        _bits(method.apply(layer, x, weights, ids, None, None), ref)
        assert torch.equal(ids, saved_ids) and torch.equal(weights, saved_weights)
    ids, weights = (tensor.clone() for tensor in cases[0])
    method.apply(layer, x, weights, ids, None, None)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, weights, ids, None, None)
    for (route, rw), ref in list(zip(cases, expected)) + list(reversed(list(zip(cases, expected)))):
        ids.copy_(route)
        weights.copy_(rw)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _bits(captured, ref)
        assert torch.equal(ids, route) and torch.equal(weights, rw)
        assert method._expert_inverse is inverse
        assert inverse.untyped_storage().data_ptr() == inverse_ptr
        assert layer.expert_map is ep_map and torch.equal(ep_map, ep_bytes)
    empty = method.apply(layer, x[:0], weights[:0], ids[:0], None, None)
    assert empty.shape == (0, HIDDEN) and empty.dtype == torch.bfloat16


@cuda
@pytest.mark.parametrize("family", ["e4m3", "value"])
@pytest.mark.parametrize("degree,rank", TP_CUTS)
def test_real_plugin_uniform_identity_is_exact_old_pure_schedule(family, degree, rank, tmp_path, monkeypatch):
    from tessera.serving import flags

    flags.reset_for_tests(moe_route.ENV_PIECE_MAJOR)
    monkeypatch.delenv(moe_route.ENV_PIECE_MAJOR, raising=False)
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3")
    method, layer, scheme, wires = _new_method(family, "uniform", degree, rank)
    _finish(method, layer, wires)
    assert scheme["expert_ids"] == list(range(EXPERTS))
    assert len(method._native.classes) == 1
    assert method._native.piece_major == (family == "e4m3")
    definition = _independent_packed(scheme, wires, degree, rank)
    _assert_loaded_coordinates(method._packed, definition, degree, rank)
    x = (torch.randn(16, HIDDEN, generator=torch.Generator().manual_seed(4108)) * 0.125).bfloat16().cuda()
    ids = torch.arange(EXPERTS, dtype=torch.int32, device="cuda").repeat(16, 1)
    weights = torch.linspace(0.125, 0.875, ids.numel(), device="cuda").reshape_as(ids)
    ref = _saved_reference(definition, x, ids, weights)
    ref, = _save_references(tmp_path, f"{family}-uniform-tp{degree}-rank{rank}", scheme, [(ids, weights)], [ref],
                            degree, rank, method._native.library)
    _bits(method.apply(layer, x, weights, ids, None, None), ref)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, weights, ids, None, None)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _bits(captured, ref)


@cuda
@pytest.mark.parametrize("family", ["e4m3", "value"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong-storage-rung"])
def test_real_plugin_load_refuses_projection_errors(family, defect):
    method, layer, scheme, wires = _new_method(family, "mixed", 2, 1)
    if defect == "duplicate":
        _callback(layer, 0, "w1", wires[(0, "w1")])
        with pytest.raises(ValueError, match="storage expert 0.*already loaded"):
            _callback(layer, 0, "w1", wires[(0, "w1")])
    elif defect == "wrong-storage-rung":
        # Original zero lives at storage four. Its valid rate-four wire does
        # not match storage zero's rate-three declaration.
        with pytest.raises(ValueError, match="rung|q256|rate"):
            _callback(layer, 0, "w1", wires[(4, "w1")])
    else:
        for (storage, shard), blob in wires.items():
            if (storage, shard) != (0, "w3"):
                _callback(layer, storage, shard, blob)
        with pytest.raises(GrammarError, match="wire length 0"):
            method.process_weights_after_loading(layer)
