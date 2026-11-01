"""Reference-only decode controls over real expert wire containers.

These CPU controls compare per-expert tiles and row scales with the encoder
stock oracle. Production WINDOW serving retains compressed planes and executes
the native class dispatcher, proved separately through real framework loading
and eager/captured device execution in the serving image.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.moe_layout import pack_moe_wires, unpack_moe_wires    # noqa: E402
from tessera.serving import moe_route                              # noqa: E402
from tessera.serving.scheme import (                               # noqa: E402
    TESSERA_FP8, validate_tessera_moe_scheme)

HIDDEN, INTER, EXPERTS, Q256 = 64, 32, 3, 1024


def _tessera():
    return (pytest.importorskip("tessera.export"), pytest.importorskip("tessera.stock"),
            pytest.importorskip("tessera.alphabet"), pytest.importorskip("tessera.fused"))


def _encode(rows, cols, name, seed):
    export, stock, alphabet, fused = _tessera()
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=generator) * 0.02
    exported, unit, forests = export.encode_linear_planes(
        w.contiguous(), grid=alphabet.E4M3_GRID, q256=Q256, name=name, verify=False)
    blob = fused.pack_fused([(name, rows, exported.blob)])
    return blob, stock.materialize_stock(unit, forests, export.DEFAULT_CODE)


def _stack(experts=EXPERTS, hidden=HIDDEN, inter=INTER):
    """E experts of (gate, up, down) wires, plus the stock reference tensors."""
    w13_blobs, w2_blobs, reference = [], [], []
    for e in range(experts):
        gate, gate_ref = _encode(inter, hidden, "gate_proj", 100 + e)
        up, up_ref = _encode(inter, hidden, "up_proj", 200 + e)
        down, down_ref = _encode(hidden, inter, "down_proj", 300 + e)
        w13_blobs.append([gate, up])
        w2_blobs.append(down)
        reference.append({"gate": gate_ref, "up": up_ref, "down": down_ref})
    scheme = {
        "family": TESSERA_FP8, "structure": "routed_moe", "grid": "E4M3", "body": "WINDOW",
        "plane": "CHANNEL", "experts": experts,
        "expert_ids": list(range(experts)),
        "expert_classes": [{"start": 0, "end": experts,
                            "q256": {"w13": [Q256, Q256], "w2": [Q256]}}],
        "groups": {
            "w13": {"rows": 2 * inter, "columns": hidden, "q256": Q256,
                    "wire_stride": max(len(b) for pair in w13_blobs for b in pair),
                    "roles": [["gate_proj", inter], ["up_proj", inter]]},
            "w2": {"rows": hidden, "columns": inter, "q256": Q256,
                   "wire_stride": max(len(b) for b in w2_blobs),
                   "roles": [["down_proj", hidden]]}},
    }
    return w13_blobs, w2_blobs, scheme, reference


def test_the_expert_stack_decodes_to_the_stock_pair_byte_for_byte():
    """Gate at rows [0:N], up at [N:2N] -- the order RoutedExperts._load_w13
    narrows to -- and every byte is materialize_stock's."""
    w13_blobs, w2_blobs, scheme, reference = _stack()
    declared = validate_tessera_moe_scheme(scheme, "m")
    prepared = moe_route.prepare_tessera_moe_experts(
        {"w13": w13_blobs, "w2": [[b] for b in w2_blobs]}, declared, "m", device="cpu")
    assert tuple(prepared.w13_weight.shape) == (EXPERTS, 2 * INTER, HIDDEN)
    assert tuple(prepared.w2_weight.shape) == (EXPERTS, HIDDEN, INTER)
    assert tuple(prepared.w13_weight_scale.shape) == (EXPERTS, 2 * INTER, 1)
    assert tuple(prepared.w2_weight_scale.shape) == (EXPERTS, HIDDEN, 1)
    assert prepared.w13_weight.dtype == torch.float8_e4m3fn
    for e, ref in enumerate(reference):
        w13 = prepared.w13_weight[e].view(torch.uint8)
        assert torch.equal(w13[:INTER], ref["gate"]["weight"].view(torch.uint8))
        assert torch.equal(w13[INTER:], ref["up"]["weight"].view(torch.uint8))
        assert torch.equal(prepared.w2_weight[e].view(torch.uint8),
                           ref["down"]["weight"].view(torch.uint8))
        scale = prepared.w13_weight_scale[e].reshape(-1)
        assert torch.equal(scale[:INTER], ref["gate"]["weight_scale"].reshape(-1).float())
        assert torch.equal(scale[INTER:], ref["up"]["weight_scale"].reshape(-1).float())
        assert torch.equal(prepared.w2_weight_scale[e].reshape(-1),
                           ref["down"]["weight_scale"].reshape(-1).float())


def test_the_wires_survive_the_padded_parameter_they_are_loaded_into():
    """Real blobs are ragged at one shape and rung; the parameter is not.  The
    round trip through the layout is byte-for-byte, and the strides the sidecar
    declares are the ones the packing derives."""
    w13_blobs, w2_blobs, scheme, _ref = _stack()
    packed = pack_moe_wires(w13_blobs, w2_blobs)
    assert packed.w13_wire.shape[2] == scheme["groups"]["w13"]["wire_stride"]
    assert packed.w2_wire.shape[1] == scheme["groups"]["w2"]["wire_stride"]
    back13, back2 = unpack_moe_wires(packed)
    assert back13 == w13_blobs and back2 == w2_blobs
    declared = validate_tessera_moe_scheme(scheme, "m")
    prepared = moe_route.prepare_tessera_moe_experts(
        {"w13": back13, "w2": [[b] for b in back2]}, declared, "m", device="cpu")
    assert prepared.experts == EXPERTS


def test_the_shard_vocabulary_is_the_groups_row_order_not_a_second_table():
    assert moe_route.SHARD_TO_GROUP == {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", 0)}


def test_a_container_that_is_not_what_the_sidecar_declared_is_refused():
    w13_blobs, w2_blobs, scheme, _ref = _stack()
    declared = validate_tessera_moe_scheme(scheme, "m")
    # gate and up swapped: the container's role name no longer matches the
    # projection the sidecar puts in that row block.
    swapped = [[pair[1], pair[0]] for pair in w13_blobs]
    with pytest.raises(ValueError, match="the container holds roles"):
        moe_route.prepare_tessera_moe_experts(
            {"w13": swapped, "w2": [[b] for b in w2_blobs]}, declared, "m", device="cpu")


def test_a_missing_expert_or_projection_is_refused_by_name():
    w13_blobs, w2_blobs, scheme, _ref = _stack()
    declared = validate_tessera_moe_scheme(scheme, "m")
    # Pin the named group/count refusal, not the explanatory suffix prose.
    with pytest.raises(ValueError, match=r"group 'w13' carries .* expert row\(s\)"):
        moe_route.prepare_tessera_moe_experts(
            {"w13": w13_blobs[:-1], "w2": [[b] for b in w2_blobs]}, declared, "m", device="cpu")
    with pytest.raises(ValueError, match="declared projection"):
        moe_route.prepare_tessera_moe_experts(
            {"w13": [[pair[0]] for pair in w13_blobs], "w2": [[b] for b in w2_blobs]},
            declared, "m", device="cpu")


def test_a_blob_longer_than_the_declared_stride_is_refused():
    w13_blobs, w2_blobs, scheme, _ref = _stack()
    scheme["groups"]["w13"]["wire_stride"] = min(len(b) for pair in w13_blobs for b in pair) - 1
    declared = validate_tessera_moe_scheme(scheme, "m")
    with pytest.raises(ValueError, match="longer than the"):
        moe_route.prepare_tessera_moe_experts(
            {"w13": w13_blobs, "w2": [[b] for b in w2_blobs]}, declared, "m", device="cpu")


# --- what a census may compare a served expert stack against ----------------
#
# The route owns its expectation, exactly as ``fp8_gemv.census_expected`` owns
# the dense FP8 route's: the dispatch is here, so the value a receipt is graded
# on is here.  What these pin is that it is not the dense route's value.

#: One routed-expert record, verbatim from the first served census of a Tessera
#: MoE checkpoint (``/mnt/shared/tessera-runs/ts5/served/census.json``: GB10,
#: vLLM 0.28, eager, ``TESSERA_SERVE_MODE=resident``, a 16-expert cut of
#: GLM-5.3-Flash-4layer, 3 of 3 stacks in both phases).
SERVED_MOE_SYMBOL = "vllm.fused_moe.modular_kernel:TRITON"
SERVED_MOE_DECODER = "torch_materialize_stock"




def test_the_served_records_symbol_reduces_into_the_expectation():
    """The runtime's backend pick is recorded, not graded.

    ``select_fp8_moe_backend`` is vLLM's predicate over the kernels it finds on
    the box; the record keeps its answer so a receipt says which backend ran,
    and the comparison is over the entry point, which is the part a route
    promises.

    The record above is the materialising launch.  It left this build's FP8
    table at contract v38 and its NVFP4 table at v39 (tessera#604), so no
    route's expectation admits it any more; the suffix rule is pinned against
    the pair the record reduces to, which is what a pre-v39 receipt carries.
    """
    from tessera.serving import nvfp4_moe_route

    served = (moe_route.census_symbol_base(SERVED_MOE_SYMBOL), SERVED_MOE_DECODER)
    from tessera.serving.scheme import MOE_GEMM_SYMBOL
    assert served == (MOE_GEMM_SYMBOL, SERVED_MOE_DECODER)
    assert served not in moe_route.census_expected(compiled=False)["batch"]
    assert served not in nvfp4_moe_route.census_expected(compiled=False)["batch"]
    # A qualified operation is not a backend-suffixed stock MoE launch.
    for other in ("torch._scaled_mm", "tessera::routed_window_classes", "tessera::another_operation"):
        assert moe_route.census_symbol_base(other) == other
        assert (moe_route.census_symbol_base(other), SERVED_MOE_DECODER) != served


def test_the_dense_fp8_expectation_would_refuse_every_served_stack():
    """THE DEFECT THIS PINS, and it is a census defect rather than a route one.

    An expert stack serves under ``TESSERA_FP8`` -- same family, same wire,
    same activation contract -- so a census that resolves a record's
    expectation from the FAMILY alone hands a stack the dense route's pair set.
    Every one of the three stacks served in
    ``/mnt/shared/tessera-runs/ts5/served/census.json`` reports the pair below,
    in both phases, and none of those pairs is in that set: the census would
    have refused a serve that did exactly what this route intends, six
    problems on a correct receipt.  The structure decides the launch, so the
    structure has to decide the expectation.
    """
    from tessera.serving import fp8_gemv

    dense = fp8_gemv.census_expected(compiled=False)
    served = (moe_route.census_symbol_base(SERVED_MOE_SYMBOL), SERVED_MOE_DECODER)
    for regime in ("decode", "batch"):
        assert served not in dense[regime]
        assert (SERVED_MOE_SYMBOL, SERVED_MOE_DECODER) not in dense[regime]
