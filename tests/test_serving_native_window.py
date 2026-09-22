"""The dense native window lane: compact load -> packed bundles -> the GEMM.

The route-level tests (``test_serving_fp8_route.py``, ``test_serving_bf16_route.py``)
drive one encoded module through the method; this file holds the lane's own
contract against the **actual** BF16 R7 checkpoint and the launch table:

* the compact reader and the materialising reader accept and refuse the same
  sidecar declarations (one comparison helper);
* the native module's forward is the retained reference preparation's product
  (fp32 accumulation of the same values), at the M tails and at TP2 cuts;
* the prepared weights stay packed across forwards -- fingerprints and byte
  count unchanged, no ``[rows, columns]`` tile anywhere;
* the launch table publishes the ``(tessera::window_gemm_dense,
  native_window_gemm)`` pair both routes stamp.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import box_artifacts

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the native window lane is a CUDA path")

BF16_ROOT = "bf16/qwen0.6b-bf16-r7-plugin"
DOWN_GROUP = "tessera_model_layers_0_mlp_down_proj"
GATE_UP_GROUP = "tessera_model_layers_0_mlp_gate_up_proj"


def _scheme(group: str) -> dict:
    model = box_artifacts.skip_now("shared_runs", BF16_ROOT, "config.json")
    config = json.loads(Path(model).read_text())
    return config["quantization_config"]["config_groups"][group]["scheme"]


def _wire(tensor: str) -> bytes:
    shard = box_artifacts.skip_now("shared_runs", BF16_ROOT, "model.safetensors")
    from safetensors import safe_open

    with safe_open(str(shard), framework="pt") as handle:
        return bytes(handle.get_tensor(tensor).detach().cpu().numpy().tobytes())


def _whole_plan(declared):
    from tessera.serving.sharding import plan_shard

    roles = [(str(n), int(r)) for n, r in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    return plan_shard("test", roles=roles, columns=columns,
                      out_partitions=[r for _, r in roles], in_size=columns,
                      tp_rank=0, tp_size=1, input_size=columns, output_size=rows)


def _row_plan(declared, tp_rank, tp_size):
    from tessera.serving.sharding import plan_shard

    roles = [(str(n), int(r)) for n, r in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    return plan_shard("test", roles=roles, columns=columns,
                      out_partitions=[r // tp_size for _, r in roles],
                      in_size=columns, tp_rank=tp_rank, tp_size=tp_size,
                      input_size=columns, output_size=rows)


def _col_plan(declared, tp_rank, tp_size):
    from tessera.serving.sharding import plan_shard

    roles = [(str(n), int(r)) for n, r in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    return plan_shard("test", roles=roles, columns=columns,
                      out_partitions=[r for _, r in roles],
                      in_size=columns // tp_size, tp_rank=tp_rank,
                      tp_size=tp_size, input_size=columns, output_size=rows)


def _reference_product(parsed_roles, plan, x):
    """The retained reference preparations' product: values * row scale, fp32."""
    import tessera.serving.bf16_route as bf16_route

    from tessera.serving.sharding import shard_parsed_roles

    roles = shard_parsed_roles(parsed_roles, plan)
    module = bf16_route.prepare_tessera_bf16_module(roles, device="cuda")
    values = module.decode()
    scale = module.row_scale()
    return (x.float() @ (values.float() * scale[:, None]).t()).bfloat16()


def _tolerance(reference):
    return 5e-3 + 1e-2 * float(reference.float().abs().max())


@cuda
def test_the_launch_table_publishes_the_native_pair():
    from tessera.serving import telemetry
    from tessera.serving.scheme import (TESSERA_BF16, TESSERA_FP8,
                                        WINDOW_GEMM_SYMBOL, launch_pairs)

    pair = (WINDOW_GEMM_SYMBOL, telemetry.DECODER_NATIVE_WINDOW_GEMM)
    for route in (TESSERA_FP8, TESSERA_BF16):
        for regime in ("decode", "batch"):
            # ATTESTED since contract v34 (tessera#545).  The pair was
            # experimental -- in the routes' census expectation and out of the
            # contract validator's default view -- until four served censuses
            # on the sm_121 serve image put all 112 declared modules on it in
            # both regimes and both residencies
            # (docs/measurements/tessera-window-gemm-census-2026-09-21.md).
            # It left scheme.EXPERIMENTAL_LAUNCHES with the four dense cells
            # that name it, so the default view and the census opt-in now
            # agree, which is what the two assertions below say.
            assert pair in launch_pairs(route, regime=regime), (route, regime)
            assert pair in launch_pairs(route, regime=regime,
                                        include_experimental=True), (route, regime)
            for mode in ("resident", "streamed"):
                assert pair in launch_pairs(route, regime=regime, mode=mode), (
                    route, regime, mode)
    assert WINDOW_GEMM_SYMBOL == "tessera::window_gemm_dense"
    assert telemetry.DECODER_NATIVE_WINDOW_GEMM in telemetry.DECODERS


@cuda
def test_the_two_readers_agree_on_an_actual_wire():
    """Compact and materialising parses accept the same bytes and refuse the
    same wrong sidecar, with the same words -- on the shipping BF16 wire."""
    from tessera.serving.scheme import (parse_compact_blob_for_scheme,
                                        parse_tessera_blob_for_scheme)

    scheme = _scheme(DOWN_GROUP)
    blob = _wire("model.layers.0.mlp.down_proj.wire_bytes")
    materialised = parse_tessera_blob_for_scheme(blob, scheme, "test")
    compact = parse_compact_blob_for_scheme(blob, scheme, "test", device="cuda")
    assert [name for name, _ in materialised] == [name for name, _ in compact]
    assert compact[0][1].role_facts == {"grid": "BF16", "body": "WINDOW",
                                        "plane": "CHANNEL", "q256": 1792,
                                        "rows": 1024, "columns": 3072, "span": 1}
    with pytest.raises(ValueError, match="sidecar scheme declares") as compact_refusal:
        parse_compact_blob_for_scheme(blob, {**scheme, "q256": 1024}, "test",
                                      device="cuda")
    with pytest.raises(ValueError, match="sidecar scheme declares") as materialised_refusal:
        parse_tessera_blob_for_scheme(blob, {**scheme, "q256": 1024}, "test")
    assert str(compact_refusal.value) == str(materialised_refusal.value)


@cuda
@pytest.mark.parametrize("m", [0, 1, 8, 9, 15, 17, 32, 128])
def test_the_native_module_serves_the_reference_product(m):
    """The actual BF16 R7 unit, whole module: every M tail equals the retained
    reference preparation's product (fp32 accumulate, row scale, one cast)."""
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import TESSERA_BF16, validate_tessera_scheme

    scheme = _scheme(DOWN_GROUP)
    declared = validate_tessera_scheme(scheme, "test")
    blob = _wire("model.layers.0.mlp.down_proj.wire_bytes")
    from tessera.serving.scheme import parse_compact_blob_for_scheme, parse_tessera_blob_for_scheme

    compact = parse_compact_blob_for_scheme(blob, scheme, "test", device="cuda")
    parsed = parse_tessera_blob_for_scheme(blob, scheme, "test", device="cuda")
    plan = _whole_plan(declared)
    module = prepare_dense_native_module(compact, plan, family=TESSERA_BF16, device="cuda")
    columns = int(declared["columns"])
    if m == 0:
        x = torch.empty(0, columns, dtype=torch.bfloat16, device="cuda")
        y = module.apply(x)
        assert tuple(y.shape) == (0, int(declared["rows"]))
        return
    x = torch.randn(m, columns, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(m))
    got = module.apply(x)
    want = _reference_product(parsed, plan, x)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    error = float((got.float() - want.float()).abs().max())
    assert error < _tolerance(want), (m, error, _tolerance(want))


@cuda
def test_native_weights_stay_packed_across_forwards():
    """No decoded weight tensor, at load or after forwards: the bundle's bytes
    and tensor identities are unchanged and stay under the 16-bit tile."""
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import TESSERA_BF16, parse_compact_blob_for_scheme, validate_tessera_scheme

    scheme = _scheme(DOWN_GROUP)
    declared = validate_tessera_scheme(scheme, "test")
    compact = parse_compact_blob_for_scheme(
        _wire("model.layers.0.mlp.down_proj.wire_bytes"), scheme, "test", device="cuda")
    module = prepare_dense_native_module(
        compact, _whole_plan(declared), family=TESSERA_BF16, device="cuda")
    rows, columns = int(declared["rows"]), int(declared["columns"])
    packed = module.packed_bytes()
    assert packed < rows * columns * 2, packed          # the wire, not the bf16 tile
    fingerprints = module.fingerprints()
    facts = module.layout_facts()
    assert len(facts) == 1 and facts[0].rates == (7,) * columns
    x = torch.randn(8, columns, dtype=torch.bfloat16, device="cuda")
    module.apply(x)
    module.apply(x)
    assert module.fingerprints() == fingerprints, "a forward rewrote the prepared weights"
    assert module.packed_bytes() == packed
    for attribute in ("weight_bf16", "weight_fp8", "decode_buf", "tile"):
        assert not hasattr(module, attribute), attribute


@cuda
def test_tp2_row_cut_carries_history_and_matches_the_reference():
    """The actual BF16 gate/up unit at TP2 rank 1: a row cut below row 0, the
    sliced unit's own register as ``initial_state``, and the reference
    product's rows."""
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import (TESSERA_BF16, parse_compact_blob_for_scheme,
                                        parse_tessera_blob_for_scheme, validate_tessera_scheme)

    scheme = _scheme(GATE_UP_GROUP)
    declared = validate_tessera_scheme(scheme, "test")
    blob = _wire("model.layers.0.mlp.gate_up_proj.wire_bytes")
    compact = parse_compact_blob_for_scheme(blob, scheme, "test", device="cuda")
    parsed = parse_tessera_blob_for_scheme(blob, scheme, "test", device="cuda")
    columns = int(declared["columns"])
    for rank in (0, 1):
        plan = _row_plan(declared, rank, 2)
        module = prepare_dense_native_module(compact, plan, family=TESSERA_BF16,
                                             device="cuda")
        facts = module.layout_facts()
        for role_facts, role in zip(facts, declared["roles"]):
            shard = plan.role(str(role[0]))
            assert int(role_facts.row_offset) == shard.lo
            if shard.lo:
                assert role_facts.has_history, "rank 1's cut carries a register"
        x = torch.randn(16, columns, dtype=torch.bfloat16, device="cuda")
        got = module.apply(x)
        want = _reference_product(parsed, plan, x)
        error = float((got.float() - want.float()).abs().max())
        assert error < _tolerance(want), (rank, error)


@cuda
def test_tp2_column_cut_matches_the_reference():
    """The actual BF16 row-parallel unit at TP2 rank 1: a column cut keeps the
    whole row axis and half the input width."""
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import (TESSERA_BF16, parse_compact_blob_for_scheme,
                                        parse_tessera_blob_for_scheme, validate_tessera_scheme)

    scheme = _scheme(DOWN_GROUP)
    declared = validate_tessera_scheme(scheme, "test")
    blob = _wire("model.layers.0.mlp.down_proj.wire_bytes")
    compact = parse_compact_blob_for_scheme(blob, scheme, "test", device="cuda")
    parsed = parse_tessera_blob_for_scheme(blob, scheme, "test", device="cuda")
    columns = int(declared["columns"])
    x_full = torch.randn(16, columns, dtype=torch.bfloat16, device="cuda")
    for rank in (0, 1):
        plan = _col_plan(declared, rank, 2)
        module = prepare_dense_native_module(compact, plan, family=TESSERA_BF16,
                                             device="cuda")
        x = x_full[:, plan.role("down_proj").lo:plan.role("down_proj").hi].contiguous()
        got = module.apply(x)
        want = _reference_product(parsed, plan, x)
        error = float((got.float() - want.float()).abs().max())
        assert error < _tolerance(want), (rank, error)


def test_compact_expert_reader_has_signature_and_refusal_parity():
    """CPU: ``parse_compact_tessera_expert_blob(blob, declared_role, target,
    device="cpu")`` accepts and refuses exactly what
    ``parse_tessera_expert_blob`` does, on an actual routed A4 expert
    container -- the shared reader the MoE owners pick up."""
    from tessera.serving.scheme import (expert_role_declarations,
                                        parse_compact_tessera_expert_blob,
                                        parse_tessera_expert_blob,
                                        validate_tessera_moe_scheme)

    config = box_artifacts.skip_now("a4_export", "config.json")
    settings = json.loads(Path(config).read_text())["quantization_config"]
    group = next(value["scheme"] for value in settings["config_groups"].values()
                 if value["scheme"].get("structure") == "routed_moe")
    declared = validate_tessera_moe_scheme(group, "test")
    declared_role = expert_role_declarations(declared["groups"]["w13"])[0]
    blob = _a4_expert_wire()
    materialised = parse_tessera_expert_blob(blob, declared_role, "test", device="cpu")
    compact = parse_compact_tessera_expert_blob(blob, declared_role, "test", device="cpu")
    assert [name for name, _ in materialised] == [name for name, _ in compact]
    assert compact[0][1].role_facts == {
        "grid": "E2M1x2", "body": "TCQ", "plane": "LUT", "q256": 896,
        "rows": 2048, "columns": 4096, "span": 2,
    }
    # A stride bound that the blob overruns: one refusal, two readers.
    with pytest.raises(ValueError) as materialised_stride:
        parse_tessera_expert_blob(blob, {**declared_role, "wire_stride": len(blob) - 1},
                                  "test", device="cpu")
    with pytest.raises(ValueError) as compact_stride:
        parse_compact_tessera_expert_blob(
            blob, {**declared_role, "wire_stride": len(blob) - 1}, "test", device="cpu")
    assert str(materialised_stride.value) == str(compact_stride.value)
    # A wrong sidecar rung: one comparison, two readers, one sentence.
    with pytest.raises(ValueError) as materialised_rung:
        parse_tessera_expert_blob(blob, {**declared_role, "role_q256": [1152]},
                                  "test", device="cpu")
    with pytest.raises(ValueError) as compact_rung:
        parse_compact_tessera_expert_blob(blob, {**declared_role, "role_q256": [1152]},
                                          "test", device="cpu")
    assert str(materialised_rung.value) == str(compact_rung.value)


def _a4_expert_wire() -> bytes:
    index_path = box_artifacts.skip_now("a4_export", "model.safetensors.index.json")
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    tensor = "model.language_model.layers.3.mlp.experts.0.gate_proj.wire"
    shard = box_artifacts.skip_now("a4_export", weight_map[tensor])
    from safetensors import safe_open

    with safe_open(str(shard), framework="pt") as handle:
        return bytes(handle.get_tensor(tensor).detach().cpu().numpy().tobytes())


@cuda
def test_the_module_owns_its_bundles_with_no_global_registry():
    """Dropping a prepared module releases its tensors: no module-level
    registry keeps a model's weights alive (the token registry this lane
    started with did exactly that)."""
    import gc
    import weakref

    from tessera.serving import native_window
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import (TESSERA_BF16, parse_compact_blob_for_scheme,
                                        validate_tessera_scheme)

    assert not hasattr(native_window, "_BUNDLES"), "a global bundle registry returned"
    scheme = _scheme(DOWN_GROUP)
    declared = validate_tessera_scheme(scheme, "test")
    compact = parse_compact_blob_for_scheme(
        _wire("model.layers.0.mlp.down_proj.wire_bytes"), scheme, "test", device="cuda")
    module = prepare_dense_native_module(
        compact, _whole_plan(declared), family=TESSERA_BF16, device="cuda")
    # The frozen bundle, by name mangling: the test wants the tensor a serve
    # would hold, not a private accessor on the class.
    role = module._PreparedDenseNativeModule__roles[0]
    words = weakref.ref(role.bundle.words)
    assert words() is not None
    del module, role, compact
    gc.collect()
    assert words() is None, "a bundle outlived its module -- something global holds it"


@cuda
def test_the_registered_op_is_functional_and_faked():
    """The custom op exists, has a fake implementation, and is what a compiled
    forward traces: compiling the module's apply over a fake-input trace
    produces the same bytes eagerly."""
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import TESSERA_BF16, parse_compact_blob_for_scheme, validate_tessera_scheme

    assert hasattr(torch.ops.tessera, "window_gemm_dense")
    scheme = _scheme(DOWN_GROUP)
    declared = validate_tessera_scheme(scheme, "test")
    compact = parse_compact_blob_for_scheme(
        _wire("model.layers.0.mlp.down_proj.wire_bytes"), scheme, "test", device="cuda")
    module = prepare_dense_native_module(
        compact, _whole_plan(declared), family=TESSERA_BF16, device="cuda")
    x = torch.randn(16, int(declared["columns"]), dtype=torch.bfloat16, device="cuda")
    eager = module.apply(x)
    compiled = torch.compile(lambda a: module.apply(a), fullgraph=True)(x)
    assert torch.equal(compiled, eager)
