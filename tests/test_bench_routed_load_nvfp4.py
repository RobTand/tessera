"""The load bench's NVFP4 routed arm: the seam, the shard vocabulary, the accounting.

``experiments/bench_routed_load.py`` grew an arm that measures what one rank's
routed NVFP4 experts occupy while a body loads (tessera#492, tessera#507).  A
footprint is only worth its receipt if the bench drives the route's OWN loader,
with the runtime's own shard ids, and counts the tiles the route actually
allocates -- a bench that quietly loaded nothing would report a very comfortable
number.  So this pins:

* the checkpoint helpers read a parts-style directory (``tessera_part_config.json``)
  as well as a merged one, and find the layers whose experts carry wires;
* ``NVFP4_SHARDS`` is the runtime's shard vocabulary (``scheme.MOE_GROUP_SHARDS``),
  not a spelling of the projection names;
* the stubbed vLLM seam and the layer stub really do drive
  ``build_tessera_nvfp4_moe_method`` -> ``create_weights`` -> ``_load_wire``,
  and the tiles come out non-zero;
* ``_nvfp4_resident`` equals the stock tile arithmetic, which is what the
  per-rank figure is summed from.

It runs on the CPU: the platform gate no-ops where there is no device, and the
device-resident half is what the GPU bench measures.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import nvfp4_moe_route                        # noqa: E402
from tessera.serving.scheme import MOE_GROUP_SHARDS                # noqa: E402

HIDDEN, INTER, EXPERTS, Q256 = 64, 64, 2, 896
LAYER = 1
TARGET = f"model.language_model.layers.{LAYER}.mlp.experts"
PROJECTIONS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN),
               ("down_proj", HIDDEN, INTER))


def _bench():
    """The experiment script, imported by path: ``experiments`` is not a package."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "bench_routed_load.py"
    spec = importlib.util.spec_from_file_location("bench_routed_load", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def restore_vllm_modules():
    """``_stub_vllm_oracle`` writes into ``sys.modules``; put it back."""
    before = dict(sys.modules)
    yield
    for name in [n for n in sys.modules if n == "vllm" or n.startswith("vllm.")]:
        del sys.modules[name]
    sys.modules.update({k: v for k, v in before.items() if k == "vllm" or k.startswith("vllm.")})


def _encode(rows, cols, name, seed):
    """One E2M1x2/896 container on the CPU."""
    export = pytest.importorskip("tessera.export")
    alphabet = pytest.importorskip("tessera.alphabet")
    fused = pytest.importorskip("tessera.fused")

    grid = alphabet.tuple_grid(alphabet.E2M1_GRID, 2)
    weight = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * 0.02
    exported, _unit, _forests = export.encode_linear_planes(
        weight.contiguous(), grid=grid, q256=Q256, name=name, verify=False)
    return fused.pack_fused([(name, rows, exported.blob)])


@pytest.fixture
def checkpoint(tmp_path):
    """A one-layer parts-style checkpoint: expert wires, A-side scales, sidecar."""
    from safetensors.torch import save_file

    tensors, strides = {}, {"w13": 0, "w2": 0}
    for expert in range(EXPERTS):
        for index, (projection, rows, cols) in enumerate(PROJECTIONS):
            blob = _encode(rows, cols, projection, seed=17 * expert + index)
            group = "w2" if projection == "down_proj" else "w13"
            strides[group] = max(strides[group], len(blob))
            tensors[f"{TARGET}.{expert}.{projection}.wire"] = torch.frombuffer(
                bytearray(blob), dtype=torch.uint8).clone()
            tensors[f"{TARGET}.{expert}.{projection}.input_global_scale"] = torch.tensor(
                [2.0 + expert + index], dtype=torch.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {key: "model.safetensors" for key in tensors}}))
    scheme = {
        "family": "TESSERA_NVFP4", "structure": "routed_moe", "grid": "E2M1x2",
        "body": "TCQ", "plane": "LUT", "experts": EXPERTS,
        "groups": {
            "w13": {"q256": Q256, "rows": 2 * INTER, "columns": HIDDEN,
                    "roles": [["gate_proj", INTER], ["up_proj", INTER]],
                    "wire_stride": strides["w13"]},
            "w2": {"q256": Q256, "rows": HIDDEN, "columns": INTER,
                   "roles": [["down_proj", HIDDEN]], "wire_stride": strides["w2"]}},
    }
    # A PART spells its sidecar this way; a merged checkpoint says config.json.
    (tmp_path / "tessera_part_config.json").write_text(json.dumps({
        "quantization_config": {
            "quant_method": "tessera", "format": "mixed-precision",
            "config_groups": {"tessera_experts": {
                "format": "TESSERA", "targets": [TARGET], "scheme": scheme}},
            "ignore": []}}))
    return tmp_path, scheme


def test_shard_vocabulary_is_the_runtimes():
    """The bench loads under ``w1``/``w3``/``w2``, which is what ``_load_wire``
    resolves through ``SHARD_TO_GROUP`` -- not the projection names."""
    bench = _bench()
    assert tuple(shard for shard, _ in bench.NVFP4_SHARDS) == (
        MOE_GROUP_SHARDS["w13"] + MOE_GROUP_SHARDS["w2"])
    assert [projection for _, projection in bench.NVFP4_SHARDS] == [
        "gate_proj", "up_proj", "down_proj"]
    for shard, _projection in bench.NVFP4_SHARDS:
        assert shard in nvfp4_moe_route.SHARD_TO_GROUP


def test_the_helpers_read_a_parts_directory(checkpoint):
    """A part carries ``tessera_part_config.json`` and its own index; the layer
    list comes off the wires, so a passthrough stack is not offered as one."""
    bench = _bench()
    data, scheme = checkpoint
    assert bench._moe_layers(data) == [LAYER]
    index = bench._checkpoint_index(data, [LAYER])
    assert set(index) == {LAYER}
    directory, weight_map = index[LAYER]
    assert directory == data
    raw, declared = bench._expert_scheme(directory, LAYER)
    assert raw == scheme
    assert declared["family"] == "TESSERA_NVFP4"
    assert declared["hidden_size"] == HIDDEN
    assert declared["intermediate_size"] == INTER
    assert bench._family(data, [LAYER]) == "TESSERA_NVFP4"

    held, wire_bytes = bench._read_layer(directory, weight_map, LAYER, EXPERTS)
    assert len(held) == EXPERTS * len(bench.NVFP4_SHARDS) * 2
    wires = [value for key, value in held.items() if key[2] == "wire"]
    assert len(wires) == EXPERTS * 3
    assert all(tensor.dtype == torch.uint8 for tensor in wires)
    assert wire_bytes == sum(int(tensor.numel()) for tensor in wires)


def test_a_layer_streams_with_a_bounded_read_ahead(checkpoint):
    bench = _bench()
    data, _scheme = checkpoint
    index = bench._checkpoint_index(data, [LAYER])
    streamed = list(bench._layer_stream(index, [LAYER], EXPERTS, read_ahead=1))
    assert [layer for layer, _held, _bytes in streamed] == [LAYER]


def test_the_stubbed_seam_drives_the_real_loader_and_the_accounting(
        checkpoint, restore_vllm_modules):
    """The bench's seam and layer stub load real wires through the route's own
    ``create_weights``/``_load_wire``, and ``_nvfp4_resident`` is the stock tile
    arithmetic -- the quantity the per-rank figure is summed from."""
    bench = _bench()
    data, scheme = checkpoint
    index = bench._checkpoint_index(data, [LAYER])
    directory, weight_map = index[LAYER]
    held, _wire_bytes = bench._read_layer(directory, weight_map, LAYER, EXPERTS)

    bench._stub_vllm_oracle()
    holder = bench._nvfp4_layer(rank=0, tp_size=1, experts=EXPERTS, topk=2, swiglu_limit=10.0)
    method = nvfp4_moe_route.build_tessera_nvfp4_moe_method(scheme, TARGET, "resident", holder)
    method.create_weights(holder, EXPERTS, HIDDEN, INTER, torch.bfloat16)
    assert torch.equal(holder.w13_weight, torch.zeros_like(holder.w13_weight))

    for expert in range(EXPERTS):
        for shard, _projection in bench.NVFP4_SHARDS:
            group = "w2" if shard == "w2" else "w13"
            param = holder.w2_wire if group == "w2" else holder.w13_wire
            assert param.weight_loader(param, held[(expert, shard, "wire")], "wire",
                                       shard, expert, return_success=True)
            scale = holder.w2_input_global_scale if group == "w2" else holder.w13_input_global_scale
            scale.weight_loader(scale, held[(expert, shard, "input_global_scale")],
                                "input_global_scale", shard, expert)

    # The decode wrote bytes into every expert's slot.
    assert (holder.w13_weight != 0).any()
    assert (holder.w2_weight != 0).any()
    for expert in range(EXPERTS):
        assert (holder.w13_weight[expert] != 0).any()
        assert float(holder.w13_weight_scale_2[expert][0]) > 0.0
        assert float(holder.w13_weight_scale_2[expert][0]) == float(
            holder.w13_weight_scale_2[expert][1])
    assert bool(torch.isfinite(holder.w13_input_global_scale).all())
    assert bool(torch.isfinite(holder.w2_input_global_scale).all())

    expected = (
        EXPERTS * 2 * INTER * (HIDDEN // 2)          # w13_weight, uint8 nibbles
        + EXPERTS * HIDDEN * (INTER // 2)            # w2_weight
        + EXPERTS * 2 * INTER * (HIDDEN // 16)       # w13_weight_scale, ue4m3
        + EXPERTS * HIDDEN * (INTER // 16)           # w2_weight_scale
        + EXPERTS * 2 * 4 + EXPERTS * 4              # weight_scale_2, fp32
        + EXPERTS * 2 * 4 + EXPERTS * 4              # input_scale, fp32
    )
    assert bench._nvfp4_resident(holder) == expected


def test_a_rank_holds_its_own_half_at_tp2(checkpoint, restore_vllm_modules):
    """At TP2 the tile is cut, so the resident figure the bench sums is the
    rank's -- which is the whole point of a per-rank fit."""
    bench = _bench()
    data, scheme = checkpoint
    index = bench._checkpoint_index(data, [LAYER])
    directory, weight_map = index[LAYER]
    held, _wire_bytes = bench._read_layer(directory, weight_map, LAYER, EXPERTS)

    bench._stub_vllm_oracle()
    resident = {}
    for rank in (0, 1):
        holder = bench._nvfp4_layer(rank=rank, tp_size=2, experts=EXPERTS, topk=2,
                                    swiglu_limit=10.0)
        method = nvfp4_moe_route.build_tessera_nvfp4_moe_method(scheme, TARGET, "resident", holder)
        method.create_weights(holder, EXPERTS, HIDDEN, INTER // 2, torch.bfloat16)
        for expert in range(EXPERTS):
            for shard, _projection in bench.NVFP4_SHARDS:
                group = "w2" if shard == "w2" else "w13"
                param = holder.w2_wire if group == "w2" else holder.w13_wire
                param.weight_loader(param, held[(expert, shard, "wire")], "wire",
                                    shard, expert, return_success=True)
        resident[rank] = bench._nvfp4_resident(holder)
        assert (holder.w13_weight != 0).any()

    single = bench._nvfp4_layer(rank=0, tp_size=1, experts=EXPERTS, topk=2, swiglu_limit=10.0)
    whole = nvfp4_moe_route.build_tessera_nvfp4_moe_method(scheme, TARGET, "resident", single)
    whole.create_weights(single, EXPERTS, HIDDEN, INTER, torch.bfloat16)
    assert resident[0] == resident[1]
    # The scalar per-expert rows (globals and A-side scales) are not cut.
    scalars = EXPERTS * (2 * 4 + 4) * 2
    assert resident[0] - scalars == (bench._nvfp4_resident(single) - scalars) // 2
