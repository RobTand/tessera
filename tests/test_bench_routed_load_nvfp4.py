"""The routed load observer reads real checkpoint frames and declared tensors.

The CPU cases check checkpoint discovery and bounded input reads. The GPU
cases drive the actual WINDOW intake and verify resident bytes and TP cuts.
The vLLM base-class substitutes do not prove production runtime integration.
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

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the route prepares wires with CUDA packers")

HIDDEN, INTER, EXPERTS, Q256 = 256, 512, 2, 896
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
    export = pytest.importorskip("tessera.export")
    alphabet = pytest.importorskip("tessera.alphabet")
    fused = pytest.importorskip("tessera.fused")
    grid = alphabet.tuple_grid(alphabet.E2M1_GRID, 2)
    recipe = export.served_recipe(grid, Q256, "routed_moe")
    weight = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * 0.02
    if torch.cuda.is_available():
        weight = weight.cuda()
    exported, _unit, _forests = export.encode_linear_planes(
        weight.contiguous(), grid=grid, q256=Q256, name=name, verify=False,
        body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma)
    return fused.pack_fused([(name, rows, exported.blob)])


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A read-only checkpoint fixture shared by this module's consumers."""
    from safetensors.torch import save_file
    tmp_path = tmp_path_factory.mktemp("routed-window-observer")

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
        "body": "WINDOW", "span": 1, "plane": "LUT", "experts": EXPERTS,
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


def _load_layer(bench, held, scheme, *, rank, tp_size, scales=True):
    """One layer through the route's own ``create_weights``/``_load_wire``."""
    holder = bench._nvfp4_layer(rank=rank, tp_size=tp_size, experts=EXPERTS, topk=2,
                                swiglu_limit=10.0)
    method = nvfp4_moe_route.build_tessera_nvfp4_moe_method(scheme, TARGET, "resident", holder)
    with torch.device("cuda"):
        method.create_weights(holder, EXPERTS, HIDDEN, INTER // tp_size, torch.bfloat16)
    for expert in range(EXPERTS):
        for shard, _projection in bench.NVFP4_SHARDS:
            group = "w2" if shard == "w2" else "w13"
            param = holder.w2_wire if group == "w2" else holder.w13_wire
            assert param.weight_loader(param, held[(expert, shard, "wire")], "wire",
                                       shard, expert, return_success=True)
            if scales:
                scale = (holder.w2_input_global_scale if group == "w2"
                         else holder.w13_input_global_scale)
                scale.weight_loader(scale, held[(expert, shard, "input_global_scale")],
                                    "input_global_scale", shard, expert)
    return holder, method


def _axis_planes(method):
    return {key: dict(axis.named_tensors()) for key, axis in method.intake_axes().items()}


def _nbytes(tensor):
    return int(tensor.numel()) * tensor.element_size()


@cuda
def test_the_stubbed_seam_drives_the_real_loader_and_the_accounting(
        checkpoint, restore_vllm_modules):
    """The bench's seam and layer stub load real wires through the route's own
    ``create_weights``/``_load_wire``, and ``_nvfp4_resident`` is the expert
    axes plus the A-side scale rows -- the quantity the per-rank figure is
    summed from."""
    bench = _bench()
    data, scheme = checkpoint
    index = bench._checkpoint_index(data, [LAYER])
    directory, weight_map = index[LAYER]
    held, _wire_bytes = bench._read_layer(directory, weight_map, LAYER, EXPERTS)

    bench._stub_vllm_oracle()
    holder, method = _load_layer(bench, held, scheme, rank=0, tp_size=1)

    from tessera.serving.residency import resident_storage_bytes
    planes = _axis_planes(method)
    declared = [(f'{group}.{field}', tensor) for group, fields in planes.items()
                for field, tensor in fields.items()]
    declared.extend([('gs13', holder.w13_input_global_scale),
                     ('gs2', holder.w2_input_global_scale)])
    assert bench._nvfp4_resident(holder, method) == resident_storage_bytes(declared)


@cuda
def test_a_rank_holds_its_own_half_at_tp2(checkpoint, restore_vllm_modules):
    """At TP2 each rank's axes hold that rank's cut, so the resident figure the
    bench sums is the rank's -- which is the whole point of a per-rank fit."""
    bench = _bench()
    data, scheme = checkpoint
    index = bench._checkpoint_index(data, [LAYER])
    directory, weight_map = index[LAYER]
    held, _wire_bytes = bench._read_layer(directory, weight_map, LAYER, EXPERTS)

    bench._stub_vllm_oracle()
    ranks = {rank: _load_layer(bench, held, scheme, rank=rank, tp_size=2, scales=False)
             for rank in (0, 1)}
    whole = _load_layer(bench, held, scheme, rank=0, tp_size=1, scales=False)

    per_rank = {rank: _axis_planes(method) for rank, (_holder, method) in ranks.items()}
    single = _axis_planes(whole[1])
    for group, fields in single.items():
        for field, tensor in fields.items():
            if field.endswith('.global_scale'):
                assert torch.equal(per_rank[0][group][field], tensor)
                assert torch.equal(per_rank[1][group][field], tensor)
            elif field.endswith('.scale_plane'):
                if group == 'w13':
                    shape = (EXPERTS, HIDDEN // 16, INTER // 4)
                    reconstructed = torch.cat([per_rank[r][group][field].reshape(shape)
                                               for r in (0, 1)], dim=2)
                else:
                    shape = (EXPERTS, INTER // 32, HIDDEN // 2)
                    reconstructed = torch.cat([per_rank[r][group][field].reshape(shape)
                                               for r in (0, 1)], dim=1)
                assert torch.equal(reconstructed.flatten(), tensor.flatten())

    resident = {rank: bench._nvfp4_resident(holder, method)
                for rank, (holder, method) in ranks.items()}
    assert resident[0] == resident[1]
    assert resident[0] < bench._nvfp4_resident(*whole)
