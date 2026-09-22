"""The load bench's NVFP4 routed arm: the seam, the shard vocabulary, the accounting.

``experiments/bench_routed_load.py`` grew an arm that measures what one rank's
routed NVFP4 experts occupy while a body loads (tessera#492, tessera#507).  A
footprint is only worth its receipt if the bench drives the route's OWN loader,
with the runtime's own shard ids, and counts the buffers the route actually
holds -- a bench that quietly loaded nothing would report a very comfortable
number.  So this pins:

* the checkpoint helpers read a parts-style directory (``tessera_part_config.json``)
  as well as a merged one, and find the layers whose experts carry wires;
* ``NVFP4_SHARDS`` is the runtime's shard vocabulary (``scheme.MOE_GROUP_SHARDS``),
  not a spelling of the projection names;
* the stubbed vLLM seam and the layer stub really do drive
  ``build_tessera_nvfp4_moe_method`` -> ``create_weights`` -> ``_load_wire``,
  and every expert's axis slot comes out non-zero;
* ``_nvfp4_resident`` is the expert axes plus the layer's A-side scale rows,
  which is what the per-rank figure is summed from.  The stock modelopt tiles
  are zero-size anchors on this route, so counting them would count nothing.

The helper tests run anywhere.  The two loader tests need CUDA: the route
prepares each wire with the span-2 CUDA packers, so there is no CPU intake to
test.
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

HIDDEN, INTER, EXPERTS, Q256 = 64, 64, 2, 896
LAYER = 1
TARGET = f"model.language_model.layers.{LAYER}.mlp.experts"
PROJECTIONS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN),
               ("down_proj", HIDDEN, INTER))
#: The stacked planes ``A4UnitStack`` takes, one per axis field.
STACK_FIELDS = ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
                "code_nibbles")


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
    axes = method.intake_axes()
    return {key: dict(axes[key].named_tensors()) for key in _bench().NVFP4_AXES}


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

    # The stock tiles are anchors; the prepared planes live on the axes.
    assert holder.w13_weight.numel() == 0 and holder.w2_weight.numel() == 0
    assert set(method.intake_axes()) == set(bench.NVFP4_AXES)
    planes = _axis_planes(method)
    for key, fields in planes.items():
        assert set(fields) == set(STACK_FIELDS) | {"globals"}, key
        assert all(tensor.device.type == "cuda" for tensor in fields.values()), key
        for expert in range(EXPERTS):
            assert (fields["nibbles"][expert] != 0).any(), (key, expert)
            assert float(fields["globals"][expert]) > 0.0, (key, expert)
    assert bool(torch.isfinite(holder.w13_input_global_scale).all())
    assert bool(torch.isfinite(holder.w2_input_global_scale).all())

    axes = sum(_nbytes(tensor) for fields in planes.values() for tensor in fields.values())
    scale_rows = EXPERTS * 2 * 4 + EXPERTS * 4       # w13/w2 input_global_scale, fp32
    assert axes > 0
    assert bench._nvfp4_resident(holder, method) == axes + scale_rows


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
    for key in bench.NVFP4_AXES:
        for field, tensor in per_rank[0][key].items():
            other = per_rank[1][key][field]
            assert (tuple(tensor.shape), tensor.dtype) == (tuple(other.shape), other.dtype), \
                (key, field)
        # The E2M1 nibble plane is rows/2 x cols/16 per expert, so a cut of
        # either the rows (w13) or the columns (w2) halves it exactly.
        assert 2 * _nbytes(per_rank[0][key]["nibbles"]) == _nbytes(single[key]["nibbles"]), key
        # The per-expert globals are not cut.
        assert _nbytes(per_rank[0][key]["globals"]) == _nbytes(single[key]["globals"]), key
        for rank in (0, 1):
            assert (per_rank[rank][key]["nibbles"] != 0).any(), (rank, key)

    resident = {rank: bench._nvfp4_resident(holder, method)
                for rank, (holder, method) in ranks.items()}
    assert resident[0] == resident[1]
    assert resident[0] < bench._nvfp4_resident(*whole)
