"""The NVFP4 expert route's load half, on real E2M1x2 wires (tessera#492).

WHAT THIS FILE CAN COVER AND WHAT IT CANNOT.  Same boundary as
``test_serving_moe_route.py``: the route's ``apply`` hands vLLM's own NVFP4
fused-MoE modular kernel vLLM's own parameters, and neither that kernel nor
``RoutedExperts.load_weights`` exists here -- vendoring the runtime is
forbidden (AGENTS.md).  So vLLM's ``oracle.nvfp4`` seam is STUBBED with a
kernel that computes the float reference from the operands it is handed (a
spy that also does the arithmetic), and what is pinned is the half that is
ours:

* the decoded tile IS ``tessera.stock.materialize_stock``'s after the
  per-expert global join (``fused.shared_lut_global``), byte for byte, expert
  by expert, gate at rows ``[0:N]`` and up at ``[N:2N]``;
* the per-expert global handed to the kernel is the MULTIPLIER (modelopt
  ``weight_scale_2``), the reciprocal of the stock tile's
  ``weight_global_scale`` divisor -- checked numerically through
  ``stock_dequant`` and not by name, because a divisor read as a multiplier
  is a plausible-looking wrong serve;
* the static A-side scale is inverted once (``input_scale = 1 /
  input_global_scale``) and a missing one refuses;
* at TP2 each rank decodes and holds its own ``w13`` rows and ``w2`` columns;
* the refusals: EP/EPLB, non-gated, streamed, another family, geometry
  disagreement, half an expert, a stock tensor name.

The load-and-execute half on the pinned image is
``experiments/nvfp4_moe_route_load_probe.py``; the A side (the kernel's own
group-16 quantisation under the static scale) is measured there, not here.
"""
from __future__ import annotations

import enum
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import nvfp4_moe_route                        # noqa: E402
from tessera.serving.scheme import (                               # noqa: E402
    STRUCTURE_ROUTED_MOE, TESSERA_NVFP4, experimental_launch_pairs, launch_pairs)

HIDDEN, INTER, EXPERTS, Q256 = 64, 64, 3, 896
SHARDS = ("w1", "w3", "w2")     # gate, up, down: the runtime's shard ids
KERNEL_PAIR = ("vllm.fused_moe.modular_kernel", "torch_materialize_stock")


def _native_prep_available() -> bool:
    """Whether this interpreter can run the native load path at all.

    ``prepare_a4_unit`` repacks the packed BODY through Triton CUDA kernels,
    so the CPU stub suite below cannot drive it; the pinned image's
    ``experiments/native_a4_serve_probe.py`` covers that ground against real
    vLLM (and is the receipt for the native load/apply protocol).
    """
    try:
        import tokenspeed_triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return torch.cuda.is_available()


needs_native_prep = pytest.mark.skipif(
    not _native_prep_available(),
    reason=("the native load path repacks the packed BODY through Triton on "
            "CUDA; this CPU suite cannot drive it -- see "
            "experiments/native_a4_serve_probe.py in the pinned image"))


def _tessera():
    return (pytest.importorskip("tessera.export"), pytest.importorskip("tessera.stock"),
            pytest.importorskip("tessera.alphabet"), pytest.importorskip("tessera.fused"))


def _encode(rows, cols, name, seed):
    """One E2M1x2/896 unit encoded on the CPU; its container and stock reference."""
    export, stock, alphabet, fused = _tessera()
    grid = alphabet.tuple_grid(alphabet.E2M1_GRID, 2)
    w = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * 0.02
    w[: rows // 4] *= 2.0 ** (seed % 3)      # units land on different globals
    exported, unit, forests = export.encode_linear_planes(
        w.contiguous(), grid=grid, q256=Q256, name=name, verify=False)
    blob = fused.pack_fused([(name, rows, exported.blob)])
    return blob, stock.materialize_stock(unit, forests, export.DEFAULT_CODE)


@pytest.fixture(scope="module")
def stack():
    """E experts of (gate, up, down) E2M1x2 wires, the scheme, and per-expert
    references already moved onto ONE global per w13 (``stock.share_global``:
    the same power-of-two move the loader makes through the LUT plane)."""
    _export, stock, _alphabet, _fused = _tessera()
    wires, reference = {}, []
    for e in range(EXPERTS):
        gate, gate_ref = _encode(INTER, HIDDEN, "gate_proj", 100 + e)
        up, up_ref = _encode(INTER, HIDDEN, "up_proj", 200 + e)
        down, down_ref = _encode(HIDDEN, INTER, "down_proj", 300 + e)
        wires[(e, "w1")], wires[(e, "w3")], wires[(e, "w2")] = gate, up, down
        moved, divisor = stock.share_global({"gate_proj": gate_ref, "up_proj": up_ref})
        reference.append({
            "gate": moved["gate_proj"], "up": moved["up_proj"], "down": down_ref,
            "w13_global": 1.0 / divisor,
            "w2_global": 1.0 / float(down_ref["weight_global_scale"].reshape(-1)[0])})
    scheme = {
        "family": TESSERA_NVFP4, "structure": STRUCTURE_ROUTED_MOE, "grid": "E2M1x2",
        "body": "TCQ", "plane": "LUT", "experts": EXPERTS,
        "groups": {
            "w13": {"rows": 2 * INTER, "columns": HIDDEN, "q256": Q256,
                    "wire_stride": max(len(wires[(e, s)])
                                       for e in range(EXPERTS) for s in ("w1", "w3")),
                    "roles": [["gate_proj", INTER], ["up_proj", INTER]]},
            "w2": {"rows": HIDDEN, "columns": INTER, "q256": Q256,
                   "wire_stride": max(len(wires[(e, "w2")]) for e in range(EXPERTS)),
                   "roles": [["down_proj", HIDDEN]]}},
    }
    return wires, scheme, reference


def _dequant(packed, scale, global_):
    """Tessera's own reading of a stock tile, with ``global_`` the MULTIPLIER."""
    from tessera.stock import stock_dequant

    return stock_dequant({"weight_packed": packed, "weight_scale": scale,
                          "weight_global_scale": torch.tensor([1.0 / float(global_)])})


@pytest.fixture
def nvfp4_runtime(monkeypatch):
    """vLLM's ``oracle.nvfp4`` seam, as the FP8 route's tests stub theirs."""
    # This seam is explicitly CPU arithmetic even on a CUDA test worker.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    names = ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe",
             "vllm.model_executor.layers.fused_moe.fused_moe_method_base",
             "vllm.model_executor.layers.fused_moe.oracle",
             "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
             "vllm.model_executor.layers.quantization",
             "vllm.model_executor.layers.quantization.utils",
             "vllm.model_executor.layers.quantization.utils.quant_utils",
             "vllm.model_executor.utils")
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    base = modules["vllm.model_executor.layers.fused_moe.fused_moe_method_base"]

    class Base:
        def __init__(self, moe):
            self.moe, self.moe_kernel, self.moe_quant_config = moe, None, None

        @property
        def is_monolithic(self):
            return False
    base.FusedMoEMethodBase = Base
    oracle = modules["vllm.model_executor.layers.fused_moe.oracle.nvfp4"]
    oracle.NvFp4MoeBackend = enum.Enum("NvFp4MoeBackend", ["FLASHINFER_CUTLASS", "MARLIN"])
    experts_cls = types.SimpleNamespace(is_monolithic=lambda: False)
    log = {"select": [], "convert": [], "finalized": [], "apply": []}

    def select(**kwargs):
        log["select"].append(kwargs)
        return oracle.NvFp4MoeBackend.FLASHINFER_CUTLASS, experts_cls
    oracle.select_nvfp4_moe_backend = select

    def convert(**kwargs):
        log["convert"].append(kwargs)
        return tuple(kwargs[k] for k in ("w13", "w13_scale", "w13_scale_2", "a13_scale",
                                         "w2", "w2_scale", "w2_scale_2", "a2_scale"))
    oracle.convert_to_nvfp4_moe_kernel_format = convert
    oracle.make_nvfp4_moe_quant_config = lambda **kwargs: kwargs

    class Kernel:
        is_monolithic = False

        def __init__(self, config):
            self.config = config
            self.fused_experts = types.SimpleNamespace(
                process_weights_after_loading=lambda layer: log["finalized"].append(layer))

        def apply(self, x, w13, w2, weights, ids, **kwargs):
            # The arithmetic the operands imply -- values x block scale x
            # global -- through Tessera's own stock_dequant, so "global" means
            # what the runtime means by weight_scale_2.  The A side is NOT
            # quantised here: that is the kernel's own arithmetic and is
            # measured in the container probe, not stubbed.
            out = torch.zeros_like(x)
            for token in range(x.shape[0]):
                for choice in range(ids.shape[1]):
                    e = int(ids[token, choice])
                    first = _dequant(w13[e], self.config["w13_scale"][e], self.config["w13_scale_2"][e])
                    second = _dequant(w2[e], self.config["w2_scale"][e], self.config["w2_scale_2"][e])
                    gate, up = (first @ x[token].float()).chunk(2)
                    out[token] += weights[token, choice] * (
                        second @ (torch.nn.functional.silu(gate) * up))
            log["apply"].append(kwargs)
            return out
    oracle.make_nvfp4_moe_kernel = lambda **kwargs: Kernel(kwargs["moe_quant_config"])
    quant = modules["vllm.model_executor.layers.quantization.utils.quant_utils"]
    quant.kNvfp4Static, quant.kNvfp4Dynamic = object(), object()
    utils = modules["vllm.model_executor.utils"]
    utils.set_weight_attrs = lambda param, attrs: [setattr(param, k, v) for k, v in attrs.items()]
    utils.replace_parameter = lambda layer, name, value: setattr(
        layer, name, torch.nn.Parameter(value, requires_grad=False))
    return oracle, log, quant


def _layer(tp_size=1, tp_rank=0, **moe):
    layer = torch.nn.Module()
    layer.moe_config = types.SimpleNamespace(
        is_act_and_mul=True, experts_per_token=2,
        moe_parallel_config=types.SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, ep_size=1, dp_size=1, pcp_size=1, sp_size=1,
            use_ep=False, enable_eplb=False),
        **moe)
    layer.activation = "silu"
    layer.global_num_experts = EXPERTS
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer._expert_routing_tables = lambda: None
    layer.swiglu_limit = 10.0        # GLM's clamp: it must reach the quant config
    return layer


def _build(scheme, layer, mode="resident"):
    return nvfp4_moe_route.build_tessera_nvfp4_moe_method(scheme, "m", mode, layer)


def _scales():
    """One static A-side scale per (expert, projection), all distinct."""
    return {(e, s): float(2 + 3 * e + i) for e in range(EXPERTS) for i, s in enumerate(SHARDS)}


def _load(method, layer, wires, *, tp_size=1, order=None, scales=None, finish=True):
    method.create_weights(layer, EXPERTS, HIDDEN, INTER // tp_size, torch.bfloat16)
    scales = _scales() if scales is None else scales
    order = order or [(e, s) for e in range(EXPERTS) for s in SHARDS]
    for e, shard in order:
        wire = layer.w2_wire if shard == "w2" else layer.w13_wire
        assert wire.weight_loader(
            wire, torch.frombuffer(bytearray(wires[(e, shard)]), dtype=torch.uint8),
            "wire", shard, e, return_success=True)
        if (e, shard) in scales:
            scale = layer.w2_input_global_scale if shard == "w2" else layer.w13_input_global_scale
            scale.weight_loader(scale, torch.tensor([scales[(e, shard)]], dtype=torch.float32),
                                "input_global_scale", shard, e)
    if finish:
        method.process_weights_after_loading(layer)
    return scales


def _expected(reference, rank=0, tp_size=1):
    """The rank's rows of w13 and columns of w2, from the joined references."""
    lo, hi = rank * INTER // tp_size, (rank + 1) * INTER // tp_size
    w13 = [torch.cat([ref["gate"]["weight_packed"][lo:hi], ref["up"]["weight_packed"][lo:hi]])
           for ref in reference]
    w13_scale = [torch.cat([ref["gate"]["weight_scale"][lo:hi], ref["up"]["weight_scale"][lo:hi]])
                 for ref in reference]
    w2 = [ref["down"]["weight_packed"][:, lo // 2: hi // 2].contiguous() for ref in reference]
    w2_scale = [ref["down"]["weight_scale"][:, lo // 16: hi // 16].contiguous() for ref in reference]
    return w13, w13_scale, w2, w2_scale


def _assert_tiles(layer, reference, rank=0, tp_size=1):
    w13, w13_scale, w2, w2_scale = _expected(reference, rank, tp_size)
    for e in range(EXPERTS):
        assert torch.equal(layer.w13_weight[e], w13[e]), e
        assert torch.equal(layer.w13_weight_scale[e].view(torch.uint8), w13_scale[e].view(torch.uint8)), e
        assert torch.equal(layer.w2_weight[e], w2[e]), e
        assert torch.equal(layer.w2_weight_scale[e].view(torch.uint8), w2_scale[e].view(torch.uint8)), e
        assert float(layer.w13_weight_scale_2[e]) == reference[e]["w13_global"], e
        assert float(layer.w2_weight_scale_2[e]) == reference[e]["w2_global"], e


# --------------------------------------------------------------------------
# the tile
# --------------------------------------------------------------------------

@needs_native_prep
def test_the_tile_is_the_stock_pair_after_the_per_expert_global_join(stack, nvfp4_runtime):
    """Gate at rows [0:N], up at [N:2N], one joined global per expert per
    group, every byte materialize_stock's -- and from finalize onward the
    layer is the modelopt parameter set and nothing else."""
    wires, scheme, reference = stack
    oracle, log, quant = nvfp4_runtime
    layer = _layer()
    method = _build(scheme, layer)
    assert log["select"] == [{"config": layer.moe_config, "weight_key": quant.kNvfp4Static,
                              "activation_key": quant.kNvfp4Dynamic}]
    assert method.supports_eplb is False
    scales = _load(method, layer, wires)
    _assert_tiles(layer, reference)
    for e in range(EXPERTS):
        assert torch.equal(layer.w13_input_scale[e], torch.tensor(
            [1.0 / scales[(e, "w1")], 1.0 / scales[(e, "w3")]], dtype=torch.float32))
        # The inverse lands in float32 (the modelopt parameter dtype), so it is
        # compared as one: 1/7 in Python is the float64 neighbour, not this value.
        assert torch.equal(layer.w2_input_scale[e],
                           torch.tensor(1.0 / scales[(e, "w2")], dtype=torch.float32))
    assert set(dict(layer.named_parameters())) == set(nvfp4_moe_route._STOCK_TILE_NAMES)
    assert layer.tessera_w13_wire_len is None and layer.tessera_w2_wire_len is None
    assert (layer.tessera_decoder, layer.tessera_backend) == ("torch_materialize_stock",
                                                              "FLASHINFER_CUTLASS")
    assert (layer.tessera_rows, layer.tessera_columns) == (2 * INTER, HIDDEN)
    assert layer.tessera_activation_contract == nvfp4_moe_route.ACTIVATION_CONTRACT
    assert (layer.tessera_family, layer.tessera_structure, layer.tessera_mode) == (
        TESSERA_NVFP4, STRUCTURE_ROUTED_MOE, "resident")
    # The modelopt mirror: the runtime's converter and quant config get the
    # runtime's own kwargs, the kernel's own finalizer runs last.
    (convert,) = log["convert"]
    assert convert["nvfp4_backend"] is oracle.NvFp4MoeBackend.FLASHINFER_CUTLASS
    assert convert["layer"] is layer and convert["is_act_and_mul"] is True
    assert convert["use_a16"] is False
    assert torch.equal(convert["w13_scale_2"], layer.w13_weight_scale_2)
    assert tuple(layer.w13_weight_scale_2.shape) == (EXPERTS,)
    assert log["finalized"] == [layer]
    config = method.moe_quant_config
    assert config["backend"] is oracle.NvFp4MoeBackend.FLASHINFER_CUTLASS
    assert config["swiglu_limit"] == 10.0 and config["use_a16"] is False
    assert config["a13_scale"] is layer.w13_input_scale and config["a2_scale"] is layer.w2_input_scale
    assert method.moe_kernel is not None


@needs_native_prep
def test_the_global_handed_to_the_kernel_is_the_multiplier(stack, nvfp4_runtime):
    """weight_scale_2 x block scale x nibble value == the stock dequant of the
    joined reference, exactly; the divisor read as a multiplier is not."""
    from tessera.stock import stock_dequant

    wires, scheme, reference = stack
    layer = _layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    for e in range(EXPERTS):
        want13 = torch.cat([stock_dequant(reference[e]["gate"]), stock_dequant(reference[e]["up"])])
        got13 = _dequant(layer.w13_weight[e], layer.w13_weight_scale[e], layer.w13_weight_scale_2[e])
        assert torch.equal(got13, want13), e
        want2 = stock_dequant(reference[e]["down"])
        got2 = _dequant(layer.w2_weight[e], layer.w2_weight_scale[e], layer.w2_weight_scale_2[e])
        assert torch.equal(got2, want2), e
        wrong = _dequant(layer.w13_weight[e], layer.w13_weight_scale[e],
                         1.0 / float(layer.w13_weight_scale_2[e]))
        assert not torch.equal(wrong, want13), "a divisor would have read as a multiplier"


def test_the_join_moved_a_half_on_at_least_one_expert(stack):
    """The fixture exercises the join: some expert's gate and up were encoded
    on different globals, so the loader had to move one of them."""
    _export, stock, _alphabet, _fused = _tessera()
    moved = 0
    for e in range(EXPERTS):
        _blob, gate_ref = _encode(INTER, HIDDEN, "gate_proj", 100 + e)
        _blob, up_ref = _encode(INTER, HIDDEN, "up_proj", 200 + e)
        own = {float(t["weight_global_scale"].reshape(-1)[0]) for t in (gate_ref, up_ref)}
        moved += len(own) > 1
    assert moved >= 1


@needs_native_prep
def test_apply_hands_the_kernel_operands_that_compute_the_stock_reference(stack, nvfp4_runtime):
    wires, scheme, reference = stack
    from tessera.stock import stock_dequant

    _oracle, log, _quant = nvfp4_runtime
    layer = _layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(4, HIDDEN, generator=generator)
    ids = torch.tensor([[0, 2], [1, 0], [2, 1], [1, 2]], dtype=torch.int32)
    weights = torch.rand(4, 2, generator=generator)
    got = method.apply(layer, x, weights, ids, None, None)
    expected = torch.zeros_like(x)
    for token in range(4):
        for choice in range(2):
            ref = reference[int(ids[token, choice])]
            gate = stock_dequant(ref["gate"]) @ x[token]
            up = stock_dequant(ref["up"]) @ x[token]
            expected[token] += weights[token, choice] * (
                stock_dequant(ref["down"]) @ (torch.nn.functional.silu(gate) * up))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    kwargs = log["apply"][-1]
    assert kwargs["global_num_experts"] == EXPERTS and kwargs["expert_map"] is None
    assert kwargs["activation"] == "silu" and kwargs["apply_router_weight_on_input"] is False


@needs_native_prep
def test_load_order_does_not_matter(stack, nvfp4_runtime):
    """Up before gate, down first, experts interleaved: the w13 join waits
    for both halves and the tile is the same."""
    wires, scheme, reference = stack
    layer = _layer()
    method = _build(scheme, layer)
    order = [(2, "w2"), (0, "w3"), (0, "w1"), (1, "w3"), (2, "w1"), (1, "w1"),
             (0, "w2"), (2, "w3"), (1, "w2")]
    _load(method, layer, wires, order=order)
    _assert_tiles(layer, reference)


@pytest.mark.parametrize("rank", [0, 1])
@needs_native_prep
def test_tp2_ranks_decode_and_hold_their_own_rows_and_columns(stack, nvfp4_runtime, rank):
    """Each rank parses every FULL container, cuts it on the group plan (rows
    of w13, columns of w2) and decodes only its own slice; the joined global
    is a whole-unit fact and is the same on both ranks."""
    from tessera.stock import stock_dequant

    wires, scheme, reference = stack
    layer = _layer(tp_size=2, tp_rank=rank)
    method = _build(scheme, layer)
    _load(method, layer, wires, tp_size=2)
    assert tuple(layer.w13_weight.shape) == (EXPERTS, INTER, HIDDEN // 2)
    assert tuple(layer.w2_weight.shape) == (EXPERTS, HIDDEN, INTER // 4)
    assert (layer.tessera_rows, layer.tessera_columns) == (INTER, HIDDEN)
    _assert_tiles(layer, reference, rank, 2)
    lo, hi = rank * INTER // 2, (rank + 1) * INTER // 2
    x = torch.randn(2, HIDDEN, generator=torch.Generator().manual_seed(11))
    ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    weights = torch.tensor([[.2, .8], [.7, .3]])
    got = method.apply(layer, x, weights, ids, None, None)
    expected = torch.zeros_like(x)
    for token in range(2):
        for choice in range(2):
            ref = reference[int(ids[token, choice])]
            gate = stock_dequant(ref["gate"])[lo:hi] @ x[token]
            up = stock_dequant(ref["up"])[lo:hi] @ x[token]
            expected[token] += weights[token, choice] * (
                stock_dequant(ref["down"])[:, lo:hi] @ (torch.nn.functional.silu(gate) * up))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------

def test_construction_refusals(stack, nvfp4_runtime):
    _wires, scheme, _reference = stack
    with pytest.raises(ValueError, match="unknown residency mode"):
        _build(scheme, _layer(), mode="cached")
    with pytest.raises(ValueError, match="resident"):
        _build(scheme, _layer(), mode="streamed")
    layer = _layer()
    layer.moe_config.is_act_and_mul = False
    with pytest.raises(ValueError, match="not gated"):
        _build(scheme, layer)
    for field, value in (("use_ep", True), ("ep_size", 2), ("enable_eplb", True)):
        layer = _layer()
        setattr(layer.moe_config.moe_parallel_config, field, value)
        with pytest.raises(ValueError, match="expert parallelism / EPLB"):
            _build(scheme, layer)
    from tests.test_serving_moe_route import _stack as fp8_stack
    _w13, _w2, fp8_scheme, _ref = fp8_stack(experts=1, hidden=HIDDEN, inter=32)
    with pytest.raises(ValueError, match="serves TESSERA_NVFP4"):
        _build(fp8_scheme, _layer())


def test_geometry_refusals_arrive_at_create_weights(stack, nvfp4_runtime):
    _wires, scheme, _reference = stack
    for args, kwargs, message in (
            ((EXPERTS + 1, HIDDEN, INTER), {}, "experts"),
            ((EXPERTS, HIDDEN, INTER), {"global_num_experts": EXPERTS + 1}, "experts"),
            ((EXPERTS, HIDDEN + 16, INTER), {}, "hidden_size"),
            ((EXPERTS, HIDDEN, INTER // 2), {}, "intermediate size"),
            ((EXPERTS, HIDDEN, INTER + 8), {}, "intermediate size")):
        layer = _layer()
        method = _build(scheme, layer)
        with pytest.raises(ValueError, match=message):
            method.create_weights(layer, *args, torch.bfloat16, **kwargs)
    layer = _layer(tp_size=2, tp_rank=0)
    method = _build(scheme, layer)
    with pytest.raises(ValueError, match="intermediate size"):
        method.create_weights(layer, EXPERTS, HIDDEN, INTER, torch.bfloat16)


@needs_native_prep
def test_wire_and_scale_loader_refusals(stack, nvfp4_runtime):
    wires, scheme, _reference = stack
    layer = _layer()
    method = _build(scheme, layer)
    method.create_weights(layer, EXPERTS, HIDDEN, INTER, torch.bfloat16)
    wire = layer.w13_wire
    good = torch.frombuffer(bytearray(wires[(0, "w1")]), dtype=torch.uint8)
    with pytest.raises(ValueError, match="shard_id"):
        wire.weight_loader(wire, good, "wire", "w9", 0)
    with pytest.raises(ValueError, match="outside"):
        wire.weight_loader(wire, good, "wire", "w1", EXPERTS)
    with pytest.raises(ValueError, match="uint8"):
        wire.weight_loader(wire, good.float(), "wire", "w1", 0)
    stride = scheme["groups"]["w13"]["wire_stride"]
    with pytest.raises(ValueError, match="wire_stride"):
        wire.weight_loader(wire, torch.zeros(stride + 1, dtype=torch.uint8), "wire", "w1", 0)
    with pytest.raises(ValueError, match="wire_stride"):
        wire.weight_loader(wire, torch.zeros(0, dtype=torch.uint8), "wire", "w1", 0)
    wire.weight_loader(wire, good, "wire", "w1", 0)
    with pytest.raises(ValueError, match="already loaded"):
        wire.weight_loader(wire, good, "wire", "w1", 0)
    scale = layer.w13_input_global_scale
    for bad, message in ((torch.tensor([0.0]), "finite positive"),
                         (torch.tensor([float("nan")]), "finite positive"),
                         (torch.tensor([-2.0]), "finite positive"),
                         (torch.tensor([1.0, 2.0]), "one floating scalar"),
                         (torch.tensor([3], dtype=torch.int64), "one floating scalar")):
        with pytest.raises(ValueError, match=message):
            scale.weight_loader(scale, bad, "input_global_scale", "w1", 0)
    scale.weight_loader(scale, torch.tensor([4.0]), "input_global_scale", "w1", 0)
    with pytest.raises(ValueError, match="already loaded"):
        scale.weight_loader(scale, torch.tensor([4.0]), "input_global_scale", "w1", 0)
    with pytest.raises(ValueError, match="does not belong to group"):
        scale.weight_loader(scale, torch.tensor([4.0]), "input_global_scale", "w2", 0)
    tile = layer.w13_weight
    with pytest.raises(ValueError, match="stock tensor"):
        tile.weight_loader(tile, torch.zeros(1, dtype=torch.uint8),
                           "experts.0.gate_proj.weight", "w1", 0)


@needs_native_prep
def test_finalize_refuses_an_incomplete_stack(stack, nvfp4_runtime):
    wires, scheme, _reference = stack
    # A missing A-side scale on one projection.
    layer = _layer()
    method = _build(scheme, layer)
    scales = _scales()
    del scales[(1, "w3")]
    with pytest.raises(ValueError, match="input_global_scale"):
        _load(method, layer, wires, scales=scales)
    # Half an expert: the up half never arrived.  The length companion refuses
    # first (``moe_layout.validate_moe_wire_lengths``: an empty wire was never
    # packable), the half-expert check stands behind it.
    from tessera.errors import GrammarError

    layer = _layer()
    method = _build(scheme, layer)
    order = [(e, s) for e in range(EXPERTS) for s in SHARDS if (e, s) != (1, "w3")]
    with pytest.raises((ValueError, GrammarError), match="wire length|one half"):
        _load(method, layer, wires, order=order)
    # A completed stack takes no more wires.
    layer = _layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    with pytest.raises(RuntimeError, match="finalize"):
        method._load_wire(None, torch.zeros(4, dtype=torch.uint8), "wire", "w1", 0)


# --------------------------------------------------------------------------
# the census expectation
# --------------------------------------------------------------------------

def test_native_method_is_modular_and_owns_no_stock_kernel(stack, nvfp4_runtime):
    """The native method answers the modular protocol from its own definition.

    Regression for the false stock-kernel ownership: the runtime's
    ``is_monolithic`` delegates to ``experts_cls`` when one is present and
    ``moe_kernel`` is None, so a selected stock backend's monolithic class
    would route this method to the obsolete ``apply_monolithic`` path.  The
    native route must own neither, and must answer ``is_monolithic`` False
    itself.
    """
    _wires, scheme, _reference = stack
    method = _build(scheme, _layer())
    assert method.moe_kernel is None
    assert method.moe_quant_config is None
    assert not hasattr(method, "experts_cls")
    assert method.is_monolithic is False
    assert method.topk_indices_dtype is None
    assert method.mk_can_overlap_shared_experts is False
    assert method.supports_eplb is False
    # the obsolete monolithic hook is gone from THIS class (the base may
    # keep its own); the quant-config hook stays abstract-satisfied, without
    # reading any stock tensor
    assert "apply_monolithic" not in type(method).__dict__
    assert "get_fused_moe_quant_config" in type(method).__dict__


def test_census_expectation_is_the_shared_launch_table():
    expected = nvfp4_moe_route.census_expected()
    assert set(expected) == {"batch", "decode"}
    for regime, pairs in expected.items():
        # The route reports the native lane's own pairs on top of the attested
        # dispatch (``experimental_launch_pairs``); the attested set is what
        # ``launch_pairs`` returns and no qualification is promoted here.
        native = experimental_launch_pairs(
            TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE, regime=regime,
            mode="resident")
        assert native, "the native lane must publish its own pairs"
        assert pairs == {KERNEL_PAIR} | native
        # the attested view itself is unchanged
        assert {KERNEL_PAIR} == launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                                             regime=regime, mode="resident")
        assert pairs.isdisjoint(launch_pairs(TESSERA_NVFP4, regime=regime))
        assert not launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                                regime=regime, mode="streamed")
    assert nvfp4_moe_route.census_expected(compiled=True) == expected
    assert nvfp4_moe_route.census_symbol_base("vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS") \
        == KERNEL_PAIR[0]
    # A platform the contract publishes as unbacked for E2M1_K2 expects nothing.
    unbacked = nvfp4_moe_route.census_expected(platform="gfx1151")
    assert set(unbacked) == set(expected) and not any(unbacked.values())
