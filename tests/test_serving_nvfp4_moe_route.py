"""The NVFP4 expert route's load and native-apply half (tessera#492).

WHAT THIS FILE CAN COVER AND WHAT IT CANNOT.  Two populations share it.

The CPU stub suite drives the route's protocol surface without vLLM:
``oracle.nvfp4`` is STUBBED with a kernel that computes the float reference
from the operands it is handed (a spy that also does the arithmetic), because
vendoring the runtime into this repository is forbidden (AGENTS.md).  What
that stub pins is construction and geometry refusals, the census expectation,
and the native method answering the modular protocol from its own definition
instead of inheriting a stock kernel's.

The eight device-driven cases marked ``needs_native_prep`` are the native
lane's own coverage and run for REAL in the pinned image: ``create_weights``
imports vLLM's own ``FusedMoEMethodBase``, ``apply`` executes the native
grouped kernels, and the A side is the runtime's registered
``scaled_fp4_quant``.  They are vLLM-exempt from PrismaBuild like every
actual vLLM run, and their references are the encoder's own
``materialize_stock`` tiles and the materializing reader's parsed units --
never the loader's own path:

* the served stacks' 16-byte scale tables and per-expert globals are the
  independent ``shared_lut_global`` join (the encoder's materialized pair
  moved onto one global by ``stock.share_global``), and the joined multiplier
  is byte-for-byte the raw-table join over independently parsed containers;
* the per-expert epilogue is the joined multiplier over the one static A-side
  scalar (the max of the loader's reciprocal, i.e. the min of the checkpoint
  scales), and the stock divisor read as a multiplier dequants differently;
* ``apply`` computes the runtime's own quantized arithmetic over those
  materialized tiles, weights applied only in the final combine, and never
  consumes the shared experts the runner owns;
* the load order moves no byte of the served stacks;
* at TP2 each rank holds its own rows of w13 and columns of w2;
* the wire/scale loaders and finalize refuse by name.

The load-and-execute receipt on the pinned image, including the CUDA-graph
capture and the fuller stock oracle, is
``experiments/native_a4_serve_probe.py``; this file is the regression suite
that must execute in that same image, not skip.
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

HIDDEN, INTER, EXPERTS, Q256 = 128, 256, 3, 896
SHARDS = ("w1", "w3", "w2")     # gate, up, down: the runtime's shard ids
KERNEL_PAIR = ("vllm.fused_moe.modular_kernel", "torch_materialize_stock")


def _native_prep_available() -> bool:
    """Whether this interpreter can run the native load path at all.

    ``prepare_a4_unit`` repacks the packed BODY through Triton CUDA kernels and
    ``apply`` executes the native span-2 grouped GEMM, so the CPU stub suite
    cannot drive either; the pinned image runs this file's device-driven cases
    for real (``experiments/native_a4_serve_probe.py`` is the fuller receipt).
    """
    try:
        import tokenspeed_triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return torch.cuda.is_available()


needs_native_prep = pytest.mark.skipif(
    not _native_prep_available(),
    reason=("the native load path imports vLLM's own runtime and repacks the "
            "packed BODY through Triton on CUDA; run this file in the pinned "
            "image, where these cases execute rather than skip -- see "
            "experiments/native_a4_serve_probe.py for the fuller receipt"))


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


# --------------------------------------------------------------------------
# the native lane's independent reference
# --------------------------------------------------------------------------

def _native_layer(tp_size=1, tp_rank=0, **moe):
    """``_layer`` with the real runtime's activation enum, for native apply."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    layer = _layer(tp_size=tp_size, tp_rank=tp_rank, **moe)
    layer.activation = MoEActivation.SILU
    return layer


def _declared_roles(scheme):
    """The sidecar's per-projection role declarations, from the shared gate."""
    from tessera.serving.scheme import expert_role_declarations, validate_tessera_moe_scheme

    declared = validate_tessera_moe_scheme(scheme, "m")
    return {group: expert_role_declarations(declared["groups"][group])
            for group in ("w13", "w2")}


def _parse_reference(blob, role, context):
    """One container through the MATERIALIZING reader, not the compact twin."""
    from tessera.serving.scheme import parse_tessera_expert_blob

    return parse_tessera_expert_blob(blob, role, context, device="cuda")[0][1]


def _independent_join(wires, roles):
    """``shared_lut_global`` over tables the materializing reader parsed.

    Gate and up of one expert are encoded on their own globals; the join is
    the loader's own rule, applied here to the encoder's containers instead
    of the loader's, so a byte difference is the loader's and not this test's.
    """
    from tessera.fused import shared_lut_global
    from tessera.lane_planes import lut_scale_bytes

    out = []
    for e in range(EXPERTS):
        gate = _parse_reference(wires[(e, "w1")], roles["w13"][0], f"reference gate {e}")
        up = _parse_reference(wires[(e, "w3")], roles["w13"][1], f"reference up {e}")
        down = _parse_reference(wires[(e, "w2")], roles["w2"][0], f"reference down {e}")
        shared, moved = shared_lut_global(
            [lut_scale_bytes(gate.unit.scale_lut), lut_scale_bytes(up.unit.scale_lut)],
            [float(gate.unit.scale_global), float(up.unit.scale_global)],
            ["gate_proj", "up_proj"])
        out.append({"shared": shared, "gate": moved[0].cpu(), "up": moved[1].cpu(),
                    "down": lut_scale_bytes(down.unit.scale_lut).cpu(),
                    "down_global": float(down.unit.scale_global)})
    return out


def _served_lut_bytes(stack, expert):
    return stack.lut_bytes[expert].view(torch.uint8)


def _served_scale_plane(stack, expert):
    """The stack's scale-index nibbles expanded through its served table.

    The compact plane is ``[groups, rows/2]`` bytes, the even row in the high
    nibble; expanding the served 16-byte table at those indices is the
    per-element E4M3 plane a stock tile carries.
    """
    rows, groups = stack.rows, stack.cols // stack.half
    packed = stack.nibbles[expert].view(torch.uint8).to(torch.int64)
    index = torch.empty((groups, rows), dtype=torch.int64, device=packed.device)
    index[:, 0::2] = (packed >> 4).reshape(groups, rows // 2)
    index[:, 1::2] = (packed & 0xF).reshape(groups, rows // 2)
    return _served_lut_bytes(stack, expert)[index].t().contiguous().cpu()


def _expected(reference, rank=0, tp_size=1):
    """The rank's rows of w13 and columns of w2, from the joined references."""
    lo, hi = rank * INTER // tp_size, (rank + 1) * INTER // tp_size
    out = []
    for ref in reference:
        out.append({
            "gate_scale": ref["gate"]["weight_scale"][lo:hi].view(torch.uint8),
            "up_scale": ref["up"]["weight_scale"][lo:hi].view(torch.uint8),
            "down_scale": ref["down"]["weight_scale"][:, lo // 16: hi // 16].view(torch.uint8),
            "w13_global": ref["w13_global"], "w2_global": ref["w2_global"]})
    return out


def _assert_native_stacks(layer, reference, rank=0, tp_size=1):
    """Every rank-local stack is the independently materialized reference."""
    gate, up, down = (layer.tessera_a4_gate_stack, layer.tessera_a4_up_stack,
                      layer.tessera_a4_down_stack)
    for e, want in enumerate(_expected(reference, rank, tp_size)):
        assert torch.equal(_served_scale_plane(gate, e), want["gate_scale"]), e
        assert torch.equal(_served_scale_plane(up, e), want["up_scale"]), e
        assert torch.equal(_served_scale_plane(down, e), want["down_scale"]), e
        assert float(gate.globals[e]) == want["w13_global"], e
        assert float(up.globals[e]) == want["w13_global"], e
        assert float(down.globals[e]) == want["w2_global"], e


def _rank_tiles(reference, rank, tp_size):
    """The rank's slice of the joined tiles: w13 rows, w2 columns."""
    lo, hi = rank * INTER // tp_size, (rank + 1) * INTER // tp_size
    tiles = []
    for ref in reference:
        tiles.append({
            "gate": {"weight_packed": ref["gate"]["weight_packed"][lo:hi].contiguous(),
                     "weight_scale": ref["gate"]["weight_scale"][lo:hi].contiguous()},
            "up": {"weight_packed": ref["up"]["weight_packed"][lo:hi].contiguous(),
                   "weight_scale": ref["up"]["weight_scale"][lo:hi].contiguous()},
            "down": {"weight_packed": ref["down"]["weight_packed"][:, lo // 2: hi // 2].contiguous(),
                     "weight_scale": ref["down"]["weight_scale"][:, lo // 16: hi // 16].contiguous()},
        })
    return tiles


def _stock_oracle(x, weights, ids, tiles, gs13, gs2, gate_epilogues, down_epilogues,
                  clamp_limit):
    """The stock lane's arithmetic for the same routing.

    The runtime's own FP4 quantizer in its ``_scaled_mm`` layout over the
    independently materialized tiles, the layer's clamped SILU, and the
    router weights applied only in the final combine.
    """
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    from tessera.serving.nvfp4_route import blocked_scales

    tiles = [{projection: {name: tensor.to(x.device) for name, tensor in tile.items()}
              for projection, tile in expert.items() if isinstance(tile, dict)}
             for expert in tiles]

    def gemm(activations, global_scale, tile, epilogue):
        a_q, a_s = torch.ops._C.scaled_fp4_quant(
            activations.contiguous(), global_scale, True)
        a_q = a_q.view(torch.float4_e2m1fn_x2)
        a_s = a_s.view(torch.uint8).view(torch.float8_e4m3fn).contiguous()
        b_q = tile["weight_packed"].view(torch.float4_e2m1fn_x2)
        b_s = blocked_scales(tile["weight_scale"].view(torch.uint8)
                             .view(torch.float8_e4m3fn))
        try:
            y = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                                 out_dtype=torch.float32)
        except RuntimeError:
            y = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                                 out_dtype=torch.bfloat16).to(torch.float32)
        return y * epilogue.reshape(1)

    out = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            gate = gemm(x[token:token + 1], gs13, tiles[expert]["gate"], gate_epilogues[expert])
            up = gemm(x[token:token + 1], gs13, tiles[expert]["up"], gate_epilogues[expert])
            gate = torch.clamp(gate, max=clamp_limit)
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
            hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            down = gemm(hidden, gs2, tiles[expert]["down"], down_epilogues[expert])
            out[token] += float(weights[token, choice]) * down[0]
    return out


# --------------------------------------------------------------------------
# the native stacks and their globals
# --------------------------------------------------------------------------

@needs_native_prep
def test_the_served_stacks_are_the_joined_stock_scale_planes(stack):
    """The load is the native one: gate and up of each expert decode onto ONE
    shared global through ``shared_lut_global`` -- byte for byte against both
    the encoder's materialized tiles and the raw tables of independently
    parsed containers -- and the down side keeps its own.  From finalize
    onward the layer holds those stacks and zero-size stock anchors, and the
    stock names carry a refusing loader rather than a decoded tile."""
    wires, scheme, reference = stack
    layer = _native_layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    _assert_native_stacks(layer, reference)
    roles = _declared_roles(scheme)
    for e, ref in enumerate(_independent_join(wires, roles)):
        assert torch.equal(_served_lut_bytes(layer.tessera_a4_gate_stack, e).cpu(),
                           ref["gate"]), e
        assert torch.equal(_served_lut_bytes(layer.tessera_a4_up_stack, e).cpu(),
                           ref["up"]), e
        assert torch.equal(_served_lut_bytes(layer.tessera_a4_down_stack, e).cpu(),
                           ref["down"]), e
        assert float(layer.tessera_a4_gate_stack.globals[e]) == ref["shared"], e
        assert float(layer.tessera_a4_down_stack.globals[e]) == ref["down_global"], e
    # The wires and A-side scales are consumed; the modelopt names survive
    # only as zero-size anchors, so no expanded expert pool is ever resident.
    assert set(dict(layer.named_parameters())) == set(nvfp4_moe_route._STOCK_TILE_NAMES)
    assert all(param.numel() == 0 for param in layer.parameters())
    assert layer.tessera_w13_wire_len is None and layer.tessera_w2_wire_len is None
    # the native lane's own labels, not a stock decoder/backend pair
    assert (layer.tessera_decoder, layer.tessera_backend) == (
        nvfp4_moe_route.DECODER_NATIVE_SPAN2_GROUPED,
        nvfp4_moe_route.A4_GROUPED_GEMM_SYMBOL)
    assert (layer.tessera_rows, layer.tessera_columns) == (2 * INTER, HIDDEN)
    assert layer.tessera_activation_contract == nvfp4_moe_route.ACTIVATION_CONTRACT
    assert (layer.tessera_family, layer.tessera_structure, layer.tessera_mode) == (
        TESSERA_NVFP4, STRUCTURE_ROUTED_MOE, "resident")
    # the method owns no stock kernel, and the runtime's config is the
    # model's swiglu facts -- a config assembled from the empty anchors would
    # be a lie about what serves
    assert method.moe_kernel is None
    assert method.moe_quant_config is not None
    assert method.moe_quant_config.gemm1_clamp_limit == 10.0


@needs_native_prep
def test_the_global_handed_to_the_kernel_is_the_multiplier(stack):
    """The joined global is the stock divisor's reciprocal, and the epilogue
    is that multiplier over the one static A-side scalar -- the max of the
    loader's reciprocal.  Read the divisor as a multiplier and the stock
    arithmetic dequants differently, which is the wrong serve this pins."""
    from tessera.stock import stock_dequant

    wires, scheme, reference = stack
    layer = _native_layer()
    method = _build(scheme, layer)
    scales = _load(method, layer, wires)
    w13_scales = [scales[(e, s)] for e in range(EXPERTS) for s in ("w1", "w3")]
    w2_scales = [scales[(e, "w2")] for e in range(EXPERTS)]
    # one quantizer scalar per GEMM: the selected backend's aggregation, the
    # max of the loader's reciprocal (capacity / amax) over the projections
    assert float(layer.tessera_a4_gs13) == min(w13_scales)
    assert float(layer.tessera_a4_gs2) == min(w2_scales)
    assert float(layer.tessera_a4_gs13) != max(w13_scales)
    assert torch.equal(layer.tessera_a4_gate_epilogues, layer.tessera_a4_up_epilogues)
    for e in range(EXPERTS):
        divisor = float(reference[e]["gate"]["weight_global_scale"].reshape(-1)[0])
        multiplier = reference[e]["w13_global"]
        assert multiplier == 1.0 / divisor
        assert float(layer.tessera_a4_gate_stack.globals[e]) == multiplier, e
        assert torch.equal(layer.tessera_a4_gate_epilogues[e],
                           layer.tessera_a4_gate_stack.globals[e] / layer.tessera_a4_gs13), e
        assert torch.equal(layer.tessera_a4_down_epilogues[e],
                           layer.tessera_a4_down_stack.globals[e] / layer.tessera_a4_gs2), e
        assert float(layer.tessera_a4_down_stack.globals[e]) == reference[e]["w2_global"], e
        # the direction, numerically: the joined multiplier used as though it
        # were the stock divisor is a different dequant, so a swapped read
        # cannot pass this test by looking plausible
        true = stock_dequant(reference[e]["gate"])
        wrong = stock_dequant({**reference[e]["gate"], "weight_global_scale": torch.tensor(
            [float(layer.tessera_a4_gate_stack.globals[e])])})
        assert not torch.equal(wrong, true), e


# --------------------------------------------------------------------------
# the forward
# --------------------------------------------------------------------------

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
def test_apply_computes_the_stock_reference_through_the_native_stages(stack):
    """``apply`` returns the runtime's own quantized arithmetic over the
    independently materialized tiles, per route, weights only in the final
    combine; the runner's shared experts are never consumed, and the two
    approximations the native contract refuses are refused by name."""
    wires, scheme, reference = stack
    layer = _native_layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    generator = torch.Generator().manual_seed(7)
    x = (torch.randn(4, HIDDEN, generator=generator) * 0.5).to("cuda").to(torch.bfloat16)
    ids = torch.tensor([[0, 2], [1, 0], [2, 1], [1, 2]], dtype=torch.int32, device="cuda")
    weights = torch.rand(4, 2, generator=generator).to("cuda")

    class Sentinel:
        called = False

        def __call__(self, *args, **kwargs):
            Sentinel.called = True

    got = method.apply(layer, x, weights, ids, Sentinel(), None)
    assert Sentinel.called is False, "apply consumed the runner's shared experts"
    expected = _stock_oracle(
        x, weights, ids, reference,
        layer.tessera_a4_gs13, layer.tessera_a4_gs2,
        layer.tessera_a4_gate_epilogues, layer.tessera_a4_down_epilogues,
        float(layer.swiglu_limit))
    assert got.dtype == x.dtype
    torch.testing.assert_close(got.float(), expected, rtol=2e-2, atol=5e-3)
    layer.apply_router_weight_on_input = True
    with pytest.raises(ValueError, match="apply_router_weight_on_input"):
        method.apply(layer, x, weights, ids, None, None)
    layer.apply_router_weight_on_input = False
    with pytest.raises(ValueError, match="routing"):
        method.apply(layer, x, weights, ids[:, :1], None, None)
    layer.expert_map = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="expert map"):
        method.apply(layer, x, weights, ids, None, None)


@needs_native_prep
def test_load_order_does_not_matter(stack):
    """Up before gate, down first, experts interleaved: the w13 join waits
    for both halves and every served byte is the same."""
    wires, scheme, reference = stack
    orders = (
        [(2, "w2"), (0, "w3"), (0, "w1"), (1, "w3"), (2, "w1"), (1, "w1"),
         (0, "w2"), (2, "w3"), (1, "w2")],
        [(e, s) for s in ("w2", "w1", "w3") for e in reversed(range(EXPERTS))],
    )
    layers = []
    for order in orders:
        layer = _native_layer()
        method = _build(scheme, layer)
        _load(method, layer, wires, order=order)
        _assert_native_stacks(layer, reference)
        layers.append(layer)
    first, second = layers
    assert torch.equal(first.tessera_a4_gate_stack.lut_bytes,
                       second.tessera_a4_gate_stack.lut_bytes)
    assert torch.equal(first.tessera_a4_up_stack.nibbles, second.tessera_a4_up_stack.nibbles)
    assert torch.equal(first.tessera_a4_down_stack.lut_bytes,
                       second.tessera_a4_down_stack.lut_bytes)
    assert torch.equal(first.tessera_a4_gate_epilogues, second.tessera_a4_gate_epilogues)
    assert torch.equal(first.tessera_a4_down_epilogues, second.tessera_a4_down_epilogues)


@pytest.mark.parametrize("rank", [0, 1])
@needs_native_prep
def test_tp2_ranks_decode_and_hold_their_own_rows_and_columns(stack, rank):
    """Each rank parses every FULL container, cuts it on the group plan (rows
    of w13, columns of w2) and decodes only its own slice; the joined global
    is a whole-unit fact and is the same on both ranks."""
    wires, scheme, reference = stack
    layer = _native_layer(tp_size=2, tp_rank=rank)
    method = _build(scheme, layer)
    _load(method, layer, wires, tp_size=2)
    local = INTER // 2
    assert (layer.tessera_a4_gate_stack.rows, layer.tessera_a4_gate_stack.cols) == (
        local, HIDDEN)
    assert (layer.tessera_a4_up_stack.rows, layer.tessera_a4_up_stack.cols) == (
        local, HIDDEN)
    assert (layer.tessera_a4_down_stack.rows, layer.tessera_a4_down_stack.cols) == (
        HIDDEN, local)
    assert (layer.tessera_rows, layer.tessera_columns) == (2 * local, HIDDEN)
    _assert_native_stacks(layer, reference, rank, 2)
    x = (torch.randn(2, HIDDEN, generator=torch.Generator().manual_seed(11))
         * 0.5).to("cuda").to(torch.bfloat16)
    ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device="cuda")
    weights = torch.tensor([[.2, .8], [.7, .3]], device="cuda")
    got = method.apply(layer, x, weights, ids, None, None)
    expected = _stock_oracle(
        x, weights, ids, _rank_tiles(reference, rank, 2),
        layer.tessera_a4_gs13, layer.tessera_a4_gs2,
        layer.tessera_a4_gate_epilogues, layer.tessera_a4_down_epilogues,
        float(layer.swiglu_limit))
    torch.testing.assert_close(got.float(), expected, rtol=2e-2, atol=5e-3)


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
def test_wire_and_scale_loader_refusals(stack):
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
def test_finalize_refuses_an_incomplete_stack(stack):
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
    # A completed stack takes no more wires or scales.
    layer = _layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    with pytest.raises(RuntimeError, match="finalize"):
        method._load_wire(None, torch.zeros(4, dtype=torch.uint8), "wire", "w1", 0)
    with pytest.raises(RuntimeError, match="finalize"):
        method._load_input_global_scale(None, torch.tensor([1.0]),
                                        "input_global_scale", "w1", 0)


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
