"""The native fused E2M1x2 WINDOW dense serving route.

The tests exercise real packed WINDOW inputs, the native FP4 launch, and
vLLM's registered activation quantizer. Only the vLLM base class and
parameter wrappers are substituted. An actual container run must prove the
production runtime integration. Per-role stock products remain independent;
no common LUT/global remap is required by the native reader.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import native_ops, telemetry                    # noqa: E402
from tessera.serving import nvfp4_route as route                     # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import (                                 # noqa: E402
    FUSED_WINDOW_DENSE_E2M1_SYMBOL, TESSERA_NVFP4, is_tessera_scheme, validate_tessera_scheme)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
GROUP = 16


def _requires_native_a4():
    """Refuse an unavailable FP4 device; a native build error fails the test."""
    if not torch.cuda.is_available():
        pytest.skip("no visible GPU: the native fused E2M1 lane is a CUDA path")
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the native fused E2M1 library requires SM121")


def _tessera():
    return pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"), \
        pytest.importorskip("tessera.stock"), pytest.importorskip("tessera.alphabet")


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


def _scheme(rows=256, columns=1024, roles=None, **over):
    s = {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "WINDOW", "plane": "LUT", "span": 1, "q256": 896,
         "rows": rows, "columns": columns, "wire_bytes": 4096,
         "roles": roles if roles is not None else [["weight", rows]]}
    s.update(over)
    return s


# --- the scheme --------------------------------------------------------------

def test_scheme_discriminator_and_normalisation():
    assert is_tessera_scheme(_scheme())
    assert not is_tessera_scheme({"family": "TCQ_E2M1_R256"})
    assert not is_tessera_scheme({"grid": "fp4"})
    norm = validate_tessera_scheme(_scheme(roles=[["q_proj", 128], ["k_proj", 128]]), "t")
    assert norm["roles"] == [("q_proj", 128), ("k_proj", 128)] and norm["plane"] == "LUT"


@pytest.mark.parametrize("bad,match", [
    ({"grid": "E4M3"}, "grid"),
    ({"plane": "CHANNEL"}, "plane"),
    ({"body": "SPIRAL"}, "body"),
    ({"columns": 1000}, "columns"),
    ({"roles": [["q", 100], ["k", 100]]}, "stack to 200"),
    ({"roles": []}, "roles must be"),
    ({"q256": 0}, "must be positive"),
    # ``routed_moe`` is SERVED since 2026-09-04, so a dense-shaped scheme
    # wearing that value is refused for the shape it is missing (an expert
    # count and two groups), not for the value.
    ({"structure": "routed_moe"}, "experts must be an integer"),
])
def test_scheme_refusals_name_the_defect(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_tessera_scheme(_scheme(**bad), "t")


def test_scheme_missing_fields_are_listed():
    s = _scheme(); del s["roles"]; del s["q256"]
    with pytest.raises(ValueError, match="missing \\['q256', 'roles'\\]"):
        validate_tessera_scheme(s, "t")


# --- the residency, and the refusals before vLLM -----------------------------

def test_the_residency_is_the_only_flag_and_it_refuses_by_name(monkeypatch):
    """Gridbook's ``GRIDBOOK_TESSERA`` enable flag is gone with the move: the
    checkpoint's ``quant_method`` selects the plugin, and the one thing the
    operator still declares is the residency, because it changes the footprint
    the artifact occupies."""
    with pytest.raises(ValueError, match=TESSERA_MODE_ENV):
        build_tessera_method(_scheme(), "test.layer")
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, "residnet")
    with pytest.raises(ValueError, match=TESSERA_MODE_ENV):
        build_tessera_method(_scheme(), "test.layer")
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="family must be one of"):
        build_tessera_method({**_scheme(), "family": "TESSERA_INT4"}, "test.layer")
    with pytest.raises(ValueError, match=f"serves {TESSERA_NVFP4}, not"):
        route.build_tessera_nvfp4_method({**_scheme(), "family": "TESSERA_FP8", "grid": "E4M3",
                                         "plane": "CHANNEL", "body": "WINDOW"},
                                        "test.layer", "resident")




# --- the numerics ------------------------------------------------------------

def _install_vllm_stubs(monkeypatch):
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.ModelWeightParameter = _param
    parameter.BasevLLMParameter = _param
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter)):
        monkeypatch.setitem(sys.modules, name, mod)




def _register_runtime_fp4_op():
    """Register the runtime's ``scaled_fp4_quant`` BEFORE the stubs go in.

    ``_install_vllm_stubs`` replaces ``sys.modules['vllm']`` with a bare module,
    so after it runs the real ``vllm._custom_ops`` cannot be imported and the
    operator the route executes never registers.  The suite only worked when an
    earlier test file had already registered it as a side effect (this file
    passes in the A4 group, where ``test_kernel_a4.py`` runs first, and fails
    when it is collected alone); the bootstrap cannot depend on test order.
    """
    if callable(getattr(torch.ops._C, "scaled_fp4_quant", None)):
        return
    try:
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    except Exception as exc:  # noqa: BLE001 -- one diagnosis for every cause
        raise RuntimeError(
            "this suite drives the route's own A side, the runtime's registered "
            f"scaled_fp4_quant, which could not be registered ({type(exc).__name__}: "
            f"{exc})") from exc


def _stock_product(x, gscale, packed, scale, epilogue):
    """The executed W4A4 product, built from the RUNTIME'S OWN operators.

    The A side is the registered ``scaled_fp4_quant`` -- the operator the route
    executes, not a written-out model of it -- held to the module's stock tile
    through ``torch._scaled_mm``, the same oracle ``tests/test_kernel_a4.py``
    uses.  A hand-written model of that operator is NOT equivalent: on the
    fused q/k/v fixture it disagrees with the operator on 172 of 32768 elements,
    every one of them exactly on a level midpoint
    (``experiments/a4_fused_discriminator.py``, PB action 5ea787ef).  Whether
    that is a tie rule or the operator's reciprocal/scale arithmetic shifting
    the nominal midpoint is NOT established by one fixture, so this oracle does
    not restate either: it uses the operator, and a claim about its rounding
    would need its own test across scales, signs and all seven boundaries.
    """
    from tessera.serving.nvfp4_route import blocked_scales

    a_q, a_s = torch.ops._C.scaled_fp4_quant(x.contiguous(), gscale, True)
    a_q = a_q.view(torch.float4_e2m1fn_x2)
    a_s = a_s.view(torch.uint8).view(torch.float8_e4m3fn).contiguous()
    b_q = packed.to("cuda").view(torch.float4_e2m1fn_x2)
    b_s = blocked_scales(scale.to("cuda").view(torch.uint8).view(torch.float8_e4m3fn))
    try:
        ref = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                               out_dtype=torch.float32)
    except RuntimeError:
        ref = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                               out_dtype=torch.bfloat16).to(torch.float32)
    return (ref * epilogue).to(torch.bfloat16)


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank: the layer's OWN TP
    coordinates, which every ``LinearBase`` sets before ``create_weights``
    and the shard plan reads (tessera#303)."""

    tp_rank, tp_size = 0, 1


def _encode_module(roles, cols=1024, q256=896, seed=0):
    """Encode each role's served WINDOW bytes and independent stock tile."""
    fused, export, stock, alphabet = _tessera()
    grid = alphabet.tuple_grid(alphabet.E2M1_GRID, 2)
    recipe = export.served_recipe(grid, q256)
    torch.manual_seed(seed)
    tensors, blobs = {}, []
    for i, (name, rows) in enumerate(roles):
        weight = torch.randn(rows, cols, device="cuda") * 0.02
        weight[:rows // 8] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            weight.contiguous(), grid=grid, q256=q256, name=name, verify=False,
            body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
            window_bits=recipe.window_bits, window_seed=recipe.window_seed,
            window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma)
        tensors[name] = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = {"family": TESSERA_NVFP4, "grid": grid.name, "body": "WINDOW", "plane": "LUT",
              "span": 1, "q256": q256, "rows": sum(r for _, r in roles), "columns": cols,
              "wire_bytes": len(blob), "roles": [[n, r] for n, r in roles]}
    packed = [tensors[name]["weight_packed"] for name, _ in roles]
    scale = [tensors[name]["weight_scale"] for name, _ in roles]
    globals_ = [1.0 / float(tensors[name]["weight_global_scale"].reshape(-1)[0])
                for name, _ in roles]
    reference = torch.cat([stock.stock_dequant(tensors[name]) for name, _ in roles])
    return blob, scheme, packed, scale, globals_, reference


def _drive(monkeypatch, mode, roles=(("weight", 256),), cols=1024, m=32, seed=0,
           input_global_scale=4.0):
    # The residency is latched to the first value this process read, so a test
    # that drives both modes clears it here rather than only between tests.
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _register_runtime_fp4_op()
    _install_vllm_stubs(monkeypatch)
    # The base-class substitutes cannot perform ABI attestation. The actual
    # registered activation operator remains the A-side numerical oracle.
    monkeypatch.setattr(native_ops, "require_native_fp4_quant", lambda context: None)
    blob, scheme, packed, scale, global_, _ref_w = _encode_module(
        list(roles), cols=cols, seed=seed)
    method = build_tessera_method(scheme, "test.layer")
    layer = _Layer()
    rows = scheme["rows"]
    method.create_weights(layer, input_size_per_partition=cols,
                          output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.trellis_input_global_scale.data = torch.tensor(
        [input_global_scale], dtype=torch.float32)
    layer.to(torch.device("cuda"))
    method.process_weights_after_loading(layer)
    gs = float(layer.trellis_input_global_scale.data.reshape(-1)[0])
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    gscale = layer.trellis_input_global_scale.data.to(torch.float32)
    want = torch.cat([_stock_product(x, gscale, packed_role, scale_role, global_role / gs)
                      for packed_role, scale_role, global_role in zip(packed, scale, global_)], dim=-1)
    return got, want, layer, method, (packed, scale, global_)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_the_native_forward_matches_the_stock_product(monkeypatch, mode):
    """The native WINDOW forward agrees with independently decoded stock roles."""
    _requires_native_a4()
    got, want, layer, _method, _reference = _drive(monkeypatch, mode)
    assert layer.tessera_decoder == telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1
    assert layer.tessera_symbol == FUSED_WINDOW_DENSE_E2M1_SYMBOL
    err = (got.float() - want.float()).abs().max().item()
    assert err <= torch.finfo(torch.bfloat16).eps * want.float().abs().max().item()




@requires_cuda
def test_fused_roles_stack_with_their_own_row_slices(monkeypatch):
    """The forward preserves each role's global and concatenates its row slice."""
    _requires_native_a4()
    roles = (("q_proj", 256), ("k_proj", 128), ("v_proj", 128))
    got, want, layer, _method, (_packed, _scale, globals_) = _drive(
        monkeypatch, MODE_RESIDENT, roles=roles, seed=3)
    assert [role.rows for role in layer.tessera_a4_roles] == [256, 128, 128]
    assert [float(role.ratio[0]) for role in layer.tessera_a4_roles] == [g / 4.0 for g in globals_]
    err = (got.float() - want.float()).abs().max().item()
    assert err <= torch.finfo(torch.bfloat16).eps * want.float().abs().max().item()


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    _requires_native_a4()
    a, _w, _l, _m, _ = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    b, _w2, _l2, _m2, _ = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_the_route_holds_packed_units_and_no_decoded_tile(monkeypatch, mode):
    """Both residencies keep the loader's packed planes: no wire parameter, no
    decoded tile, no prepared-tile wrapper -- and a forward rewrites none of
    the unit's tensors."""
    _requires_native_a4()
    _g, _w, layer, method, _r = _drive(monkeypatch, mode,
                                       roles=(("weight", 128),), cols=512)
    for name in ("wire_bytes", "weight_fp4", "tessera_prepared", "decode_buf"):
        assert not hasattr(layer, name), name
    before = {name: tensor.detach().clone() for name, tensor in method.resident_tensors(layer)}
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    first = method.apply(layer, x)
    second = method.apply(layer, x)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))
    for name, tensor in method.resident_tensors(layer):
        assert torch.equal(before[name].view(torch.uint8), tensor.view(torch.uint8)), name


@requires_cuda
def test_scheme_and_blob_must_agree(monkeypatch):
    from tessera.serving.scheme import (parse_compact_blob_for_scheme,
                                        parse_tessera_blob_for_scheme)
    blob, scheme, *_ = _encode_module([("weight", 256)], cols=512)
    parse_tessera_blob_for_scheme(blob, scheme, "t")
    parse_compact_blob_for_scheme(blob, scheme, "t", device="cuda")
    with pytest.raises(ValueError, match="wire_bytes"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "wire_bytes": len(blob) + 1}, "t")
    with pytest.raises(ValueError, match="wire_bytes"):
        parse_compact_blob_for_scheme(blob, {**scheme, "wire_bytes": len(blob) + 1},
                                      "t", device="cuda")
    with pytest.raises(ValueError, match="roles"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "roles": [["other", 256]]}, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "columns": 1024}, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_compact_blob_for_scheme(blob, {**scheme, "columns": 1024}, "t",
                                      device="cuda")


@requires_cuda
def test_route_record_names_the_family_mode_symbol_and_decoder(monkeypatch):
    from tessera.serving.telemetry import read_route
    _requires_native_a4()
    _g, _w, layer, _m, _ = _drive(monkeypatch, MODE_RESIDENT)
    rec = read_route(layer)
    assert rec is not None and rec["policy"] == f"{TESSERA_NVFP4}:resident"
    assert rec["state"] == "served"
    assert rec["contract"] == route.ACTIVATION_CONTRACT
    assert rec["symbol"] == FUSED_WINDOW_DENSE_E2M1_SYMBOL == layer.tessera_symbol
    assert rec["decoder"] == telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1 == layer.tessera_decoder
    assert rec["decoder"] in telemetry.DECODERS


# --- the A-side scale gates --------------------------------------------------

def _create_layer(monkeypatch, mode):
    """The real method and the real ``create_weights``, no forward: what the
    load hook is driven through for the A-side scale gate below.

    No runtime op registration here (tessera#561): the scale gate refuses
    before the container parse, and the op attestation
    (``require_native_fp4_quant``) runs after it, so neither is reachable
    from this test -- its ``_no_parse`` guard is the proof, and
    ``create_weights`` touches no operator.  Registering the real
    ``scaled_fp4_quant`` needs vLLM installed and turned the CPU-pool
    population red; the executed-A-side proofs that do need the operator
    live in ``_drive`` under the CUDA gate.
    """
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    method = build_tessera_method(_scheme(), "test.layer")
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=1024,
                          output_partition_sizes=[256],
                          input_size=1024, output_size=256, params_dtype=torch.bfloat16)
    return method, layer


def test_an_unloaded_input_global_scale_is_refused_by_the_gates_own_predicate(monkeypatch):
    """The A-side scale must fail ``gs > 0`` until a loader writes it."""
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, MODE_RESIDENT)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_ops, "require_native_fp4_quant", lambda context: None)
    method = build_tessera_method(_scheme(), "test.layer")
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=1024, output_partition_sizes=[256],
                          input_size=1024, output_size=256, params_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="trellis_input_global_scale"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 0.0, -1.0],
                         ids=["inf", "-inf", "nan", "zero", "negative"])
def test_a_nonfinite_or_nonpositive_activation_scale_is_refused_at_load(monkeypatch, mode, bad):
    """#202: the gate requires a finite positive scalar, refuses by the tensor's
    name, and runs BEFORE the container parse, so nothing expensive is spent on
    a checkpoint that will be refused."""
    method, layer = _create_layer(monkeypatch, mode)

    def _no_parse(*_a, **_k):
        raise AssertionError("the scale gate must refuse before the container parse")

    monkeypatch.setattr(route, "parse_compact_blob_for_scheme", _no_parse)
    layer.trellis_input_global_scale.data = torch.tensor([bad], dtype=torch.float32)
    with pytest.raises(ValueError, match="trellis_input_global_scale must be a finite "
                                         "positive scalar"):
        method.process_weights_after_loading(layer)


@requires_cuda
def test_a_valid_finite_scale_still_loads_and_sets_the_epilogue(monkeypatch):
    """Each role's ratio uses its own weight global and the static input scale."""
    _requires_native_a4()
    _got, _want, layer, _method, (_packed, _scale, globals_) = _drive(
        monkeypatch, MODE_STREAMED, input_global_scale=4.0)
    assert [float(role.ratio[0]) for role in layer.tessera_a4_roles] == [g / 4.0 for g in globals_]
