"""The Tessera NVFP4 W4A4 dense serving route — the NATIVE A4 lane.

WHAT THIS FILE IS NOW.  The route loads through the compact reader
(``scheme.parse_compact_blob_for_scheme`` + ``native_a4.prepare_a4_unit``) and
serves the packed span-2 GEMM (``tessera.kernel_a4``); its forward is held to
the stock product built from ``tessera.stock.materialize_stock`` on the same
encoded roles (the A-side value through the test's own reference quantizer).
The retired whole-weight expansion — the decoded stock tile, the load-time
reference cross-check and the named pure-torch fallback — is covered by
``tests/nvfp4_reference.py`` as a test asset where its oracle survives, and by
``tests/test_span2_start_state.py`` / ``tests/test_lane_planes_refusals.py``
for the ``slice_unit`` and admission properties it used to exercise.

STUBBED: vLLM's ``LinearMethodBase`` / parameters and the ABI attestation that
would otherwise reach for the real vLLM operator library.  The native GEMM and
the native quantizer are real.  vLLM loading this route owes a container run.
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
    A4_DENSE_GEMM_SYMBOL, TESSERA_NVFP4, is_tessera_scheme, validate_tessera_scheme)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
GROUP = 16


def _requires_native_a4():
    """Skip where the native A4 backend is ABSENT; a broken build is a FAILURE.

    ``prepare_a4_unit`` repacks the packed BODY through Triton kernels and the
    GEMM needs a Triton build that lowers block-scaled FP4 MMA.  A box without
    one cannot exercise this route at all; the A4 owner's own suite records the
    backend it ran against.
    """
    if not torch.cuda.is_available():
        pytest.skip("no visible GPU: the native A4 lane is a CUDA path")
    from tessera import kernel_a4

    if kernel_a4.native_fp4_backend() is None:
        pytest.skip("no importable Triton with FP4 MMA support on this host")


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
    s = {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT", "q256": 896,
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
    ({"grid": "E4M3"}, "E2M1-based"),
    ({"plane": "CHANNEL"}, "no NVFP4 tile"),
    ({"body": "SPIRAL"}, "body must be"),
    ({"columns": 1000}, "K % 16"),
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
    assert not hasattr(serving_lane, "TESSERA_FLAG")
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


def test_the_route_record_carries_which_decoder_ran():
    """A receipt must never read a torch-materialised serve as a native one."""
    assert "decoder" in telemetry.ROUTE_FIELDS
    assert telemetry.DECODER_NATIVE_SPAN2_GEMM != telemetry.DECODER_TORCH_STOCK
    assert {telemetry.DECODER_NATIVE_SPAN2_GEMM, telemetry.DECODER_TORCH_STOCK} <= \
        telemetry.DECODERS


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


_ATTESTED = []


def _reference_fp4_quant_value(x, global_scale):
    """The A-side VALUE matrix, one group-16 E2M1 quantisation at the static
    global -- the arithmetic vLLM's ``scaled_fp4_quant`` publishes, written out
    so the expectation is the tensor the route's own quantizer stands for."""
    m, k = x.shape
    groups = k // GROUP
    xf = x.float().view(m, groups, GROUP)
    amax = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    sf = (amax / 6.0 * float(global_scale)).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    q = (xf * float(global_scale) / sf_f).clamp(-6.0, 6.0)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          dtype=torch.float32, device=x.device)
    idx = (q.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    vals = levels[idx] * torch.sign(q)
    return (vals * sf_f).view(m, k)


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank: the layer's OWN TP
    coordinates, which every ``LinearBase`` sets before ``create_weights``
    and the shard plan reads (tessera#303)."""

    tp_rank, tp_size = 0, 1


def _encode_module(roles, cols=1024, q256=896, seed=0):
    """Encode ``roles`` = [(name, rows)] with Tessera; return the container blob,
    the scheme, and the stock reference (shared global applied)."""
    fused, export, stock, alphabet = _tessera()
    K2 = alphabet.tuple_grid(alphabet.E2M1_GRID, 2)
    torch.manual_seed(seed)
    tensors, blobs = {}, []
    for i, (name, rows) in enumerate(roles):
        w = (torch.randn(rows, cols, device="cuda") * 0.02)
        w[: rows // 8] *= 2.0 ** (i + 1)            # roles land on different globals
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=K2, q256=q256, name=name, verify=False)
        tensors[name] = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
    shared, divisor = stock.share_global(tensors)
    blob = fused.pack_fused(blobs)
    scheme = {"family": TESSERA_NVFP4, "grid": K2.name, "body": "TCQ", "plane": "LUT",
              "q256": q256, "rows": sum(r for _, r in roles), "columns": cols,
              "wire_bytes": len(blob), "roles": [[n, r] for n, r in roles]}
    packed = torch.cat([shared[n]["weight_packed"] for n, _ in roles])
    scale = torch.cat([shared[n]["weight_scale"] for n, _ in roles])
    ref_w = torch.cat([stock.stock_dequant(shared[n]) for n, _ in roles])
    return blob, scheme, packed, scale, 1.0 / divisor, ref_w


def _drive(monkeypatch, mode, roles=(("weight", 256),), cols=1024, m=32, seed=0,
           input_global_scale=4.0):
    # The residency is latched to the first value this process read, so a test
    # that drives both modes clears it here rather than only between tests.
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    # With sys.modules['vllm'] stubbed, the real operator library is not
    # importable; record that the route ATTESTs the ABI rather than executing
    # an attestation the stub cannot satisfy.  See the report.
    _ATTESTED.clear()
    monkeypatch.setattr(native_ops, "require_native_fp4_quant",
                        lambda context: _ATTESTED.append(context))
    blob, scheme, packed, scale, global_, ref_w = _encode_module(
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
    # ``_reference_fp4_quant_value`` is the A-side VALUES the quantizer stands
    # for, which carry the static global (``sf = amax/6 * gs``).  The route's
    # epilogue divides that global back out (``A4Unit.epilogue_for``:
    # ``global_scale / input_global_scale``), so the expectation divides it too
    # -- without this the reference is ``gs`` times the product and the
    # comparison fails at exactly ``(gs-1)/gs``.
    want = ((_reference_fp4_quant_value(x, gs) / float(gs)) @ ref_w.t()).to(torch.bfloat16)
    return got, want, layer, method, (packed, scale, global_)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_the_native_forward_matches_the_stock_product(monkeypatch, mode):
    """Both residencies serve the stock product: the same packed wire, the
    static A-side global, and the span-2 GEMM's fp32 epilogue."""
    _requires_native_a4()
    got, want, layer, _m, (_packed, _scale, global_) = _drive(monkeypatch, mode)
    assert layer.tessera_decoder == telemetry.DECODER_NATIVE_SPAN2_GEMM
    assert layer.tessera_symbol == A4_DENSE_GEMM_SYMBOL
    assert layer.tessera_global_scale_real == global_
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_the_route_attests_the_native_a_side_abi(monkeypatch):
    """The A side is vLLM's own compiled NVFP4 quantiser; a missing ABI is a
    model-load error, never an implementation switch."""
    _requires_native_a4()
    _drive(monkeypatch, MODE_RESIDENT)
    assert len(_ATTESTED) == 1 and "test.layer" in _ATTESTED[0]


@requires_cuda
def test_fused_roles_stack_with_their_own_row_slices(monkeypatch):
    """A fused container's roles are one module: the native units stack in row
    order, share the moved global, and the forward concatenates them."""
    _requires_native_a4()
    roles = (("q_proj", 256), ("k_proj", 128), ("v_proj", 128))
    got, want, layer, _m, (_packed, _scale, global_) = _drive(
        monkeypatch, MODE_RESIDENT, roles=roles, seed=3)
    # The route records the role names in module order; the container is a
    # list, and what the record owes a reader is the order, not the spelling.
    assert list(layer.tessera_roles) == ["q_proj", "k_proj", "v_proj"]
    assert layer.tessera_global_scale_real == global_
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


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
    units = layer.tessera_a4_units
    assert units and layer.tessera_a4_epilogues
    before = [(t.data_ptr(), t._version) for t in (units[0].select, units[0].nibbles)]
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    method.apply(layer, x)
    after = [(t.data_ptr(), t._version) for t in (units[0].select, units[0].nibbles)]
    assert before == after, "a forward rewrote the packed planes"


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
    assert rec["symbol"] == A4_DENSE_GEMM_SYMBOL == layer.tessera_symbol
    assert rec["decoder"] == telemetry.DECODER_NATIVE_SPAN2_GEMM == layer.tessera_decoder
    assert rec["decoder"] in telemetry.DECODERS


# --- the A-side scale gates --------------------------------------------------

def _create_layer(monkeypatch, mode):
    """The real method and the real ``create_weights``, no forward: what the
    load hook is driven through for the A-side scale gate below."""
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
    gs = layer.trellis_input_global_scale.data
    assert torch.isnan(gs).all(), "the unloaded scale must be a value the gate refuses"
    assert not float(gs.reshape(-1)[0]) > 0.0


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
    """The control, through the same hook with a real wire: a finite positive
    scale is accepted and the epilogue factor is the module's shared weight
    global over it."""
    _requires_native_a4()
    _g, _w, layer, _m, (_packed, _scale, global_) = _drive(
        monkeypatch, MODE_STREAMED, input_global_scale=4.0)
    assert layer.tessera_epilogue_scale == global_ / 4.0
    assert layer.tessera_global_scale_real == global_
