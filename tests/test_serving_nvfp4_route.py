"""The Tessera NVFP4 W4A4 dense serving route.

Scope, stated up front.  Exercised for real on a CUDA box with ``tessera``
importable: the container parse, the shared-global shift, the native decode
into the stock tile, the blocking of the scale plane, ``torch._scaled_mm``,
the scalar epilogue, both residency modes, the named pure-torch fallback and
the refusals.  The load-bearing assertion is BYTE identity of the decoded tile
with ``tessera.stock.materialize_stock`` after ``share_global`` -- the same
tensors the compressed-tensors stock lane served on vanilla vLLM -- so the
numbers this route produces are, by construction, the stock lane's served
numbers.  STUBBED: vLLM's ``LinearMethodBase`` / parameters, the NVFP4
activation quantiser (an audited upstream op) and the ABI attestation that
would otherwise reach for the real vLLM operator library.  vLLM loading this
route owes a container run, exactly as for the Gridbook lane it replaces.

Ported from Gridbook's ``test_tessera_nvfp4_lane.py``.  Gone with the code:
the decode pool (there is none) and the ``GRIDBOOK_TESSERA`` enable flag (the
checkpoint selects the plugin).  New here: the residency is the only flag, and
``allow_torch_fallback`` is the route's, not the decoder's, decision.
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
from tessera.serving.nvfp4_route import blocked_scales               # noqa: E402
from tessera.serving.scheme import (                                 # noqa: E402
    TESSERA_NVFP4, is_tessera_scheme, validate_tessera_scheme)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
GROUP = 16


def _requires_native_ext():
    """Skip where the toolchain is ABSENT; a broken build is a FAILURE.

    The STREAMED residency decodes inside the forward, where there is no
    fallback by design (``prepare_tessera_module``'s ``allow_torch_fallback``
    is the resident path's decision alone) -- so a box without a CUDA toolkit
    cannot exercise it at all, and a skip says so.  Resident-mode tests are
    unaffected: they take the named pure-torch fallback, which is the point of
    having one.

    The two cases are separated deliberately, because collapsing them is how a
    route stops being tested with nobody seeing a red line.  It happened here:
    a blanket ``skip if get_tessera_ext() is None`` was green on BOTH boxes,
    and neither was missing a compiler -- ``/usr/local/cuda`` pointed at a
    partial install one directory away from a complete CUDA (sparky), and
    ``ninja`` was off a non-login ssh's PATH (sparklina).  Both are now
    resolved by ``ext.toolchain_report``/``_resolve_cuda_home``, so on either
    box these RUN.  If the compiler is there and the build still fails, that
    is a regression and it fails loudly, naming what it found.
    """
    torch = pytest.importorskip("torch")
    from tessera.serving.ext import get_tessera_ext, toolchain_report

    if not torch.cuda.is_available():
        pytest.skip("no visible GPU: the extension is compiled for the LIVE device's capability, "
                    "so there is no defensible target to build for")
    found = toolchain_report(torch)
    if not found["complete"]:
        pytest.skip(f"no CUDA build toolchain on this host (nvcc={found['nvcc']}, "
                    f"ninja={found['ninja']}); the streamed residency decodes in-forward "
                    "and has no fallback")
    if get_tessera_ext() is None:
        pytest.fail(f"the CUDA toolchain IS present (nvcc={found['nvcc']}, "
                    f"ninja={found['ninja']}) but Tessera's span-2 NVFP4 decode extension did "
                    "not build; the build error is on stderr above. A broken build, not an "
                    "absent kernel.")


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
    ({"structure": "routed_moe"}, "routed_moe"),
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
    """A receipt must never read a pure-torch fallback serve as a native one."""
    assert "decoder" in telemetry.ROUTE_FIELDS
    assert telemetry.DECODER_NATIVE_SPAN2 != telemetry.DECODER_TORCH_STOCK
    assert {telemetry.DECODER_NATIVE_SPAN2, telemetry.DECODER_TORCH_STOCK} <= telemetry.DECODERS


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


_LAST_A = {}
_ATTESTED = []


def _reference_fp4_quant(x, global_scale):
    """Group-16 static-global-scale E2M1 (the arithmetic of vLLM's op), recording
    the value it represents so the expectation is the A side the route consumed."""
    m, k = x.shape
    groups = k // GROUP
    xf = x.float().view(m, groups, GROUP)
    amax = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    sf = (amax / 6.0 * global_scale.float()).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    q = (xf * global_scale.float() / sf_f).clamp(-6.0, 6.0)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=x.device)
    idx = (q.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    vals = levels[idx] * torch.sign(q)
    codes = idx.to(torch.uint8) | (q < 0).to(torch.uint8) * 8
    codes = torch.where(vals == 0, torch.zeros_like(codes), codes).view(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    _LAST_A["value"] = (vals * sf_f).view(m, k)
    return packed, blocked_scales(sf.view(m, groups))


class _Layer(torch.nn.Module):
    pass


def _encode_module(roles, cols=1024, q256=896, seed=0):
    """Encode ``roles`` = [(name, rows)] with Tessera; return the container blob,
    the scheme, and the stock reference (shared global applied)."""
    fused, export, stock, alphabet = _tessera()
    K2 = alphabet.tuple_grid(alphabet.E2M1_GRID, 2)
    torch.manual_seed(seed)
    members, tensors, blobs = [], {}, []
    for i, (name, rows) in enumerate(roles):
        w = (torch.randn(rows, cols, device="cuda") * 0.02)
        w[: rows // 8] *= 2.0 ** (i + 1)            # roles land on different globals
        exported, unit, forests = export.encode_linear_planes(w.contiguous(), grid=K2, q256=q256, name=name, verify=False)
        tensors[name] = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
    shared, divisor = stock.share_global(tensors)
    blob = fused.pack_fused(blobs)
    scheme = {"family": TESSERA_NVFP4, "grid": K2.name, "body": "TCQ", "plane": "LUT", "q256": q256,
              "rows": sum(r for _, r in roles), "columns": cols, "wire_bytes": len(blob),
              "roles": [[n, r] for n, r in roles]}
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
    monkeypatch.setattr(native_ops, "native_fp4_quant", _reference_fp4_quant)
    # With sys.modules['vllm'] stubbed, the real operator library is not
    # importable; record that the route ATTESTS the ABI rather than executing
    # an attestation the stub cannot satisfy.  See the report.
    _ATTESTED.clear()
    monkeypatch.setattr(native_ops, "require_native_fp4_quant",
                        lambda context: _ATTESTED.append(context))
    blob, scheme, packed, scale, global_, ref_w = _encode_module(list(roles), cols=cols, seed=seed)
    method = build_tessera_method(scheme, "test.layer")
    layer = _Layer()
    rows = scheme["rows"]
    method.create_weights(layer, input_size_per_partition=cols, output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.trellis_input_global_scale.data = torch.tensor([input_global_scale], dtype=torch.float32)
    layer.to(torch.device("cuda"))
    method.process_weights_after_loading(layer)
    gs = layer.trellis_input_global_scale.data.reshape(())
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    want = ((_LAST_A["value"] / float(gs)) @ ref_w.t()).to(torch.bfloat16)
    return got, want, layer, method, (packed, scale, global_)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_tile_is_the_stock_tile_byte_for_byte(monkeypatch, mode):
    """The decoded tile IS materialize_stock's after share_global: codes, scale bytes, global."""
    if mode == MODE_STREAMED:
        _requires_native_ext()
    got, want, layer, method, (packed, scale, global_) = _drive(monkeypatch, mode)
    if mode == MODE_RESIDENT:
        tile, scale_b = layer.weight_fp4.view(torch.uint8), layer.scale_b
    else:
        tile, scales = layer.tessera_prepared.decode()
        scale_b = blocked_scales(scales.view(torch.float8_e4m3fn))
    assert torch.equal(tile, packed)
    assert torch.equal(scale_b.view(torch.uint8), blocked_scales(scale).view(torch.uint8))
    assert layer.tessera_global_scale_real == global_
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_the_route_attests_the_native_a_side_abi(monkeypatch):
    """The A side is vLLM's own compiled NVFP4 quantiser; a missing ABI is a
    model-load error, never an implementation switch."""
    _drive(monkeypatch, MODE_RESIDENT)
    assert len(_ATTESTED) == 1 and "test.layer" in _ATTESTED[0]


@requires_cuda
def test_fused_roles_decode_into_row_slices_on_one_global(monkeypatch):
    roles = (("q_proj", 256), ("k_proj", 128), ("v_proj", 128))
    got, want, layer, _m, (packed, scale, global_) = _drive(monkeypatch, MODE_RESIDENT, roles=roles, seed=3)
    assert layer.tessera_roles == ("q_proj", "k_proj", "v_proj")
    assert torch.equal(layer.weight_fp4.view(torch.uint8), packed)
    assert torch.equal(layer.scale_b.view(torch.uint8), blocked_scales(scale).view(torch.uint8))
    assert layer.tessera_global_scale_real == global_
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    _requires_native_ext()
    a, _w, _l, _m, _ = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    b, _w2, _l2, _m2, _ = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(a, b)


@requires_cuda
def test_streamed_holds_the_wire_and_no_resident_tile(monkeypatch):
    """Streamed: the prepared planes are the layer's only weight state; the
    tile is decoded into fresh tensors each forward (a functional op), so no
    layer holds a decoded tile, a scale plane, or a slice of a shared pool."""
    _requires_native_ext()
    _g, _w, a, _m, _ = _drive(monkeypatch, MODE_STREAMED, roles=(("weight", 128),), cols=512)
    for name in ("wire_bytes", "weight_fp4", "scale_b", "decode_buf", "scale_scratch"):
        assert not hasattr(a, name), name
    prepared = a.tessera_prepared
    assert prepared is not None and prepared.wire_bytes_resident() < 128 * 512 * 4.25 / 8 + 4096
    p1, s1 = prepared.decode()
    p2, s2 = prepared.decode()
    assert torch.equal(p1, p2) and torch.equal(s1, s2)
    assert p1.data_ptr() != p2.data_ptr() or p1 is not p2


@requires_cuda
def test_resident_drops_the_wire(monkeypatch):
    _g, _w, r, _m, _ = _drive(monkeypatch, MODE_RESIDENT)
    assert not hasattr(r, "wire_bytes") and r.tessera_prepared is None
    assert r.weight_fp4.numel() == r.tessera_rows * r.tessera_columns // 2


@requires_cuda
def test_scheme_and_blob_must_agree(monkeypatch):
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 256)], cols=512)
    parse_tessera_blob_for_scheme(blob, scheme, "t")
    with pytest.raises(ValueError, match="wire_bytes"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "wire_bytes": len(blob) + 1}, "t")
    with pytest.raises(ValueError, match="roles"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "roles": [["other", 256]]}, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "q256": 640}, "t")


@requires_cuda
def test_route_record_names_the_family_mode_and_decoder(monkeypatch):
    from tessera.serving.telemetry import read_route
    _g, _w, layer, _m, _ = _drive(monkeypatch, MODE_RESIDENT)
    rec = read_route(layer)
    assert rec is not None and rec["policy"] == f"{TESSERA_NVFP4}:resident" and rec["state"] == "served"
    assert rec["contract"] == route.ACTIVATION_CONTRACT
    # Which decoder ran is a fact of the box (whether the extension built), not
    # of this test; what must hold is that the record and the layer agree.
    assert rec["decoder"] == layer.tessera_decoder
    assert rec["decoder"] in telemetry.DECODERS


# --- the named pure-torch fallback -------------------------------------------

@requires_cuda
def test_a_load_time_decode_may_take_the_named_torch_fallback(monkeypatch):
    """``prepare_tessera_module(allow_torch_fallback=True)`` decodes once, at
    load, through ``tessera.stock.materialize_stock`` -- the same bytes -- and
    SAYS SO on the prepared module, so a census cannot read it as native."""
    from tessera.serving import ext, ops
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    blob, scheme, packed, scale, _global, _ref = _encode_module([("weight", 128)], cols=512)
    roles = parse_tessera_blob_for_scheme(blob, scheme, "t")
    monkeypatch.setattr(ext, "get_tessera_ext", lambda: None)
    prepared = ops.prepare_tessera_module(roles, device="cuda", allow_torch_fallback=True)
    assert prepared.decoder == telemetry.DECODER_TORCH_STOCK
    got_packed, got_scales = prepared.decode()
    assert torch.equal(got_packed, packed)
    assert torch.equal(got_scales, scale.view(torch.uint8))
    # It decodes once, at load: there is no per-call decode target.
    with pytest.raises(RuntimeError, match="no per-call decode target"):
        prepared.decode_out(*prepared.empty_tile())
    with pytest.raises(ext.NativeKernelUnavailableError):
        ops.prepare_tessera_module(roles, device="cuda", allow_torch_fallback=False)


@requires_cuda
def test_the_streamed_residency_refuses_the_fallback(monkeypatch):
    """The route passes ``allow_torch_fallback=(mode == resident)``: the
    streamed mode decodes inside a traced forward, where the pure-torch path's
    data-dependent shapes cannot run, so substituting it would substitute a
    residency the operator did not ask for."""
    from tessera.serving import ext

    monkeypatch.setattr(ext, "get_tessera_ext", lambda: None)
    with pytest.raises(ext.NativeKernelUnavailableError):
        _drive(monkeypatch, MODE_STREAMED, roles=(("weight", 128),), cols=512)
