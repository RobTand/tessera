"""The Tessera FP8 W8A8 dense serving route.

Exercised for real on a CUDA box with ``tessera`` importable: the container
parse, the packed-window decode, the reference-decoder cross-check at
preparation, the per-row ``scale_b``, ``torch._scaled_mm`` W8A8, both
residency modes, the compiled decode and the refusals.  The load-bearing
assertion is BYTE identity of the decoded pair with ``tessera.stock.
materialize_stock`` -- the tensors the compressed-tensors stock lane served
on vanilla vLLM -- so the numbers this route produces are, by construction,
the stock lane's served numbers.  STUBBED: vLLM's ``LinearMethodBase`` /
parameters, the per-token FP8 activation quantiser (an audited upstream op)
and the ABI attestation that would otherwise reach for the real vLLM operator
library.  vLLM loading this route owes a container run, as for every route.

Ported from Gridbook's ``test_tessera_fp8_lane.py``; the enable flag is gone
(the checkpoint selects the plugin) and the residency is the only flag left.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import fp8_route as route                       # noqa: E402
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import native_ops, telemetry                    # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import (                                 # noqa: E402
    TESSERA_FP8, TESSERA_NVFP4, validate_tessera_scheme)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.stock"), pytest.importorskip("tessera.alphabet"))


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


def _scheme(rows=256, columns=1024, roles=None, **over):
    s = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
         "rows": rows, "columns": columns, "wire_bytes": 4096,
         "roles": roles if roles is not None else [["weight", rows]]}
    s.update(over)
    return s


# --- the scheme --------------------------------------------------------------

def test_fp8_scheme_normalises_and_refuses_the_other_routes_vocabulary():
    norm = validate_tessera_scheme(_scheme(roles=[["q_proj", 128], ["k_proj", 128]]), "t")
    assert norm["family"] == TESSERA_FP8 and norm["plane"] == "CHANNEL"
    assert norm["roles"] == [("q_proj", 128), ("k_proj", 128)]
    with pytest.raises(ValueError, match="scalar E4M3 grid"):
        validate_tessera_scheme(_scheme(grid="E2M1x2"), "t")
    with pytest.raises(ValueError, match="no FP8 tile"):
        validate_tessera_scheme(_scheme(plane="LUT"), "t")
    with pytest.raises(ValueError, match="K % 16"):
        validate_tessera_scheme(_scheme(columns=1000), "t")
    with pytest.raises(ValueError, match="no NVFP4 tile"):
        validate_tessera_scheme({**_scheme(), "family": TESSERA_NVFP4, "grid": "E2M1x2"}, "t")


def test_the_route_refuses_before_vllm_and_the_family_picks_the_route(monkeypatch):
    """FAMILY = ROUTE.  The checkpoint's fact, not the operator's: the only
    thing left for the operator to declare is the residency."""
    assert not hasattr(serving_lane, "TESSERA_FLAG")
    with pytest.raises(ValueError, match=TESSERA_MODE_ENV):
        build_tessera_method(_scheme(), "test.layer")
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="family must be one of"):
        build_tessera_method({**_scheme(), "family": "TESSERA_INT4"}, "test.layer")
    with pytest.raises(ValueError, match=f"serves {TESSERA_FP8}, not"):
        # A COHERENT NVFP4 scheme, handed to the wrong builder.  q256 is 896
        # because that is the only rate the E2M1x2 reader takes -- leaving the
        # FP8 default here would be refused by the rung gate first, which is a
        # true refusal but not the one under test.
        route.build_tessera_fp8_method({**_scheme(), "family": TESSERA_NVFP4, "grid": "E2M1x2",
                                        "plane": "LUT", "body": "TCQ", "q256": 896},
                                       "test.layer", "resident")


def test_the_fp8_routes_decoder_is_pure_torch():
    """No CUDA extension at all on this route: the window decoder is torch."""
    assert route.ACTIVATION_CONTRACT == "fp8_per_token_dynamic"
    assert telemetry.DECODER_TORCH_WINDOW in telemetry.DECODERS


def _selected_module(expert, names=('gate', 'up')):
    from tessera.serving.window import prepare_window
    generator = torch.Generator().manual_seed(907 + expert)
    roles = []
    for position, name in enumerate(names):
        window = prepare_window(
            torch.randint(0, 4, (16, 8), generator=generator, dtype=torch.uint8),
            [2] * 8, 8, torch.randperm(256, generator=generator).to(torch.uint8), 'cpu')
        roles.append(route._Fp8Role(name, position * 16, 16, window))
    return route.PreparedTesseraFp8Module(
        roles, rows=len(names) * 16, columns=8,
        scale=torch.arange(len(names) * 16, dtype=torch.float32) + expert + 1,
        device=torch.device('cpu'))


def test_selected_expert_modules_preserve_distinct_roles_scales_and_id_order():
    modules = [_selected_module(expert) for expert in range(4)]
    batch = route.PreparedTesseraFp8Module.stack(modules)
    ids = torch.tensor([3, 0, 2, 3])
    expected = torch.stack([m.decode() for m in modules]).index_select(0, ids)
    assert torch.equal(batch.decode(ids, max_experts_per_chunk=2), expected)
    assert torch.equal(batch.row_scale(ids), torch.stack([m.row_scale() for m in modules]).index_select(0, ids))
    assert batch.role_names == ('gate', 'up')
    assert batch.decode(ids[:0], max_experts_per_chunk=2).shape == (0, 32, 8)
    first = batch.decode(ids, max_experts_per_chunk=1)
    second = batch.decode(ids, max_experts_per_chunk=4)
    assert first.data_ptr() != second.data_ptr()
    assert batch.resident_bytes() == batch.wire_bytes_resident() + 4 * 32 * 4
    modules[0]._PreparedTesseraFp8Module__scale.zero_()
    assert batch.row_scale(torch.tensor([0]))[0, 0].item() == 1


def test_selected_expert_modules_refuse_mixed_role_order_and_empty_stack():
    with pytest.raises(ValueError, match='at least one'):
        route.PreparedTesseraFp8Module.stack([])
    with pytest.raises(ValueError, match='roles'):
        route.PreparedTesseraFp8Module.stack([_selected_module(0), _selected_module(1, ('up', 'gate'))])


def test_concatenated_roles_share_packed_windows_and_preserve_scale_order():
    modules = [_selected_module(0, ('gate',)), _selected_module(1, ('up',))]
    combined = route.PreparedTesseraFp8Module.concatenate(modules)
    assert torch.equal(combined.decode(), torch.cat([m.decode() for m in modules]))
    assert torch.equal(combined.row_scale(), torch.cat([m.row_scale() for m in modules]))
    actual = combined._PreparedTesseraFp8Module__roles
    assert all(a.window is m._PreparedTesseraFp8Module__roles[0].window
               for a, m in zip(actual, modules))
    assert [r.row_offset for r in actual] == [0, 16]
    with pytest.raises(ValueError, match='at least one'):
        route.PreparedTesseraFp8Module.concatenate([])
    with pytest.raises(ValueError, match='distinct names'):
        route.PreparedTesseraFp8Module.concatenate([modules[0], modules[0]])


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
FP8_MAX = 448.0


def _reference_fp8_quant(x):
    """Per-token dynamic E4M3 (the arithmetic of vLLM's op), recording the value
    it represents so the expectation is the A side the route consumed."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    _LAST_A["value"] = q.float() * scale
    return q, scale.contiguous()


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank: the layer's OWN TP
    coordinates, which every ``LinearBase`` sets before ``create_weights``
    and the shard plan reads (tessera#303)."""

    tp_rank, tp_size = 0, 1


def _encode_module(roles, cols=1024, q256=1024, seed=0):
    """Encode ``roles`` = [(name, rows)] with Tessera on the E4M3 grid; return
    the container blob, the scheme and the stock reference pair."""
    fused, export, stock, alphabet = _tessera()
    torch.manual_seed(seed)
    tensors, blobs = {}, []
    for i, (name, rows) in enumerate(roles):
        w = (torch.randn(rows, cols, device="cuda") * 0.02)
        w[: rows // 8] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.E4M3_GRID, q256=q256, name=name, verify=False)
        tensors[name] = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme(rows=sum(r for _, r in roles), columns=cols, wire_bytes=len(blob),
                     roles=[[n, r] for n, r in roles], q256=q256)
    weight = torch.cat([tensors[n]["weight"].view(torch.uint8) for n, _ in roles])
    scale = torch.cat([tensors[n]["weight_scale"].reshape(-1) for n, _ in roles])
    ref_w = torch.cat([stock.stock_dequant(tensors[n]) for n, _ in roles])
    return blob, scheme, weight, scale, ref_w


def _drive(monkeypatch, mode, roles=(("weight", 256),), cols=1024, m=32, seed=0, q256=1024):
    # The residency is latched to the first value this process read, so a test
    # that drives both modes clears it here rather than only between tests.
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_ops, "native_fp8_quant", _reference_fp8_quant)
    # With sys.modules['vllm'] stubbed, the real operator library is not
    # importable; record that the route ATTESTS the ABI rather than executing
    # an attestation the stub cannot satisfy.  See the report.
    _ATTESTED.clear()
    monkeypatch.setattr(native_ops, "require_native_fp8_quant",
                        lambda context: _ATTESTED.append(context))
    blob, scheme, weight, scale, ref_w = _encode_module(list(roles), cols=cols, seed=seed, q256=q256)
    method = build_tessera_method(scheme, "test.layer")
    assert type(method).__name__ == "TesseraFp8LinearMethod"
    layer = _Layer()
    rows = scheme["rows"]
    method.create_weights(layer, input_size_per_partition=cols, output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.to(torch.device("cuda"))
    method.process_weights_after_loading(layer)
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    want = (_LAST_A["value"] @ ref_w.t()).to(torch.bfloat16)
    return got, want, layer, method, (weight, scale)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_pair_is_the_stock_pair_byte_for_byte(monkeypatch, mode):
    """The retained reference pair IS ``materialize_stock``'s, the route's
    scale is beside it, and the packed native forward matches the stock
    product.

    The route no longer materialises a tile (``native_window`` holds packed
    bundles), so the byte-for-byte claim is made where it still has a subject:
    the RETAINED ``prepare_tessera_fp8_module`` preparation, which stays in the
    tree as the oracle.  The route's own claim is the forward parity at the
    end, against the stock pair's product.
    """
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    got, want, layer, method, (weight, scale) = _drive(monkeypatch, mode)
    blob, scheme, *_ = _encode_module([("weight", 256)], cols=1024)
    reference = route.prepare_tessera_fp8_module(
        parse_tessera_blob_for_scheme(blob, scheme, "t"), device="cuda")
    tile = reference.decode()
    assert torch.equal(tile, weight)
    assert torch.equal(reference.row_scale(), scale)
    assert torch.equal(layer.scale_b.reshape(-1), scale)
    assert layer.scale_b.dtype == torch.float32 and tuple(layer.scale_b.shape) == (1, weight.shape[0])
    assert layer.tessera_native is not None
    assert not hasattr(layer, "weight_fp8") and not hasattr(layer, "tessera_prepared")
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_the_route_attests_the_native_a_side_abi(monkeypatch):
    _drive(monkeypatch, MODE_RESIDENT)
    # Two attestations of the same ABI, and both are the point: the prepared
    # bundle attests the quantizer it will accept bf16 through, and the route
    # attests the A side it actually runs, once each at load.
    assert len(_ATTESTED) == 2
    assert any("test.layer" in context for context in _ATTESTED)
    assert any("quantizer" in context for context in _ATTESTED)


@requires_cuda
def test_fused_roles_decode_into_row_slices_with_their_own_row_scales(monkeypatch):
    roles = (("q_proj", 256), ("k_proj", 128), ("v_proj", 128))
    got, want, layer, _m, (weight, scale) = _drive(monkeypatch, MODE_RESIDENT, roles=roles, seed=3)
    assert layer.tessera_roles == ("q_proj", "k_proj", "v_proj")
    assert layer.tessera_native is not None
    assert torch.equal(layer.scale_b.reshape(-1), scale)
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    blob, scheme, *_ = _encode_module(list(roles), cols=1024, seed=3)
    reference = route.prepare_tessera_fp8_module(
        parse_tessera_blob_for_scheme(blob, scheme, "t"), device="cuda")
    assert torch.equal(reference.decode(), weight)
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_a_mixed_rate_schedule_decodes_through_the_group_permutation(monkeypatch):
    got, want, layer, _m, (weight, scale) = _drive(monkeypatch, MODE_STREAMED, cols=512, seed=4, q256=1000)
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    blob, scheme, *_ = _encode_module([("weight", 256)], cols=512, seed=4, q256=1000)
    reference = route.prepare_tessera_fp8_module(
        parse_tessera_blob_for_scheme(blob, scheme, "t"), device="cuda")
    assert torch.equal(reference.decode(), weight)
    assert len(set(layer.tessera_native.layout_facts()[0].rates)) > 1, "not a mixed schedule"


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    a, _w, _l, _m, _ = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    b, _w2, _l2, _m2, _ = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(a, b)


@requires_cuda
def test_streamed_holds_the_packed_wire_and_no_resident_tile(monkeypatch):
    _g, _w, a, _m, _ = _drive(monkeypatch, MODE_STREAMED, roles=(("weight", 512),), cols=512)
    for name in ("wire_bytes", "weight_fp8", "decode_buf", "tessera_prepared", "tessera_gemv"):
        assert not hasattr(a, name), name
    native = a.tessera_native
    assert native is not None
    # The packed wire half (words + tables + the fp32 row scale), not the
    # 8-bit tile: a generous cap over the body's own bytes.
    assert native.packed_bytes() < 512 * 512 * 4.5 / 8 + 65536, native.packed_bytes()
    fingerprints = native.fingerprints()
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    q, s = _reference_fp8_quant(x)
    p1, p2 = native.apply(q, s), native.apply(q, s)
    assert torch.equal(p1, p2) and p1.data_ptr() != p2.data_ptr()
    assert native.fingerprints() == fingerprints, "a forward changed the prepared weights"


@requires_cuda
def test_resident_drops_the_wire(monkeypatch):
    _g, _w, r, _m, _ = _drive(monkeypatch, MODE_RESIDENT)
    assert not hasattr(r, "wire_bytes") and not hasattr(r, "weight_fp8")
    native = r.tessera_native
    assert native is not None
    # Resident mode holds the packed repack, not an 8-bit tile: the wire's own
    # words over the layout's padded rows (``rep.rows_p``), plus the small
    # tables and the per-column bookkeeping.
    facts = native.layout_facts()[0]
    cap = facts.rows_p * facts.cols * 4.5 / 8 + 65536
    assert native.packed_bytes() < cap, (native.packed_bytes(), cap)


@requires_cuda
def test_streamed_decode_traces_under_torch_compile(monkeypatch):
    _g, _w, layer, _m, (weight, _s) = _drive(monkeypatch, MODE_STREAMED, cols=512, seed=9, q256=1000)
    native = layer.tessera_native
    assert native is not None
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    q, s = _reference_fp8_quant(x)
    eager = native.apply(q, s)
    compiled = torch.compile(lambda a, b: native.apply(a, b), fullgraph=True)(q, s)
    assert torch.equal(compiled, eager)


@requires_cuda
def test_preparation_refuses_a_decoder_that_disagrees_with_the_reference(monkeypatch):
    import tessera.decode as tdecode
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 128)], cols=512)
    roles = parse_tessera_blob_for_scheme(blob, scheme, "t")
    real = tdecode.materialize_fp8

    def _wrong(unit, forest, code):
        b, s = real(unit, forest, code)
        b = b.clone(); b[0, 0] ^= 1
        return b, s

    monkeypatch.setattr(tdecode, "materialize_fp8", _wrong)
    with pytest.raises(RuntimeError, match="disagrees with tessera.decode.materialize_fp8 on 1 of"):
        route.prepare_tessera_fp8_module(roles, device="cuda")


@requires_cuda
def test_scheme_and_blob_must_agree(monkeypatch):
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 128)], cols=512)
    from tessera.serving.scheme import parse_compact_blob_for_scheme

    parse_tessera_blob_for_scheme(blob, scheme, "t")
    parse_compact_blob_for_scheme(blob, scheme, "t", device="cuda")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "q256": 896}, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_compact_blob_for_scheme(blob, {**scheme, "q256": 896}, "t", device="cuda")
    with pytest.raises(ValueError, match="TESSERA_FP8 serves WINDOW bodies"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "body": "TCQ"}, "t")
    with pytest.raises(ValueError, match="TESSERA_FP8 serves WINDOW bodies"):
        parse_compact_blob_for_scheme(blob, {**scheme, "body": "TCQ"}, "t", device="cuda")


@requires_cuda
def test_route_record_names_the_family_mode_contract_and_decoder(monkeypatch):
    from tessera.serving.telemetry import read_route
    _g, _w, layer, _m, _ = _drive(monkeypatch, MODE_STREAMED)
    from tessera.serving.scheme import WINDOW_GEMM_SYMBOL

    rec = read_route(layer)
    assert rec is not None and rec["policy"] == f"{TESSERA_FP8}:streamed" and rec["state"] == "served"
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "fp8_per_token_dynamic"
    # The packed native GEMM, in both residencies and at every M: the tile is
    # never materialised, so the record names the op that ran and the decoder
    # that ran it.
    assert rec["symbol"] == WINDOW_GEMM_SYMBOL == "tessera::window_gemm_dense"
    assert rec["decoder"] == telemetry.DECODER_NATIVE_WINDOW_GEMM == layer.tessera_decoder
    assert rec["decoder"] in telemetry.DECODERS
