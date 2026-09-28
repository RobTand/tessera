"""The folded native BF16 dense route and retained window-GEMV references.

The serve now prepares one packed native window GEMM for both residencies
and every M; it folds each weight's row scale before the bf16 dot. The CPU
eligibility tests describe the retained GEMV module, not the current serve.
Its precision, host-metadata and graph-capture checks prepare a reference
holder directly from the wire, preserving the older operator's unfolded
arithmetic without pretending that the serving route still creates it.

STUBBED: vLLM's ``LinearMethodBase`` and parameters, as in
``test_serving_bf16_route``.  There is no A-side quantiser to stub -- the A
side is bf16 as it arrives, which is the whole of this route's activation
contract.
"""
from __future__ import annotations

import sys
import types

import pytest

from tessera.serving.scheme import WINDOW_GEMM_SYMBOL

torch = pytest.importorskip("torch")

from tessera.serving import bf16_route as route                      # noqa: E402
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import telemetry                                # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import (                                # noqa: E402
    TESSERA_BF16, parse_tessera_blob_for_scheme,
)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.decode"), pytest.importorskip("tessera.alphabet"))


def _kg():
    return pytest.importorskip("tessera.kernel_window_gemv")


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


# --- the eligibility rule, on real units --------------------------------------
#
# CPU throughout: eligibility reads the unit's rates, window bits and start
# state -- no device, no extension -- so small CPU encodes decide it.


def _encode_cpu(q256, rows=32, cols=64, seed=0):
    fused, export, decode, alphabet = _tessera()
    torch.manual_seed(seed)
    w = torch.randn(rows, cols) * 0.02
    exported, unit, forests = export.encode_linear_planes(
        w, grid=alphabet.BF16_GRID, q256=q256, name="unit", verify=False)
    return exported, unit, forests


def _parse(exported):
    """The PARSED object off the exported bytes: since #264 the gate reads
    the published predicate, which includes the grid, so it takes a parse."""
    from tessera.unit_artifact import parse_unit_artifact

    return parse_unit_artifact(exported.blob, device="cpu")


def _probe(rates, window_bits=14, initial_state=None):
    """A parsed-unit stand-in carrying every fact the published predicate
    reads (``scheme.wire_facts_of_parsed``'s vocabulary)."""
    from types import SimpleNamespace

    unit = SimpleNamespace(
        body="WINDOW", scale_plane="CHANNEL",
        release_index=torch.zeros(0, dtype=torch.int64), diagonals=None,
        rotation=None, rates=tuple(rates), window_bits=window_bits,
        initial_state=initial_state)
    return SimpleNamespace(unit=unit, grid=SimpleNamespace(arity=1))


def test_eligibility_is_derived_from_the_kernel_constants():
    """The rule, not a roster: every rate and window THE KERNEL DECLARES is
    eligible.

    This used to open by restating ``(1, 2, 4)`` and ``(14,)`` -- the shape
    AGENTS.md rule 3 names, a test that passes on the day the list is wrong.
    The constants are now the parse of ``csrc/window_gemv.cu``'s own
    declaration (issue #145) and ``tests/test_kernel_roster.py`` owns that
    derivation, so what is left here is the rule this module is about: the
    eligibility gate says yes to everything the kernel has a lane for.
    """
    kg = _kg()
    top = max(kg.SUPPORTED_RATES)
    for rate in kg.SUPPORTED_RATES:
        for window_bits in kg.WINDOW_BITS_SUPPORTED:
            assert route.gemv_eligible_for_unit(
                _probe((rate,) * 8, window_bits=window_bits)), (rate, window_bits)
    outside = max(kg.WINDOW_BITS_SUPPORTED) + 1
    assert not route.gemv_eligible_for_unit(_probe((top,) * 8, window_bits=outside))
    assert not route.gemv_eligible_for_unit(
        _probe((max(kg.SUPPORTED_RATES) + 1,) * 8,
               window_bits=kg.WINDOW_BITS_SUPPORTED[0]))


def test_a_bresenham_mix_inside_the_supported_set_is_eligible():
    """Rates are per column: a schedule mixing supported rates is in range even
    though no uniform rung sits between them."""
    assert route.gemv_eligible_for_unit(_probe((2,) * 4 + (4,) * 4))


@pytest.mark.parametrize("q256", [256, 512, 1024])
def test_low_rungs_are_eligible(q256):
    """Real BF16 units at rate 1, 2 and 4: the kernel reads all three."""
    exported, unit, _forests = _encode_cpu(q256)
    assert set(int(r) for r in unit.rates) <= {1, 2, 4}, [int(r) for r in unit.rates]
    assert route.gemv_eligible_for_unit(_parse(exported))


@pytest.mark.parametrize("q256", [768, 1792])
def test_a_rate_outside_the_supported_set_is_not_eligible(q256):
    """R = 3 has no lane here, and neither does R = 7: the torch path serves."""
    exported, _unit, _forests = _encode_cpu(q256)
    assert not route.gemv_eligible_for_unit(_parse(exported))


def test_a_mixed_schedule_with_one_unsupported_column_is_not_eligible():
    """q256=896 mixes 3 and 4: the 4-columns are readable, the unit is not --
    dispatch is per module, so one unsupported column keeps the torch path."""
    exported, unit, _forests = _encode_cpu(896, cols=96)
    assert set(int(r) for r in unit.rates) == {3, 4}
    assert not route.gemv_eligible_for_unit(_parse(exported))


def test_a_window_outside_the_supported_set_is_not_eligible():
    assert not route.gemv_eligible_for_unit(_probe((4,) * 8, window_bits=15))


def test_a_shard_start_state_is_not_eligible():
    """A TP row shard starts mid-stream; the kernel supplies state_{-1} = 0
    itself, so a shard keeps the torch lane that threads the state."""
    from tessera.layout import slice_unit
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    torch.manual_seed(0)
    fused, export, decode, alphabet = _tessera()
    weight = torch.randn(32, 96) * 0.02
    exported, _unit, _forests = export.encode_linear_planes(
        weight, grid=alphabet.BF16_GRID, q256=512, name="unit", verify=False)
    parsed = parse_unit_artifact(exported.blob, device="cpu")
    assert route.gemv_eligible_for_unit(parsed)
    shard = slice_unit(parsed, rows=(8, 24))
    manifest = parsed.manifest
    _m, _region, blob = build_unit_artifact(
        shard, "rank", parsed.forests, int(manifest.branch.root_q256),
        parsed.code or ConvCode(),
        superblock=int(manifest.geometry.superblock_columns),
        container=manifest.branch.container)
    reparsed = parse_unit_artifact(blob, device="cpu")
    assert reparsed.unit.initial_state is not None
    assert not route.gemv_eligible_for_unit(reparsed)


# --- the lane's names ----------------------------------------------------------

def test_gemv_max_m_is_the_kernel_build_limit():
    """A wider kernel widens this route with no edit: one place to remember."""
    kg = _kg()
    assert route.GEMV_MAX_M == kg.GEMV_MAX_M == 8


def test_gemv_symbol_and_module_name_are_the_kernel_and_contract_values():
    """The record's symbol is the op the lane invokes; the module name is the
    contract table's constant, not a second literal."""
    from tessera.serving import ext
    assert route.GEMV_MODULE_NAME == ext.WINDOW_GEMV_MODULE_NAME == "tessera_window_gemv"
    assert route.GEMV_SYMBOL == "tessera_window_gemv::gemv"


def test_the_census_expectations_come_from_the_route():
    """What each REGIME may report, and since tessera#538 it is ONE launch.

    ``decode`` is the one-row forward and ``batch`` is every M > 1
    (``contract.CENSUS_PHASE_REGIMES``), which is the vocabulary a census
    record is stamped in.  This function derives its answer from
    ``scheme.ROUTE_LAUNCHES``, and that table used to carry the window-GEMV
    lane's three dense rows: the lane's own ``gemv`` in both regimes, the torch
    window decode in both, and the kernel-decoded tile under the stock GEMM in
    ``batch`` alone.

    ``1b767a207`` left ``bf16_route.apply`` making one launch -- the packed native window
    GEMM, at every M and in both residencies -- and contract v31 dropped the
    retired rows from the table, so the expectation a census compares a served
    record against became that one pair.  Contract v43 added the fused window
    kernel's dense identity (``native_fused_window_dense_folded``) as a second
    launch the route decides per module at weight load, so the expectation is
    now exactly the route's own ``DENSE_LAUNCHES`` -- two pairs, of which any
    one module stamps one.  Asserted as EQUALITY, because the defect this whole
    file is about was an expectation wider than the dispatch.

    A note on where this function lives, which the equality makes visible: it
    still belongs to ``bf16_route``, and ``bf16_route.apply`` does not import it.  The census tool
    reads it all the same, so it is right about the serve and housed in the
    wrong module; moving it is follow-up, not part of the withdrawal.
    """
    # On the folded arithmetic's own decoder since tessera#614.
    assert route.DENSE_LAUNCH == (WINDOW_GEMM_SYMBOL, telemetry.DECODER_NATIVE_WINDOW_GEMM_FOLDED)
    expected = set(route.DENSE_LAUNCHES)
    assert expected == {route.DENSE_LAUNCH, route.DENSE_FUSED_LAUNCH} and len(expected) == 2
    go = route.census_expected(compiled=False)
    assert go["decode"] == expected
    assert go["batch"] == expected
    gc = route.census_expected(compiled=True)
    assert (route.COMPILED_SYMBOL, route.COMPILED_DECODER) in gc["decode"]
    assert (route.COMPILED_SYMBOL, route.COMPILED_DECODER) in gc["batch"]
    assert route.GEMM_SYMBOL == "torch.mm"


def test_m_tile_is_the_kernel_build_rule():
    """The telemetry's tile is the lane's rule, off the lane itself."""
    kg = _kg()
    for m in (1, 2, 3, 4, 5, 8):
        assert route.m_tile(m) == kg._m_tile(m)


def _synthetic_holder(rate_one=False):
    """A one-role holder of small CPU tensors: only the shapes and the meta
    the dispatch reads, for the kernel-free tests below."""
    words = torch.zeros(8, dtype=torch.int32)
    items = torch.zeros(2, 8, dtype=torch.int32)
    perm = torch.arange(32, dtype=torch.int32)
    table = torch.ones(64, dtype=torch.bfloat16)
    scale = torch.ones(16, dtype=torch.float32)
    runs = torch.zeros(1, 4, dtype=torch.int32)
    tensors = (words, items, items[:0].clone(), perm, table, scale, runs)
    meta = (4, 16, 6, 16, 8, 4, 32, 0, int(rate_one), 1, 1)
    role = route._Bf16GemvRole("weight", 0, tensors, meta)
    return route.PreparedBf16Gemv([role], rows=16, columns=32, device=torch.device("cpu"))


def test_decode_is_gemv_is_the_m_rule_in_one_place():
    """M past the lane's max is prefill; M >= 4 over a rate-1 column has no
    8-row lane; everything else in the decode regime is the GEMV."""
    plain = _synthetic_holder(rate_one=False)
    assert route.GEMV_MAX_M == 8
    for m in (1, 2, 3, 4, 5, 8):
        assert route.decode_is_gemv(plain, m), m
    assert not route.decode_is_gemv(plain, 9)
    assert not route.decode_is_gemv(plain, 64)
    racy = _synthetic_holder(rate_one=True)
    assert racy.rate_one
    assert route.decode_is_gemv(racy, 2)
    assert not route.decode_is_gemv(racy, 4)
    assert not route.decode_is_gemv(racy, 8)


def test_streamed_apply_routes_by_m_and_rate_one(monkeypatch):
    """The custom op's dispatch, with the kernel behind fakes: the GEMV branch
    in the decode regime, the kernel-decode + GEMM branch past the lane's max
    and over a rate-1 column at M >= 4.  What the fakes return is distinctive
    per branch, so the assertions read which path ran."""
    kg = _kg()
    calls = []

    def _fake_gemv_concrete(x, *args):
        calls.append("gemv")
        return torch.ones(x.shape[0], int(args[7]), dtype=torch.float32) * 2.0

    class _FakeExt:
        def window_decode(self, words, tile_words, n_tiles, runs, perm, table,
                          window_bits, out):
            calls.append("decode")
            out.fill_(3.0)

    monkeypatch.setattr(kg, "_gemv_concrete", _fake_gemv_concrete)
    monkeypatch.setattr(kg, "_ext", lambda: _FakeExt())
    # ``torch.mm(..., out_dtype=)`` over bf16 is a CUDA kernel; on this CPU box
    # the materialised branch needs the same product spelled promotably.
    _real_mm = torch.mm
    monkeypatch.setattr(torch, "mm",
                        lambda a, b, **kw: _real_mm(a.float(), b.float())
                        if kw.get("out_dtype") is not None else _real_mm(a, b, **kw))

    plain = _synthetic_holder(rate_one=False)
    tensors, meta, rows, cols = plain.op_args()
    x = torch.ones(2, cols, dtype=torch.bfloat16)
    y = route.streamed_apply(x, tensors, meta, rows, cols)
    assert calls == ["gemv"] and tuple(y.shape) == (2, 16) and y.dtype == torch.bfloat16
    assert bool((y == torch.tensor(2.0, dtype=torch.bfloat16)).all())

    calls.clear()
    y64 = route.streamed_apply(torch.ones(64, cols, dtype=torch.bfloat16),
                               tensors, meta, rows, cols)
    assert calls == ["decode"] and tuple(y64.shape) == (64, 16)
    # tile 3.0, scale 1.0: each output is 32 * 3.0, in bf16.
    assert bool((y64 == torch.tensor(96.0, dtype=torch.bfloat16)).all())

    racy = _synthetic_holder(rate_one=True)
    rtensors, rmeta, rrows, rcols = racy.op_args()
    calls.clear()
    y2 = route.streamed_apply(torch.ones(2, cols, dtype=torch.bfloat16),
                              rtensors, rmeta, rrows, rcols)
    assert calls == ["gemv"] and tuple(y2.shape) == (2, 16)
    calls.clear()
    y4 = route.streamed_apply(torch.ones(4, cols, dtype=torch.bfloat16),
                              rtensors, rmeta, rrows, rcols)
    assert calls == ["decode"] and tuple(y4.shape) == (4, 16)


# --- the serve, on a CUDA box ---------------------------------------------------
#
# Everything below drives the route for real.  Without a GPU these skip; the
# CPU tests above are what pin the rule here.


def _install_vllm_stubs(monkeypatch):
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.__dict__["LinearMethodBase"] = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.__dict__.update(ModelWeightParameter=_param, BasevLLMParameter=_param)
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter)):
        monkeypatch.setitem(sys.modules, name, mod)


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank: the layer's OWN TP
    coordinates, which every ``LinearBase`` sets before ``create_weights``
    and the shard plan reads (tessera#303)."""

    tp_rank, tp_size = 0, 1


def _scheme(rows, columns, roles, q256, wire_bytes):
    return {"family": TESSERA_BF16, "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
            "q256": q256, "rows": rows, "columns": columns, "wire_bytes": wire_bytes,
            "roles": roles}


def _drive(monkeypatch, mode, roles=(("weight", 64),), cols=256, m=4, seed=0,
           q256=1024, *, with_gemv_reference=False):
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    fused, export, decode, alphabet = _tessera()
    torch.manual_seed(seed)
    values, scales, blobs = [], [], []
    for i, (name, rows) in enumerate(roles):
        w = torch.randn(rows, cols, device="cuda") * 0.02
        w[: max(1, rows // 8)] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.BF16_GRID, q256=q256, name=name, verify=False)
        tile, scale = decode.materialize_bf16(unit, forests, export.DEFAULT_CODE)
        values.append(tile)
        scales.append(scale.reshape(-1))
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    total = sum(r for _, r in roles)
    scheme = _scheme(total, cols, [[n, r] for n, r in roles], q256, len(blob))
    method = build_tessera_method(scheme, "test.layer")
    assert type(method).__name__ == "TesseraBf16LinearMethod"
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=cols,
                          output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=total, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.to(torch.device("cuda"))
    if with_gemv_reference:
        # Explicit test-owned reference; never a serving-route attribute.
        parsed = parse_tessera_blob_for_scheme(blob, scheme, "test.reference")
        layer.reference_gemv = route.prepare_bf16_gemv(
            parsed, device="cuda", expected=(torch.cat(values), torch.cat(scales)))
    method.process_weights_after_loading(layer)
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    return got, layer, method, x, (torch.cat(values), torch.cat(scales))


def _fp32_bound(tile_f32, scale, x):
    """A deterministic fp32 accumulation bound: ``2K * 2^-23 * sum_j |w_ij x_j|``
    (each of the K partial sums is rounded once, in either order; the factor 2
    covers the reference's own rounding of the same size).  The kernel's own
    GEMV tests derive this same bound; it is not a picked tolerance."""
    K = x.shape[1]
    mag = (tile_f32 * scale[:, None]).abs().double() @ x.abs().double().t()
    return (2 * K * 2.0 ** -23) * mag.t() + 1e-30


def _dense_reference(values, scale, x):
    """The served weight is bf16(value * scale), rounded once before the dot."""
    folded = (values.float().to(x.device) * scale.float().to(x.device)[:, None]).bfloat16().float()
    exact = x.float() @ folded.t()
    bound = _fp32_bound(folded, torch.ones(folded.shape[0], device=x.device), x)
    return exact, bound + 2.0 ** -8 * exact.abs()


@requires_cuda
def test_streamed_prepares_folded_native_without_materialized_planes(monkeypatch):
    _g, layer, _m, _x, (_values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024)
    assert layer.tessera_native is not None
    for name in ("tessera_gemv", "tessera_prepared", "weight_bf16", "wire_bytes"):
        assert not hasattr(layer, name)
    assert layer.tessera_decoder == route.DENSE_LAUNCH[1]
    assert torch.equal(layer.tessera_native.row_scale(), scale.cuda())


@requires_cuda
def test_rung_outside_reference_gemv_range_uses_dense_native(monkeypatch):
    """R=7 is outside the old GEMV range, not the native reader's range.

    This component test does not assert matched-cell serving qualification.
    """
    from tessera.serving.telemetry import read_route
    got, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1792,
                                                m=2, cols=512)
    assert layer.tessera_native is not None
    rec = read_route(layer)
    assert rec is not None
    assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert rec["state"] == "served"
    exact, bound = _dense_reference(values, scale, x)
    assert bool(((got.float() - exact).abs() <= bound).all())


@requires_cuda
def test_supported_rungs_declare_the_same_dense_compile_graph(monkeypatch):
    """Both rungs now choose the same packed op, not a per-unit GEMV fallback.

    Shape guards are the compiler's; the declared dispatch identity names
    the actual op rather than a legacy eligibility decision.
    """
    import json

    from tessera.serving import compile_identity as ci

    def _identity(mp, q256, cols, m):
        ci.reset_for_tests()
        cfg = types.SimpleNamespace(
            additional_config={},
            compilation_config=types.SimpleNamespace(
                mode=types.SimpleNamespace(name="VLLM_COMPILE")))
        ci.declare_compile_identity_in(cfg, serve_mode=MODE_STREAMED)
        _g, layer, _m, _x, _r = _drive(mp, MODE_STREAMED, q256=q256, cols=cols, m=m)
        return layer, json.dumps(cfg.additional_config, sort_keys=True)

    with pytest.MonkeyPatch.context() as mp:
        in_range, gemv = _identity(mp, 1024, 256, 4)
    with pytest.MonkeyPatch.context() as mp:
        out_of_range, torch_lane = _identity(mp, 1792, 512, 2)
    ci.reset_for_tests()

    assert in_range.tessera_native is not None
    assert out_of_range.tessera_native is not None
    assert WINDOW_GEMM_SYMBOL in gemv
    assert route.STREAMED_APPLY_OP not in gemv
    assert gemv == torch_lane, "the same native dispatch must declare the same graph"


@requires_cuda
def test_resident_also_holds_only_the_packed_native_bundle(monkeypatch):
    _g, layer, _m, _x, _r = _drive(monkeypatch, MODE_RESIDENT, q256=1024)
    assert layer.tessera_native is not None
    for name in ("tessera_gemv", "tessera_prepared", "weight_bf16", "wire_bytes"):
        assert not hasattr(layer, name)


@requires_cuda
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_decode_regime_serves_the_folded_native_window_gemm(monkeypatch, m):
    """Keep the dtype-derived error bar, against the current folded weights."""
    from tessera.serving.telemetry import read_route
    got, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                                m=m, seed=11)
    rec = read_route(layer)
    assert rec is not None
    assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert rec["state"] == "served"
    exact, bound = _dense_reference(values, scale, x)
    assert bool(((got.float() - exact).abs() <= bound).all())
    assert layer.tessera_native is not None


@requires_cuda
@pytest.mark.parametrize("m", [16, 64])
def test_prefill_keeps_the_folded_native_window_path(monkeypatch, m):
    """Prefill runs the same packed op without materializing a weight tile."""
    from tessera.serving.telemetry import read_route
    got, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                                m=m, seed=12)
    rec = read_route(layer)
    assert rec is not None
    assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH
    assert rec["state"] == "served"
    exact, bound = _dense_reference(values, scale, x)
    assert bool(((got.float() - exact).abs() <= bound).all())


@requires_cuda
def test_gemv_and_torch_agree_with_measured_differences(monkeypatch):
    """Same bytes, two engines: the decode-regime GEMV against the torch
    decode + torch.mm on the same module, with the exact max absolute and
    relative differences reported.

    Bit-exactness is asserted where the bytes are exact (the lane's decode of
    the tile); the two fp32 products may differ by summation order, so they
    are held to the derived fp32 bound -- and the measured differences are
    printed for the record.
    """
    _g, layer, _m, x, (values, scale) = _drive(
        monkeypatch, MODE_STREAMED, q256=1024, m=4, seed=13, with_gemv_reference=True)
    holder = layer.reference_gemv
    assert holder is not None
    tensors, meta, rows, cols = holder.op_args()
    y_gemv = route.streamed_apply(x.contiguous(), tensors, meta, rows, cols)
    tile, scl = values.cuda(), scale.float().cuda()
    y_torch = (torch.mm(x.contiguous(), tile.t(), out_dtype=torch.float32)
               * scl).to(torch.bfloat16)
    absdiff = (y_gemv.float() - y_torch.float()).abs()
    max_abs = float(absdiff.max())
    denom = float(y_torch.float().abs().max())
    print(f"\nBF16 GEMV-vs-torch on 64x256 R=4 M=4: max_abs={max_abs:.3e} "
          f"max_rel={max_abs / max(denom, 1e-9):.3e}")
    ref = (x.float() @ (tile.float() * scl[:, None]).t())
    bound = _fp32_bound(tile.float(), scl, x) + 2.0 ** -8 * ref.abs()
    assert bool(((y_gemv.float() - ref).abs() <= bound).all())
    assert bool(((y_torch.float() - ref).abs() <= bound).all())


@requires_cuda
def test_rate1_columns_keep_dense_dispatch_across_decode_sizes(monkeypatch):
    """The current dense reader does not inherit the reference GEMV's M limit."""
    from tessera.serving.telemetry import read_route
    for m, seed in ((2, 20), (4, 21)):
        got, layer, _method, x, (values, scale) = _drive(
            monkeypatch, MODE_STREAMED, q256=256, m=m, seed=seed)
        assert layer.tessera_native is not None
        rec = read_route(layer)
        assert rec is not None
        assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH
        assert tuple(got.shape) == (m, layer.tessera_rows)
        exact, bound = _dense_reference(values, scale, x)
        assert bool(((got.float() - exact).abs() <= bound).all())


@requires_cuda
def test_the_retained_run_descriptor_lives_on_the_host(monkeypatch):
    """#203: ``runs`` is host-read metadata -- ``ext.window_decode`` moves it
    to the CPU and reads its scalars through ``.item()`` to build the kernel
    launches -- so the holder keeps it on CPU from preparation, the way
    ``fp8_gemv.prepare_fp8_gemv`` already does.  Retaining the repack-device
    tensor made every materialised forward a synchronous device-to-host copy.
    Every other retained tensor stays on the device."""
    _g, layer, _m, _x, _r = _drive(
        monkeypatch, MODE_STREAMED, q256=1024, with_gemv_reference=True)
    holder = layer.reference_gemv
    assert holder is not None
    tensors, _meta, _rows, _cols = holder.op_args()
    n = len(route._ROLE_TENSORS)
    for i in range(len(tensors) // n):
        for key, t in zip(route._ROLE_TENSORS, tensors[n * i:n * (i + 1)], strict=True):
            if key == "runs":
                assert t.device.type == "cpu", \
                    "the run descriptor is read on the host; keeping it on CUDA is an " \
                    "in-forward device-to-host copy (#203)"
            else:
                assert t.is_cuda, key


@requires_cuda
@pytest.mark.parametrize("q256,m", [(1024, 16), (256, 4)],
                         ids=["prefill-m16", "rate1-fallback-m4"])
def test_the_materialised_branch_survives_cuda_graph_capture(monkeypatch, q256, m):
    """#203's consequence, measured: the two materialised branches -- M past
    the lane's max, and M >= 4 over a rate-1 column -- capture into a
    ``torch.cuda.CUDAGraph`` and replay to the right numbers.  With the run
    descriptor retained on CUDA, ``ext.window_decode``'s
    ``runs.to(torch::kCPU)`` is a synchronous copy the capture refuses, so
    the serving graph fails to initialise at exactly the batch shapes vLLM
    captures."""
    _got, layer, _method, _x, (values, scale) = _drive(
        monkeypatch, MODE_STREAMED, q256=q256, m=2, seed=30 + m, with_gemv_reference=True)
    holder = layer.reference_gemv
    assert holder is not None
    tensors, meta, rows, cols = holder.op_args()
    if q256 == 256:
        assert holder.rate_one
    assert not route.decode_is_gemv(holder, m), "these shapes are the materialised branch"
    x = torch.randn(m, cols, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(40 + m)).bfloat16()
    # Warm up on a side stream (allocator and cuBLAS workspace state), the
    # standard capture prologue.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    y_eager = None
    with torch.cuda.stream(side):
        for _ in range(3):
            y_eager = route.streamed_apply(x, tensors, meta, rows, cols)
    torch.cuda.current_stream().wait_stream(side)
    assert y_eager is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_cap = route.streamed_apply(x, tensors, meta, rows, cols)
    y_cap.zero_()          # replay must be what rewrites it
    graph.replay()
    torch.cuda.synchronize()
    tile, scl = values.float().cuda(), scale.float().cuda()
    exact = (x.float() @ (tile * scl[:, None]).t())
    bound = _fp32_bound(tile, scl, x) + 2.0 ** -8 * exact.abs()
    assert bool(((y_eager.float() - exact).abs() <= bound).all())
    assert bool(((y_cap.float() - exact).abs() <= bound).all())


@requires_cuda
def test_optional_gemv_extension_is_not_needed_by_dense_dispatch(monkeypatch):
    """The old reference extension cannot select a materialized serve fallback."""
    from tessera import kernel_window_gemv
    from tessera.serving.telemetry import read_route

    def _no_toolchain():
        raise RuntimeError("no nvcc on this box")

    monkeypatch.setattr(kernel_window_gemv, "_ext", _no_toolchain)
    got, layer, _m, _x, _r = _drive(monkeypatch, MODE_STREAMED, q256=1024, m=2, seed=14)
    assert layer.tessera_native is not None
    assert not hasattr(layer, "tessera_gemv")
    rec = read_route(layer)
    assert rec is not None
    assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH


@requires_cuda
def test_the_dispatch_survives_a_compiled_forward_with_a_dynamic_token_dim(monkeypatch):
    """One folded native graph serves M = 1..8 and M = 64 without recompile."""
    from tessera.serving.telemetry import read_route
    torch._dynamo.reset()
    _got, layer, method, _x, _r = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                         m=2, seed=15)
    assert layer.tessera_native is not None
    compiled = torch.compile(lambda x: method.apply(layer, x))

    def x_for(M, seed):
        x = torch.randn(M, layer.tessera_columns, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed)).bfloat16()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        return x

    for i, M in enumerate((2, 1, 4, 8, 64)):
        with torch._dynamo.config.patch(error_on_recompile=(i > 0)):
            y = compiled(x_for(M, 500 + M))
        assert tuple(y.shape) == (M, layer.tessera_rows) and y.dtype == torch.bfloat16
    rec = read_route(layer)
    assert rec is not None
    assert str(rec["shape"]).startswith("M*:")
    assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH
