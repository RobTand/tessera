"""The Tessera 16-bit W16A16 dense serving route.

Exercised for real on a CUDA box: the container parse, the packed-window
decode, the reference-decoder cross-check at preparation, the fp32 row-scale
epilogue, both residency modes, the compiled decode and the refusals.

**The load-bearing assertion is that the row scale never touches the tile.**
That is what the family is for: a CHANNEL scale is an output-row factor and
commutes with the matmul, so a lane holding the wire runs the GEMM on the raw
table values -- exact, since every entry is already a bf16 word -- and applies
the scale to the fp32 output.  Folding instead adds one bf16 rounding
(~0.0011-0.0022 absolute on GLM expert rows at any rate).  Its 15.4% *share*
at R = 7 composes in quadrature -- a 1.2% error gap, 2.4% squared -- and served
at R = 7 the twin's KL is 1.0011x the route's on ``all`` and 0.9961x on
``confident``, i.e. below what the corpus resolves, so no fold win is claimed
here (#45).  Two tests hold the line directly: the decoded tile equals
``materialize_bf16``'s VALUES (not the folded twin's tensor), and the route's
output is closer to the exact fp32 product than the folded rendering is --
both in weight space, which is where the fold is visible.

STUBBED: vLLM's ``LinearMethodBase`` and parameters.  There is no A-side
quantiser to stub -- the A side is bf16 as it arrives, which is the whole of
this route's activation contract, and one test asserts the route reaches for no
such op.  vLLM loading this route owes a container run, as for every route.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import bf16_route as route                      # noqa: E402
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import telemetry                                # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import (                                 # noqa: E402
    TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4, validate_tessera_scheme)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")

#: Small, and a real rung: q256=1792 is R=7, the rate the alphabet-floor
#: measurement singles out as where an 8-bit tile has stopped paying and this
#: family has not.
Q256 = 1792


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.decode"), pytest.importorskip("tessera.alphabet"))


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


def _scheme(rows=64, columns=512, roles=None, **over):
    s = {"family": TESSERA_BF16, "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
         "q256": Q256, "rows": rows, "columns": columns, "wire_bytes": 4096,
         "roles": roles if roles is not None else [["weight", rows]]}
    s.update(over)
    return s


# --- the scheme --------------------------------------------------------------

def test_bf16_scheme_normalises_and_refuses_the_other_routes_vocabulary():
    norm = validate_tessera_scheme(_scheme(roles=[["gate_proj", 32], ["up_proj", 32]]), "t")
    assert norm["family"] == TESSERA_BF16 and norm["plane"] == "CHANNEL"
    assert norm["roles"] == [("gate_proj", 32), ("up_proj", 32)]
    with pytest.raises(ValueError, match="scalar BF16 grid"):
        validate_tessera_scheme(_scheme(grid="E4M3"), "t")
    with pytest.raises(ValueError, match="no BF16 tile"):
        validate_tessera_scheme(_scheme(plane="LUT"), "t")
    with pytest.raises(ValueError, match=f"serves {TESSERA_BF16}, not"):
        route.build_tessera_bf16_method(
            {**_scheme(), "family": TESSERA_FP8, "grid": "E4M3", "q256": 1024},
            "test.layer", "resident")


def test_a_wide_column_count_is_not_refused_for_a_group_this_tile_has_not_got():
    """A bf16 tile is one word a weight, so the GEMM takes any K.

    The other two routes decode to a PACKED tile whose mainloop reads groups --
    a nibble pair, a group-16 block scale -- and refuse K % 16.  Copying that
    quantum here would refuse geometries this route serves, which is a refusal
    with no mechanism behind it.
    """
    assert validate_tessera_scheme(_scheme(columns=1000), "t")["columns"] == 1000
    with pytest.raises(ValueError, match="K % 16"):
        validate_tessera_scheme({**_scheme(), "family": TESSERA_FP8, "grid": "E4M3",
                                 "q256": 1024, "columns": 1000}, "t")


def test_the_reader_rate_range_gates_this_family_too():
    """The rung gate resolves by (route, grid); BF16 has its own range."""
    assert validate_tessera_scheme(_scheme(q256=4096), "t")["q256"] == 4096
    with pytest.raises(ValueError, match="outside the rungs this build's decoder reads"):
        validate_tessera_scheme(_scheme(q256=4097), "t")
    with pytest.raises(ValueError, match="outside the rungs this build's decoder reads"):
        validate_tessera_scheme(_scheme(q256=255), "t")


def test_the_bf16_route_declares_an_unquantised_a_side():
    """W16A16, and the contract says so with a VALUE.

    A gate reading ``activation_contract`` has to be able to tell "unquantised
    by design" from "nobody filled it in", and only a value carries that.
    """
    assert route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert route.ACTIVATION_CONTRACT in telemetry.ROUTE_CONTRACTS


def test_the_route_contract_set_is_derived_from_the_table():
    from tessera.serving import scheme as sch
    assert telemetry.ROUTE_CONTRACTS == {
        r["activation_contract"] for r in sch.ROUTES.values()}, \
        "a hand-written contract set is a second place to remember"


def test_every_families_grid_description_comes_off_its_route():
    """The refusal that names the grid a family holds must not be an if-chain.

    An if-chain describes every family it has not heard of as the family it was
    written for -- which is how a BF16 scheme would have been refused with the
    words "the scalar E4M3 grid".
    """
    from tessera.serving import scheme as sch
    for family, route_ in sch.ROUTES.items():
        assert route_["grid_kind"], family
        with pytest.raises(ValueError, match=route_["grid_kind"]):
            validate_tessera_scheme(
                {**_scheme(), "family": family, "grid": "NOT_A_GRID",
                 "plane": route_["plane"]}, "t")


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


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank: the layer's OWN TP
    coordinates, which every ``LinearBase`` sets before ``create_weights``
    and the shard plan reads (tessera#303)."""

    tp_rank, tp_size = 0, 1


def _encode_module(roles, cols=512, q256=Q256, seed=0):
    """Encode ``roles`` = [(name, rows)] on the BF16 grid; return the container
    blob, the scheme, the reference pair and the FOLDED twin tensor."""
    fused, export, decode, alphabet = _tessera()
    torch.manual_seed(seed)
    values, scales, folded, blobs = [], [], [], []
    for i, (name, rows) in enumerate(roles):
        w = (torch.randn(rows, cols, device="cuda") * 0.02)
        w[: max(1, rows // 8)] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.BF16_GRID, q256=q256, name=name, verify=False)
        tile, scale = decode.materialize_bf16(unit, forests, export.DEFAULT_CODE)
        values.append(tile)
        scales.append(scale.reshape(-1))
        folded.append(decode.materialize_bf16_folded(unit, forests, export.DEFAULT_CODE))
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme(rows=sum(r for _, r in roles), columns=cols, wire_bytes=len(blob),
                     roles=[[n, r] for n, r in roles], q256=q256)
    return (blob, scheme, torch.cat(values), torch.cat(scales), torch.cat(folded))


def _drive(monkeypatch, mode, roles=(("weight", 64),), cols=512, m=8, seed=0, q256=Q256):
    # The residency is latched to the first value this process read, so a test
    # that drives both modes clears it here rather than only between tests.
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    blob, scheme, values, scale, folded = _encode_module(list(roles), cols=cols, seed=seed,
                                                        q256=q256)
    method = build_tessera_method(scheme, "test.layer")
    assert type(method).__name__ == "TesseraBf16LinearMethod"
    layer = _Layer()
    rows = scheme["rows"]
    method.create_weights(layer, input_size_per_partition=cols,
                          output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.to(torch.device("cuda"))
    method.process_weights_after_loading(layer)
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                   generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    return got, layer, method, x, (values, scale, folded)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_the_tile_is_the_reference_values_and_the_scale_is_beside_it(monkeypatch, mode):
    """The decoded tile IS ``materialize_bf16``'s values -- never the fold."""
    _got, layer, _m, _x, (values, scale, folded) = _drive(monkeypatch, mode)
    tile = layer.weight_bf16 if mode == MODE_RESIDENT else layer.tessera_prepared.decode()
    assert tile.dtype == torch.bfloat16
    assert torch.equal(tile, values)
    assert torch.equal(layer.row_scale, scale)
    assert layer.row_scale.dtype == torch.float32
    assert tuple(layer.row_scale.shape) == (values.shape[0],)
    # And it is NOT the folded twin: if it were, this whole family would be
    # paying the fold it exists to avoid, and every other assertion here would
    # still pass.
    assert not torch.equal(tile, folded), "the tile has the row scale folded into it"


@requires_cuda
def test_the_route_beats_the_fold_it_refuses_to_do(monkeypatch):
    """Not folding is measurably better in weight space, which is why the rule exists.

    The exact answer is the fp32 product of the unfolded pair.  The route's
    output and the folded twin's are both approximations of it; the route's
    must be the closer one, or the pair is costing bytes for nothing.  (Served
    KL at R = 7 does not resolve the difference -- #45 -- so this weight-space
    ordering is the claim, not a served win.)
    """
    got, _layer, _m, x, (values, scale, folded) = _drive(monkeypatch, MODE_RESIDENT, m=8)
    exact = (x.float() @ (values.float() * scale[:, None]).t())
    twin = (x @ folded.t()).float()
    err_route = (got.float() - exact).norm() / exact.norm()
    err_twin = (twin - exact).norm() / exact.norm()
    assert err_route < err_twin, f"route {err_route:.3e} is not better than the fold {err_twin:.3e}"


@requires_cuda
def test_the_epilogue_keeps_the_gemms_own_accumulator(monkeypatch):
    """fp32 out, scale, one rounding -- not bf16 out, scale, two roundings.

    Rounding the GEMM to bf16 before applying the row scale would put a second
    rounding between the accumulator and the answer, which is most of what not
    folding the scale was bought to avoid.
    """
    got, _layer, _m, x, (values, scale, _folded) = _drive(monkeypatch, MODE_STREAMED, m=8)
    exact = (x.float() @ (values.float() * scale[:, None]).t())
    rounded_first = ((x @ values.t()).float() * scale).to(torch.bfloat16)
    err_route = (got.float() - exact).norm() / exact.norm()
    err_rounded = (rounded_first.float() - exact).norm() / exact.norm()
    assert err_route < err_rounded, (
        f"route {err_route:.3e} is no better than rounding before the scale {err_rounded:.3e}")


@requires_cuda
def test_fused_roles_stack_with_their_own_row_scales(monkeypatch):
    roles = (("gate_proj", 64), ("up_proj", 64))
    _got, layer, _m, _x, (values, scale, _f) = _drive(monkeypatch, MODE_RESIDENT, roles=roles,
                                                      seed=3)
    assert layer.tessera_roles == ("gate_proj", "up_proj")
    assert torch.equal(layer.weight_bf16, values)
    assert torch.equal(layer.row_scale, scale)
    # The two roles were scaled apart by construction; a single shared scale
    # would pass every equality above if the encoder had produced one.
    assert layer.row_scale[:64].mean() != layer.row_scale[64:].mean()


@requires_cuda
def test_a_mixed_rate_schedule_decodes_through_the_group_permutation(monkeypatch):
    _got, layer, _m, _x, (values, _s, _f) = _drive(monkeypatch, MODE_STREAMED, cols=512,
                                                   seed=4, q256=1700)
    assert torch.equal(layer.tessera_prepared.decode(), values)


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    a, *_ = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    b, *_ = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(a, b)


@requires_cuda
def test_streamed_holds_the_packed_wire_and_no_resident_tile(monkeypatch):
    rows, cols = 256, 2048
    _g, layer, _m, _x, _r = _drive(monkeypatch, MODE_STREAMED, roles=(("weight", rows),),
                                   cols=cols, m=2)
    for name in ("wire_bytes", "weight_bf16"):
        assert not hasattr(layer, name), name
    prepared = layer.tessera_prepared
    assert prepared is not None
    # Every term named, none of them a round number someone liked: the tile is
    # 16 bits a weight, the body is the rung's own 7, the table is 2^L bf16
    # words once per unit, and the rest is the per-column gather bookkeeping.
    tile_bytes = rows * cols * 2
    body_bytes = rows * cols * 7 // 8
    table_bytes = (1 << 14) * 2
    resident = prepared.wire_bytes_resident()
    assert resident < tile_bytes, f"streamed holds {resident} of the tile's {tile_bytes}"
    assert resident < body_bytes + table_bytes + tile_bytes // 16, (
        f"streamed holds {resident}, more than the body ({body_bytes}) plus the table "
        f"({table_bytes}) plus a generous allowance for the gather bookkeeping")
    p1, p2 = prepared.decode(), prepared.decode()
    assert torch.equal(p1, p2) and p1.data_ptr() != p2.data_ptr()


@requires_cuda
def test_the_windows_table_is_a_fixed_cost_and_a_small_unit_does_not_pay_for_it():
    """Streaming is not free at every size, and the size it costs at is stated.

    The window table is ``2^L`` bf16 words -- 32 KB at L=14 -- per PREPARED
    UNIT, whatever the unit's shape, while the body saves ``16 - R`` bits a
    weight.  At R=7 that is 1.125 bytes a weight against a 32768-byte constant,
    so the streamed mode holds MORE than the tile below ~29k weights.  A
    64x512 unit is 32768 weights, i.e. right on the crossover, and an early
    version of the test above asserted a win there and got 70912 bytes against
    the tile's 65536.

    Nothing is wrong: real Linears are two orders of magnitude past the
    crossover.  But a footprint claim that is only true above a threshold has
    to say the threshold, or it is the kind of claim that gets quoted at a
    shape it is false at.
    """
    from tessera.serving.window import prepare_window

    torch.manual_seed(0)
    L, R, steps, cols = 14, 7, 8, 16
    body = torch.randint(0, 1 << R, (steps, cols), dtype=torch.uint8)
    table = torch.randn(1 << L, dtype=torch.bfloat16)
    prepared = prepare_window(body, [R] * cols, L, table, "cpu")
    resident = prepared.resident_bytes()
    table_bytes = (1 << L) * 2
    assert resident > table_bytes, "the table is in the footprint"
    assert resident - table_bytes < table_bytes // 20, (
        f"on a {steps}x{cols} unit the table is {table_bytes} of {resident} bytes -- it is the "
        "whole footprint, and the tile it replaces is 256")
    assert resident > steps * cols * 2, (
        "and it is larger than the tile: streaming a unit this small COSTS memory")


@requires_cuda
def test_resident_drops_the_wire(monkeypatch):
    _g, layer, _m, _x, _r = _drive(monkeypatch, MODE_RESIDENT)
    assert not hasattr(layer, "wire_bytes") and layer.tessera_prepared is None
    assert layer.weight_bf16.dtype == torch.bfloat16
    assert tuple(layer.weight_bf16.shape) == (layer.tessera_rows, layer.tessera_columns)


@requires_cuda
def test_streamed_decode_traces_under_torch_compile(monkeypatch):
    _g, layer, _m, _x, (values, _s, _f) = _drive(monkeypatch, MODE_STREAMED, cols=512, seed=9,
                                                 q256=1700)
    compiled = torch.compile(layer.tessera_prepared.decode, fullgraph=True)
    assert torch.equal(compiled(), values)


@requires_cuda
def test_preparation_refuses_a_decoder_that_disagrees_with_the_reference(monkeypatch):
    import tessera.decode as tdecode
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 32)], cols=256)
    roles = parse_tessera_blob_for_scheme(blob, scheme, "t")
    real = tdecode.materialize_bf16

    def _wrong(unit, forest, code):
        v, s = real(unit, forest, code)
        v = v.clone()
        v[0, 0] = v[0, 0] + 1.0
        return v, s

    monkeypatch.setattr(tdecode, "materialize_bf16", _wrong)
    with pytest.raises(RuntimeError, match="disagrees with tessera.decode.materialize_bf16 on 1 of"):
        route.prepare_tessera_bf16_module(roles, device="cuda")


@requires_cuda
def test_scheme_and_blob_must_agree(monkeypatch):
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 32)], cols=256)
    parse_tessera_blob_for_scheme(blob, scheme, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "q256": 1024}, "t")


@requires_cuda
def test_route_record_names_the_family_mode_contract_and_decoder(monkeypatch):
    from tessera.serving.telemetry import read_route
    _g, layer, _m, _x, _r = _drive(monkeypatch, MODE_STREAMED)
    rec = read_route(layer)
    assert rec is not None and rec["policy"] == f"{TESSERA_BF16}:streamed"
    assert rec["state"] == "served"
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert rec["symbol"] == "torch.mm"
    assert rec["decoder"] == telemetry.DECODER_TORCH_WINDOW == layer.tessera_decoder


@requires_cuda
def test_the_route_reaches_for_no_activation_quantiser(monkeypatch):
    """W16A16 means there is nothing to quantise, and the code must show it.

    ``native_ops`` is where the other routes' A-side quantiser and its ABI
    attestation live; a BF16 route that touched it would either be quantising
    an activation it declares unquantised, or attesting an ABI it never calls.
    """
    from tessera.serving import native_ops

    calls = []
    monkeypatch.setattr(native_ops, "native_fp8_quant",
                        lambda *a, **k: calls.append("quant"))
    monkeypatch.setattr(native_ops, "require_native_fp8_quant",
                        lambda *a, **k: calls.append("attest"))
    _drive(monkeypatch, MODE_RESIDENT)
    assert calls == []


@requires_cuda
def test_a_non_channel_plane_is_refused_at_preparation(monkeypatch):
    """The route applies ONE factor per output row; any other plane has none."""
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.manifest import ScalePlaneKind
    blob, scheme, *_ = _encode_module([("weight", 32)], cols=256)
    roles = parse_tessera_blob_for_scheme(blob, scheme, "t")
    object.__setattr__(roles[0][1].unit, "scale_plane", ScalePlaneKind.S6B)
    with pytest.raises(ValueError, match="CHANNEL plane"):
        route.prepare_tessera_bf16_module(roles, device="cuda")
