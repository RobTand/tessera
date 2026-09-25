"""The Tessera 16-bit W16A16 dense serving route.

Exercised for real on a CUDA box: the container parse, the packed-window
decode, the reference-decoder cross-check at preparation, the folded row
scale, both residency modes, the compiled decode and the refusals.

**The load-bearing assertion is that the route serves the folded tile**
(tessera#606, #614): each weight is ``bf16(value * row_scale)``, rounded once,
before the dot, and there is no scale on the output -- the tile
``materialize_bf16_folded`` renders, and the arithmetic the routed BF16 stack
serves.  The reason is pricing identity: a consumer that prices a BF16 rung
prices that tile, so the served function must be that one.  The fold costs one
bf16 rounding (~0.0011-0.0022 absolute on GLM expert rows at any rate), and
served at R = 7 the folded twin's KL was 1.0011x the epilogue route's on
``all`` and 0.9961x on ``confident``, below what the corpus resolves (#45), so
neither arithmetic is claimed to win on quality.  Two tests hold the line: the
served output is the folded product (and measurably not the epilogue one), and
the RETAINED reference preparation still returns ``materialize_bf16``'s
unfolded pair, which is what the fold is taken of.

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


def _selected_bf16_module(expert, names=('gate', 'up'), rate=2):
    """Small prepared windows with distinct BF16 tables and row scales."""
    from tessera.serving.window import prepare_window

    generator = torch.Generator().manual_seed(1201 + expert)
    roles = []
    for position, name in enumerate(names):
        table = (torch.arange(256, dtype=torch.float32) + expert + position).to(torch.bfloat16)
        window = prepare_window(
            torch.randint(0, 1 << rate, (16, 8), generator=generator, dtype=torch.uint8),
            [rate] * 8, 8, table, 'cpu')
        roles.append(route._Bf16Role(name, position * 16, 16, window))
    return route.PreparedTesseraBf16Module(
        roles, rows=len(names) * 16, columns=8,
        scale=torch.arange(len(names) * 16, dtype=torch.float32) + expert + 1,
        device=torch.device('cpu'))


def test_selected_bf16_windows_preserve_values_scales_and_global_expert_ids():
    modules = [_selected_bf16_module(expert) for expert in range(4)]
    batch = route.PreparedTesseraBf16Module.stack(modules)
    ids = torch.tensor([3, 0, 2, 3], dtype=torch.int32)
    expected_values = torch.stack([m.decode() for m in modules]).index_select(0, ids.long())
    expected_scale = torch.stack([m.row_scale() for m in modules]).index_select(0, ids.long())
    assert torch.equal(batch.decode(ids, max_experts_per_chunk=2), expected_values)
    assert torch.equal(batch.row_scale(ids), expected_scale)
    assert torch.equal(batch.decode_folded(ids, max_experts_per_chunk=2),
                       (expected_values.float() * expected_scale[:, :, None]).to(torch.bfloat16))
    assert batch.decode(ids[:0], max_experts_per_chunk=2).shape == (0, 32, 8)
    assert batch.resident_bytes() == batch.wire_bytes_resident() + 4 * 32 * 4
    assert batch.decode(ids, max_experts_per_chunk=1).data_ptr() != batch.decode(
        ids, max_experts_per_chunk=4).data_ptr()
    modules[0]._PreparedTesseraBf16Module__scale.zero_()
    assert batch.row_scale(torch.tensor([0]))[0, 0].item() == 1


def test_selected_bf16_windows_refuse_incompatible_layouts():
    with pytest.raises(ValueError, match='at least one'):
        route.PreparedTesseraBf16Module.stack([])
    with pytest.raises(ValueError, match='roles'):
        route.PreparedTesseraBf16Module.stack([
            _selected_bf16_module(0), _selected_bf16_module(1, ('up', 'gate'))])
    with pytest.raises(ValueError, match='layout'):
        route.PreparedTesseraBf16Module.stack([
            _selected_bf16_module(0), _selected_bf16_module(1, rate=3)])


def test_selected_bf16_fold_bounds_raw_and_fp32_temporaries_by_chunk(monkeypatch):
    batch = route.PreparedTesseraBf16Module.stack([
        _selected_bf16_module(expert) for expert in range(4)])
    real_decode, real_scale = batch.decode, batch.row_scale
    seen = []

    def bounded_decode(ids, **kwargs):
        seen.append(int(ids.numel()))
        assert ids.numel() <= 2, "fold decoded an unbounded raw selected stack"
        return real_decode(ids, **kwargs)

    def bounded_scale(ids):
        assert ids.numel() <= 2, "fold expanded unbounded fp32 row scales"
        return real_scale(ids)

    monkeypatch.setattr(batch, "decode", bounded_decode)
    monkeypatch.setattr(batch, "row_scale", bounded_scale)
    ids = torch.tensor([3, 0, 1, 2, 3], dtype=torch.int32)
    actual = batch.decode_folded(ids, max_experts_per_chunk=2)
    expected = torch.stack([
        (module.decode().float() * module.row_scale()[:, None]).to(torch.bfloat16)
        for module in (_selected_bf16_module(expert) for expert in range(4))
    ]).index_select(0, ids.long())
    assert torch.equal(actual, expected)
    assert seen == [2, 2, 1]


def test_selected_bf16_folded_tile_matches_the_joint_screens_wire_reader():
    fused, export, _decode, alphabet = _tessera()
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.unit_artifact import read_unit_artifact

    modules, rendered = [], []
    for expert in range(2):
        weight = torch.randn(32, 64, generator=torch.Generator().manual_seed(expert + 51))
        written, _unit, _forests = export.encode_linear_planes(
            weight, grid=alphabet.BF16_GRID, q256=512, name='weight',
            window_bits=8, verify=False)
        blob = fused.pack_fused([('weight', 32, written.blob)])
        scheme = _scheme(rows=32, columns=64, roles=[['weight', 32]],
                         q256=512, wire_bytes=len(blob))
        parsed = parse_tessera_blob_for_scheme(blob, scheme, f'expert {expert}')
        modules.append(route.prepare_tessera_bf16_module(parsed, device='cpu'))
        rendered.append(read_unit_artifact(written.blob).to(torch.bfloat16))
    ids = torch.tensor([1, 0, 1], dtype=torch.int32)
    selected = route.PreparedTesseraBf16Module.stack(modules).decode_folded(
        ids, max_experts_per_chunk=2)
    assert torch.equal(selected, torch.stack(rendered).index_select(0, ids.long()))


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
    """The RETAINED reference tile is ``materialize_bf16``'s values (never the
    fold), with the same fp32 row scale beside it that the served GEMM folds
    into each weight.

    The route no longer materialises a tile -- ``native_window`` holds packed
    bundles -- so the byte-for-byte claim is made where it still has a subject,
    the retained ``prepare_tessera_bf16_module`` preparation, which stays as
    the oracle.
    """
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    _got, layer, _m, _x, (values, scale, folded) = _drive(monkeypatch, mode)
    blob, scheme, *_ = _encode_module([("weight", 64)], cols=512)
    reference = route.prepare_tessera_bf16_module(
        parse_tessera_blob_for_scheme(blob, scheme, "t"), device="cuda")
    tile = reference.decode()
    assert tile.dtype == torch.bfloat16
    assert torch.equal(tile, values)
    assert torch.equal(reference.row_scale(), scale)
    assert torch.equal(layer.row_scale, scale)
    assert layer.row_scale.dtype == torch.float32
    assert tuple(layer.row_scale.shape) == (values.shape[0],)
    assert layer.tessera_native is not None
    assert not hasattr(layer, "weight_bf16") and not hasattr(layer, "tessera_prepared")
    # And the reference pair is NOT the folded twin: the fold is taken once,
    # in the served kernel, of exactly this pair -- a reference that had
    # already folded would be folded twice there, and every other assertion
    # here would still pass.
    assert not torch.equal(tile, folded), "the tile has the row scale folded into it"


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_the_route_serves_the_folded_tile_and_not_the_epilogue(monkeypatch, mode):
    """The served output IS ``x @ materialize_bf16_folded(...)^T``, and is not
    the epilogue arithmetic (tessera#614).

    ``folded`` is ``decode.materialize_bf16_folded``'s tile, so the reference is
    the one definition of the fold in the tree.  Both references are the
    arithmetic's own answer, rounded to bf16 once, as the served GEMM rounds:
    the products are exact in fp32, so the served output differs from its own
    arithmetic's reference only by fp32 summation order -- at most one bf16
    ulp, and in most elements not at all.  Compared against unrounded fp32
    products instead, the output's own bf16 rounding (unit roundoff 2^-8)
    swamps the fold's per-weight rounding and neither ordering can be read.
    On a build that serves the epilogue the ordering reverses, which is what
    makes this bite.
    """
    got, layer, _m, x, (values, scale, folded) = _drive(monkeypatch, mode, m=8)
    folded_ref = (x.float() @ folded.float().t()).bfloat16()
    epilogue_ref = ((x.float() @ values.float().t()) * scale).bfloat16()
    # The two arithmetics really are different functions of these bytes.
    assert not torch.equal(folded_ref, epilogue_ref)
    gap = (got.float() - folded_ref.float()).abs()
    bound = folded_ref.float().abs() * 2 ** -7 + 1e-4 * float(folded_ref.float().abs().max())
    assert bool((gap <= bound).all()), (
        f"served output is up to {float((gap - bound).max()):.3e} past one bf16 ulp of the "
        "folded product")
    err_folded = _rel(got, folded_ref)
    err_epilogue = _rel(got, epilogue_ref)
    assert err_folded < err_epilogue, (
        f"served output is closer to the epilogue product ({err_epilogue:.3e}) than to the "
        f"folded one ({err_folded:.3e}): the route is not serving the fold")
    assert layer.tessera_native.arithmetic == "folded"


@requires_cuda
def test_fused_roles_stack_with_their_own_row_scales(monkeypatch):
    roles = (("gate_proj", 64), ("up_proj", 64))
    _got, layer, _m, _x, (values, scale, _f) = _drive(monkeypatch, MODE_RESIDENT, roles=roles,
                                                      seed=3)
    assert layer.tessera_roles == ("gate_proj", "up_proj")
    assert layer.tessera_native is not None
    assert torch.equal(layer.row_scale, scale)
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    blob, scheme, *_ = _encode_module(list(roles), cols=512, seed=3)
    reference = route.prepare_tessera_bf16_module(
        parse_tessera_blob_for_scheme(blob, scheme, "t"), device="cuda")
    assert torch.equal(reference.decode(), values)
    # The two roles were scaled apart by construction; a single shared scale
    # would pass every equality above if the encoder had produced one.
    assert layer.row_scale[:64].mean() != layer.row_scale[64:].mean()


@requires_cuda
def test_a_mixed_rate_schedule_decodes_through_the_group_permutation(monkeypatch):
    _got, layer, _m, _x, (values, _s, _f) = _drive(monkeypatch, MODE_STREAMED, cols=512,
                                                   seed=4, q256=1700)
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    blob, scheme, *_ = _encode_module([("weight", 64)], cols=512, seed=4, q256=1700)
    reference = route.prepare_tessera_bf16_module(
        parse_tessera_blob_for_scheme(blob, scheme, "t"), device="cuda")
    assert torch.equal(reference.decode(), values)
    assert len(set(layer.tessera_native.layout_facts()[0].rates)) > 1, "not a mixed schedule"


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
    for name in ("wire_bytes", "weight_bf16", "tessera_prepared", "tessera_gemv"):
        assert not hasattr(layer, name), name
    native = layer.tessera_native
    assert native is not None
    # Every term named, none of them a round number someone liked: the tile is
    # 16 bits a weight, the body is the rung's own 7, the table is 2^L bf16
    # words once per unit, and the rest is the per-column gather bookkeeping.
    tile_bytes = rows * cols * 2
    # The repack pads rows to the 512-row tile, so the body is priced over
    # ``rep.rows_p`` (that padding IS the layout, not an accident).
    facts = native.layout_facts()[0]
    padded_rows = int(facts.rows_p)
    body_bytes = padded_rows * cols * 7 // 8
    table_bytes = (1 << 14) * 2
    resident = native.packed_bytes()
    assert resident < tile_bytes, f"streamed holds {resident} of the tile's {tile_bytes}"
    assert resident < body_bytes + table_bytes + cols * 32 + 65536, (
        f"streamed holds {resident}, more than the padded body ({body_bytes}) plus the table "
        f"({table_bytes}) plus a generous allowance for the gather bookkeeping")
    fingerprints = native.fingerprints()
    x = torch.randn(4, cols, dtype=torch.bfloat16, device="cuda")
    p1, p2 = native.apply(x), native.apply(x)
    assert torch.equal(p1, p2) and p1.data_ptr() != p2.data_ptr()
    assert native.fingerprints() == fingerprints, "a forward changed the prepared weights"


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
    assert not hasattr(layer, "wire_bytes") and not hasattr(layer, "weight_bf16")
    native = layer.tessera_native
    assert native is not None
    # Resident mode holds the packed repack, not the 16-bit tile.  The layout
    # pads rows to the 512-row tile, so the footprint is priced over
    # ``rep.rows_p`` and no decoded ``[rows, columns]`` tensor appears at all.
    facts = native.layout_facts()[0]
    cap = int(facts.rows_p) * facts.cols * 2 + (1 << 14) * 2 + facts.cols * 32
    assert native.packed_bytes() <= cap, (native.packed_bytes(), cap)


@requires_cuda
def test_streamed_decode_traces_under_torch_compile(monkeypatch):
    _g, layer, _m, _x, (values, _s, _f) = _drive(monkeypatch, MODE_STREAMED, cols=512, seed=9,
                                                 q256=1700)
    native = layer.tessera_native
    assert native is not None
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    eager = native.apply(x)
    compiled = torch.compile(lambda a: native.apply(a), fullgraph=True)(x)
    assert torch.equal(compiled, eager)


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
    from tessera.serving.scheme import parse_compact_blob_for_scheme, parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 32)], cols=256)
    parse_tessera_blob_for_scheme(blob, scheme, "t")
    parse_compact_blob_for_scheme(blob, scheme, "t", device="cuda")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "q256": 1024}, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_compact_blob_for_scheme(blob, {**scheme, "q256": 1024}, "t", device="cuda")


@requires_cuda
def test_route_record_names_the_family_mode_contract_and_decoder(monkeypatch):
    from tessera.serving.telemetry import read_route
    _g, layer, _m, _x, _r = _drive(monkeypatch, MODE_STREAMED)
    from tessera.serving.scheme import WINDOW_GEMM_SYMBOL

    rec = read_route(layer)
    assert rec is not None and rec["policy"] == f"{TESSERA_BF16}:streamed"
    assert rec["state"] == "served"
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert rec["symbol"] == WINDOW_GEMM_SYMBOL == "tessera::window_gemm_dense"
    assert rec["decoder"] == telemetry.DECODER_NATIVE_WINDOW_GEMM_FOLDED == layer.tessera_decoder
    assert (rec["symbol"], rec["decoder"]) == route.DENSE_LAUNCH
    assert rec["decoder"] in telemetry.DECODERS


@requires_cuda
def test_a_bundle_on_another_arithmetic_is_refused_at_load(monkeypatch):
    """``apply`` stamps ``DENSE_LAUNCH``; a prepared module on the epilogue
    arithmetic would serve one function and record another, so loading refuses
    it by name rather than emitting a record the kernel did not earn."""
    from tessera.serving import native_window

    monkeypatch.setitem(native_window.NATIVE_WINDOW_ARITHMETIC, TESSERA_BF16, "epilogue")
    with pytest.raises(RuntimeError, match="native_window_gemm_folded"):
        _drive(monkeypatch, MODE_RESIDENT)


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
