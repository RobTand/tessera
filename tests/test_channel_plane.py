"""Schema minor 3: the per-channel scale plane, and the wire recipe.

The 8-bit headline (``docs/measurements/tessera-window-body-2026-09-02.md``)
was measured under one scale per output row -- the layout the FP8 tensor
core consumes -- and the wire could not spell it.  Now it can, with elements
it already had: ``ScalePlaneKind.CHANNEL`` puts the row scale on the DIAG_SV
plane (one fp16 per row) over the unit's fp32 global, with SCALE_BASE,
SCALE_REFINE and DIAG_SU absent.  These tests hold the seam to what the
measurement relied on and to what the earlier minors promised:

  * the seam round-trips on both tiles and both bodies, the header says
    minor 3, the accountant agrees with the bytes to the bit, and the reader
    needs nothing but bytes;
  * an E4M3 unit materialises into the stock per-channel FP8 pair, exactly;
  * the refit is monotone; the profile id binds the plane kind; the reader
    fails closed on a manifest whose plane counts disagree with the kind and
    on a header too old to name the plane;
  * every S6b/LUT artifact keeps its bytes and its minor;
  * the window body may spend the grid's whole width (its shaping is not a
    code bit), and the exporter's defaults are ``wire_recipe``'s.
"""
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import E2M1_GRID, E4M3_GRID, build_forest, tuple_grid
from tessera.calculator import terminal_rate
from tessera.container import parse, serialize
from tessera.decode import materialize_fp8, reconstruct_unit, unit_scale_field
from tessera.encode import encode_unit
from tessera.errors import GrammarError, ManifestError, TesseraError
from tessera.export import (
    TCQ_RECIPE,
    WireRecipe,
    _plan_for,
    encode_linear,
    encode_settings_from_config,
    export_checkpoint,
    read_checkpoint_config,
    wire_recipe,
)
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import BodyKind, Manifest, ScalePlane, ScalePlaneKind
from tessera.planes import PlaneKind, PlaneLayout
from tessera.scale_channel import default_channel_sigma, initial_channel_scale, refit_channel_scale
from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, encoder_profile_id, read_unit_artifact

CODE = ConvCode(memory=6)
K2 = tuple_grid(E2M1_GRID, 2)
CHANNEL = ScalePlaneKind.CHANNEL
WINDOW = BodyKind.WINDOW


def _weights(rows=64, cols=512, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, cols) * 0.02


def _elements(blob: bytes, kind: PlaneKind) -> int:
    art = parse(blob)
    return art.terminal.plane_elements[art.manifest.plane_order.index(kind)]


def _forests(grid, rates):
    from tessera.alphabet import GAUSSIAN_SOURCE

    sigma = default_channel_sigma(grid)
    return {r: build_forest(r, samples=GAUSSIAN_SOURCE(1 << 14, sigma), grid=grid)
            for r in sorted(set(rates))}


# ------------------------------------------------------------- round trip


@pytest.mark.parametrize(
    "grid,q256,body,window",
    [
        (E4M3_GRID, 4 * 256, WINDOW, 8),
        (E4M3_GRID, int(4.5 * 256), BodyKind.TCQ, 0),
        (K2, 7 * 256, WINDOW, 10),
        (K2, int(3.5 * 256), BodyKind.TCQ, 0),
        (E2M1_GRID, 3 * 256, BodyKind.TCQ, 0),
        (E2M1_GRID, 2 * 256, WINDOW, 6),
    ],
)
def test_wire_round_trip_of_a_channel_plane(grid, q256, body, window):
    w = _weights()
    rows, cols = w.shape
    cap = grid.payload_bits if body is WINDOW else grid.rate_cap
    rates = bresenham_rate_schedule(root_from_q256(q256), cols, cap=cap)
    forest = grid if body is WINDOW else _forests(grid, rates)
    unit = encode_unit(w, forest, rates, CODE, body=body, window_bits=window,
                       scale_plane=CHANNEL, scale_refit=1, completion=0)
    assert unit.scale_rows is not None and unit.scale_rows.dtype == torch.float16
    assert unit.scale_rows.numel() == rows
    assert unit.scale_base.numel() == 0 and unit.scale_refine.numel() == 0
    # This assertion prices the CHANNEL record, not the encoder-identity
    # envelope.  Ask for the born-against spelling explicitly.
    manifest, region, blob = build_unit_artifact(
        unit, "unit0", forest, q256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    recovered = read_unit_artifact(blob)
    assert torch.equal(recovered, reconstruct_unit(unit, forest, CODE))
    assert blob[10] == 3, "schema minor 3"
    art = parse(blob)
    assert art.manifest.scale_plane.kind is CHANNEL
    assert art.manifest.scale_plane.table == b""
    assert float(art.manifest.scale_plane.global_scale) == unit.scale_global
    assert _elements(blob, PlaneKind.SCALE_BASE) == 0
    assert _elements(blob, PlaneKind.SCALE_REFINE) == 0
    assert _elements(blob, PlaneKind.DIAG_SU) == 0
    assert _elements(blob, PlaneKind.DIAG_SV) == rows
    # the decoded weight is value(code) * global * sv[row], nothing else
    expected_scale = unit.scale_rows.float() * unit.scale_global
    field = unit_scale_field(unit, rows, cols)
    assert torch.equal(field[:, 0], expected_scale)
    # the accountant prices the plane exactly as the wire charges it
    predicted = terminal_rate(
        q256, rows, cols, with_scale_base=False, with_scale_refine=False,
        with_row_scale=True, cap=cap, arity=grid.arity,
        window_bits=window if body is WINDOW else 0,
    )
    # ``terminal_rate`` sizes the window table but not a TCQ forest's blob
    # planes (per-unit ALPHABET/DESCENDANT bytes, negligible at real widths).
    table_bits = (8 << window) if body is WINDOW else 8 * (
        _elements(blob, PlaneKind.ALPHABET) + _elements(blob, PlaneKind.DESCENDANT))
    forest_bits = 0 if body is WINDOW else table_bits
    assert predicted + Fraction(forest_bits, w.numel()) == art.terminal.exact_bpp
    body_bits = sum(rates) * (rows // grid.arity)
    completion_bits = _elements(blob, PlaneKind.COMPLETION)
    assert art.terminal.exact_bpp == Fraction(
        table_bits + body_bits + completion_bits + 16 * rows, w.numel()
    )


def test_the_refit_is_monotone_and_the_rows_land_on_stored_words():
    w = _weights(rows=32, cols=256)
    rates = (4,) * 256
    errs = []
    for refit in (0, 1, 3):
        unit = encode_unit(w, E4M3_GRID, rates, CODE, body=WINDOW, window_bits=8,
                           scale_plane=CHANNEL, scale_refit=refit)
        hat = reconstruct_unit(unit, E4M3_GRID, None)
        errs.append(float((hat - w).norm()))
    assert errs[1] <= errs[0] and errs[2] <= errs[1]
    # one least-squares step on random codes never worsens a row
    units = torch.randn(32, 256)
    stored = torch.full((32,), 1.0, dtype=torch.float16)
    before = ((w - units * (stored.float() * 0.5).unsqueeze(1)) ** 2).sum(1)
    new_stored, new_eff = refit_channel_scale(w, units, stored, 0.5)
    after = ((w - units * new_eff.unsqueeze(1)) ** 2).sum(1)
    assert torch.all(after <= before + 1e-9)
    assert torch.equal(new_eff, new_stored.float() * 0.5)


def test_materialize_fp8_is_the_stock_per_channel_tensor():
    w = _weights(rows=32, cols=256, seed=3)
    rates = (4,) * 256
    unit = encode_unit(w, E4M3_GRID, rates, CODE, body=WINDOW, window_bits=8,
                       scale_plane=CHANNEL, scale_refit=2)
    raw, scale = materialize_fp8(unit, E4M3_GRID, None)
    assert raw.dtype == torch.uint8 and raw.shape == (32, 256)
    assert scale.dtype == torch.float32 and scale.shape == (32,)
    assert not any(int(b) in (0x7F, 0xFF) for b in raw.flatten().tolist())
    served = raw.view(torch.float8_e4m3fn).float() * scale.unsqueeze(1)
    assert torch.equal(served, reconstruct_unit(unit, E4M3_GRID, None))
    with pytest.raises(GrammarError, match="per output row"):
        lut = encode_unit(w, E4M3_GRID, rates, CODE, body=WINDOW, window_bits=8,
                          scale_plane=ScalePlaneKind.LUT)
        materialize_fp8(lut, E4M3_GRID, None)


# ------------------------------------------------------ identity & refusals


def test_the_profile_id_binds_the_plane_kind():
    rates = (4,) * 512
    ids = {
        kind: encoder_profile_id(None, rates, E4M3_GRID, 1, kind, WINDOW, 8)
        for kind in ScalePlaneKind
    }
    assert len(set(ids.values())) == 3


def test_a_channel_plane_refuses_segment_2a_and_block_words():
    w = _weights()
    with pytest.raises(GrammarError, match="DIAG_SV"):
        encode_unit(w, E4M3_GRID, (4,) * 512, CODE, body=WINDOW, window_bits=8,
                    scale_plane=CHANNEL, with_diagonals=True)
    unit = encode_unit(w, E4M3_GRID, (4,) * 512, CODE, body=WINDOW, window_bits=8,
                       scale_plane=CHANNEL)
    unit.scale_refine = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(GrammarError, match="block-scale"):
        build_unit_artifact(unit, "unit0", E4M3_GRID, 4 * 256, CODE)


def test_the_reader_fails_closed_on_planes_that_disagree_with_the_kind():
    w = _weights()
    unit = encode_unit(w, E4M3_GRID, (4,) * 512, CODE, body=WINDOW, window_bits=8,
                       scale_plane=CHANNEL)
    manifest, region, blob = build_unit_artifact(
        unit, "unit0", E4M3_GRID, 4 * 256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    # a LUT unit's bytes relabelled as CHANNEL: the terminal declares a
    # refinement plane a CHANNEL plane cannot have
    lut = encode_unit(w, E4M3_GRID, (4,) * 512, CODE, body=WINDOW, window_bits=8,
                      scale_plane=ScalePlaneKind.LUT)
    lm, lregion, _ = build_unit_artifact(lut, "unit0", E4M3_GRID, 4 * 256, CODE)
    forged = Manifest(
        encoder_profile_id=encoder_profile_id(None, lm.rates, E4M3_GRID, 1, CHANNEL, WINDOW, 8),
        branch=lm.branch, geometry=lm.geometry, arrangement=lm.arrangement,
        rates=lm.rates, planes=lm.planes, terminals=lm.terminals,
        payload_digest=lm.payload_digest, span=1,
        scale_plane=ScalePlane.channel(lut.scale_global), body=WINDOW, window_bits=8,
    )
    with pytest.raises(GrammarError, match="block-scale"):
        read_unit_artifact(serialize(forged, lregion))
    # a header too old to name the plane
    stale = bytearray(blob)
    stale[10] = 2
    with pytest.raises(TesseraError):
        read_unit_artifact(bytes(stale))
    with pytest.raises(ManifestError, match="minor 3"):
        manifest.encode(2)


def test_the_span2_lane_refuses_a_channel_plane():
    pytest.importorskip("triton")
    from tessera.kernel import pack_unit_for_kernel

    w = _weights()
    forest = _forests(K2, (7,))
    unit = encode_unit(w, forest, (7,) * 512, CODE, span=2, scale_plane=CHANNEL,
                       completion=0)
    with pytest.raises(TesseraError):
        pack_unit_for_kernel(unit, forest[7], CODE)


@pytest.mark.parametrize("grid,rate,window", [(E4M3_GRID, 4, 8), (K2, 7, 8), (E4M3_GRID, 5, 10),
                                              (E4M3_GRID, 4, 14), (E4M3_GRID, 8, 14)],
                         ids=["e4m3-r4-L8", "k2-r7-L8", "e4m3-r5-L10",
                              "e4m3-r4-L14-the-default", "e4m3-r8-L14-the-ceiling"])
def test_the_window_lane_decodes_a_channel_plane(grid, rate, window):
    """The window GEMV under a CHANNEL plane: the reader's bytes, bit for bit.

    One-hot columns compare the kernel's decode against ``read_unit_artifact``
    with ``torch.equal`` (the row scale is an epilogue in the reader's own
    fp32 expression); a random vector holds the lane's ``rel < 1e-5``.
    """
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the kernel lane runs on CUDA")
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    rows, cols = 256, 512
    w = _weights(rows, cols, seed=3).to("cuda")
    w[: rows // 8] *= 4.0
    rates = (rate,) * cols
    unit = encode_unit(w, grid, rates, CODE, body=WINDOW, window_bits=window,
                       scale_plane=CHANNEL, scale_refit=1, completion=0)
    _manifest, _region, blob = build_unit_artifact(
        unit, "unit0", grid, rate * 256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    assert blob[10] == 3
    reference = read_unit_artifact(blob, device="cuda")
    packed = pack_unit_for_kernel(unit, grid, CODE)
    assert packed["kind"] == "window" and packed["row_scale"] is not None
    assert packed["global_scale"] == 1.0 and packed["scale_table"] is None
    for k in (0, 1, 15, 16, 17, cols // 2, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = gemv_from_packed(x, packed, lanes=8, split_k=4)
        assert torch.equal(got, reference[:, k]), f"column {k}"
    x = torch.randn(cols, device="cuda")
    got = gemv_from_packed(x, packed, lanes=8, split_k=4)
    expect = reference @ x
    assert (got - expect).norm() / expect.norm() < 1e-5


# ----------------------------------------------------- what did not change


def test_block_plane_artifacts_keep_their_bytes_and_their_minor():
    w = _weights()
    forests = {7: build_forest(7, grid=K2)}
    tcq = encode_unit(w, forests, (7,) * 512, CODE, span=2, scale_plane=ScalePlaneKind.LUT)
    m, _, blob = build_unit_artifact(
        tcq, "unit0", forests, 7 * 256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    assert blob[10] == 1 and m.schema_minor == 1 and tcq.scale_rows is None
    win = encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=9,
                      scale_plane=ScalePlaneKind.LUT)
    m, _, blob = build_unit_artifact(
        win, "unit0", K2, 7 * 256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    assert blob[10] == 2 and m.schema_minor == 2
    legacy = encode_unit(w, forests, (7,) * 512, CODE)
    m, _, blob = build_unit_artifact(
        legacy, "unit0", forests, 7 * 256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    assert blob[10] == 0 and m.schema_minor == 0
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(legacy, forests, CODE))


# ------------------------------------------------ the window body's cap


def test_a_window_body_may_spend_the_grids_whole_width():
    w = _weights()
    rates, forest = _plan_for(K2, 4 * 256, 512, WINDOW)
    assert set(rates) == {8} and forest is K2
    unit = encode_unit(w, K2, rates, CODE, body=WINDOW, window_bits=12,
                       scale_plane=ScalePlaneKind.LUT)
    # the artifact declares the per-CODE rate: 8 bits per pair
    _, _, blob = build_unit_artifact(unit, "unit0", K2, 8 * 256, CODE)
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, K2, None))
    with pytest.raises(GrammarError):
        _plan_for(K2, 4 * 256, 512, BodyKind.TCQ)      # the TCQ cap is one bit lower


# --------------------------------------------------------- the recipe


def test_the_exporter_writes_the_recipe_and_replays_it(tmp_path):
    from tessera.export import E2M1X2_SUBCAP_RECIPE, E4M3_RECIPE, tcq_cap_q256

    # E2M1 keeps the coset trellis; E2M1x2 is the window below its cap and
    # the coset trellis at it; E4M3 is the window over CHANNEL at every rung.
    assert wire_recipe(E2M1_GRID) == wire_recipe(E2M1_GRID, 512) == TCQ_RECIPE
    assert tcq_cap_q256(K2) == 896
    assert wire_recipe(K2) == wire_recipe(K2, 896) == TCQ_RECIPE
    assert wire_recipe(K2, 895) == wire_recipe(K2, 256) == E2M1X2_SUBCAP_RECIPE
    assert E2M1X2_SUBCAP_RECIPE.body is WINDOW and E2M1X2_SUBCAP_RECIPE.scale_plane is ScalePlaneKind.LUT
    assert wire_recipe(E4M3_GRID) == wire_recipe(E4M3_GRID, 1024) == E4M3_RECIPE
    assert E4M3_RECIPE.body is WINDOW and E4M3_RECIPE.scale_plane is CHANNEL
    assert E4M3_RECIPE.window_bits == 14 and E4M3_RECIPE.span == 1
    assert TCQ_RECIPE.body is BodyKind.TCQ and TCQ_RECIPE.span == 2
    assert TCQ_RECIPE.scale_plane is ScalePlaneKind.LUT
    with pytest.raises(GrammarError):
        WireRecipe(WINDOW, 1, ScalePlaneKind.LUT)            # no width
    with pytest.raises(GrammarError):
        WireRecipe(WINDOW, 2, ScalePlaneKind.LUT, window_bits=8)
    w = _weights()
    unit = encode_linear(w, grid=E4M3_GRID, q256=4 * 256, scale_refit=0)
    art = parse(unit.blob).manifest
    assert art.body is E4M3_RECIPE.body and art.span == E4M3_RECIPE.span
    assert art.scale_plane.kind is E4M3_RECIPE.scale_plane
    assert art.window_bits == E4M3_RECIPE.window_bits
    unit = encode_linear(w, grid=K2, q256=896, scale_refit=0)
    art = parse(unit.blob).manifest
    assert art.body is TCQ_RECIPE.body and art.span == TCQ_RECIPE.span

    plan = {"a": 4 * 256}
    export_checkpoint({"a": w}, plan, tmp_path, grid=E4M3_GRID, scale_refit=1,
                      body=WINDOW, window_bits=8, scale_plane=CHANNEL)
    config = read_checkpoint_config(tmp_path)
    assert config["scale"]["plane"] == "channel"
    assert config["scale"]["sigma"] == pytest.approx(default_channel_sigma(E4M3_GRID))
    settings = encode_settings_from_config(config)
    assert settings["scale_plane"] is CHANNEL and settings["body"] is WINDOW
    assert settings["channel_sigma"] == pytest.approx(default_channel_sigma(E4M3_GRID))
    replay = encode_linear(w, grid=E4M3_GRID, q256=4 * 256, verify=False, **settings)
    from tessera.export import load_tessera_weight

    assert torch.equal(load_tessera_weight(tmp_path, "a"), read_unit_artifact(replay.blob))
    # a config written before the plane existed still means what it meant
    old = encode_settings_from_config({"scale": {"plane": "lut16"}})
    assert old["scale_plane"] is ScalePlaneKind.LUT and old["channel_sigma"] is None


def test_the_initial_plane_keeps_every_row_inside_the_reach():
    """A row whose largest weight would land past the body's reach starts at
    the sigma that puts it exactly on the reach; rows inside it are the plain
    RMS start byte for byte (``tessera-dense-outlier-mechanism``)."""
    torch.manual_seed(3)
    work = torch.randn(6, 512)
    work[1, 7] = 30.0 * work[1].pow(2).mean().sqrt()       # a 30-sigma weight
    work[4, 100] = 9.0 * work[4].pow(2).mean().sqrt()      # a 9-sigma weight
    sigma, reach = 94.2, 384.0
    plain_stored, plain_eff, plain_global = initial_channel_scale(work, sigma)
    stored, eff, global_scale = initial_channel_scale(work, sigma, reach=reach)
    units = work.abs().amax(dim=1) / eff
    assert bool((units <= reach * (1 + 2.0 ** -10)).all()), units.tolist()
    rms = work.pow(2).mean(dim=1).sqrt()
    inside = (work.abs().amax(dim=1) * sigma <= reach * rms)
    assert inside.tolist() == [True, False, True, True, False, True]
    # Inside the reach the effective scale is the plain start, whatever the
    # global landed on; past it the row's largest weight lands on the reach.
    assert torch.allclose(eff[inside], plain_eff[inside], rtol=1e-6, atol=0)
    for r in (1, 4):
        assert abs(float(units[r]) - reach) / reach < 2.0 ** -9
    plain_units = work.abs().amax(dim=1) / plain_eff
    assert float(plain_units[1]) > reach and float(plain_units[4]) > reach
    assert initial_channel_scale(work, sigma, reach=None)[1].equal(plain_eff)
    with pytest.raises(GrammarError):
        initial_channel_scale(work, sigma, reach=0.0)


def test_a_raised_row_lands_at_or_inside_the_reach():
    """Issue #87: the reach start's lower bound is held exactly.

    ``initial_channel_scale`` raises a row whose loudest weight would land past
    the body's reach to ``scale = amax / reach``.  It first casts that scale to
    the nearest fp16 word, then steps upward if the word is below the bound;
    landing below it clips the very weight the raise exists for.  The source is
    heavy-tailed on purpose: a light-tailed fixture raises nothing and therefore
    asserts nothing.  There is no tolerance here because one fp16 ulp is the
    defect this test exists to catch.
    """
    from tessera.encode import grid_vector_table, window_table

    torch.manual_seed(87)
    rows, cols = 1024, 256
    w = torch.randn(rows, cols)
    w[torch.arange(rows), torch.randint(cols, (rows,))] *= torch.linspace(1.0, 40.0, rows)
    sigma = default_channel_sigma(E4M3_GRID)
    table = window_table(E4M3_GRID, 14, sigma=sigma, seed=0, half=16)
    reach = float(grid_vector_table(E4M3_GRID)[table.long()].abs().max())

    amax = w.abs().amax(dim=1)
    rms = w.pow(2).mean(dim=1).sqrt()
    over = amax * sigma > reach * rms
    assert 0 < int(over.sum()) < rows, "the fixture must raise some rows and not others"

    stored, effective, global_scale = initial_channel_scale(w, sigma, reach=reach)
    landed = amax[over] / effective[over]
    assert bool((landed <= reach).all()), (
        f"{int((landed > reach).sum())} of {int(over.sum())} raised rows land "
        f"past the reach; worst {float(landed.max()) / reach - 1:.3e} over"
    )

    # Rows the reach raise did not touch retain the ordinary RMS start byte for
    # byte.  Rounding those boundary rows upward would be a wider policy.
    from tessera.scale_channel import land_channel_scale

    plain, _ = land_channel_scale(rms / sigma, global_scale)
    assert torch.equal(stored[~over], plain[~over])

    # The raise is minimal: it lands at its floor or one fp16 ulp above it.
    floor = amax / reach
    assert bool((effective[over] >= floor[over]).all())
    assert bool((effective[over] <= floor[over] * (1 + 2.0 ** -10)).all())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_refit_keeps_an_improvement_smaller_than_the_loss_ulp(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    # Exact binary inputs: the nearest fp16 word is 1, but subtracting two
    # rounded quadratic losses can erase its strict improvement.
    work = torch.tensor([[1 + 2**-11 - 2**-16]], device=device)
    units = torch.ones_like(work)
    stored = torch.tensor([1 + 2**-10], dtype=torch.float16, device=device)
    new_stored, effective = refit_channel_scale(work, units, stored, 1.0)
    assert new_stored.item() == 1.0
    old_loss = (work.double() - stored.double().reshape(-1, 1)).square().sum()
    new_loss = (work.double() - effective.double().reshape(-1, 1)).square().sum()
    assert new_loss < old_loss
