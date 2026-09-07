"""The batch axis of the encoder writes each unit's bytes, not nearly its bytes.

``encode.encode_units`` and ``export.encode_linears`` (tessera#385) run ``B``
same-shape units through one LDLQ schedule with every Viterbi call joined
along the column axis.  The contract is *identity*: the blob each unit gets
out of the batch is byte for byte the blob ``encode_linear`` writes for it
alone, at every shipping family, with LDLQ on and the default refit
schedule.  Columns are independent inside both trellises, which is the
property the whole batch rests on -- and a trellis is a chain of decisions,
so anything less than identity here is a different artifact.

Pre-fix, on the base without the batch axis, every test in this file fails
at import: ``ImportError: cannot import name 'encode_units' from
'tessera.encode'``.
"""
import pytest
import torch

from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.encode import encode_unit, encode_units
from tessera.errors import GrammarError
from tessera.export import (
    ActivationSource, DEFAULT_CODE, DEFAULT_SCALE_REFIT, encode_linear,
    encode_linear_planes, encode_linears, encode_linears_planes, tcq_cap_q256,
    wire_recipe)
from tessera.manifest import BodyKind

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the trellises are CUDA")

ROWS, COLS = 64, 256
K2 = tuple_grid(E2M1_GRID, 2)
#: One rung per shipping family, at the rung the campaign encodes them
#: (PrismaQuant #275): BF16_K1@1792 and E4M3_K1@1024 are the window body over
#: the CHANNEL plane; E2M1_K2@896 is the coset trellis over the LUT plane at
#: its cap.  E4M3@1042 is the Bresenham mix of two rates, so the per-(block,
#: rate) column index is exercised with two rates inside every block.
FAMILIES = [
    ("BF16_K1@1792", BF16_GRID, 1792),
    ("E4M3_K1@1024", E4M3_GRID, 1024),
    ("E2M1_K2@896", K2, 896),
    ("E4M3_K1@1042", E4M3_GRID, 1042),
]
assert FAMILIES[2][2] == tcq_cap_q256(K2)
assert wire_recipe(K2, 896).body is BodyKind.TCQ


def _weights(seed, rows=ROWS, cols=COLS):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.02
    w[3, 7] = 0.6                     # a heavy tail, so the planes have work
    return w.to(device="cuda", dtype=torch.bfloat16).contiguous()


def _hessian(seed, cols=COLS):
    """A PSD input Hessian with real off-block structure: iid inputs would
    make H the identity, under which LDLQ compensates nothing."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(4 * cols, cols, generator=g)
    mix = torch.eye(cols) + torch.randn(cols, cols, generator=g) / cols ** 0.5
    x = x @ mix
    x[:, ::29] *= 5.0
    return (x.T @ x).to(device="cuda", dtype=torch.float32)


def _source(count, seed=100):
    return ActivationSource(
        hessians={f"u{i}": _hessian(seed + i) for i in range(count)},
        provenance={"text_sha256": "0" * 64, "fit_tokens": 4 * COLS,
                    "fit_ids_sha256": "1" * 64},
    )


def _per_unit(source, weights, recipe):
    return [
        source.for_unit(f"u{i}", COLS, "cuda", scale_plane=recipe.scale_plane, weight=w)
        for i, w in enumerate(weights)
    ]


@cuda
@pytest.mark.parametrize("label,grid,q256", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_the_batch_writes_each_units_own_bytes(label, grid, q256):
    """Two units through ``encode_linears`` against each alone through
    ``encode_linear``: the same bytes, unit by unit, with the exporter's
    round-trip verify on both sides."""
    recipe = wire_recipe(grid, q256)
    weights = [_weights(0), _weights(1)]
    per_unit = _per_unit(_source(2), weights, recipe)
    assert all(kw["ldl"] is not None for kw in per_unit), "LDLQ must be on"
    alone = [
        encode_linear(w, grid=grid, q256=q256, name=f"u{i}",
                      scale_refit=DEFAULT_SCALE_REFIT, **kw)
        for i, (w, kw) in enumerate(zip(weights, per_unit))
    ]
    together = encode_linears(
        weights, grid=grid, q256=q256, names=["u0", "u1"], per_unit=per_unit,
        scale_refit=DEFAULT_SCALE_REFIT)
    assert len(together) == 2
    for one, batched in zip(alone, together):
        assert batched.blob == one.blob
        assert batched.exact_bytes == one.exact_bytes
        assert batched.name == one.name
    # Two different units: a batch that wrote unit 0's bytes twice would
    # have passed the loop above only if the units were the same.
    assert together[0].blob != together[1].blob


@cuda
def test_three_units_at_two_rates_through_the_plane_entry():
    """``encode_linears_planes`` with explicit per-unit sequences, at the
    mixed-rate E4M3 rung, at B=3: codes, planes and sse per unit equal the
    unit alone."""
    grid, q256 = E4M3_GRID, 1042
    recipe = wire_recipe(grid, q256)
    weights = [_weights(s) for s in (5, 6, 7)]
    per_unit = _per_unit(_source(3, seed=200), weights, recipe)
    ldl = [kw["ldl"] for kw in per_unit]
    metric = [kw["refit_metric"] for kw in per_unit]
    block = {kw["ldl_block"] for kw in per_unit}
    assert len(block) == 1
    alone = [
        encode_linear_planes(w, grid=grid, q256=q256, name=f"u{i}", ldl=l,
                             ldl_block=next(iter(block)), refit_metric=m)
        for i, (w, l, m) in enumerate(zip(weights, ldl, metric))
    ]
    together = encode_linears_planes(
        weights, grid=grid, q256=q256, ldl=ldl, ldl_block=next(iter(block)),
        refit_metric=metric)
    for (exported, unit, _), (exported_b, unit_b, _) in zip(alone, together):
        assert exported_b.blob == exported.blob
        assert torch.equal(unit_b.codes, unit.codes)
        assert torch.equal(unit_b.scale_rows, unit.scale_rows)
        assert unit_b.scale_global == unit.scale_global
        assert unit_b.sse == unit.sse


@cuda
def test_encode_unit_is_the_batch_at_one():
    from tessera.manifest import ScalePlaneKind

    w = _weights(9)
    kw = dict(body=BodyKind.WINDOW, window_bits=10, span=1,
              scale_plane=ScalePlaneKind.CHANNEL, trellis_weighting="scale")
    one = encode_unit(w, E4M3_GRID, (4,) * COLS, DEFAULT_CODE, **kw)
    batch = encode_units([w], E4M3_GRID, (4,) * COLS, DEFAULT_CODE, **kw)
    assert len(batch) == 1
    assert torch.equal(one.codes, batch[0].codes)
    assert torch.equal(one.scale_rows, batch[0].scale_rows)
    assert one.sse == batch[0].sse


@cuda
def test_a_batch_that_is_not_one_call_is_refused():
    from tessera.manifest import ScalePlaneKind

    kw = dict(body=BodyKind.WINDOW, window_bits=10, span=1,
              scale_plane=ScalePlaneKind.CHANNEL, trellis_weighting="scale")
    rates = (4,) * COLS
    a, b = _weights(1), _weights(2)
    with pytest.raises(GrammarError, match="one shape"):
        encode_units([a, _weights(3, rows=2 * ROWS)], E4M3_GRID, rates, DEFAULT_CODE, **kw)
    with pytest.raises(GrammarError, match="one LDLQ schedule"):
        encode_units([a, b], E4M3_GRID, rates, DEFAULT_CODE,
                     ldl=[torch.eye(COLS, device="cuda"), None], **kw)
    with pytest.raises(GrammarError, match="entries for 2 units"):
        encode_units([a, b], E4M3_GRID, rates, DEFAULT_CODE, ldl=[None], **kw)
    with pytest.raises(GrammarError, match="at least one unit"):
        encode_units([], E4M3_GRID, rates, DEFAULT_CODE, **kw)


@cuda
def test_per_unit_settings_that_disagree_are_refused_by_key():
    grid, q256 = E4M3_GRID, 1024
    recipe = wire_recipe(grid, q256)
    weights = [_weights(0), _weights(1)]
    per_unit = _per_unit(_source(2), weights, recipe)
    per_unit[1]["refit_reach_floor"] = not per_unit[0]["refit_reach_floor"]
    with pytest.raises(GrammarError, match="refit_reach_floor"):
        encode_linears(weights, grid=grid, q256=q256, per_unit=per_unit)
    per_unit = _per_unit(_source(2), weights, recipe)
    with pytest.raises(ValueError, match="one spelling"):
        encode_linears(weights, grid=grid, q256=q256, per_unit=per_unit,
                       ldl=[kw["ldl"] for kw in per_unit])
    per_unit[0]["unrouted"] = 1
    with pytest.raises(GrammarError, match="unrouted"):
        encode_linears(weights, grid=grid, q256=q256, per_unit=per_unit)
