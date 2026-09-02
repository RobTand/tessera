"""LDLQ on the window body: the scheduling change, and what it may not move.

The three properties this file pins:

* an LDLQ pass whose factor has no off-diagonal blocks is the ordinary pass,
  **bit for bit** -- codes, row scales and the reported sse.  That is the
  statement that block-sequential encoding is a schedule and not a second
  encoder; it holds because the Viterbi carries no state across columns.
* the refit metrics are the quadratics they claim to be, and the monotone
  guard holds under each of them.
* the reach floor does what it is for: no row's target lands outside the
  body's reach after a refit that was asked to keep it inside.
"""
import math

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.compensate import block_ldl, regularize_hessian
from tessera.encode import encode_unit, window_table
from tessera.errors import GrammarError
from tessera.export import (
    DEFAULT_CODE, DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA, DEFAULT_REFIT_OBJECTIVE,
    encode_linear_planes)
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.scale_channel import (
    default_channel_sigma, land_at_least, refit_channel_scale)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused window Viterbi is CUDA")

ROWS, COLS, BITS = 64, 256, 10


def _weights(seed=0, rows=ROWS, cols=COLS, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.02
    # A heavy tail, so the reach floor has something to hold.
    w[3, 7] = 0.6
    return w.to(device=device, dtype=torch.float32).contiguous()


def _encode(w, **kw):
    """The FP8 route's encoder settings: E4M3 over the CHANNEL plane, window
    body, span 1 -- what ``wire_recipe(E4M3, 1024)`` resolves to."""
    return encode_unit(
        w, E4M3_GRID, (4,) * w.shape[1], DEFAULT_CODE,
        body=BodyKind.WINDOW, window_bits=BITS, span=1,
        scale_plane=ScalePlaneKind.CHANNEL, trellis_weighting="scale", **kw,
    )


@cuda
@pytest.mark.parametrize("block", [32, 64, 128])
def test_block_diagonal_ldlq_is_the_plain_pass(block):
    """L with zero off-diagonal blocks compensates nothing, so it must encode
    exactly what the single whole-matrix pass encodes."""
    w = _weights()
    plain = _encode(w, scale_refit=4)
    eye = torch.eye(COLS, device=w.device)
    same = _encode(w, scale_refit=4, ldl=eye, ldl_block=block)
    assert torch.equal(plain.codes, same.codes)
    assert torch.equal(plain.scale_rows, same.scale_rows)
    assert plain.scale_global == same.scale_global
    assert plain.sse == same.sse


@cuda
def test_ldlq_changes_the_codes_and_keeps_the_wire():
    """A real Hessian moves the encode, and the artifact still round-trips."""
    from tessera.unit_artifact import read_unit_artifact

    torch.manual_seed(1)          # mix/x/x_ev below draw from the global RNG
    w = _weights(seed=1)
    # Correlated inputs: an iid x makes H the identity, under which LDLQ has
    # nothing to compensate and the arms differ only by rounding noise.
    mix = torch.randn(COLS, COLS, device=w.device) / math.sqrt(COLS)
    x = torch.randn(4096, COLS, device=w.device) @ mix
    x_ev = torch.randn(4096, COLS, device=w.device) @ mix
    H = regularize_hessian(x.T @ x, count=x.shape[0], sigma_reg=1.0)
    L = block_ldl(H, 64)
    ex_plain, plain, _ = encode_linear_planes(w, grid=E4M3_GRID, q256=1024, name="u")
    ex_comp, comp, _ = encode_linear_planes(
        w, grid=E4M3_GRID, q256=1024, name="u", ldl=L, ldl_block=64)
    assert not torch.equal(plain.codes, comp.codes)
    assert len(ex_plain.blob) == len(ex_comp.blob)          # same wire, same bytes
    hat = read_unit_artifact(ex_comp.blob, device=w.device)
    assert hat.shape == w.shape
    # LDLQ trades plain weight error for output error, so the check is the
    # output error -- on rows the factor was not fit on.
    y = x_ev @ w.T
    def outerr(hat_):
        return float((x_ev @ hat_.T - y).norm() / y.norm())
    hat_plain = read_unit_artifact(ex_plain.blob, device=w.device)
    assert outerr(hat) < outerr(hat_plain)


@cuda
def test_a_block_size_disagreement_is_refused():
    """A COARSER schedule over a finer factor is the silent-wrong pair: the
    factor already spent compensation on columns the schedule then quantises
    together, and the arithmetic stays well-formed. The reverse (a finer
    schedule over a coarser factor) only compensates less, so it is allowed."""
    torch.manual_seed(3)
    w = _weights(seed=3)
    x = torch.randn(1024, COLS, device=w.device)
    H = regularize_hessian(x.T @ x, count=1024, sigma_reg=3.0)
    with pytest.raises(GrammarError, match="diagonal blocks are not the identity"):
        _encode(w, scale_refit=4, ldl=block_ldl(H, 32), ldl_block=64)
    with pytest.raises(GrammarError, match="not a multiple of the LDLQ block"):
        _encode(w, scale_refit=4, ldl=block_ldl(H, 64), ldl_block=96)
    _encode(w, scale_refit=4, ldl=block_ldl(H, 64), ldl_block=64)   # the matching pair


@cuda
def test_ldlq_is_deterministic():
    torch.manual_seed(2)
    w = _weights(seed=2)
    x = torch.randn(1024, COLS, device=w.device)
    L = block_ldl(regularize_hessian(x.T @ x, count=1024, sigma_reg=3.0), 64)
    a = _encode(w, scale_refit=4, ldl=L, ldl_block=64)
    b = _encode(w, scale_refit=4, ldl=L, ldl_block=64)
    assert torch.equal(a.codes, b.codes) and torch.equal(a.scale_rows, b.scale_rows)


def test_refit_metric_forms_agree_and_stay_monotone():
    """A diagonal metric is the full-Hessian form with a diagonal H, and the
    guard never lets a row end worse under the metric it was given."""
    torch.manual_seed(0)
    work = torch.randn(16, 64)
    units = torch.randint(-8, 8, (16, 64)).float()
    stored = torch.full((16,), 0.05).to(torch.float16)
    h = torch.rand(64) + 0.1
    _, eff_diag = refit_channel_scale(work, units, stored, 1.0, metric=h)
    _, eff_full = refit_channel_scale(work, units, stored, 1.0, metric=torch.diag(h))
    assert torch.equal(eff_diag, eff_full)
    mix = torch.randn(64, 64)
    for metric in (None, h, torch.diag(h) + 0.01 * mix @ mix.T):
        _, eff = refit_channel_scale(work, units, stored, 1.0, metric=metric)
        M = (torch.eye(64) if metric is None else
             (torch.diag(metric) if metric.ndim == 1 else metric))
        def loss(s):
            e = work - s.reshape(-1, 1) * units
            return ((e @ M) * e).sum(dim=1)
        assert bool((loss(eff) <= loss(stored.float()) + 1e-6).all())


def test_land_at_least_never_lands_short():
    floor = torch.tensor([1e-3, 0.017, 3.25, 61234.0, 1.0 / 3])
    stored, eff = land_at_least(floor, 1.0)
    assert bool((eff >= floor).all())
    assert stored.dtype is torch.float16


@cuda
def test_reach_floor_keeps_the_target_inside_the_body():
    w = _weights(seed=3)
    sigma = default_channel_sigma(E4M3_GRID)
    table = window_table(E4M3_GRID, BITS, sigma=sigma, seed=0, half=16, device=w.device)
    from tessera.encode import grid_vector_table
    reach = float(grid_vector_table(E4M3_GRID, w.device)[table.long()].abs().max())
    for floor_on in (False, True):
        unit = _encode(w, scale_refit=4, refit_reach_floor=floor_on)
        scale = unit.scale_rows.float().to(w.device) * unit.scale_global
        over = (w.abs().amax(dim=1) / scale > reach * (1 + 1e-6))
        if floor_on:
            assert not bool(over.any()), f"{int(over.sum())} rows clip after a floored refit"


def test_the_activation_aware_defaults_are_the_measured_ones():
    """The recipe an exporter applies when it is handed a Hessian.

    Pinned because these three numbers are a measurement, not a taste: on
    Qwen3-0.6B's 4.07-bpp FP8 wire they take served KL-vs-BF16 from 0.1512 to
    0.1046 at identical bytes, and on six GLM experts the out-space geomean to
    0.932x (docs/measurements/tessera-ldlq-window-served-2026-09-02.md).
    Changing one means re-running that gate.
    """
    assert (DEFAULT_LDLQ_SIGMA, DEFAULT_LDLQ_BLOCK, DEFAULT_REFIT_OBJECTIVE) == (1.0, 32, "hessian")


@cuda
def test_a_weights_only_encode_is_untouched_by_the_defaults():
    """No Hessian, no change: the levers default off inside the encoder, so an
    export that was not given activations is the artifact it always was."""
    w = _weights(seed=7)
    a = encode_linear_planes(w, grid=E4M3_GRID, q256=1024, name="u")[0]
    b = encode_linear_planes(
        w, grid=E4M3_GRID, q256=1024, name="u",
        ldl=None, refit_metric=None, refit_reach_floor=False)[0]
    assert a.blob == b.blob
