"""The trailing refit's objective is a schedule, not a second encoder (issue #75).

Issue #75's screen showed the full-H objective's worth sitting almost entirely
in the trailing refit -- but the swap arm ran five refits against the control's
four.  The fair pair is ``T R_h T R_h T R_h T R_H`` against
``T R_h T R_h T R_h T R_h``: same pass count, only the trailing refit's
objective swapped.  The encoder could not express that: one ``refit_metric``
ran on every pass.  ``refit_metric_trailing`` is the missing half -- inner
passes minimise ``refit_metric``, the trailing refit minimises
``refit_metric_trailing``, ``None`` is the encode that was already there.

Encoder-side and opt-in like ``refit_gauss_seidel``: no ``ActivationSource``
field, no config entry, no wire change, refused wherever it would be silently
ignored.  CPU-only: the schedule is visible in bytes at 32x64 without a GPU.
"""
import pytest
import torch

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.errors import GrammarError
from tessera.export import encode_linear, encode_linear_planes, tcq_cap_q256
from tessera.manifest import ScalePlaneKind

K2 = tuple_grid(E2M1_GRID, 2)
CAP = tcq_cap_q256(K2)          # 896: the 4.0 bpp wire
ROWS, COLS = 32, 64


def _weights(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(ROWS, COLS, generator=g) * 0.02).to(dtype=torch.float32)


def _hessian(seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(4 * COLS, COLS, generator=g)
    return ((x.T @ x) / x.shape[0]).to(dtype=torch.float32)


def _units(w, **kw):
    kw.setdefault("scale_refit", 2)
    _, unit, _ = encode_linear_planes(
        w, grid=K2, q256=CAP, name="u", verify=False, **kw)
    return unit


def _same(a, b):
    return (torch.equal(a.codes, b.codes)
            and torch.equal(a.scale_lut, b.scale_lut)
            and torch.equal(a.scale_refine, b.scale_refine)
            and a.scale_global == b.scale_global
            and a.sse == b.sse)


def test_trailing_none_is_the_encode_that_was_there():
    """The default is byte for byte the old call: same codes, table, sse."""
    w, H = _weights(), _hessian()
    h = H.diagonal() / H.diagonal().mean()
    old = _units(w, refit_metric=h)
    new = _units(w, refit_metric=h, refit_metric_trailing=None)
    assert _same(old, new)


def test_a_trailing_equal_to_the_base_collapses_to_the_uniform_schedule():
    """A schedule whose trailing objective is the base objective is uniform."""
    w, H = _weights(seed=3), _hessian(seed=3)
    h = H.diagonal() / H.diagonal().mean()
    uniform = _units(w, refit_metric=H)
    collapsed = _units(w, refit_metric=h, refit_metric_trailing=h)
    uniform_h = _units(w, refit_metric=h)
    assert _same(collapsed, uniform_h)
    assert not _same(uniform, uniform_h)  # the two objectives differ here


def test_the_fair_pair_is_expressible_at_equal_pass_count():
    """``T R_h T R_H`` against ``T R_h T R_h``: same passes, swapped trailing
    objective, byte-identical length -- otherwise the arm names a setting that
    did nothing, the failure ``test_both_levers_reach_the_bytes`` guards."""
    w, H = _weights(seed=5), _hessian(seed=5)
    h = H.diagonal() / H.diagonal().mean()
    control = encode_linear(w, grid=K2, q256=CAP, name="x", verify=False,
                            scale_refit=2, refit_metric=h).blob
    swapped = encode_linear(w, grid=K2, q256=CAP, name="x", verify=False,
                            scale_refit=2, refit_metric=h,
                            refit_metric_trailing=H).blob
    assert control != swapped
    assert len(control) == len(swapped), "the schedule never changes the wire"


def test_the_trailing_objective_reaches_the_channel_plane():
    """The schedule is per-pass, not per-plane: CHANNEL's row refit reads it
    on the trailing pass too."""
    from tessera.alphabet import E4M3_GRID

    w, H = _weights(seed=7), _hessian(seed=7)
    h = H.diagonal() / H.diagonal().mean()
    kw = dict(grid=E4M3_GRID, q256=4 * 256, name="x", verify=False, scale_refit=2)
    control = encode_linear(w, **kw, refit_metric=h).blob
    swapped = encode_linear(w, **kw, refit_metric=h, refit_metric_trailing=H).blob
    assert control != swapped
    assert len(control) == len(swapped)


def test_a_trailing_metric_is_refused_where_it_would_be_ignored():
    """Every context where the trailing objective shapes no refit, refused by
    its own message rather than accepted as a silent no-op."""
    w, H = _weights(seed=9), _hessian(seed=9)
    h = H.diagonal() / H.diagonal().mean()
    with pytest.raises(GrammarError, match="S6b"):
        _units(w, refit_metric=h, refit_metric_trailing=H,
               scale_plane=ScalePlaneKind.S6B)
    with pytest.raises(GrammarError, match="scale_refit=0 runs none"):
        encode_linear(w, grid=K2, q256=CAP, name="x", verify=False,
                      scale_refit=0, refit_metric_trailing=H)
    with pytest.raises(GrammarError, match="one weight per input column"):
        _units(w, refit_metric=h,
               refit_metric_trailing=torch.ones(COLS + 1))


def test_gauss_seidel_reads_coupling_from_either_schedule_leg():
    """The sweep is refused only where it would be the parallel step under
    another name: a coupled trailing leg makes it meaningful even when the
    inner legs are separable, and two separable legs still refuse."""
    from tessera.compensate import block_ldl, regularize_hessian

    g = torch.Generator(device="cpu").manual_seed(11)
    w = (torch.randn(64, 256, generator=g) * 0.02).to(dtype=torch.float32)
    gh = torch.Generator(device="cpu").manual_seed(12)
    x = torch.randn(4 * 256, 256, generator=gh)
    H = ((x.T @ x) / x.shape[0]).to(dtype=torch.float32)
    h = H.diagonal() / H.diagonal().mean()
    h2 = (H.diagonal() / H.diagonal().mean()).pow(2.0)
    L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
    kw = dict(grid=K2, q256=CAP, name="x", verify=False, scale_refit=2,
              ldl=L, ldl_block=32, refit_metric=h, refit_metric_trailing=H)
    # Coupled trailing leg: accepted, and the sweep reaches the bytes.
    jac = encode_linear(w, **kw).blob
    gs = encode_linear(w, **kw, refit_gauss_seidel=True).blob
    assert jac != gs
    assert len(jac) == len(gs)
    # No coupled leg anywhere: the flag would name an arm that changed nothing.
    with pytest.raises(GrammarError, match="separable"):
        encode_linear(w, grid=K2, q256=CAP, name="x", verify=False,
                      scale_refit=2, refit_metric=h, refit_metric_trailing=h2,
                      refit_gauss_seidel=True)


def test_the_diagnostic_records_the_optimiser_that_ran():
    """``refit_gauss_seidel`` rides the whole schedule, and on the inner passes
    the metric is 1-D, where the sweep is provably the parallel step -- the
    encoder reads the flag only on the coupled branch.  The record must say
    what the refit DID, or a measurement reads three sweeps that never ran.

    The numbers prove it: the inner refits of a swept trailing schedule are
    identical, field for field, to the un-swept control's.
    """
    from tessera.compensate import block_ldl, regularize_hessian
    from tessera.encode import refit_diagnostics

    g = torch.Generator(device="cpu").manual_seed(21)
    w = (torch.randn(64, 256, generator=g) * 0.02).to(dtype=torch.float32)
    gh = torch.Generator(device="cpu").manual_seed(22)
    x = torch.randn(4 * 256, 256, generator=gh)
    H = ((x.T @ x) / x.shape[0]).to(dtype=torch.float32)
    h = H.diagonal() / H.diagonal().mean()
    L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
    kw = dict(grid=K2, q256=CAP, name="x", verify=False, scale_refit=4,
              ldl=L, ldl_block=32)

    def records(**extra):
        with refit_diagnostics() as diag:
            encode_linear(w, **kw, **extra)
        return [dict(d) for d in diag]

    control = records(refit_metric=h)
    swept = records(refit_metric=h, refit_metric_trailing=H,
                    refit_gauss_seidel=True)
    assert len(control) == len(swept) == 4
    # What ran: three separable steps, then one sweep.
    assert [d["gauss_seidel"] for d in swept] == [False, False, False, True]
    # What was asked for, kept so the arm is still identifiable.
    assert all(d["gauss_seidel_requested"] for d in swept)
    assert not any(d["gauss_seidel_requested"] for d in control)
    # And the claim the record makes is true of the numbers.
    for i in range(3):
        for k in ("before", "stepped", "continuous", "landed", "reverted"):
            assert control[i][k] == swept[i][k], (i, k)
