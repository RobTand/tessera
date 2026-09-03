"""LDLQ and the metric-aware refit on the LUT plane -- the 4-bit route's wire.

The FP8 route got both levers first because its CHANNEL plane has no column
axis.  ``encode_unit`` refused them under a block plane on the reasoning that
the plane's within-row column spans "would have to be scheduled with the
blocks".  They do not: the plane is read once per pass, before the block loop,
and refit once after it, so the schedule and the plane never interleave.  This
file is the evidence for that, on the two bodies the E2M1x2 grid ships:

* the **TCQ cap wire** (``q256 = 896``, span 2, LUT16) -- the 4.0 bpp wire; and
* the **sub-cap window body** (``q256 < 896``, L=12, LUT16).

and for the block-scale refit's closed form: that it is the plain refit when
the metric is the identity, the diagonal form when the metric is diagonal, and
that on a Hessian with real off-block coupling it *moves* and *lowers the
metric's own error* -- the last because a Jacobi step that the monotone guard
rejects every pass would leave the arm encoding identically to LDLQ alone and
raise nothing.
"""
import pytest
import torch

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.compensate import block_ldl, regularize_hessian
from tessera.encode import _lut_values, _refit_scales_lut
from tessera.errors import GrammarError
from tessera.export import (
    DEFAULT_CODE, encode_linear, encode_linear_planes, tcq_cap_q256,
    wire_recipe)
from tessera.manifest import BodyKind, ScalePlaneKind

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the Viterbi is CUDA")

K2 = tuple_grid(E2M1_GRID, 2)
CAP = tcq_cap_q256(K2)          # 896: the 4.0 bpp wire
SUBCAP = 768                    # the window body's range
ROWS, COLS = 64, 256


def _weights(seed=0, rows=ROWS, cols=COLS, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.02).to(
        device=device, dtype=torch.float32).contiguous()


def _hessian(cols=COLS, seed=1, device="cuda", coupling=1.0):
    """A PSD input Hessian with genuine off-BLOCK structure.

    A block-diagonal H would make the coordinate step exact and prove nothing
    about the guard, so the mixing matrix is dense and a few columns are made
    loud, which is the shape a real activation Hessian has.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(4 * cols, cols, generator=g)
    mix = torch.eye(cols) + coupling * torch.randn(cols, cols, generator=g) / cols ** 0.5
    x = x @ mix
    x[:, ::29] *= 5.0
    return ((x.T @ x) / x.shape[0]).to(device=device, dtype=torch.float32)


def _encode(w, q256, **kw):
    """The unit ``wire_recipe(E2M1x2, q256)`` resolves to.

    Through ``encode_linear_planes`` rather than ``encode_unit`` because the
    TCQ body needs its anchor forests built for the rung, and the point of
    these arms is the wire the exporter writes, not a hand-assembled one.
    """
    return encode_linear_planes(
        w.bfloat16(), grid=K2, q256=q256, name="u", verify=False, **kw)[1]


# --------------------------------------------------------------------------
# LDLQ is a schedule, not a second encoder -- on the block plane too.
# --------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("q256", [CAP, SUBCAP])
@pytest.mark.parametrize("block", [32, 64, 128])
def test_identity_factor_is_the_plain_pass_on_the_lut_plane(q256, block):
    """A factor with no off-diagonal blocks compensates nothing, so it must
    encode bit for bit what the ordinary whole-matrix pass encodes -- codes,
    scale table, indices and the reported sse."""
    w = _weights()
    plain = _encode(w, q256, scale_refit=4)
    eye = torch.eye(COLS, device=w.device)
    same = _encode(w, q256, scale_refit=4, ldl=eye, ldl_block=block)
    assert torch.equal(plain.codes, same.codes)
    assert torch.equal(plain.scale_lut, same.scale_lut)
    assert torch.equal(plain.scale_refine, same.scale_refine)
    assert plain.scale_global == same.scale_global
    assert plain.sse == same.sse


@cuda
@pytest.mark.parametrize("q256", [CAP, SUBCAP])
def test_ldlq_moves_the_encode_and_keeps_the_wire(q256):
    """A real Hessian changes the codes, and the unit still serialises."""
    w = _weights(seed=5)
    L = block_ldl(regularize_hessian(_hessian(), sigma_reg=1.0), 32)
    plain = _encode(w, q256, scale_refit=4)
    ldlq = _encode(w, q256, scale_refit=4, ldl=L, ldl_block=32)
    assert not torch.equal(plain.codes, ldlq.codes)
    assert plain.codes.shape == ldlq.codes.shape
    assert plain.scale_refine.shape == ldlq.scale_refine.shape
    assert plain.scale_lut.numel() == ldlq.scale_lut.numel()
    again = _encode(w, q256, scale_refit=4, ldl=L, ldl_block=32)
    assert torch.equal(ldlq.codes, again.codes)          # deterministic


@cuda
def test_the_block_size_guard_still_holds_under_a_block_plane():
    w = _weights(seed=3)
    H = regularize_hessian(_hessian(seed=3), sigma_reg=3.0)
    with pytest.raises(GrammarError, match="diagonal blocks are not the identity"):
        _encode(w, CAP, scale_refit=4, ldl=block_ldl(H, 32), ldl_block=64)
    with pytest.raises(GrammarError, match="not a multiple of the LDLQ block"):
        _encode(w, CAP, scale_refit=4, ldl=block_ldl(H, 64), ldl_block=96)
    _encode(w, CAP, scale_refit=4, ldl=block_ldl(H, 64), ldl_block=64)


# --------------------------------------------------------------------------
# The block-scale refit's closed form.
# --------------------------------------------------------------------------


def _lut_fixture(seed=0, rows=12, cols=64, half=16, device="cpu"):
    """A LUT plane and codes to refit it against, without running a trellis."""
    g = torch.Generator().manual_seed(seed)
    work = (torch.randn(rows, cols, generator=g) * 0.02).to(device)
    units = torch.randint(-6, 7, (rows, cols), generator=g).float().to(device)
    from tessera.encode import _pack_scales_lut
    table, index, effective, glob = _pack_scales_lut(work, half, peak=6.0)
    return work, units, half, table.to(device), index.to(device), effective.to(device), glob


def _cost(work, units, effective, half, M):
    e = work - effective.reshape(work.shape[0], -1).repeat_interleave(half, dim=1) * units
    return float(((e @ M) * e).sum())


def test_an_identity_metric_is_the_plain_refit():
    """The derivation's own claim: at ``H = I`` the cross-block terms vanish
    because blocks have disjoint support, and ``s* = <w,u>/<u,u>`` again."""
    work, units, half, table, index, eff, glob = _lut_fixture()
    plain = _refit_scales_lut(work, units, half, table, index, eff, glob)
    eye = torch.eye(work.shape[1])
    metric = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=eye)
    assert torch.equal(plain[0], metric[0])
    assert torch.equal(plain[1], metric[1])
    assert torch.allclose(plain[2], metric[2], rtol=1e-6, atol=0)


def test_a_diagonal_metric_is_the_full_form_with_a_diagonal_hessian():
    work, units, half, table, index, eff, glob = _lut_fixture(seed=2)
    h = torch.rand(work.shape[1]) + 0.1
    a = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=h)
    b = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=torch.diag(h))
    assert torch.equal(a[1], b[1])
    assert torch.allclose(a[2], b[2], rtol=1e-5, atol=0)


def test_the_metric_refit_moves_scales_and_lowers_the_metric_cost():
    """The no-op this test exists to catch: a Jacobi step that the monotone
    guard rejects on every pass leaves the metric arm encoding *identically* to
    the arm without it, and raises nothing.  So assert both halves -- scales
    move, and the metric's own error strictly falls."""
    work, units, half, table, index, eff, glob = _lut_fixture(seed=4, cols=128)
    H = _hessian(cols=128, seed=4, device="cpu")
    before = _cost(work, units, eff, half, H)
    _, new_index, new_eff = _refit_scales_lut(
        work, units, half, table, index, eff, glob, metric=H)
    assert not torch.equal(new_index, index), "no scale moved: the guard rejected every step"
    assert _cost(work, units, new_eff, half, H) < before


def test_the_metric_refit_is_monotone_under_its_own_metric():
    """Whatever the metric, the plane never ends worse than it began under it:
    the plane the unit already has is one of the candidates."""
    work, units, half, table, index, eff, glob = _lut_fixture(seed=6, cols=128)
    h = torch.rand(128) + 0.05
    for metric in (h, _hessian(cols=128, seed=7, device="cpu"),
                   _hessian(cols=128, seed=8, device="cpu", coupling=4.0)):
        M = torch.diag(metric) if metric.ndim == 1 else metric
        _, _, new_eff = _refit_scales_lut(
            work, units, half, table, index, eff, glob, metric=metric)
        assert _cost(work, units, new_eff, half, M) <= _cost(work, units, eff, half, M) + 1e-9


def test_a_metric_refit_needs_its_blocks_inside_a_row():
    work, units, half, table, index, eff, glob = _lut_fixture(seed=9, rows=8, cols=64)
    with pytest.raises(GrammarError, match="within one row"):
        _refit_scales_lut(work, units, 24, table, index, eff, glob,
                          metric=torch.eye(64))


# --------------------------------------------------------------------------
# End to end, through the exporter's own entry point.
# --------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("q256", [CAP, SUBCAP])
def test_both_levers_reach_the_bytes_on_the_four_bit_wire(q256):
    """Each lever alone, and the two together, must produce three encodings
    that differ from the baseline and from each other -- otherwise the arm is
    named after a setting that did nothing."""
    w = _weights(seed=8).bfloat16()
    H = _hessian(seed=8)
    L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
    base = encode_linear(w, grid=K2, q256=q256, name="x").blob
    ldlq = encode_linear(w, grid=K2, q256=q256, name="x", ldl=L, ldl_block=32).blob
    refit = encode_linear(w, grid=K2, q256=q256, name="x", refit_metric=H).blob
    both = encode_linear(w, grid=K2, q256=q256, name="x",
                         ldl=L, ldl_block=32, refit_metric=H).blob
    assert len({base, ldlq, refit, both}) == 4
    assert len(base) == len(ldlq) == len(refit) == len(both), "the wire is unchanged"


@cuda
def test_the_body_the_recipe_picks_is_the_body_measured():
    """The two arms above are the two bodies the grid ships, not one twice."""
    assert wire_recipe(K2, CAP).body is BodyKind.TCQ
    assert wire_recipe(K2, CAP).scale_plane is ScalePlaneKind.LUT
    assert wire_recipe(K2, SUBCAP).body is BodyKind.WINDOW
    assert wire_recipe(K2, SUBCAP).scale_plane is ScalePlaneKind.LUT


@cuda
def test_the_streaming_export_carries_a_hessian_onto_the_four_bit_wire(tmp_path):
    """``export_glm53_tessera.py``'s path, at its own grid.

    That driver streams shards through ``export_checkpoint_streaming`` on
    E2M1_K2, and it is the driver whose ``--hessian`` was withheld while the
    block plane refused both levers.  So pin the whole leg: the Hessian
    reaches the encoder (the bytes move), the wire does not grow, and the
    capture's identity lands in the config a merge guard reads.
    """
    import json

    from safetensors.torch import save_file

    from tessera.export import ActivationSource, export_checkpoint_streaming

    g = torch.Generator().manual_seed(21)
    tensors = {f"model.layers.0.mlp.{p}.weight": torch.randn(64, 256, generator=g).bfloat16()
               for p in ("gate_proj", "up_proj")}
    src = tmp_path / "src"
    src.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    plan = {name: CAP for name in tensors}
    hessians = {ActivationSource.unit_name(n): _hessian(cols=256, seed=i, device="cpu")
                for i, n in enumerate(tensors)}
    provenance = {"source": "wikitext-2 train", "text_sha256": "c" * 64,
                  "fit_tokens": 16384, "fit_ids_sha256": "d" * 64}
    extra = {"source_model": str(src), "prismaquant_plan": "smoke", "inherits": {}}

    export_checkpoint_streaming(src, tmp_path / "plain", plan, grid=K2,
                                copy_aux=False, extra_config=extra)
    export_checkpoint_streaming(
        src, tmp_path / "aware", plan, grid=K2, copy_aux=False, extra_config=extra,
        activation=ActivationSource(hessians=hessians, provenance=provenance))

    plain = (tmp_path / "plain" / "model.safetensors").read_bytes()
    aware = (tmp_path / "aware" / "model.safetensors").read_bytes()
    assert plain != aware, "the Hessian was recorded but never reached the encoder"
    assert len(plain) == len(aware), "both levers are encoder-side; the wire is the same"
    config = json.loads((tmp_path / "aware" / "tessera_config.json").read_text())
    assert config["activation_aware"]["hessian"]["fit_ids_sha256"] == "d" * 64
    assert config["activation_aware"]["refit_objective"]["lut16"] == "h^1.0", (
        "the 4-bit route's own plane is what this export used, and the config "
        "must say which objective wrote the bytes")
    assert json.loads(
        (tmp_path / "plain" / "tessera_config.json").read_text())["activation_aware"] is None


@cuda
def test_the_refit_objective_reaches_the_lut_plane_and_changes_the_bytes(tmp_path):
    """A named objective the encoder ignores is the silent no-op this task closed.

    ``refit_objective`` is the one setting whose *value* the LUT plane's new
    code reads: ``"hessian"`` hands ``for_unit`` the whole matrix, ``"h^ALPHA"``
    a diagonal power, ``"plain"`` nothing at all.  If any two of those reached
    the same bytes, an arm named after one of them would be measuring another
    -- so assert all three are distinct, at the same length, and that the
    config records which one wrote them.
    """
    import json

    from safetensors.torch import save_file

    from tessera.export import ActivationSource, export_checkpoint_streaming

    g = torch.Generator().manual_seed(34)
    tensors = {"model.layers.0.mlp.gate_proj.weight":
               torch.randn(64, 256, generator=g).bfloat16()}
    src = tmp_path / "src"
    src.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    plan = {name: CAP for name in tensors}
    hessians = {ActivationSource.unit_name(n): _hessian(cols=256, seed=7, device="cpu")
                for n in tensors}
    provenance = {"source": "wikitext-2 train", "text_sha256": "e" * 64,
                  "fit_tokens": 16384, "fit_ids_sha256": "f" * 64}

    blobs = {}
    for objective in ("plain", "h^1.0", "hessian"):
        out = tmp_path / objective.replace("^", "")
        export_checkpoint_streaming(
            src, out, plan, grid=K2, copy_aux=False,
            extra_config={"source_model": str(src), "inherits": {}},
            activation=ActivationSource(hessians=hessians, provenance=provenance,
                                        refit_objective=objective))
        blobs[objective] = (out / "model.safetensors").read_bytes()
        config = json.loads((out / "tessera_config.json").read_text())
        assert config["activation_aware"]["refit_objective"] == objective

    assert len(set(blobs.values())) == 3, (
        "two refit objectives wrote the same bytes: one of them is not being read")
    assert len({len(b) for b in blobs.values()}) == 1, "the refit never changes the wire"


def test_a_per_plane_objective_needs_the_plane_and_will_not_guess():
    """The map's two measured answers disagree, so a caller that does not say
    which plane it is encoding on cannot be handed either of them.

    This is the silent-no-op shape again, one level up: a ``for_unit`` that
    quietly picked (say) the CHANNEL entry would encode a LUT-plane unit at an
    objective measured somewhere else and raise nothing.
    """
    from tessera.export import DEFAULT_REFIT_OBJECTIVE, ActivationSource
    from tessera.manifest import ScalePlaneKind

    provenance = {"text_sha256": "a" * 64, "fit_tokens": 1024,
                  "fit_ids_sha256": "b" * 64}
    source = ActivationSource(hessians={}, provenance=provenance)
    assert source.refit_objective == DEFAULT_REFIT_OBJECTIVE

    with pytest.raises(GrammarError, match="scale plane"):
        source.objective_for(None)
    assert source.objective_for(ScalePlaneKind.LUT) == "h^1.0"
    assert source.objective_for(ScalePlaneKind.CHANNEL) == "hessian"
    assert source.objective_for(ScalePlaneKind.S6B) == "plain"

    # A plain string still means every plane, and needs no plane to resolve.
    assert ActivationSource(hessians={}, provenance=provenance,
                            refit_objective="plain").objective_for(None) == "plain"

    # A map that does not name the plane in use refuses rather than falling
    # back to another plane's measurement.
    partial = ActivationSource(hessians={}, provenance=provenance,
                               refit_objective={"channel": "hessian"})
    with pytest.raises(GrammarError, match="lut16"):
        partial.objective_for(ScalePlaneKind.LUT)


@cuda
def test_the_default_on_this_plane_is_the_arm_that_was_measured(tmp_path):
    """The exported default and the measured arm must be the same bytes.

    The served arm below was exported with an explicit ``--refit-metric
    h^1.0``, and the per-plane default was set to that value afterwards. If the
    default resolved to anything else on the LUT plane -- the CHANNEL entry,
    a fallback, the old single constant -- then the recipe the exporter applies
    by default would not be the recipe the receipt measured, and nothing would
    say so.
    """
    from safetensors.torch import save_file

    from tessera.export import ActivationSource, export_checkpoint_streaming

    g = torch.Generator().manual_seed(55)
    tensors = {"model.layers.0.mlp.gate_proj.weight":
               torch.randn(64, 256, generator=g).bfloat16()}
    src = tmp_path / "src"
    src.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    plan = {name: CAP for name in tensors}
    hessians = {ActivationSource.unit_name(n): _hessian(cols=256, seed=3, device="cpu")
                for n in tensors}
    provenance = {"text_sha256": "a" * 64, "fit_tokens": 16384,
                  "fit_ids_sha256": "b" * 64}
    extra = {"source_model": str(src), "inherits": {}}

    out = {}
    for tag, objective in (("default", None), ("explicit", "h^1.0"),
                           ("full_h", "hessian")):
        kw = {} if objective is None else {"refit_objective": objective}
        d = tmp_path / tag
        export_checkpoint_streaming(
            src, d, plan, grid=K2, copy_aux=False, extra_config=extra,
            activation=ActivationSource(hessians=hessians, provenance=provenance, **kw))
        out[tag] = (d / "model.safetensors").read_bytes()

    assert out["default"] == out["explicit"], (
        "the LUT plane's default is not the h^1.0 arm the receipt measured")
    assert out["default"] != out["full_h"], (
        "the two objectives reach the same bytes, so the plane is not reading one")


# --------------------------------------------------------------------------
# Issue #35: the Gauss-Seidel sweep of the same block-scale refit.
#
# It is a different OPTIMISER for one objective, opt-in and encoder-side.  So
# the tests are of three kinds: it must still be monotone in the metric it
# claims to descend, it must be a strictly better STEP than the Jacobi one it
# replaces (that is the whole hypothesis, and it is checkable before any wire
# is involved), and it must refuse every context where it would silently be
# the parallel step under another name.
# --------------------------------------------------------------------------


def _step_record(fixture, H, gauss_seidel):
    """The refit's pre-landing point: step + line search, no table.

    Read out of the refit's own diagnostic sink rather than reimplemented,
    because a test that recomputes the arithmetic it is checking tests the
    test.
    """
    from tessera.encode import refit_diagnostics
    work, units, half, table, index, eff, glob = fixture
    with refit_diagnostics() as diag:
        _refit_scales_lut(work, units, half, table, index, eff, glob,
                          metric=H, gauss_seidel=gauss_seidel)
    assert len(diag) == 1
    return diag[0]


def test_gauss_seidel_is_a_strictly_better_step_than_jacobi():
    """The hypothesis, isolated from the table and the trellis.

    Under a coupled metric the sequential sweep solves each block against a
    residual that already carries the blocks before it, so the point the STEP
    reaches is better than the parallel step's.  ``stepped`` is that point and
    is the only leg #35 is about: what happens after it -- the non-positive
    revert and the sixteen-entry landing -- is common to both optimisers and
    can give the whole gain back, which is a measurement, not a bug in this
    assertion.
    """
    fixture = _lut_fixture(seed=11, cols=128)
    H = _hessian(cols=128, seed=11, device="cpu", coupling=4.0)
    jac = _step_record(fixture, H, False)
    gs = _step_record(fixture, H, True)
    assert jac["before"] == pytest.approx(gs["before"])
    assert gs["stepped"] < jac["stepped"] < gs["before"]


def test_gauss_seidel_is_monotone_under_its_own_metric():
    """The guard is the same guard: the plane the unit has is a candidate."""
    work, units, half, table, index, eff, glob = _lut_fixture(seed=12, cols=128)
    for H in (_hessian(cols=128, seed=12, device="cpu"),
              _hessian(cols=128, seed=13, device="cpu", coupling=4.0)):
        _, _, new_eff = _refit_scales_lut(
            work, units, half, table, index, eff, glob,
            metric=H, gauss_seidel=True)
        assert _cost(work, units, new_eff, half, H) <= _cost(work, units, eff, half, H) + 1e-9


def test_gauss_seidel_off_is_the_refit_that_was_there():
    """The byte claim, at the function it is made about."""
    work, units, half, table, index, eff, glob = _lut_fixture(seed=14, cols=128)
    H = _hessian(cols=128, seed=14, device="cpu", coupling=4.0)
    a = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H)
    b = _refit_scales_lut(work, units, half, table, index, eff, glob,
                          metric=H, gauss_seidel=False)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


@cuda
def test_the_sweep_reaches_the_bytes_and_leaves_the_wire_alone():
    """An arm named after a setting that changed no byte is the failure this
    catches; a wire that changed length would be the other one."""
    w = _weights(seed=15).bfloat16()
    H = _hessian(seed=15)
    L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
    jac = encode_linear(w, grid=K2, q256=CAP, name="x",
                        ldl=L, ldl_block=32, refit_metric=H).blob
    gs = encode_linear(w, grid=K2, q256=CAP, name="x", ldl=L, ldl_block=32,
                       refit_metric=H, refit_gauss_seidel=True).blob
    assert jac != gs
    assert len(jac) == len(gs), "the wire is unchanged"


@cuda
def test_the_sweep_is_refused_where_it_would_be_the_parallel_step():
    """Every context where a sequential sweep computes the parallel numbers,
    refused by its own message rather than accepted as a silent no-op."""
    w = _weights(seed=16).bfloat16()
    H = _hessian(seed=16)
    h = H.diagonal() / H.diagonal().mean()
    with pytest.raises(GrammarError, match="without refit_metric"):
        encode_linear(w, grid=K2, q256=CAP, name="x", refit_gauss_seidel=True)
    with pytest.raises(GrammarError, match="separable"):
        encode_linear(w, grid=K2, q256=CAP, name="x",
                      refit_metric=h, refit_gauss_seidel=True)
    with pytest.raises(GrammarError, match="LUT plane"):
        encode_linear(w, grid=K2, q256=CAP, name="x", refit_metric=H,
                      scale_plane=ScalePlaneKind.CHANNEL, refit_gauss_seidel=True)


def test_the_diagnostic_sink_is_off_unless_asked_for():
    """It is a measurement instrument, not a side effect of encoding."""
    from tessera import encode as enc
    from tessera.encode import refit_diagnostics
    assert enc._REFIT_DIAG is None
    with refit_diagnostics() as diag:
        assert enc._REFIT_DIAG is diag
    assert enc._REFIT_DIAG is None


# --------------------------------------------------------------------------
# Issue #50: the coupled landing.  The refit's landing puts each block at the
# table entry nearest its OWN continuous target; once its neighbours have
# landed somewhere else that entry is no longer the block's conditional
# minimiser under a coupled metric.  ``coupled_landing`` re-assigns until it
# is.  Tests pin the property (no block left first-order improvable), the
# monotone cost, the off-is-identical claim, the sink's separation, and the
# refusals where the landing is already exact.
# --------------------------------------------------------------------------


def _remaining_gain(work, units, half, eff, table_bytes, glob, H):
    """``(sum of the exact per-block decreases still available, cost)``.

    A block's error with the others held is ``A_b (c_b - s_b)^2 + const``;
    the decrease from moving it to the table entry nearest ``s_b`` is exact.
    Summed over blocks it is the first-order money the landing left on the
    table -- zero, up to fp32 resolution on the cost, for a landing that is
    conditionally optimal block by block.  Blocks whose conditional optimum
    is non-positive are excluded: the refit holds them on purpose.
    """
    from tessera.encode import _lut_values
    rows, cols = work.shape
    nb = cols // half
    W, U = work.float(), units.float()
    Ub = U.reshape(rows, nb, half)
    C = eff.reshape(rows, nb)
    T = _lut_values(table_bytes, glob)
    Hd = torch.diagonal(H.reshape(nb, half, nb, half), dim1=0, dim2=2).permute(2, 0, 1)
    A = torch.einsum("rbi,bij,rbj->rb", Ub, Hd, Ub)
    E = W - C.repeat_interleave(half, dim=1) * U
    G = E @ H
    s = C + (G.reshape(rows, nb, half) * Ub).sum(dim=2) / A.clamp_min(1e-30)
    best = T[(s[:, :, None] - T[None, None, :]).abs().argmin(dim=2)]
    gain = A * ((C - s) ** 2 - (best - s) ** 2)
    # Blocks whose conditional optimum is non-positive are held by the refit's
    # revert rule, so they are not money the landing left on the table.
    gain = torch.where((A > 0) & (s > 0), gain, torch.zeros_like(gain)).clamp_min(0.0)
    return float(gain.sum()), float((G * E).sum())


def test_the_coupled_landing_leaves_no_block_first_order_improvable():
    """The property, on the fixture the sweep tests use.

    The separable landing leaves blocks that would lower the quadratic by
    moving one table entry over; the coupled landing leaves none beyond what
    fp32 can resolve on the cost -- the exact stop rule the sweep runs to.
    Asserting both halves is what makes this a test of the landing and not of
    the fixture.
    """
    work, units, half, table, index, eff, glob = _lut_fixture(seed=11, cols=128)
    H = _hessian(cols=128, seed=11, device="cpu", coupling=4.0)
    eps = torch.finfo(torch.float32).eps
    for gs in (False, True):
        sep = _refit_scales_lut(work, units, half, table, index, eff, glob,
                                metric=H, gauss_seidel=gs)
        cou = _refit_scales_lut(work, units, half, table, index, eff, glob,
                                metric=H, gauss_seidel=gs, coupled_landing=True)
        left_sep, cost_sep = _remaining_gain(work, units, half, sep[2], sep[0], glob, H)
        left_cou, cost_cou = _remaining_gain(work, units, half, cou[2], cou[0], glob, H)
        assert left_sep > 1e-3 * cost_sep, (
            "the separable landing left nothing to re-assign on this fixture; "
            "it cannot show the coupled one does anything")
        assert left_cou <= eps * cost_cou, (left_cou, cost_cou)
        assert cost_cou < cost_sep


def test_the_coupled_landing_is_monotone_and_below_the_separable_one():
    """Fourth candidate, at or below the third, never above the plane the
    unit already had -- the guard is the same guard."""
    for seed in (12, 13, 14):
        work, units, half, table, index, eff, glob = _lut_fixture(seed=seed, cols=128)
        H = _hessian(cols=128, seed=seed, device="cpu", coupling=4.0)
        before = _cost(work, units, eff, half, H)
        sep = _cost(work, units, _refit_scales_lut(
            work, units, half, table, index, eff, glob, metric=H, gauss_seidel=True)[2], half, H)
        cou = _cost(work, units, _refit_scales_lut(
            work, units, half, table, index, eff, glob, metric=H, gauss_seidel=True,
            coupled_landing=True)[2], half, H)
        assert cou <= sep + 1e-9 * sep
        assert sep <= before + 1e-9 * before


def test_the_coupled_landing_off_is_the_refit_that_was_there():
    """The byte claim, at the function it is made about, for both sweep orders."""
    work, units, half, table, index, eff, glob = _lut_fixture(seed=15, cols=128)
    H = _hessian(cols=128, seed=15, device="cpu", coupling=4.0)
    for gs in (False, True):
        a = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H, gauss_seidel=gs)
        b = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H, gauss_seidel=gs,
                              coupled_landing=False)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_the_coupled_landing_is_a_no_op_under_a_separable_metric():
    """Under a 1-D metric nearest-in-linear already is each block's
    conditional minimiser, so the coupled sweep has nothing to move and the
    function returns the separable landing bit for bit.  ``encode_unit``
    refuses the flag there for exactly this reason; this pins the reason."""
    from tessera.encode import _refit_scales_lut_metric
    work, units, half, table, index, eff, glob = _lut_fixture(seed=16, cols=128)
    h = torch.rand(128) + 0.1
    a = _refit_scales_lut_metric(work, units, half, table, index, eff, glob, h)
    b = _refit_scales_lut_metric(work, units, half, table, index, eff, glob, h, coupled_landing=True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


@cuda
def test_the_trailing_mode_lands_the_last_refit_only():
    """``"trailing"`` leaves every inner pass's refit exactly as the plain
    full-H arm computed it and re-lands the last one; ``True`` re-lands them
    all.  Read from the sink, which records ``coupled`` only where the sweep
    ran, and from the inner passes matching the plain arm's to the float."""
    from tessera.encode import refit_diagnostics
    w = _weights(seed=20).bfloat16()
    H = _hessian(seed=20, coupling=4.0)
    kw = dict(grid=K2, q256=CAP, name="x", scale_refit=3, refit_metric=H,
              refit_gauss_seidel=True)
    runs = {}
    for mode in (False, "trailing", True):
        with refit_diagnostics() as sink:
            encode_linear(w, refit_coupled_landing=mode, **kw)
        runs[mode] = [dict(r) for r in sink]
    assert [len(r) for r in runs.values()] == [3, 3, 3]
    assert ["coupled" in r for r in runs[False]] == [False, False, False]
    assert ["coupled" in r for r in runs["trailing"]] == [False, False, True]
    assert ["coupled" in r for r in runs[True]] == [True, True, True]
    # The trailing run's inner passes ARE the plain run's inner passes, and
    # its last pass starts where the plain run's last pass started.
    for a, b in zip(runs[False][:2], runs["trailing"][:2]):
        assert a["before"] == b["before"] and a["landed"] == b["landed"]
    assert runs["trailing"][2]["before"] == runs[False][2]["before"]
    assert runs["trailing"][2]["landed"] == runs[False][2]["landed"]
    assert runs["trailing"][2]["coupled"] <= runs["trailing"][2]["landed"]
    with pytest.raises(GrammarError, match="expected False, True"):
        encode_linear(w, refit_coupled_landing="always", **kw)


def test_the_sink_keeps_the_separable_landing_next_to_the_coupled_one():
    """With the option on, the record's ``landed`` is still the separable
    landing and ``coupled`` is the plane returned -- so the instrument that
    found the loss can still read it after the fix.  With it off, no
    ``coupled`` key at all."""
    from tessera.encode import refit_diagnostics
    work, units, half, table, index, eff, glob = _lut_fixture(seed=17, cols=128)
    H = _hessian(cols=128, seed=17, device="cpu", coupling=4.0)
    with refit_diagnostics() as off:
        _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H, gauss_seidel=True)
    with refit_diagnostics() as on:
        out = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H,
                                gauss_seidel=True, coupled_landing=True)
    assert "coupled" not in off[0]
    assert on[0]["landed"] == pytest.approx(off[0]["landed"])
    assert on[0]["coupled"] == pytest.approx(_cost(work, units, out[2], half, H), rel=1e-5)
    assert on[0]["coupled"] <= on[0]["landed"]
    assert on[0]["coupled_moves"] > 0 and on[0]["coupled_sweeps"] >= 1
    assert on[0]["candidate"].endswith("+coupled")


@cuda
def test_the_coupled_landing_reaches_the_bytes_and_leaves_the_wire_alone():
    w = _weights(seed=18).bfloat16()
    H = _hessian(seed=18)
    L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
    gs = encode_linear(w, grid=K2, q256=CAP, name="x", ldl=L, ldl_block=32,
                       refit_metric=H, refit_gauss_seidel=True).blob
    cou = encode_linear(w, grid=K2, q256=CAP, name="x", ldl=L, ldl_block=32,
                        refit_metric=H, refit_gauss_seidel=True, refit_coupled_landing=True).blob
    assert gs != cou
    assert len(gs) == len(cou), "the wire is unchanged"


@cuda
def test_the_coupled_landing_is_refused_where_the_landing_is_already_exact():
    w = _weights(seed=19).bfloat16()
    H = _hessian(seed=19)
    h = H.diagonal() / H.diagonal().mean()
    with pytest.raises(GrammarError, match="without refit_metric"):
        encode_linear(w, grid=K2, q256=CAP, name="x", refit_coupled_landing=True)
    with pytest.raises(GrammarError, match="separable"):
        encode_linear(w, grid=K2, q256=CAP, name="x", refit_metric=h, refit_coupled_landing=True)
    with pytest.raises(GrammarError, match="no table"):
        encode_linear(w, grid=K2, q256=CAP, name="x", refit_metric=H,
                      scale_plane=ScalePlaneKind.CHANNEL, refit_coupled_landing=True)


# The landing ceiling (issue #50): what the sixteen-entry table costs.
# --------------------------------------------------------------------------


def test_the_landing_context_is_off_unless_asked_for():
    """It is a measurement instrument, not a side effect of encoding."""
    from tessera import encode as enc
    from tessera.encode import lut_landing

    assert enc._LUT_LANDING == "table" and enc._LUT_LANDING_SINK is None
    with lut_landing("none") as sink:
        assert enc._LUT_LANDING == "none" and enc._LUT_LANDING_SINK is sink
    assert enc._LUT_LANDING == "table" and enc._LUT_LANDING_SINK is None
    with pytest.raises(GrammarError, match="unknown LUT landing mode"):
        with lut_landing("continuous"):
            pass


def test_the_default_landing_is_the_refit_that_was_there():
    """The byte claim, at the function it is made about.

    A ceiling read is only believable if the arm it is read against is the
    encode that was already there, so ``"table"`` inside the context has to be
    identical to no context at all -- table bytes, index and effective scales.
    """
    from tessera.encode import lut_landing

    work, units, half, table, index, eff, glob = _lut_fixture(seed=50, cols=128)
    H = _hessian(cols=128, seed=50, device="cpu", coupling=4.0)
    a = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H)
    with lut_landing("table"):
        b = _refit_scales_lut(work, units, half, table, index, eff, glob, metric=H)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


@pytest.mark.parametrize("mode", ["grid", "none"])
def test_a_free_landing_is_monotone_and_is_not_the_table(mode):
    """The two properties a ceiling arm has to have.

    **Monotone**: the plane the unit holds is still a candidate and the
    candidate is still scored on the full quadratic, so a free landing can
    never raise the metric's own error -- the same guarantee the wire arm has,
    which is what makes the pair a matched comparison rather than two encoders.

    **Not the table**: a mode that silently landed on the sixteen entries
    anyway would report a ceiling of exactly zero and look like a closed issue.
    """
    from tessera.encode import lut_landing

    work, units, half, table, index, eff, glob = _lut_fixture(seed=51, cols=128)
    H = _hessian(cols=128, seed=51, device="cpu", coupling=4.0)
    with lut_landing(mode):
        _, _, free = _refit_scales_lut(work, units, half, table, index, eff, glob,
                                       metric=H)
    assert _cost(work, units, free, half, H) <= _cost(work, units, eff, half, H) + 1e-9
    _, _, landed = _refit_scales_lut(work, units, half, table, index, eff, glob,
                                     metric=H)
    assert not torch.equal(free, landed)
    on_table = torch.isin(free, _lut_values(table, glob))
    if mode == "none":
        assert not bool(on_table.all()), "a continuous landing that lands on the table"
    else:
        # ``grid`` is E4M3 by construction, so what it must NOT be is confined
        # to the sixteen entries the unit's table holds.
        assert not bool(on_table.all())


@cuda
def test_a_free_landing_is_refused_where_it_would_mean_something_else():
    """Every context where the ceiling would be read off the wrong quantity,
    refused by its own message rather than accepted as a silent no-op."""
    from tessera.encode import lut_landing

    w = _weights(seed=52).bfloat16()
    H = _hessian(seed=52)
    h = H.diagonal() / H.diagonal().mean()
    with lut_landing("none"):
        with pytest.raises(GrammarError, match="LUT plane"):
            encode_linear(w, grid=K2, q256=CAP, name="x", refit_metric=H,
                          scale_plane=ScalePlaneKind.CHANNEL)
        with pytest.raises(GrammarError, match="without refit_metric"):
            encode_linear(w, grid=K2, q256=CAP, name="x")
        with pytest.raises(GrammarError, match="scale_refit=0"):
            encode_linear(w, grid=K2, q256=CAP, name="x", refit_metric=h,
                          scale_refit=0)


@cuda
def test_the_sink_is_the_wire_on_a_table_arm_and_is_not_on_a_free_one():
    """The identity that licenses reading a ceiling arm off the sink at all.

    A ``table`` arm is the wire, so the reconstruction the encoder hands back
    must be the one ``stock_dequant`` recovers from the bytes.  A free landing
    builds a unit whose scale plane is NOT what it held, so the same comparison
    has to disagree -- if it agreed, the sink would be recording the landed
    plane and the ceiling would read zero for the wrong reason.
    """
    from tessera.encode import lut_landing
    from tessera.export import encode_linear_planes as _elp
    from tessera.stock import materialize_stock, stock_dequant

    w = _weights(seed=53).bfloat16()
    H = _hessian(seed=53)
    L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
    kw = dict(grid=K2, q256=CAP, name="x", verify=False, ldl=L, ldl_block=32,
              refit_metric=H)
    got = {}
    for mode in ("table", "none"):
        with lut_landing(mode) as sink:
            _, unit, forests = _elp(w, **kw)
        wire = stock_dequant(materialize_stock(unit, forests, DEFAULT_CODE))
        got[mode] = (sink, wire.to(sink["work_reconstruction"].device).float())
    sink, wire = got["table"]
    assert sink["serialisable"] is True
    assert torch.allclose(wire, sink["work_reconstruction"], rtol=0, atol=1e-6)
    sink, wire = got["none"]
    assert sink["serialisable"] is False
    assert not torch.allclose(wire, sink["work_reconstruction"], rtol=0, atol=1e-6)
