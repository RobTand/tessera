"""The trailing refit's objective is a schedule, not a second encoder (issue #75).

Issue #75's screen showed the full-H objective's worth sitting almost entirely
in the trailing refit -- but the swap arm ran five refits against the control's
four.  The fair pair is ``T R_h T R_h T R_h T R_H`` against
``T R_h T R_h T R_h T R_h``: same pass count, only the trailing refit's
objective swapped.  The encoder could not express that: one ``refit_metric``
ran on every pass.  ``refit_metric_trailing`` is the missing half -- inner
passes minimise ``refit_metric``, the trailing refit minimises
``refit_metric_trailing``, ``None`` is the encode that was already there.

Opt-in like ``refit_gauss_seidel``, and no wire change: the schedule is a
schedule, so the bytes move and their length does not.  It is refused wherever
it would be silently ignored.  It stopped being encoder-only with tessera#103,
which gave ``ActivationSource`` a ``refit_objective_trailing`` field, put it in
the exported config and made the merge guard compare it; the exporter flag that
sets it is pinned at the bottom of this file.  CPU-only throughout: the
schedule is visible in bytes at 32x64 without a GPU.
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


# --------------------------------------------------------------------------
# the trailing objective has to reach an ARTIFACT, not only ``encode_linear``
# --------------------------------------------------------------------------
#
# ``ActivationSource.refit_objective_trailing`` landed with tessera#103, and
# the merge guard compares it.  But an exporter still had no way to *set* it:
# ``experiments/export_tessera_serving.py`` plumbed ``--refit-metric`` alone,
# so #75's B-Jac arm -- the only one that clears the GLM gate -- could be
# encoded in a measurement script and not in a checkpoint.  These two pin the
# flag end to end: it must reach the recorded config AND the bytes.

import importlib.util
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parents[1]
_BODY = "model.language_model.layers.0."
_ROUTED = _BODY + "mlp.down_proj"
_UNROUTED = _BODY + "self_attn.o_proj"


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving",
        _ROOT / "experiments" / "export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """A two-Linear GLM checkpoint; only ``down_proj`` is routed (#99)."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    g = torch.Generator(device="cpu").manual_seed(31)
    save_file({_ROUTED + ".weight":
               (torch.randn(32, 64, generator=g) * 0.02).to(torch.bfloat16),
               _UNROUTED + ".weight":
               (torch.randn(32, 64, generator=g) * 0.02).to(torch.bfloat16)},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": 64, "moe_intermediate_size": 32},
    }))
    return src


def _tiny_capture(tmp_path: Path) -> Path:
    """A ``capture_h_full.py``-shaped payload for the one routed unit."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "h.pt"
    torch.save({"H": {_ROUTED: _hessian(seed=33)},
                "provenance": {"text_sha256": "0" * 64, "fit_tokens": 4096,
                               "fit_ids_sha256": "1" * 64}}, str(path))
    return path


def _export(tmp_path: Path, monkeypatch, name: str, *extra: str) -> Path:
    out = tmp_path / name
    monkeypatch.setattr(
        "sys.argv",
        ["export_tessera_serving.py", str(_tiny_checkpoint(tmp_path)), str(out),
         "--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--no-verify",
         "--passthrough-unrouted", "--hessian", str(_tiny_capture(tmp_path)),
         "--refit-metric", "h^1.0", *extra])
    _exporter().main()
    return out


def _wire(out: Path) -> bytes:
    with safe_open(str(out / "model.safetensors"), framework="np") as handle:
        return np.asarray(handle.get_tensor(_ROUTED + ".wire_bytes")).tobytes()


def test_the_export_flag_records_the_trailing_objective(tmp_path, monkeypatch):
    """The block an auditor reads must name what ran.

    Without this the flag would be a shell argument: the bytes would carry a
    trailing full-H refit and ``activation_aware`` would say ``None``, which
    is the config-lying-about-the-bytes failure tessera#103 exists to stop.
    Checked on the TWIN's manifest as well as the wire's, because a served KL
    is read off the twin and quoted long afterwards -- two arms of an A/B whose
    one difference is an ``activation_aware`` field must be tellable apart from
    the artifacts that produced the numbers (tessera#60).
    """
    def blocks(out: Path) -> "tuple[dict, dict]":
        wire = json.loads((out / "tessera_serving_manifest.json").read_text())
        twin = json.loads(
            (out.parent / "twin" / "tessera_stock_twin_manifest.json").read_text())
        return wire["activation_aware"], twin["activation_aware"]

    control_wire, control_twin = blocks(
        _export(tmp_path / "a", monkeypatch, "out",
                "--stock-twin", str(tmp_path / "a" / "twin")))
    swapped_wire, swapped_twin = blocks(
        _export(tmp_path / "b", monkeypatch, "out",
                "--stock-twin", str(tmp_path / "b" / "twin"),
                "--refit-metric-trailing", "hessian"))
    for control, swapped in ((control_wire, swapped_wire),
                             (control_twin, swapped_twin)):
        assert control["refit_objective"] == swapped["refit_objective"] == "h^1.0"
        assert control["refit_objective_trailing"] is None   # the old encode
        assert swapped["refit_objective_trailing"] == "hessian"


def test_the_export_flag_reaches_the_bytes_and_moves_no_others(tmp_path, monkeypatch):
    """#75's matched pair, at the exporter: same wire length, different wire.

    A flag that changed the config and not the bytes would be worse than
    absent -- it would label an artifact with a recipe it was not encoded
    under.  The equal length is the pair's other half: the schedule is a
    schedule, so the wire does not move and a served A/B is at identical
    bytes.
    """
    control = _wire(_export(tmp_path / "a", monkeypatch, "out"))
    swapped = _wire(_export(tmp_path / "b", monkeypatch, "out",
                            "--refit-metric-trailing", "hessian"))
    assert control != swapped, "the trailing objective reached no byte"
    assert len(control) == len(swapped), "the schedule changed the wire"
