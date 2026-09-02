"""The merge guard, field by field -- and the proof it is not vacuous.

`merge_tessera_parts.py` refuses to join two halves of a shard-split export
that were not encoded identically.  It has been vacuous before: eight of the
thirteen names it compared existed nowhere in the config the exporter wrote,
so they compared ``None`` to ``None`` and passed, including the two that catch
encoder drift.  A guard is only worth what its weakest name is worth, so the
first test here is not a mismatch test at all -- it asserts that every dotted
path the guard compares **resolves in a config the exporter actually wrote**.
Every mismatch test below it would pass against a field nobody writes; only
that one says the guard has teeth.

The rest give each activation-aware field its own failing case, because a
refusal that says "the configs disagree" is not actionable when thirty fields
are compared, and a single combined case cannot tell a working comparison from
one that short-circuits on the first field.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.errors import GrammarError
from tessera.export import (
    ActivationSource,
    encode_settings_from_config,
    export_checkpoint,
)

_spec = importlib.util.spec_from_file_location(
    "merge_tessera_parts",
    Path(__file__).resolve().parents[1] / "experiments" / "merge_tessera_parts.py",
)
merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge)

ROWS, COLS, Q256 = 64, 256, 1024

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="encoder is a GPU job")


def _provenance(**over):
    """What ``capture_h_full.py`` writes; the three identity fields plus extras."""
    return dict({
        "source": "wikitext-2 train",
        "text_sha256": "a" * 64,
        "fit_tokens": 131072,
        "fit_ids_sha256": "b" * 64,
        "model": "Qwen/Qwen3-0.6B",
        "seqlen": 512,
    }, **over)


def _hessians(names, seed=0, cols=COLS):
    """One PSD input Hessian per unit name (tensor name minus ``.weight``)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name in names:
        x = torch.randn(4 * cols, cols, generator=g)
        out[ActivationSource.unit_name(name)] = (x.T @ x) / (4 * cols)
    return out


def _tensors():
    g = torch.Generator().manual_seed(7)
    return {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            ROWS, COLS, generator=g).bfloat16(),
        "model.layers.0.mlp.up_proj.weight": torch.randn(
            ROWS, COLS, generator=g).bfloat16(),
    }


#: ``source_model``, ``prismaquant_plan`` and ``inherits`` are in the guard's
#: ``SHARED`` but are **not** written by ``_write_config``: they arrive through
#: ``extra_config`` from ``experiments/export_glm53_tessera.py``, the driver
#: whose shard-split parts this merge exists for.  A part written by a bare
#: ``export_checkpoint`` therefore has no ``source_model`` and the guard refuses
#: it -- correctly, since it cannot say what those parts are halves *of*.  The
#: fixtures build the config the way that driver does, so the anti-vacuity test
#: below checks the guard against the config shape it actually guards.
DRIVER_EXTRA = {"source_model": "/mnt/shared/models/GLM-5.3-Flash-BF16",
                "prismaquant_plan": "everything-eligible",
                "inherits": {"vision": "bf16 passthrough"}}


def _export(out_dir, *, activation=None, tensors=None):
    tensors = _tensors() if tensors is None else tensors
    plan = {name: Q256 for name in tensors}
    export_checkpoint(tensors, plan, out_dir, grid=E4M3_GRID, activation=activation,
                      extra_config=DRIVER_EXTRA)
    return json.loads((Path(out_dir) / "tessera_config.json").read_text())


def _source(tensors=None, **over):
    tensors = _tensors() if tensors is None else tensors
    return ActivationSource(
        hessians=_hessians(tensors), provenance=_provenance(), **over)


def _set(config, path, value):
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


# --------------------------------------------------------------------------
# The anti-vacuity test.  Everything below it depends on this one.
# --------------------------------------------------------------------------


@cuda
def test_every_guarded_field_is_written_by_the_exporter(tmp_path):
    """No name the guard compares may be absent from a real exported config.

    An absent name compares ``_MISSING`` to ``_MISSING`` across two parts and
    passes -- which is exactly how eight of thirteen went unenforced.  This
    reads a config the exporter wrote, activation-aware so the new block is
    populated, and refuses to let any guarded path resolve to nothing.
    """
    config = _export(tmp_path, activation=_source())
    guarded = (merge.SHARED + merge.SHARED_WHEN_WRITTEN + merge.SHARED_ACTIVATION)
    unwritten = [f for f in guarded if merge.dotted(config, f) is merge._MISSING]
    assert not unwritten, (
        f"the merge guard compares {unwritten}, which the exporter does not "
        f"write: those comparisons pass vacuously")


@cuda
def test_the_config_names_which_hessian_shaped_the_bytes(tmp_path):
    """An auditor can read the capture's identity off the artifact."""
    config = _export(tmp_path, activation=_source())
    block = config["activation_aware"]
    assert block["hessian"]["text_sha256"] == "a" * 64
    assert block["hessian"]["fit_ids_sha256"] == "b" * 64
    assert block["hessian"]["fit_tokens"] == 131072
    assert (block["ldlq_sigma"], block["ldlq_block"]) == (1.0, 32)
    assert block["refit_objective"] == "hessian"


# --------------------------------------------------------------------------
# One failing case per field.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aware_config(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("encoder is a GPU job")
    return _export(tmp_path_factory.mktemp("aware"), activation=_source())


@pytest.fixture(scope="module")
def plain_config(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("encoder is a GPU job")
    return _export(tmp_path_factory.mktemp("plain"))


@pytest.mark.parametrize("field,other", [
    ("activation_aware.ldlq_sigma", 3.0),
    ("activation_aware.ldlq_block", 128),
    ("activation_aware.refit_objective", "plain"),
    ("activation_aware.refit_reach_floor", True),
    ("activation_aware.hessian.text_sha256", "c" * 64),
    ("activation_aware.hessian.fit_tokens", 65536),
    ("activation_aware.hessian.fit_ids_sha256", "d" * 64),
])
def test_each_activation_field_refuses_on_its_own(aware_config, field, other):
    """Every field gets its own case: a combined one cannot tell a working
    comparison from one that short-circuits on the first difference."""
    a = copy.deepcopy(aware_config)
    b = copy.deepcopy(aware_config)
    assert merge.dotted(b, field) != other, f"{field} sentinel equals the real value"
    _set(b, field, other)
    with pytest.raises(SystemExit, match=field):
        merge.check_configs([("partA", a), ("partB", b)])
    merge.check_configs([("partA", a), ("partB", copy.deepcopy(aware_config))])


def test_matching_activation_aware_parts_merge(aware_config):
    base = merge.check_configs([("partA", copy.deepcopy(aware_config)),
                                ("partB", copy.deepcopy(aware_config))])
    assert base["activation_aware"]["hessian"]["fit_ids_sha256"] == "b" * 64


def test_weights_only_parts_merge(plain_config):
    """``activation_aware: null`` in every part is a consistent weights-only
    merge, not a missing field."""
    assert plain_config["activation_aware"] is None
    merge.check_configs([("partA", copy.deepcopy(plain_config)),
                         ("partB", copy.deepcopy(plain_config))])


def test_an_aware_half_and_a_plain_half_are_two_artifacts(aware_config, plain_config):
    with pytest.raises(SystemExit, match="never saw"):
        merge.check_configs([("partA", copy.deepcopy(aware_config)),
                             ("partB", copy.deepcopy(plain_config))])
    with pytest.raises(SystemExit, match="never saw"):
        merge.check_configs([("partA", copy.deepcopy(plain_config)),
                             ("partB", copy.deepcopy(aware_config))])


def test_a_part_from_an_older_exporter_is_refused(aware_config):
    """No ``activation_aware`` key at all means the part cannot say whether a
    Hessian shaped it -- which is not the same as saying none did."""
    old = copy.deepcopy(aware_config)
    del old["activation_aware"]
    with pytest.raises(SystemExit, match="different exporters"):
        merge.check_configs([("partA", copy.deepcopy(aware_config)), ("partB", old)])
    with pytest.raises(SystemExit, match="different exporters"):
        merge.check_configs([("partA", old), ("partB", copy.deepcopy(aware_config))])


def test_no_part_carrying_the_key_is_noted_not_refused(plain_config, capsys):
    """Pre-2026-09-02 parts predate the field; they merge, and say so."""
    old = copy.deepcopy(plain_config)
    del old["activation_aware"]
    merge.check_configs([("partA", old), ("partB", copy.deepcopy(old))])
    assert "unrecorded" in capsys.readouterr().out


@pytest.mark.parametrize("field", merge.SHARED_ACTIVATION)
def test_an_activation_block_missing_a_guarded_field_is_refused(aware_config, field):
    """A field absent from *every* part compares equal and passes; the guard
    refuses on the first part instead, the same way it does for ``SHARED``."""
    a, b = copy.deepcopy(aware_config), copy.deepcopy(aware_config)
    for config in (a, b):
        node, parts = config, field.split(".")
        for part in parts[:-1]:
            node = node[part]
        del node[parts[-1]]
    with pytest.raises(SystemExit, match="cannot certify"):
        merge.check_configs([("partA", a), ("partB", b)])


# --------------------------------------------------------------------------
# The producer side: what may become an ActivationSource at all.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["text_sha256", "fit_tokens", "fit_ids_sha256"])
def test_a_hessian_without_an_identity_is_refused(field):
    """``payload.get("provenance")`` yields ``None`` on an old capture, and two
    ``None`` identities compare equal -- the vacuity, one level up."""
    prov = _provenance()
    del prov[field]
    with pytest.raises(GrammarError, match=field):
        ActivationSource(hessians={}, provenance=prov)
    with pytest.raises(GrammarError, match="provenance"):
        ActivationSource(hessians={}, provenance=None)


def test_bad_activation_settings_are_refused():
    with pytest.raises(GrammarError, match="must be positive"):
        ActivationSource(hessians={}, provenance=_provenance(), ldlq_sigma=0.0)
    with pytest.raises(GrammarError, match="at least one column"):
        ActivationSource(hessians={}, provenance=_provenance(), ldlq_block=0)
    with pytest.raises(GrammarError, match="unknown refit objective"):
        ActivationSource(hessians={}, provenance=_provenance(),
                         refit_objective="output_mse")
    with pytest.raises(GrammarError, match="diagonal power"):
        ActivationSource(hessians={}, provenance=_provenance(), refit_objective="h^x")


@cuda
def test_a_missing_hessian_refuses_rather_than_encoding_weights_only(tmp_path):
    """A wrong key renders the unit RTN and raises nothing (see the memory note
    ``render-activations-keyed-by-qname``); the export must refuse instead."""
    tensors = _tensors()
    partial = ActivationSource(
        hessians={k: v for k, v in _hessians(tensors).items() if "gate" in k},
        provenance=_provenance())
    with pytest.raises(GrammarError, match="up_proj"):
        _export(tmp_path, activation=partial, tensors=tensors)


@cuda
def test_a_hessian_of_the_wrong_width_is_refused(tmp_path):
    tensors = _tensors()
    wrong = ActivationSource(hessians=_hessians(tensors, cols=128),
                             provenance=_provenance())
    with pytest.raises(GrammarError, match="H is"):
        _export(tmp_path, activation=wrong, tensors=tensors)


@cuda
def test_the_hessian_key_is_the_tensor_name_minus_one_weight(tmp_path):
    """There is one key transform and no fallback: a dict keyed by tensor name
    would partially match a dict keyed by module name and encode the rest
    weights-only."""
    assert ActivationSource.unit_name("m.0.up_proj.weight") == "m.0.up_proj"
    tensors = _tensors()
    by_tensor = ActivationSource(
        hessians={n: h for n, h in zip(tensors, _hessians(tensors).values())},
        provenance=_provenance())
    with pytest.raises(GrammarError, match="minus one"):
        _export(tmp_path, activation=by_tensor, tensors=tensors)


@cuda
def test_an_activation_aware_export_writes_different_bytes(tmp_path):
    """The library path must SHAPE the bytes, not merely record that it meant to.

    Every other test here proves the plumbing is called and that the config
    says so.  None of them would notice a refactor that accepted an
    ``ActivationSource``, wrote its block, and dropped the kwargs on the floor
    -- the artifact would then claim a Hessian it never used, which is worse
    than not having the path at all.  The bytes are the claim: same wire, same
    length, different codes.
    """
    tensors = _tensors()
    _export(tmp_path / "plain", tensors=tensors)
    _export(tmp_path / "aware", tensors=tensors, activation=_source(tensors))
    plain = (tmp_path / "plain" / "model.safetensors").read_bytes()
    aware = (tmp_path / "aware" / "model.safetensors").read_bytes()
    assert plain != aware, "the Hessian was recorded but never reached the encoder"
    assert len(plain) == len(aware), "both levers are encoder-side; the wire is the same"


@cuda
def test_the_streaming_export_takes_the_same_source(tmp_path):
    """The 100B path is the streaming one; it must not be the path that
    silently encodes weights-only."""
    from safetensors.torch import save_file

    from tessera.export import export_checkpoint_streaming

    src = tmp_path / "src"
    src.mkdir()
    tensors = _tensors()
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    plan = {name: Q256 for name in tensors}

    export_checkpoint_streaming(
        src, tmp_path / "aware", plan, grid=E4M3_GRID, copy_aux=False,
        activation=_source(tensors), extra_config=DRIVER_EXTRA)
    config = json.loads((tmp_path / "aware" / "tessera_config.json").read_text())
    assert config["activation_aware"]["hessian"]["fit_ids_sha256"] == "b" * 64

    partial = ActivationSource(
        hessians={k: v for k, v in _hessians(tensors).items() if "gate" in k},
        provenance=_provenance())
    with pytest.raises(GrammarError, match="up_proj"):
        export_checkpoint_streaming(src, tmp_path / "part", plan, grid=E4M3_GRID,
                                    copy_aux=False, activation=partial)


@cuda
def test_a_plane_without_a_metric_refit_refuses_one_rather_than_dropping_it(tmp_path):
    """Every plane that does not read an argument refuses it.

    The CHANNEL plane's row-scale refit and the LUT plane's per-16 block-scale
    refit both implement ``refit_metric``; S6b does not, and the reach floor is
    a CHANNEL mechanism on any plane.  Dropped silently, either would let an
    activation-aware export ship weights-only bytes and raise nothing, which is
    the failure this plumbing exists to prevent."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import encode_linear
    from tessera.manifest import ScalePlaneKind

    K2 = tuple_grid(E2M1_GRID, 2)
    g = torch.Generator().manual_seed(11)
    w = torch.randn(ROWS, COLS, generator=g).bfloat16()
    H = next(iter(_hessians(["x.weight"]).values()))
    with pytest.raises(GrammarError, match="S6b"):
        encode_linear(w, grid=K2, q256=896, name="x", refit_metric=H,
                      scale_plane=ScalePlaneKind.S6B)
    with pytest.raises(GrammarError, match="CHANNEL-plane mechanism"):
        encode_linear(w, grid=K2, q256=896, name="x", refit_reach_floor=True)
    with pytest.raises(GrammarError, match="scale_refit=0 runs none"):
        encode_linear(w, grid=E4M3_GRID, q256=Q256, name="x",
                      scale_refit=0, refit_metric=H)
    # ...and the LUT plane, which now implements it, does not refuse.
    encode_linear(w, grid=K2, q256=896, name="x", refit_metric=H)


# --------------------------------------------------------------------------
# Replay.
# --------------------------------------------------------------------------


def test_replaying_an_activation_aware_config_is_refused(aware_config):
    """The bytes depend on a Hessian, and no dict of encode keywords carries
    one: handing back the weights-only settings would replay a different
    artifact and raise nothing."""
    with pytest.raises(GrammarError, match="activation-aware"):
        encode_settings_from_config(aware_config)


def test_a_weights_only_config_still_replays(plain_config):
    settings = encode_settings_from_config(plain_config, q256=Q256)
    assert "ldl" not in settings and "refit_metric" not in settings
    assert settings["scale_refit"] > 0
