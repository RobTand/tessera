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


#: ``source_model``, ``prismaquant_plan`` and ``inherits`` are **not** written
#: by ``_write_config``: they arrive through ``extra_config`` from
#: ``experiments/export_glm53_tessera.py``, the driver whose shard-split parts
#: this merge exists for.  The guard used to *require* them, so a part written
#: by a bare ``export_checkpoint`` was refused with a message blaming the
#: exporter (tessera#137); it now finds them by subtracting the exporter's own
#: fields and compares them when every part has one.  The fixtures build the
#: config the way that driver does, so the anti-vacuity test below checks the
#: guard against the config shape it actually guards.
DRIVER_EXTRA = {"source_model": "/models/GLM-5.3-Flash-BF16",
                "prismaquant_plan": "everything-eligible",
                "inherits": {"vision": "bf16 passthrough"}}


def _export(out_dir, *, activation=None, tensors=None, extra=DRIVER_EXTRA):
    tensors = _tensors() if tensors is None else tensors
    plan = {name: Q256 for name in tensors}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    export_checkpoint(tensors, plan, out_dir, grid=E4M3_GRID, activation=activation,
                      extra_config=extra)
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
    guarded = (merge.shared_fields() + merge.SHARED_WHEN_WRITTEN
               + merge.SHARED_ACTIVATION)
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
    # The objective is per scale plane, so the whole map travels: two parts
    # built with different maps are two artifacts even where they happen to
    # agree on the plane one of them used.
    assert block["refit_objective"] == {"channel": "hessian", "lut16": "h^1.0",
                                        "s6b": "plain"}
    # The trailing leg defaults to the uniform schedule: unset is today's
    # encode, byte for byte, and the config says so as null, not by omission
    # (an omitted key would compare vacuous across parts).
    assert block["refit_objective_trailing"] is None
    assert block["refit_gauss_seidel"] is False


# --------------------------------------------------------------------------
# The guard's field set is the exporter's, not a roster beside it (#137).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bare_config(tmp_path_factory):
    """What a plain ``export_checkpoint`` writes: no driver ``extra_config``.

    CPU, and small: the question is which fields the config carries, so the
    trellis is incidental and a GPU gate would leave the answer unmeasured on
    the population that actually runs these files.
    """
    return _export(tmp_path_factory.mktemp("bare"), extra=None)


def test_two_bare_export_parts_are_a_legal_merge(bare_config):
    """The defect: the guard required three fields no exporter writes.

    ``source_model``, ``prismaquant_plan`` and ``inherits`` come from
    ``export_glm53_tessera.py``'s ``extra_config``.  Requiring them refused
    every pair of parts a plain ``export_checkpoint_streaming`` produced --
    and the refusal told the operator the exporter had stopped writing fields
    it never wrote (tessera#137).
    """
    base = merge.check_configs([("partA", copy.deepcopy(bare_config)),
                                ("partB", copy.deepcopy(bare_config))])
    assert base["quant_method"] == "tessera"


def test_the_guard_requires_only_fields_the_exporter_writes(bare_config):
    """Every guarded name resolves in a config the exporter alone wrote.

    The complement of the anti-vacuity test above: that one says no guarded
    field is missing from a *driver's* config, this one says none of them is a
    driver's field.  Both are needed -- a name absent from every part compares
    ``_MISSING`` to ``_MISSING`` and passes, and a name only one driver writes
    refuses a merge that is fine.
    """
    unwritten = [f for f in merge.shared_fields()
                 if merge.dotted(bare_config, f) is merge._MISSING]
    assert not unwritten, (
        f"the merge guard requires {unwritten}, which the exporter does not "
        f"write: no plain export can be merged")


def test_the_guarded_set_is_exactly_the_encoding_half_of_the_config(bare_config):
    """Derived, not restated: the two lists are one list.

    ``tessera.export`` checks its declaration against the dict it just built on
    every export, so this binds the guard to the bytes rather than to a tuple
    someone maintained.
    """
    from tessera.export import (CONFIG_ACTIVATION_FIELD, CONFIG_PER_PART_FIELDS,
                                _config_leaves)

    stop = frozenset(CONFIG_PER_PART_FIELDS) | {CONFIG_ACTIVATION_FIELD}
    encoding = _config_leaves(bare_config, stop) - stop
    assert set(merge.shared_fields()) | set(merge.SHARED_WHEN_WRITTEN) == encoding


def test_a_driver_field_only_one_part_carries_is_refused(bare_config, tmp_path):
    """Two drivers, two artifacts -- found by subtraction, named by nobody."""
    driven = _export(tmp_path / "driven")
    with pytest.raises(SystemExit, match="different drivers"):
        merge.check_configs([("partA", driven), ("partB", copy.deepcopy(bare_config))])


def test_parts_that_disagree_on_a_driver_field_are_refused(tmp_path):
    """A driver field every part carries is compared like any other."""
    a = _export(tmp_path / "a")
    b = _export(tmp_path / "b")
    b["source_model"] = "/models/somewhere-else"
    with pytest.raises(SystemExit, match="source_model"):
        merge.check_configs([("partA", a), ("partB", b)])


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
    ("activation_aware.refit_objective_trailing", "hessian"),
    ("activation_aware.refit_reach_floor", True),
    ("activation_aware.refit_gauss_seidel", True),
    ("activation_aware.hessian.text_sha256", "c" * 64),
    ("activation_aware.hessian.fit_tokens", 65536),
    ("activation_aware.hessian.fit_ids_sha256", "d" * 64),
    ("activation_aware.hessian.model", "/models/somewhere-else"),
    ("activation_aware.hessian.seqlen", 1024),
    ("activation_aware.hessian.capture_sha256", "e" * 64),
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


def test_parts_that_disagree_only_off_their_own_plane_still_refuse(aware_config):
    """The map is compared whole, not at the plane this export happened to use.

    Both parts here encode on the CHANNEL plane and agree about it; they
    disagree about ``lut16``. Comparing only the plane in use would call them
    the same artifact -- and the next export from the same source, on a grid
    whose recipe is LUT, would silently be a third one.
    """
    a = copy.deepcopy(aware_config)
    b = copy.deepcopy(aware_config)
    assert b["activation_aware"]["refit_objective"]["channel"] == "hessian"
    b["activation_aware"]["refit_objective"] = dict(
        b["activation_aware"]["refit_objective"], lut16="hessian")
    with pytest.raises(SystemExit, match="refit_objective"):
        merge.check_configs([("partA", a), ("partB", b)])


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


def test_parts_cut_by_different_encoders_are_refused(aware_config):
    """The field tessera#101 added: same settings, same profile ids, different
    encoder.  Nothing else in the config can tell these two apart, which is
    exactly the merge that went through once already (tessera#78)."""
    a = copy.deepcopy(aware_config)
    b = copy.deepcopy(aware_config)
    assert len(a["encoder_fixture_id"]) == 64
    b["encoder_fixture_id"] = "f" * 64
    with pytest.raises(SystemExit, match="encoder_fixture_id"):
        merge.check_configs([("partA", a), ("partB", b)])
    merge.check_configs([("partA", a), ("partB", copy.deepcopy(aware_config))])


def test_one_part_predating_the_encoder_identity_is_refused(aware_config):
    """Written by some parts and not others means two exporters, and the older
    one cannot say which encoder cut it."""
    old = copy.deepcopy(aware_config)
    del old["encoder_fixture_id"]
    with pytest.raises(SystemExit, match="different exporters"):
        merge.check_configs([("partA", copy.deepcopy(aware_config)), ("partB", old)])


def test_no_part_carrying_the_encoder_identity_is_noted(aware_config, capsys):
    """Parts that both predate the field merge, and the note says what went
    unchecked rather than reading as a clean comparison."""
    a = copy.deepcopy(aware_config)
    b = copy.deepcopy(aware_config)
    del a["encoder_fixture_id"], b["encoder_fixture_id"]
    merge.check_configs([("partA", a), ("partB", b)])
    assert "whether one encoder cut both parts is unrecorded" in capsys.readouterr().out


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


# --------------------------------------------------------------------------
# Tessera#103: the trailing refit leg as an exportable setting.
#
# CPU-only by construction: everything here stops at the kwargs and the
# config, so it runs where the encoder cannot.  The GPU half -- that a set
# trailing leg reaches the bytes -- is the encoder's own contract
# (``test_refit_trailing.py``); the exporter half is that the leg is named,
# recorded and compared.
# --------------------------------------------------------------------------


def _trailing_hessians(cols=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * cols, cols, generator=g)
    return {"u": (x.T @ x) / (4 * cols)}


def test_the_trailing_leg_round_trips_through_config_block():
    """What the exporter records is what the guard compares, verbatim."""
    source = ActivationSource(
        hessians=_trailing_hessians(), provenance=_provenance(),
        refit_objective_trailing={"channel": "hessian", "lut16": "h^1.0",
                                  "s6b": "plain"},
        refit_gauss_seidel=True)
    block = source.config_block()
    assert block["refit_objective_trailing"] == {"channel": "hessian",
                                                "lut16": "h^1.0",
                                                "s6b": "plain"}
    assert block["refit_gauss_seidel"] is True
    # Serialisable as written: a ``MappingProxyType`` left unconverted would
    # fail the config write, not the guard.
    json.loads(json.dumps(block))


def test_the_trailing_leg_defaults_to_the_encode_that_was_there():
    """Unset is the uniform schedule: null on the config, nothing new for the
    encoder but the sweep flag at its own default."""
    from tessera.manifest import ScalePlaneKind

    source = ActivationSource(hessians=_trailing_hessians(),
                              provenance=_provenance())
    assert source.refit_objective_trailing is None
    assert source.trailing_objective_for(ScalePlaneKind.CHANNEL) is None
    assert source.config_block()["refit_objective_trailing"] is None
    assert source.config_block()["refit_gauss_seidel"] is False
    kwargs = source.for_unit("u.weight", 32,
                             scale_plane=ScalePlaneKind.CHANNEL)
    assert kwargs.get("refit_metric_trailing") is None
    assert kwargs["refit_gauss_seidel"] is False


def test_a_set_trailing_leg_reaches_the_encoder_kwargs():
    """The exporter half of the bytes claim: the leg ``for_unit`` hands over
    is the one the config records."""
    from tessera.manifest import ScalePlaneKind

    H = _trailing_hessians()["u"]
    source = ActivationSource(
        hessians={"u": H}, provenance=_provenance(),
        refit_objective_trailing="hessian", refit_gauss_seidel=True)
    kwargs = source.for_unit("u.weight", 32,
                             scale_plane=ScalePlaneKind.CHANNEL)
    assert torch.equal(kwargs["refit_metric_trailing"], H)
    assert kwargs["refit_gauss_seidel"] is True
    # A diagonal trailing power is the same spelling as the base leg's.
    diagonal = ActivationSource(
        hessians={"u": H}, provenance=_provenance(),
        refit_objective_trailing={"channel": "h^2.0"})
    kw = diagonal.for_unit("u.weight", 32,
                           scale_plane=ScalePlaneKind.CHANNEL)
    h = H.diagonal()
    assert torch.equal(kw["refit_metric_trailing"], (h / h.mean()).pow(2.0))


def test_a_bad_trailing_leg_is_refused_by_name():
    """The trailing leg is checked like the base one, and the refusal says
    which leg it is."""
    with pytest.raises(GrammarError, match="unknown refit objective"):
        ActivationSource(hessians={}, provenance=_provenance(),
                         refit_objective_trailing="output_mse")
    with pytest.raises(GrammarError, match="refit_objective_trailing"):
        ActivationSource(hessians={}, provenance=_provenance(),
                         refit_objective_trailing={"nope": "hessian"})
    with pytest.raises(GrammarError, match="refit_objective_trailing"):
        ActivationSource(hessians={}, provenance=_provenance(),
                         refit_objective_trailing={})
    from tessera.manifest import ScalePlaneKind

    partial = ActivationSource(hessians={}, provenance=_provenance(),
                               refit_objective_trailing={"channel": "hessian"})
    with pytest.raises(GrammarError, match="lut16"):
        partial.trailing_objective_for(ScalePlaneKind.LUT)


@pytest.fixture
def guard_template(bare_config):
    """A config carrying every dotted path the guard compares, so a refusal
    below names a compared field and never a missing one.

    Assembled from a real export, a driver's ``extra_config`` and an
    ``ActivationSource``'s own block -- never typed out.  A hand-written
    template restates today's field list and goes stale the day one is added:
    this one lacked ``scale.sigma``, and would have said the guard passed over
    a field it compares (AGENTS.md principle 3).
    """
    config = copy.deepcopy(bare_config)
    config.update(copy.deepcopy(DRIVER_EXTRA))
    config["activation_aware"] = ActivationSource(
        hessians={}, provenance=_provenance()).config_block()
    return config


@pytest.mark.parametrize("field,other", [
    ("activation_aware.refit_objective_trailing", "hessian"),
    ("activation_aware.refit_objective_trailing",
     {"channel": "hessian", "lut16": "hessian", "s6b": "plain"}),
    ("activation_aware.refit_gauss_seidel", True),
])
def test_the_guard_refuses_a_trailing_leg_it_did_not_compare_before(guard_template,
                                                                     field, other):
    """Two parts encoded under different trailing schedules are two
    artifacts: the guard compares the leg, it does not pass over it."""
    a = copy.deepcopy(guard_template)
    b = copy.deepcopy(guard_template)
    assert merge.dotted(b, field) != other
    _set(b, field, other)
    with pytest.raises(SystemExit, match=field):
        merge.check_configs([("partA", a), ("partB", b)])
    merge.check_configs([("partA", a), ("partB", copy.deepcopy(guard_template))])


def test_a_plain_trailing_leg_over_a_weighted_base_is_refused():
    """#103: the config must not be able to describe bytes that never existed.

    ``encode_unit`` reads ``refit_metric_trailing=None`` as "use the base
    leg's metric", so there is no way to ask for an *un-weighted* last pass
    over a weighted base.  Before this refusal, ``for_unit`` passed no kwarg
    for a trailing objective of ``"plain"`` and the encode weighted every pass
    -- while ``config_block`` recorded ``refit_objective_trailing="plain"``.
    """
    import torch

    from tessera.export import ActivationSource, GrammarError
    from tessera.manifest import ScalePlaneKind

    source = ActivationSource(
        hessians={"unit": torch.eye(4)},
        provenance={"text_sha256": "a", "fit_tokens": 1, "fit_ids_sha256": "b"},
        refit_objective="hessian",
        refit_objective_trailing="plain",
        ldlq_sigma=None,           # a 4-column fixture, no LDL block to fit
    )
    with pytest.raises(GrammarError) as caught:
        source.for_unit("unit.weight", 4, scale_plane=ScalePlaneKind.CHANNEL)
    message = str(caught.value)
    assert "'plain'" in message and "hessian" in message


def test_a_plain_trailing_leg_over_a_plain_base_is_legal():
    """The S6B row of the measured default: both legs plain is one encode.

    Nothing is un-weighted here, so the config and the bytes agree by both
    readings -- and refusing it would refuse the shipped default.
    """
    import torch

    from tessera.export import ActivationSource, DEFAULT_REFIT_OBJECTIVE
    from tessera.manifest import ScalePlaneKind

    assert DEFAULT_REFIT_OBJECTIVE["s6b"] == "plain"
    source = ActivationSource(
        hessians={"unit": torch.eye(4)},
        provenance={"text_sha256": "a", "fit_tokens": 1, "fit_ids_sha256": "b"},
        refit_objective=DEFAULT_REFIT_OBJECTIVE,
        refit_objective_trailing={"s6b": "plain"},
        ldlq_sigma=None,
    )
    kwargs = source.for_unit("unit.weight", 4, scale_plane=ScalePlaneKind.S6B)
    assert "refit_metric" not in kwargs
    assert "refit_metric_trailing" not in kwargs


# --------------------------------------------------------------------------
# Tessera#214: calibration identity is the capture, not three token fields.
#
# CPU-only by construction, like the trailing-leg block above: everything here
# stops at ``config_block`` and the guard, and ``guard_template`` carries the
# exact block ``_write_config`` embeds.
# --------------------------------------------------------------------------


def test_parts_at_different_capture_seqlens_are_two_artifacts(guard_template):
    """The #214 repro: ``capture_h_full.py`` reshapes ONE token prefix to
    ``[-1, seqlen]`` and hashes the flat ids (no shape), so captures at 512
    and 1024 share ``text_sha256``, ``fit_tokens`` and ``fit_ids_sha256``
    while running different attention contexts and producing different H.
    Three token fields cannot certify identical Hessians; the guard reads the
    sequence layout beside them and refuses by name."""
    a = copy.deepcopy(guard_template)
    b = copy.deepcopy(guard_template)
    assert merge.dotted(a, "activation_aware.hessian.seqlen") == 512
    _set(b, "activation_aware.hessian.seqlen", 1024)
    with pytest.raises(SystemExit, match="activation_aware.hessian.seqlen"):
        merge.check_configs([("part512", a), ("part1024", b)])
    merge.check_configs([("part512", a), ("part512b", copy.deepcopy(guard_template))])


def test_two_captures_with_the_same_token_metadata_are_told_apart_by_content(guard_template):
    """The actual capture content is sealed and compared: two H populations
    that agree on every recorded token field are still two Hessians, and no
    metadata coincidence may certify them identical."""
    a = copy.deepcopy(guard_template)
    b = copy.deepcopy(guard_template)
    a["activation_aware"] = ActivationSource(
        hessians=_hessians(["u.weight"], seed=1, cols=8),
        provenance=_provenance()).config_block()
    b["activation_aware"] = ActivationSource(
        hessians=_hessians(["u.weight"], seed=2, cols=8),
        provenance=_provenance()).config_block()
    with pytest.raises(SystemExit, match="capture_sha256"):
        merge.check_configs([("partA", a), ("partB", b)])


def test_an_identical_capture_across_disjoint_parts_still_merges(guard_template):
    block = ActivationSource(hessians=_hessians(["u.weight"], seed=1, cols=8),
                             provenance=_provenance()).config_block()
    a = copy.deepcopy(guard_template)
    b = copy.deepcopy(guard_template)
    a["activation_aware"] = copy.deepcopy(block)
    b["activation_aware"] = copy.deepcopy(block)
    base = merge.check_configs([("partA", a), ("partB", b)])
    assert base["activation_aware"]["hessian"]["capture_sha256"] == \
        block["hessian"]["capture_sha256"]


def test_the_capture_seal_is_stamped_by_the_source_itself():
    """``config_block`` is what ``_write_config`` embeds, so the seal and its
    context ride in every activation-aware config.  Content, model identity
    and sequence layout each move the seal; nothing else stamps it."""
    def source_with(seed=1, **over):
        return ActivationSource(hessians=_hessians(["u.weight"], seed=seed, cols=8),
                                provenance=_provenance(**over))

    block = source_with().config_block()
    seal = block["hessian"]["capture_sha256"]
    assert isinstance(seal, str) and len(seal) == 64
    assert block["hessian"]["model"] == "Qwen/Qwen3-0.6B"
    assert block["hessian"]["seqlen"] == 512
    json.loads(json.dumps(block))
    assert source_with().config_block()["hessian"]["capture_sha256"] == seal
    assert source_with(seed=2).config_block()["hessian"]["capture_sha256"] != seal
    assert source_with(seqlen=1024).config_block()["hessian"]["capture_sha256"] != seal
    assert source_with(model="/models/somewhere-else").config_block()[
        "hessian"]["capture_sha256"] != seal


# --------------------------------------------------------------------------
# The sweep is per scale plane (tessera#107).
# --------------------------------------------------------------------------


def test_the_sweep_map_serves_each_plane_the_answer_it_was_given():
    """#107: one bool cannot be true on a checkpoint that spans two planes.

    ``encode_linear`` refuses ``refit_gauss_seidel`` off the LUT plane rather
    than ignoring it -- deliberately, since a sequential sweep with nothing to
    sweep is the parallel step under another name, and a silently dropped
    encoder setting is how an export ships bytes its config misdescribes.  So
    a bare ``True`` is unreachable on any checkpoint whose units do not all
    sit on the LUT plane: ``experiments/export_tessera_serving.py`` reads
    ``(grid, q256)`` per member and encodes GLM's attention on E4M3/CHANNEL
    beside its experts on E2M1x2/LUT16, from ONE ``ActivationSource``, so the
    first CHANNEL unit refuses.  The map is the one value both halves carry.
    """
    from tessera.manifest import ScalePlaneKind

    source = _source(refit_gauss_seidel={"lut16": True})
    assert source.gauss_seidel_for(ScalePlaneKind.LUT) is True
    # Not a refusal, and not another plane's answer: the field's own default.
    # This is the asymmetry with ``objective_for``, which refuses an unnamed
    # plane because no objective is neutral -- ``False`` here is.
    assert source.gauss_seidel_for(ScalePlaneKind.CHANNEL) is False
    assert source.gauss_seidel_for(ScalePlaneKind.S6B) is False


def test_a_bare_sweep_flag_still_means_every_plane():
    """The map must not have quietly turned ``True`` into "the LUT plane only".

    A bare bool is the *whole model's* setting, so on a CHANNEL unit it still
    reaches the encoder as ``True`` and still refuses there.  Resolving it to
    ``False`` instead would be the silent no-op the refusal exists to prevent,
    dressed up as a fix.
    """
    from tessera.manifest import ScalePlaneKind

    source = _source(refit_gauss_seidel=True)
    assert source.gauss_seidel_for(ScalePlaneKind.CHANNEL) is True
    assert source.gauss_seidel_for(ScalePlaneKind.LUT) is True
    assert source.gauss_seidel_for(None) is True


def test_the_sweep_map_is_what_for_unit_hands_the_encoder():
    """The plane the encode resolved decides the kwarg, unit by unit."""
    from tessera.manifest import ScalePlaneKind

    tensors = _tensors()
    source = _source(tensors, refit_objective="hessian",
                     refit_gauss_seidel={"lut16": True})
    name = next(iter(tensors))
    on_lut = source.for_unit(name, COLS, scale_plane=ScalePlaneKind.LUT)
    on_channel = source.for_unit(name, COLS, scale_plane=ScalePlaneKind.CHANNEL)
    assert on_lut["refit_gauss_seidel"] is True
    assert on_channel["refit_gauss_seidel"] is False
    # The rest of the recipe is untouched: the same Hessian shapes both.
    assert torch.equal(on_lut["refit_metric"], on_channel["refit_metric"])


def test_the_sweep_map_round_trips_through_config_block():
    """The config records which planes swept, not one bool that cannot be true.

    The whole map travels, not the value at the plane this part used: the
    merge guard compares this field across parts, and a part that recorded
    only its own plane's answer would compare unequal to a part that recorded
    only the other's while both were built from one setting.
    """
    from tessera.manifest import ScalePlaneKind

    block = _source(refit_gauss_seidel={"lut16": True}).config_block()
    assert block["refit_gauss_seidel"] == {"lut16": True}
    json.loads(json.dumps(block))       # a MappingProxyType would fail the write
    # The default is unchanged, and it is still a bool: an artifact that never
    # asked for the sweep records what it always recorded.
    assert _source().config_block()["refit_gauss_seidel"] is False
    assert _source().gauss_seidel_for(ScalePlaneKind.LUT) is False


def test_a_sweep_map_needs_the_plane_it_is_asked_about():
    """A per-plane setting resolved without a plane would silently be off."""
    source = _source(refit_gauss_seidel={"lut16": True})
    with pytest.raises(GrammarError, match="scale plane"):
        source.gauss_seidel_for(None)
    with pytest.raises(GrammarError, match="refit_gauss_seidel"):
        source.gauss_seidel_for(None)


@pytest.mark.parametrize("bad,message", [
    ({"nope": True}, "are not scale planes"),
    ({"lut16": 1}, "on or off on a plane"),
    ({"lut16": "yes"}, "on or off on a plane"),
    ("yes", "must be a bool or a plane"),
    (1, "must be a bool or a plane"),
    (None, "must be a bool or a plane"),
    ({}, "turns the sweep on nowhere"),
    ({"lut16": False}, "turns the sweep on nowhere"),
    ({"lut16": False, "channel": False}, "turns the sweep on nowhere"),
])
def test_a_bad_sweep_setting_is_refused_by_name(bad, message):
    """A map with no plane set to ``True`` is refused rather than accepted as
    a second spelling of ``False``: it encodes the same bytes while comparing
    unequal to ``False`` at the merge guard, so two parts spelling one default
    two ways would refuse over bytes that agree."""
    with pytest.raises(GrammarError, match=message):
        ActivationSource(hessians={}, provenance=_provenance(),
                         refit_gauss_seidel=bad)


def test_the_guard_refuses_two_parts_that_swept_different_planes(guard_template):
    """The map is compared like the objective map: whole, and by value."""
    a = copy.deepcopy(guard_template)
    b = copy.deepcopy(guard_template)
    _set(a, "activation_aware.refit_gauss_seidel", {"lut16": True})
    _set(b, "activation_aware.refit_gauss_seidel", {"channel": True})
    with pytest.raises(SystemExit, match="refit_gauss_seidel"):
        merge.check_configs([("partA", a), ("partB", b)])
    merge.check_configs([("partA", a), ("partB", copy.deepcopy(a))])
