"""The encoder identity: derived from behaviour, and pinned as a rule.

``encoder_profile_id`` is input-only by decision, so an encoder change moves
the bytes at an unchanged id and nothing on the artifact says so (tessera#101,
tessera#78).  ``tessera.encoder_identity`` closes that by encoding a fixed
fixture set and hashing the result.

What these tests pin is the **rule**, never the digest.  A test asserting
``encoder_fixture_id() == "b2ce..."`` would have to be edited every time the
encoder legitimately moves, which is precisely the hand-maintained discipline
the issue refuses -- and a discipline that fails does so silently.  So instead:
the identity moves when behaviour moves, the untagged spelling writes no byte,
the coverage set is derived from the modules that own it, and the field
round-trips.
"""

import dataclasses
import hashlib

import pytest
import torch

from tessera import encoder_identity as ei
from tessera.alphabet import E4M3_GRID, SERIALISABLE_GRIDS
from tessera.container import SCHEMA_MINOR, SCHEMA_MINORS_READ, parse
from tessera.errors import ManifestError, TesseraError
from tessera.export import encode_linear, encode_linear_planes, recipe_table
from tessera.manifest import Manifest
from tessera.unit_artifact import _resolve_fixture_id, build_unit_artifact


@pytest.fixture(scope="module")
def identity():
    """The digest, computed once: the fixture encodes are the expensive part."""
    return ei.encoder_fixture_id()


# --------------------------------------------------------------------------
# It is derived from behaviour.  This is the whole claim.
# --------------------------------------------------------------------------


def test_the_identity_moves_when_the_encoder_moves(identity, monkeypatch):
    """Perturb what the encoder *does* and the digest follows, with nothing
    declared anywhere.

    The perturbation is one fp16 ulp on every stored CHANNEL row word -- the
    smallest byte-moving change the plane admits, and the same class as the
    reach-floor landing this identity was built to let land (tessera#87).
    """
    import tessera.scale_channel as sc

    plain = sc.land_channel_scale

    def bumped(scale, global_scale):
        stored, _effective = plain(scale, global_scale)
        stored = (stored.contiguous().view(torch.int16) + 1).view(torch.float16)
        return stored, stored.float() * global_scale

    monkeypatch.setattr(sc, "land_channel_scale", bumped)
    monkeypatch.setattr(ei, "_MEMO", [])
    assert ei.encoder_fixture_id() != identity


def test_identity_sees_upward_landing_of_an_unraised_boundary(
        identity, monkeypatch):
    """Issue #116: the fixture reaches #115's exact residual branch.

    Arm B changes only an unraised CHANNEL row whose nearest fp16 word landed
    below ``amax / reach``.  If that policy can move artifact bytes, the
    behaviour-derived identity must distinguish them from current arm A.
    """
    import tessera.scale_channel as sc

    current = sc.initial_channel_scale

    def upward_for_every_below_floor(work, sigma, reach=None):
        stored, effective, global_scale = current(work, sigma, reach=reach)
        if reach is None:
            return stored, effective, global_scale
        floor = work.float().abs().amax(dim=1) / float(reach)
        stored, effective = sc._bump_below_floor(
            stored, effective, floor, global_scale,
        )
        return stored, effective, global_scale

    monkeypatch.setattr(sc, "initial_channel_scale", upward_for_every_below_floor)
    monkeypatch.setattr(ei, "_MEMO", [])
    assert ei.encoder_fixture_id() != identity


def test_reach_floor_moves_identity_and_refuses_an_untagged_resume(identity):
    """Issue #87 is a byte mover, so #101's derived gate must see it.

    ``UNTAGGED_ENCODER_ID`` is the historical encoder before the upward reach
    landing.  The fixed encoder must differ without a maintained version bump,
    and a cached artifact carrying that absent spelling must be refused.
    """

    class Untagged:
        encoder_fixture_id = None

    assert identity != ei.UNTAGGED_ENCODER_ID
    assert not ei.resumable(Untagged())


def test_an_unread_slot_does_not_move_the_identity(identity, monkeypatch):
    """The other half of the claim: it moves on behaviour and on nothing else.

    Rebinding a name the encoder never calls is the in-process stand-in for
    the docstring edit measured on the branch -- both change the module and
    neither changes what the encoder emits.
    """
    import tessera.scale_channel as sc

    monkeypatch.setattr(sc, "__doc__", "rewritten by a test; moves no byte")
    monkeypatch.setattr(sc, "_TS101_UNUSED", object(), raising=False)
    monkeypatch.setattr(ei, "_MEMO", [])
    assert ei.encoder_fixture_id() == identity


def test_the_fixture_weights_do_not_ride_torch_rng():
    """A torch upgrade must not re-label every artifact on disk.

    ``torch.randn``'s stream is not promised across versions; the fixture is
    built from ``random.Random`` instead, whose ``random()`` is.  Seeding torch
    differently on either side of the call proves the weights do not read it.
    """
    torch.manual_seed(1)
    first = ei._fixture_weight()
    torch.manual_seed(999)
    assert torch.equal(first, ei._fixture_weight())
    assert first.device.type == "cpu"


def test_unraised_boundary_fixture_reaches_the_exact_residual_branch():
    """The #116 witness is unraised, but its nearest word is below reach."""
    import tessera.scale_channel as sc

    weight, sigma, reach = ei._unraised_boundary_fixture()
    w = weight.float()
    rms = w.pow(2).mean(dim=1).sqrt()
    amax = w.abs().amax(dim=1)
    over = amax * sigma > reach * rms
    floor = amax / reach
    stored, effective, global_scale = sc.initial_channel_scale(
        w, sigma, reach=reach,
    )
    residual = (~over) & (effective < floor)

    assert torch.equal(torch.nonzero(residual).flatten(), torch.tensor([0]))
    assert not bool(over[0])
    assert float(global_scale) == 1.0
    assert float(stored[0]) == 1.0


def test_every_compatibility_witness_is_neutral_for_the_current_encoder(identity):
    """Coverage added after #101 must not relabel bytes it did not change.

    Every witness, not the first one: a set membership test would pass with a
    wrong constant on any case it did not reach, and a wrong constant is the
    one failure this mechanism has -- it silently turns a neutral witness into
    an Option-A re-base, relabelling every artifact on disk.
    """
    witnesses = [
        case for case in ei.fixtures()
        if case.compatibility_baseline is not None
    ]
    assert witnesses
    for case in witnesses:
        encoded = ei._encode_fixture(case)
        computed = hashlib.sha256(encoded).hexdigest()
        assert computed == case.compatibility_baseline, (
            f"{case.label} encodes to {computed}, not its recorded "
            f"compatibility baseline {case.compatibility_baseline}: it "
            f"therefore contributes its bytes and re-bases the identity, "
            f"relabelling every artifact on disk"
        )
        assert ei._identity_contribution(case) == b""

    old_payload = b"".join(
        ei._encode_fixture(case) for case in ei.fixtures()
        if case.compatibility_baseline is None
    )
    assert identity == hashlib.sha256(ei._DOMAIN + old_payload).digest()


def test_the_identity_sees_the_s6b_scale_plane(identity, monkeypatch):
    """tessera#143: the plane no recipe selects and every reader still decodes.

    ``encode._refit_scales`` is the ``else`` arm of the refit -- the CHANNEL and
    LUT planes have their own -- so it runs on the S6b case and on no other.
    Disabling it is a real byte move that stays internally consistent (the
    plane it returns is the one ``_pack_scales`` built), and before
    ``e2m1-768/s6b`` existed it moved no fixture's bytes and the digest did not
    follow.  The counter is asserted because a monkeypatch nothing calls is a
    test that proves nothing.
    """
    import tessera.encode as enc

    calls = []

    def refit_nothing(work, units, group, half, base_byte, refine, effective):
        calls.append(1)
        return base_byte, refine, effective

    monkeypatch.setattr(enc, "_refit_scales", refit_nothing)
    monkeypatch.setattr(ei, "_MEMO", [])
    moved = ei.encoder_fixture_id()
    assert calls, "the S6b refit was never called, so nothing was perturbed"
    assert moved != identity


def _written_planes(case):
    """The plane kinds one fixture's artifact actually carries.

    Read off the artifact's own ``plane_order`` zipped against its terminal's
    ``plane_elements`` -- the pair every reader indexes -- so what a case
    covers is measured on its bytes and never declared beside it.
    """
    with ei._fixture_build():
        art = parse(ei._fixture_blob(case))
    return frozenset(
        kind
        for kind, count in zip(art.manifest.plane_order, art.terminal.plane_elements)
        if count
    )


def test_every_override_fixture_writes_a_plane_the_wire_cases_do_not():
    """Non-vacuity, derived: a case off the wire has to earn its place.

    A fixture that overrides the encode is in the set for one reason -- it
    reaches a plane ``wire_recipe`` never writes -- so if every plane it
    carries is already carried by a wire case, it costs an encode and watches
    nothing. Measured off each artifact's own manifest rather than declared,
    and stated over whatever ``fixtures()`` holds, so it covers the next such
    case without being edited.
    """
    off_wire = [case for case in ei.fixtures() if not case.covers_wire]
    assert off_wire
    on_wire = frozenset().union(
        *(_written_planes(case) for case in ei.fixtures() if case.covers_wire)
    )
    for case in off_wire:
        earned = _written_planes(case) - on_wire
        assert earned, (
            f"{case.label} writes only planes the wire cases already write "
            f"({sorted(k.name for k in _written_planes(case))}), so it is an "
            f"encode that watches nothing"
        )


# --------------------------------------------------------------------------
# Coverage is enforced, not asserted.
# --------------------------------------------------------------------------


def test_every_shipping_structure_has_a_fixture():
    """A fixture hash is blind to what it does not encode, so the set has to
    span the grids, bodies and scale planes that ship -- and the expected set
    is derived from ``recipe_table`` over ``SERIALISABLE_GRIDS``, never
    restated.  Adding a grid, a body or a plane fails here until a fixture
    covers it.
    """
    covered = frozenset(
        case.structure for case in ei.fixtures() if case.covers_wire
    )
    missing = ei.shipping_structures() - covered
    assert not missing, (
        f"these shipping structures have no fixture, so a change that moved "
        f"only their bytes would leave the encoder identity unmoved: "
        f"{sorted(str(s) for s in missing)}"
    )


def test_no_fixture_covers_a_structure_nothing_ships():
    """The converse, so the set cannot quietly grow past what it must cover."""
    covered = frozenset(
        case.structure for case in ei.fixtures() if case.covers_wire
    )
    assert not (covered - ei.shipping_structures())


def test_the_identity_sees_the_segment_2a_diagonals(identity, monkeypatch):
    """tessera#143: the DIAG planes' *other* producer.

    The CHANNEL cases fill DIAG_SV with a row scale; ``fit_diagonals`` fits the
    rank-1 magnitude field and is reached only through ``with_diagonals=``.
    The perturbation is one fp16 ulp on every stored input-channel word -- the
    smallest byte-moving change the plane admits, the same class as the CHANNEL
    one above -- and it stays self-consistent, because the encoder codes the
    residual of exactly the fit it is handed. Before ``e2m1-768/diagonals``
    existed, ``fit_diagonals`` was never called at all.

    Dropping a Sinkhorn sweep was tried first and is *not* a byte move here:
    eight sweeps and seven agree to fp16 on this fixture, so the digest did not
    follow and the test read as a failure of the fixture rather than of the
    perturbation. A perturbation has to be shown to move bytes, not assumed to.
    """
    import tessera.encode as enc

    plain = enc.fit_diagonals
    calls = []

    def one_ulp_on_su(weights, **kwargs):
        calls.append(1)
        fit = plain(weights, **kwargs)
        su = (fit.su.contiguous().view(torch.int16) + 1).view(torch.float16)
        return dataclasses.replace(fit, su=su)

    monkeypatch.setattr(enc, "fit_diagonals", one_ulp_on_su)
    monkeypatch.setattr(ei, "_MEMO", [])
    moved = ei.encoder_fixture_id()
    assert calls, "fit_diagonals was never called, so nothing was perturbed"
    assert moved != identity


def test_the_identity_sees_the_completion_axis(identity, monkeypatch):
    """tessera#143: the second rate axis's only decision.

    ``_completion_choice`` runs on every TCQ column group, but at
    ``completion=0`` -- the exporter's default, and every rung at its body cap
    -- the reachable set has one member and the argmin returns zeros whatever
    the metric says. So "it was called" proves nothing here; the guard is that
    some fixture offered it more than one descendant, and before
    ``e2m1-256/completion`` existed none did.

    Rotating the pick by one is a byte move that stays legal: the completion
    bits index the same reachable table the decoder rebuilds from the
    descendant plane, so the artifact still round-trips.
    """
    import tessera.encode as enc

    plain = enc._completion_choice
    widths = []

    def next_descendant(want, per_pos, weights, steps, arity):
        widths.append(per_pos.shape[2])
        chosen = plain(want, per_pos, weights, steps, arity)
        return (chosen + 1) % per_pos.shape[2]

    monkeypatch.setattr(enc, "_completion_choice", next_descendant)
    monkeypatch.setattr(ei, "_MEMO", [])
    moved = ei.encoder_fixture_id()
    assert max(widths, default=0) > 1, (
        f"no fixture offered the completion argmin more than one descendant "
        f"(widths seen: {sorted(set(widths))}), so rotating its pick was a "
        f"no-op and this asserts nothing"
    )
    assert moved != identity


def test_an_override_fixture_is_not_counted_as_wire_coverage():
    """A case that overrides the encode writes different planes than the wire.

    ``e2m1-768/s6b`` resolves to a *shipping* structure -- it would be counted
    -- and writes SCALE_BASE where the wire writes SCALE_REFINE, so counting
    it would let the coverage rule above be satisfied by a fixture that never
    encodes the plane the rule is about.
    """
    s6b = next(c for c in ei.fixtures() if c.label == "e2m1-768/s6b")
    assert s6b.structure in ei.shipping_structures()
    assert not s6b.covers_wire
    assert all(c.covers_wire for c in ei.fixtures() if not c.encode)


def test_the_structure_set_is_read_off_the_owning_modules():
    """``shipping_structures`` derives; it does not restate.

    Hiding a grid from ``SERIALISABLE_GRIDS`` must shrink the set, which is
    what makes the coverage test above fail on the day a grid is added rather
    than pass against yesterday's list.
    """
    full = ei.shipping_structures()
    assert {name for name, _body, _plane in full} == {
        grid.name for grid in SERIALISABLE_GRIDS.values()
    }
    assert full == frozenset(
        (grid.name, row.recipe.body, row.recipe.scale_plane)
        for grid in SERIALISABLE_GRIDS.values()
        for row in recipe_table(grid)
    )


def test_each_fixture_contributes_its_own_bytes(identity):
    """Per-case digests exist and differ, so a bisect can say which plane moved
    rather than only that the encoder did."""
    per_case = ei.fixture_digests()
    assert len(per_case) == len(ei.fixtures())
    assert len(set(per_case.values())) == len(per_case)


# --------------------------------------------------------------------------
# The fixture build stamps nothing, so the digest is not a function of itself.
# --------------------------------------------------------------------------


def test_the_fixture_build_stamps_no_identity():
    seen = []

    with ei._fixture_build():
        assert ei.building()
        seen.append(ei.stamped_fixture_id())
    assert not ei.building()
    assert seen == [None]


def test_re_entering_the_fixture_build_is_refused():
    with ei._fixture_build():
        with pytest.raises(RuntimeError, match="re-entered"):
            with ei._fixture_build():
                pass


# --------------------------------------------------------------------------
# The untagged spelling writes no byte.
# --------------------------------------------------------------------------


def test_the_untagged_identity_writes_no_field():
    """The rule, not the value: whatever the encoder's digest is, an artifact
    cut by the *untagged* encoder carries no field and the minor it always had.
    """
    assert _resolve_fixture_id(ei.UNTAGGED_ENCODER_ID) is None
    other = bytes(range(32))
    assert _resolve_fixture_id(other) == other
    assert _resolve_fixture_id(None) is None


def test_the_field_is_absent_exactly_when_it_is_the_untagged_encoder():
    """The one that makes ``UNTAGGED_ENCODER_ID`` a fact rather than a knob.

    If the constant no longer names this encoder, every artifact this exporter
    writes declares minor 6 -- which is correct, and is the automatic
    behaviour the issue asked for.  This asserts the *equivalence*, so neither
    half can drift from the other.
    """
    weight = ei._fixture_weight()
    exported = encode_linear(weight, grid=E4M3_GRID, q256=1024)
    manifest = parse(exported.blob).manifest
    untagged = ei.encoder_fixture_id() == ei.UNTAGGED_ENCODER_ID
    assert (manifest.encoder_fixture_id is None) is untagged
    if untagged:
        assert manifest.schema_minor < 6
    else:
        assert manifest.schema_minor == 6
        assert manifest.encoder_fixture_id == ei.encoder_fixture_id()


def test_the_field_moves_no_plane_byte():
    """Stamping an identity changes the manifest and nothing the decoder reads.

    A minor bump that moved the plane region would make every byte comparison
    across the bump meaningless, which is the opposite of what this field is
    for.
    """
    weight = ei._fixture_weight()
    _exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=1024
    )
    plain = build_unit_artifact(unit, "u", forests, 1024, fixture_id=None)
    tagged = build_unit_artifact(unit, "u", forests, 1024, fixture_id=bytes(range(32)))
    assert plain[1] == tagged[1]
    assert plain[0].payload_digest == tagged[0].payload_digest
    assert plain[0].schema_minor < 6 and tagged[0].schema_minor == 6
    assert len(tagged[2]) > len(plain[2])


# --------------------------------------------------------------------------
# The wire.
# --------------------------------------------------------------------------


def test_the_identity_round_trips_through_the_container():
    weight = ei._fixture_weight()
    _exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=1024
    )
    stamp = hashlib.sha256(b"a foreign encoder").digest()
    _m, _r, blob = build_unit_artifact(unit, "u", forests, 1024, fixture_id=stamp)
    assert blob[10] == 6
    back = parse(blob).manifest
    assert back.encoder_fixture_id == stamp
    # The profile id is untouched: the identity is a sibling, never an input.
    assert back.encoder_profile_id == _m.encoder_profile_id


def test_minor_six_is_readable_and_current():
    assert SCHEMA_MINOR == 6
    assert 6 in SCHEMA_MINORS_READ
    assert tuple(SCHEMA_MINORS_READ) == tuple(range(SCHEMA_MINOR + 1))


def test_a_malformed_identity_is_refused():
    """A width no reader can step over is refused where the field has a name."""
    weight = ei._fixture_weight()
    _exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=1024
    )
    manifest = build_unit_artifact(unit, "u", forests, 1024, fixture_id=None)[0]
    with pytest.raises(ManifestError, match="encoder_fixture_id"):
        dataclasses.replace(manifest, encoder_fixture_id=b"\x00" * 8)


# --------------------------------------------------------------------------
# The consumers.
# --------------------------------------------------------------------------


def test_resumable_reads_absent_as_the_untagged_encoder():
    """One home for "may this cached unit be reused", so a resume, a merge
    guard and an A/B cannot disagree about what one encoder means."""

    class _M:
        encoder_fixture_id = None

    class _Foreign:
        encoder_fixture_id = hashlib.sha256(b"another encoder").digest()

    class _Mine:
        encoder_fixture_id = ei.encoder_fixture_id()

    assert ei.resumable(_M()) is (
        ei.encoder_fixture_id() == ei.UNTAGGED_ENCODER_ID
    )
    assert ei.resumable(_Foreign()) is False
    assert ei.resumable(_Mine()) is True


def test_a_presence_flag_that_is_not_a_bool_is_refused():
    """The shard, reach and fixture records are flagged present with one
    canonical bool.  A flag byte of 2 is not a bool, and a reader that treats
    it as "present" would accept two byte strings for one manifest -- the
    canonical encoding's one rule.  The reader has to refuse it, not truthy it."""
    weight = ei._fixture_weight()
    _exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=1024
    )
    manifest = build_unit_artifact(
        unit, "u", forests, 1024, fixture_id=bytes(range(32))
    )[0]
    data = manifest.encode(6)
    assert Manifest.decode(data, 6) == manifest
    # The fixture record is the last field: one flag byte, then its digest.
    assert data[-33] == 1
    patched = data[:-33] + b"\x02" + data[-32:]
    with pytest.raises(TesseraError, match="bool"):
        Manifest.decode(patched, 6)
