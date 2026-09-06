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
from tessera.planes import PlaneLayout
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


def test_a_compatibility_witness_contributes_exactly_when_its_bytes_moved(identity):
    """The mechanism, pinned: a witness contributes the empty string while its
    encoded contribution matches the baseline it recorded, and its
    self-delimiting bytes otherwise -- and the identity is the digest of
    exactly those contributions, in fixture order.

    Every witness, not the first one: a set membership test would pass with a
    wrong constant on any case it did not reach.

    The schema-minor-7 transition (tessera#144) moved every witness then
    present by reordering its terminal record. Later witnesses, including the
    refit boundary (#360), record their own historical baseline once. The
    baselines are measured history and are **not** advanced to bless new bytes
    (``encoder_identity`` docstring, schema §1g). A witness contributes whenever
    its current bytes differ from its own baseline.
    """
    witnesses = [
        case for case in ei.fixtures()
        if case.compatibility_baseline is not None
    ]
    assert witnesses
    for case in witnesses:
        encoded = ei._encode_fixture(case)
        computed = hashlib.sha256(encoded).hexdigest()
        neutral = computed == case.compatibility_baseline
        contribution = ei._identity_contribution(case)
        assert (contribution == b"") is neutral, case.label
        if not neutral:
            assert contribution.endswith(encoded), case.label
        assert not neutral, (
            f"{case.label} encodes to its minor 0-6 compatibility baseline "
            f"{computed} on the minor-7 wire: the baseline was advanced, or "
            "the layout stopped moving the terminal record"
        )

    payload = b"".join(ei._identity_contribution(case) for case in ei.fixtures())
    assert identity == hashlib.sha256(ei._DOMAIN + payload).digest()


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


def test_the_identity_sees_the_release_placement(identity, monkeypatch):
    """tessera#143: the plane the exporter has no keyword for.

    ``encode._canonical_release_order`` runs only when ``released_positions``
    is set, and before ``e2m1-768/release`` existed no fixture could set it --
    ``encode_linear`` takes no such argument. Reversing the order is a real
    placement change: the reader regenerates this order from the placed count,
    so a different order is a different set of protected positions.
    """
    import tessera.encode as enc

    plain = enc._canonical_release_order
    calls = []

    def reversed_order(decoded, cols, superblock, total):
        calls.append(1)
        return plain(decoded, cols, superblock, total).flip(0)

    monkeypatch.setattr(enc, "_canonical_release_order", reversed_order)
    monkeypatch.setattr(ei, "_MEMO", [])
    moved = ei.encoder_fixture_id()
    assert calls, (
        "no fixture released a position, so the placement rule was never asked "
        "for an order"
    )
    assert moved != identity


def test_the_release_fixture_is_the_exporters_own_call():
    """The one hand-assembled encode, pinned against the exporter's.

    ``encode_linear`` cannot express a release, so the release fixture builds
    the ``encode_unit`` call itself -- which is a re-implementation, and a
    re-implementation drifts. At zero releases the two paths must agree byte
    for byte, so a keyword the exporter starts passing and this call does not
    fails here instead of silently pricing a different encoder.
    """
    case = next(c for c in ei.fixtures() if c.released_positions is not None)
    with ei._fixture_build():
        hand_assembled = ei._fixture_blob(
            dataclasses.replace(case, released_positions=0)
        )
        exporter = ei._fixture_blob(
            dataclasses.replace(case, released_positions=None)
        )
    assert hand_assembled == exporter


def test_the_release_fixture_actually_places_releases():
    """Non-vacuity: zero releases would make the perturbation above a no-op."""
    from tessera.planes import PlaneKind

    case = next(c for c in ei.fixtures() if c.released_positions is not None)
    assert PlaneKind.RELEASE in _written_planes(case)


def test_the_identity_sees_the_shard_state_replay(identity, monkeypatch):
    """tessera#143: the second byte-producing path.

    ``slicing._initial_state`` replays the trellis register a shard's first row
    is entered from -- the one thing a cut is not a restriction of -- and
    ``slice_unit`` is reached by no encode. Permuting the per-column start
    states is a real byte move on the INITIAL_STATE plane; the guard is that
    the replay produced states that a permutation actually changes, because a
    uniform state would make the perturbation a no-op.
    """
    import tessera.slicing as sl

    plain = sl._initial_state
    perturbed = []

    def permuted(unit, steps0, arity, code, parent_state):
        state = plain(unit, steps0, arity, code, parent_state)
        if state is None or state.numel() == 0:
            return state
        flipped = state.flip(0)
        perturbed.append(not torch.equal(state, flipped))
        return flipped

    monkeypatch.setattr(sl, "_initial_state", permuted)
    monkeypatch.setattr(ei, "_MEMO", [])
    moved = ei.encoder_fixture_id()
    assert any(perturbed), (
        f"no fixture replayed a start state a permutation changes "
        f"(seen: {perturbed}), so nothing was perturbed"
    )
    assert moved != identity


def test_the_fixture_set_writes_every_plane_a_reader_reads():
    """Coverage over planes, derived the way the structure rule is derived.

    ``planes.SHARD_PLANE_ORDER`` is the wire order every reader indexes
    ``plane_elements`` by, so it is the set of planes an artifact can carry. A
    plane no fixture writes is a plane whose packing can move at an unmoved
    identity -- which is what SCALE_BASE, DIAG_SU, COMPLETION, RELEASE and
    INITIAL_STATE each were (tessera#143). Occupancy is measured off each
    artifact's own manifest, never declared beside the case, and the expected
    set comes from the module that owns it, so a plane added to the wire fails
    here until a fixture writes it.
    """
    from tessera.planes import SHARD_PLANE_ORDER

    written = frozenset().union(
        *(_written_planes(case) for case in ei.fixtures())
    )
    missing = frozenset(SHARD_PLANE_ORDER) - written
    assert not missing, (
        f"no fixture writes {sorted(k.name for k in missing)}, so a change to "
        f"how those planes are packed moves real bytes and leaves "
        f"encoder_fixture_id where it was"
    )


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
    # And it is a different encode, measured: the wire case at the same
    # structure writes a different set of planes, which is the whole reason
    # one cannot stand in for the other.
    wire_case = next(
        c for c in ei.fixtures()
        if c.covers_wire and c.structure == s6b.structure
    )
    assert _written_planes(s6b) != _written_planes(wire_case)


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
    # Since minor 7 the minor is the plane layout's, so the field's presence
    # no longer moves it on a fresh artifact; the equivalence that remains is
    # field <-> encoder.  The minor-6 envelope is pinned on the LEGACY layout
    # in ``test_the_field_moves_no_plane_byte``.
    assert manifest.schema_minor == SCHEMA_MINOR
    if not untagged:
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
    # On the LEGACY layout the field is what moves the minor, so the envelope
    # is visible; on the current layout both sides are minor 7 and the field
    # still moves no plane byte.
    for layout in (PlaneLayout.LEGACY, PlaneLayout.LADDER):
        plain = build_unit_artifact(
            unit, "u", forests, 1024, fixture_id=None, layout=layout
        )
        tagged = build_unit_artifact(
            unit, "u", forests, 1024, fixture_id=bytes(range(32)), layout=layout
        )
        assert plain[1] == tagged[1]
        assert plain[0].payload_digest == tagged[0].payload_digest
        if layout is PlaneLayout.LEGACY:
            assert plain[0].schema_minor < 6 and tagged[0].schema_minor == 6
        else:
            assert plain[0].schema_minor == tagged[0].schema_minor == SCHEMA_MINOR
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
    assert blob[10] == SCHEMA_MINOR
    back = parse(blob).manifest
    assert back.encoder_fixture_id == stamp
    # The profile id is untouched: the identity is a sibling, never an input.
    assert back.encoder_profile_id == _m.encoder_profile_id


def test_the_identity_minor_is_readable_and_the_current_one_carries_it():
    # Minor 6 is the identity-bearing envelope; minor 7 (the LADDER layout,
    # tessera#144) is current and carries the same field.
    assert SCHEMA_MINOR == 7
    assert 6 in SCHEMA_MINORS_READ and 7 in SCHEMA_MINORS_READ
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
    data = manifest.encode(SCHEMA_MINOR)
    assert Manifest.decode(data, SCHEMA_MINOR) == manifest
    # The fixture record is the last field: one flag byte, then its digest.
    assert data[-33] == 1
    patched = data[:-33] + b"\x02" + data[-32:]
    with pytest.raises(TesseraError, match="bool"):
        Manifest.decode(patched, SCHEMA_MINOR)


def test_identity_sees_refit_improvements_below_the_rounded_loss_ulp(identity, monkeypatch):
    import tessera.scale_channel as sc

    current = sc.refit_channel_scale

    def rounded_loss_guard(work, units, stored, global_scale, metric=None, floor=None):
        new_stored, effective = current(work, units, stored, global_scale, metric, floor)
        if metric is not None or floor is not None:
            return new_stored, effective
        W, U = work.float(), units.float()
        A, B = (U * U).sum(dim=1), (W * U).sum(dim=1)
        old = stored.float() * global_scale
        old_loss = A * old * old - 2 * B * old
        new_loss = A * effective * effective - 2 * B * effective
        held = torch.where(new_loss < old_loss, new_stored, stored)
        return held, held.float() * global_scale

    monkeypatch.setattr(sc, "refit_channel_scale", rounded_loss_guard)
    monkeypatch.setattr(ei, "_MEMO", [])
    assert ei.encoder_fixture_id() != identity
