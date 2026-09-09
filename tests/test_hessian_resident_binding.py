"""A producer that already holds the committed H consumes it without paying twice.

The defect these pin (tessera#440) is that neither existing construction gets
both halves.  A plain mapping seals by digesting the whole population, which a
producer that just wrote those exact commitments has already paid for.  A
``ReferenceHessians`` seals from metadata but reads every unit back off disk to
serve it.  ``bind_resident`` is the third case: the owner's commitments and
checks, the caller's own tensors.
"""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tessera import cached_unit as cached_unit_module
from tessera.alphabet import E4M3_GRID
from tessera.cached_unit import encoding_input_identity, tensor_identity
from tessera.errors import GrammarError
from tessera.export import ActivationSource
from tessera.hessian_capture import ReferenceHessians

from test_hessian_reference_capture import reference  # noqa: F401  (fixture)


@pytest.fixture
def resident(reference):  # noqa: F811
    """The handoff path, the payload, and H as the caller still holds it.

    The fixture's H are the tensors its commitments were computed from, which
    is exactly the producer's position after it writes the reference document.
    """
    handoff, payload, H, _, _ = reference
    return handoff, payload, H


def counted(monkeypatch, module, name):
    """Count calls to ``module.name`` and keep its behaviour."""
    original = getattr(module, name)
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0] if args else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, observed)
    return calls


def bound_source(handoff, H, **settings):
    settings.setdefault('ldlq_sigma', None)
    return ActivationSource.from_capture(handoff, resident_hessians=H, **settings)


def encode_kwargs(source, unit='a.weight', columns=4):
    """``for_unit`` as the existing reference tests call it on this fixture."""
    from tessera.manifest import ScalePlaneKind

    return source.for_unit(unit, columns, 'cpu', scale_plane=ScalePlaneKind.CHANNEL)


# --- the regression itself -------------------------------------------------

def test_sealing_a_resident_binding_digests_no_unit(resident, monkeypatch):
    """The seal is metadata only, and it is the same seal.

    Before the fix a plain-mapping source digests every unit here.  That cost
    is the whole defect: the caller computed those digests to write the
    reference document, and sealing recomputed them over the same tensors.
    """
    handoff, payload, H = resident
    digests = counted(monkeypatch, cached_unit_module, 'tensor_identity')
    source = bound_source(handoff, H)
    assert source.capture_sha256() == payload['capture_sha256']
    assert digests == [], (
        'sealing a resident binding must read commitments, not tensors')
    # The roster and the identity fields are the document's, not the mapping's.
    assert set(source.hessians) == set(H)
    source.hessians.close()


def test_consuming_a_bound_unit_reads_no_canonical_payload(resident, monkeypatch):
    """``for_unit`` serves the caller's object, and touches no source file.

    Before the fix the only source that seals cheaply is one that reads every
    H back through ``torch.load``.
    """
    handoff, _, H = resident
    loads = counted(monkeypatch, torch, 'load')
    source = bound_source(handoff, H)
    served = source.hessians['a']
    assert served is H['a'], 'the bound owner must serve the caller\'s exact object'
    kwargs = encode_kwargs(source)
    assert kwargs['refit_metric'] is served, (
        'the encoder must receive the caller\'s own tensor, not a copy of it')
    assert loads == [], 'a bound owner must not read the canonical H payloads'
    source.hessians.close()


def test_the_checkpoint_identity_path_sees_the_same_object(resident, monkeypatch):
    """``cached_unit`` reads ``activation.hessians[name]`` directly, not ``for_unit``.

    A resident mapping consulted only inside ``for_unit`` would leave this
    path reading canonical payloads and minting a new CPU tensor per unit.
    """
    handoff, _, H = resident
    loads = counted(monkeypatch, torch, 'load')
    source = bound_source(handoff, H)
    seen = []
    original = cached_unit_module.tensor_identity

    def observed(tensor):
        seen.append(tensor)
        return original(tensor)

    monkeypatch.setattr(cached_unit_module, 'tensor_identity', observed)
    identity = encoding_input_identity(
        torch.zeros(8, 4), 'a.weight', E4M3_GRID, 1024, activation=source)
    assert identity['calibration'] is not None
    assert any(t is H['a'] for t in seen), (
        'the identity path must digest the caller\'s own tensor')
    assert loads == [], 'the identity path must not read a canonical payload'
    source.hessians.close()


# --- what the binding refuses ---------------------------------------------

def test_bind_refuses_a_roster_that_gained_or_lost_a_unit(resident):
    handoff, _, H = resident
    owner = ReferenceHessians(handoff)
    with pytest.raises(GrammarError) as gained:
        owner.bind_resident(dict(H, c=torch.eye(4)))
    assert "gained ['c']" in str(gained.value)
    with pytest.raises(GrammarError) as lost:
        owner.bind_resident({'a': H['a']})
    assert "lost ['b']" in str(lost.value)
    owner.close()


@pytest.mark.parametrize('name,wrong', [
    ('dtype', lambda: torch.eye(4, dtype=torch.float64)),
    ('shape', lambda: torch.eye(8)),
    ('type', lambda: [[1.0]]),
    ('contiguity', lambda: torch.eye(4).t().flip(0)),
])
def test_bind_refuses_a_tensor_its_commitment_does_not_describe(resident, name, wrong):
    handoff, _, H = resident
    owner = ReferenceHessians(handoff)
    with pytest.raises(GrammarError, match='resident Hessian'):
        owner.bind_resident(dict(H, a=wrong()))
    owner.close()


def test_bind_refuses_a_view_that_does_not_own_its_storage(resident):
    """A view pins a buffer larger than the unit, so the owner would understate."""
    handoff, _, H = resident
    buffer = torch.zeros(2, 4, 4)
    buffer[0] = H['a']
    owner = ReferenceHessians(handoff)
    with pytest.raises(GrammarError, match='own its storage'):
        owner.bind_resident(dict(H, a=buffer[0]))
    owner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a device')
def test_a_resident_cuda_h_is_served_as_itself(resident):
    """The population this exists for is on CUDA, and it is not staged to bind.

    A CPU-only pass cannot certify this; the skip is the honest answer on the
    CPU fleet, and root's native harness runs it.
    """
    handoff, _, H = resident
    device = {name: value.cuda() for name, value in H.items()}
    source = bound_source(handoff, device)
    served = source.hessians['a']
    assert served is device['a']
    assert served.device.type == 'cuda'
    kwargs = encode_kwargs(source)
    assert kwargs['refit_metric'] is served
    assert source.hessians.receipt()['loaded_entries'] == 0
    source.hessians.close()


def test_bind_refuses_a_device_with_no_bytes(resident):
    handoff, _, H = resident
    owner = ReferenceHessians(handoff)
    with pytest.raises(GrammarError, match='lives on meta'):
        owner.bind_resident(dict(H, a=torch.eye(4, device='meta')*3))
    owner.close()


def test_close_releases_the_owners_hold_on_the_population(resident):
    """The owner promises its memory lifetime ends at ``close()``.

    A held descriptor and a held tensor are the same promise.  The caller's own
    reference is its business; what is proved here is that nothing survives on
    the owner's side once the caller lets go.
    """
    import gc
    import weakref

    handoff, _, H = resident
    owner = ReferenceHessians(handoff)
    caller = dict(H)
    owner.bind_resident(caller)
    watch = weakref.ref(caller['a'])
    assert owner['a'] is not None

    owner.close()
    caller.clear()
    del H
    gc.collect()
    assert watch() is None
    assert owner.receipt()['resident_bound'] is True


def test_closing_does_not_reach_into_the_callers_mapping(resident):
    handoff, _, H = resident
    owner = ReferenceHessians(handoff)
    caller = dict(H)
    owner.bind_resident(caller)
    owner.close()
    assert set(caller) == set(H)
    assert caller['a'] is H['a']


def test_consuming_a_bound_unit_twice_leaves_the_callers_bytes_alone(resident):
    """The file path handed out a clone, so nothing downstream could touch it.

    A bound owner hands out the caller's own object, and ``H.to('cpu', float32)``
    on a CPU float32 tensor is that same object again.  If any step of a unit's
    encode wrote through it, the caller's H would be corrupt and the NEXT
    consumption of that unit would refuse against its own commitment.  So the
    second consumption is the check: it exercises the metric, the LDL factor and
    the rotated diagonals, and the digest is what decides.
    """
    from tessera.manifest import ScalePlaneKind

    handoff, _, H = resident
    before = tensor_identity(H['a'])['sha256']

    # The LDL leg regularises and factorises; the diagonals leg transports the
    # metric through Du.  Both take the served object as their input.
    arms = [dict(ldlq_sigma=0.025, ldlq_block=2), dict(ldlq_sigma=None)]
    for settings in arms:
        source = ActivationSource.from_capture(handoff, resident_hessians=H, **settings)
        for _ in range(2):
            kwargs = source.for_unit('a.weight', 4, 'cpu',
                                     scale_plane=ScalePlaneKind.CHANNEL,
                                     with_diagonals=settings['ldlq_sigma'] is None,
                                     weight=(None if settings['ldlq_sigma'] is not None
                                             else torch.randn(8, 4)))
            assert kwargs['refit_metric'] is not None
            assert tensor_identity(H['a'])['sha256'] == before
        source.hessians.close()


def test_bind_is_one_shot_and_ends_with_the_owner(resident):
    handoff, _, H = resident
    owner = ReferenceHessians(handoff)
    owner.bind_resident(dict(H))
    with pytest.raises(GrammarError, match='already bound'):
        owner.bind_resident(dict(H))
    owner.close()
    with pytest.raises(GrammarError, match='closed'):
        owner.bind_resident(dict(H))
    with pytest.raises(GrammarError, match='closed'):
        owner['a']


def test_an_h_mutated_after_binding_is_refused_where_the_bytes_are_decided(resident):
    """The per-unit digest is unmoved: ``_require_sealed_unit`` still runs."""
    handoff, _, H = resident
    source = bound_source(handoff, H)
    assert source.capture_sha256()
    H['a'][0, 0] = 99.0
    with pytest.raises(GrammarError, match='differs from what capture_sha256|sealed'):
        encode_kwargs(source)
    source.hessians.close()


def test_a_nonfinite_resident_h_is_refused_on_consumption(resident):
    handoff, _, H = resident
    source = bound_source(handoff, H)
    H['a'][0, 0] = float('nan')
    with pytest.raises(GrammarError, match='nonfinite'):
        source.hessians['a']
    source.hessians.close()


def test_a_unit_that_goes_nonfinite_after_a_good_consumption_is_still_refused(resident):
    """Finiteness is memoised on first use, so this is caught by the digest.

    Which is the point of memoising it: the check that decides the bytes is the
    commitment comparison, and a NaN written into a bound tensor moves the
    bytes.  Nothing rests on re-scanning a unit the owner already passed.
    """
    handoff, _, H = resident
    source = bound_source(handoff, H)
    encode_kwargs(source)
    H['a'][1, 1] = float('nan')
    with pytest.raises(GrammarError):
        encode_kwargs(source)
    source.hessians.close()


def test_an_h_that_is_not_what_the_document_commits_is_refused_at_consumption(resident):
    """Bound bytes are never trusted: the commitment decides, at the unit."""
    handoff, _, H = resident
    wrong = dict(H, a=torch.eye(4) * 5)
    source = bound_source(handoff, wrong)
    with pytest.raises(GrammarError):
        encode_kwargs(source)
    source.hessians.close()


def test_replacing_the_callers_entry_after_binding_does_not_change_what_is_served(resident):
    """The bound set is fixed at bind, so a later swap cannot reach the encoder."""
    handoff, _, H = resident
    caller = dict(H)
    owner = ReferenceHessians(handoff)
    owner.bind_resident(caller)
    caller['a'] = torch.eye(4) * 11
    served = owner['a']
    assert served is H['a'], 'the owner must retain the object it was handed'
    torch.testing.assert_close(served, torch.eye(4) * 3, rtol=0, atol=0)
    owner.close()


def test_resident_hessians_needs_a_reference_document(resident, tmp_path):
    handoff, _, H = resident
    legacy = tmp_path / 'capture.pt'
    torch.save({'H': H, 'provenance': {}}, legacy)
    with pytest.raises(GrammarError, match='resident_hessians'):
        ActivationSource.from_capture(legacy, resident_hessians=H)


# --- what the run records --------------------------------------------------

def test_the_receipt_says_which_path_ran(resident):
    handoff, _, H = resident
    source = bound_source(handoff, H)
    receipt = source.hessians.receipt()
    assert receipt['resident_bound'] is True
    assert receipt['resident_units_observed'] == []
    source.hessians['a']
    receipt = source.hessians.receipt()
    assert receipt['resident_units_observed'] == ['a']
    assert receipt['loaded_entries'] == 0 and receipt['source_read_bytes'] == 0
    assert receipt['verified_units'] == [], (
        'verified_units claims this owner read and compared the bytes, which '
        'the bound path does not do; observed and authenticated are two claims '
        'and the receipt must not merge them')
    source.hessians.close()


def test_the_metadata_a_bound_source_adds_is_exactly_hessian_role_and_path(resident):
    """Named because the merge guard compares every field of this block."""
    handoff, payload, H = resident
    plain = ActivationSource(H, {k: v for k, v in payload['provenance'].items()
                                 if k != 'hessian_role'})
    source = bound_source(handoff, H)
    assert source.capture_sha256() == plain.capture_sha256()
    added = set(source.config_block()['hessian']) - set(plain.config_block()['hessian'])
    assert added == {'hessian_role', 'path'}
    assert source.reference_binding() is not None and plain.reference_binding() is None
    source.hessians.close()
