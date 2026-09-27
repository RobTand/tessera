"""Reader admission is explicit; it never relaxes authenticated wire inputs."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tessera.cached_unit import CachedUnitBundle, verify_cached_unit
from test_cached_producer import _record, encoded  # noqa: F401 (shared fixture)
from test_rooted_cached_bundle import rooted


def bound(tmp_path, name, document):
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(document))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def unproven(manifest):
    manifest['reuse_authority']['encoder_source_proofs'] = []
    manifest['encoder_adoptions']['expert']['encoder_source_proof'] = None


def load(manifest, tmp_path, mode='permissive'):
    return CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'],
                            encoder_source_proof_mode=mode)


@pytest.mark.parametrize('proof_state', ['absent', 'unlisted', 'wrong_pins', 'failed'])
def test_unproven_adoption_requires_explicit_mode_and_retains_warning(tmp_path, proof_state):
    manifest, _ = rooted(tmp_path)
    adoption = manifest['encoder_adoptions']['expert']
    if proof_state in ('absent', 'unlisted'):
        manifest['reuse_authority']['encoder_source_proofs'] = []
        if proof_state == 'absent':
            adoption['encoder_source_proof'] = None
    else:
        document = json.loads(Path(adoption['encoder_source_proof']['path']).read_text())
        if proof_state == 'wrong_pins':
            document['pins']['new']['encoder_source_sha256'] = '0' * 64
        else:
            document['ok'] = False
        proof = bound(tmp_path, 'altered-proof', document)
        manifest['reuse_authority']['encoder_source_proofs'] = [proof]
        adoption['encoder_source_proof'] = proof
    bundle = load(manifest, tmp_path)
    assert bundle.encoder_source_proof_mode == 'permissive'
    assert bundle.warnings == [{
        'schema': 'tessera.cached_unit_warning.v1',
        'code': 'encoder_source_proof_not_authorized', 'unit': 'expert',
        'reason': 'cached unit encoder source proof does not authorize this adoption',
        'proof_status': proof_state if proof_state in ('absent', 'unlisted') else 'not_authorizing',
        'encoder_source_proof': adoption['encoder_source_proof']}]
    for kwargs in ({}, {'encoder_source_proof_mode': 'strict'}):
        with pytest.raises(ValueError, match='proof does not authorize'):
            CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'], **kwargs)


def test_composition_propagates_mode_and_warning_without_mutating_children(tmp_path):
    manifest, _ = rooted(tmp_path)
    unproven(manifest)
    original = copy.deepcopy(manifest)
    child = bound(tmp_path, 'child', manifest)
    composed = {'schema': 'tessera.cached_units.v3', 'source': manifest['source'],
                'children': [{'manifest': child, 'producer_package': None}]}
    bundle = load(composed, tmp_path)
    assert bundle.warnings == bundle.child_manifests[0]['warnings'] == load(manifest, tmp_path).warnings
    assert len(bundle.warnings) == 1
    assert manifest == original
    with pytest.raises(ValueError, match='proof does not authorize'):
        load(composed, tmp_path, 'strict')


@pytest.mark.parametrize('mode', ['', 'dev', None, True])
def test_unknown_proof_mode_refuses(tmp_path, mode):
    manifest, _ = rooted(tmp_path)
    with pytest.raises(ValueError, match='encoder_source_proof_mode'):
        load(manifest, tmp_path, mode)


@pytest.mark.parametrize('field', ['unit', 'source', 'calibration', 'encoder_fixture_id', 'projection'])
def test_permissive_adoption_still_refuses_identity_changes(tmp_path, field):
    manifest, _ = rooted(tmp_path)
    unproven(manifest)
    assert len(load(manifest, tmp_path).warnings) == 1
    manifest['encoder_adoptions']['expert']['reference_encoding_identity'][field] = {'changed': True}
    with pytest.raises(ValueError, match='identities differ|changed ' + field):
        load(manifest, tmp_path)


@pytest.mark.parametrize('authority', ['catalog_extension', 'candidate_overlay', 'encoder_source_proofs'])
def test_permissive_adoption_still_authenticates_bound_documents(tmp_path, authority):
    manifest, _ = rooted(tmp_path)
    assert load(manifest, tmp_path).warnings == []
    target = manifest['reuse_authority'][authority]
    if isinstance(target, list):
        target = target[0]
    Path(target['path']).write_text('{}')
    with pytest.raises(ValueError, match='SHA256 differs'):
        load(manifest, tmp_path)


@pytest.mark.parametrize('damage', ['bytes', 'digest', 'size', 'shape', 'projection', 'geometry'])
def test_permissive_adoption_never_relaxes_wire_verification(tmp_path, encoded, damage):
    manifest, roots = rooted(tmp_path)
    unproven(manifest)
    record, identity = _record(encoded)
    identity['unit'] = 'expert'
    identity['encoder_source_sha256'] = 'b' * 64
    record['identity'] = copy.deepcopy(identity)
    manifest['units']['expert'] = record
    adoption = manifest['encoder_adoptions']['expert']
    adoption['candidate_encoding_identity'] = record['identity']
    adoption['reference_encoding_identity'] = {**copy.deepcopy(identity), 'encoder_source_sha256': 'a' * 64}
    (roots['added'] / record['file']).write_bytes(encoded[1])
    bundle = load(manifest, tmp_path)
    blob, observed = bundle.read('expert')
    assert len(bundle.warnings) == 1
    assert verify_cached_unit(blob, observed, identity).blob == blob
    if damage == 'bytes':
        blob = blob[:-1] + bytes([blob[-1] ^ 1])
    elif damage == 'digest':
        observed['blob_sha256'] = '0' * 64
    elif damage == 'size':
        observed['blob_bytes'] += 1
    else:
        identity = copy.deepcopy(identity)
        if damage == 'shape':
            identity['source']['shape'] = [32]
        elif damage == 'projection':
            identity['projection']['rows'] = 64
        else:
            identity['source']['shape'] = [64, 32]
            identity['projection']['rows'] = 64
        observed['identity'] = identity
    with pytest.raises(ValueError, match='sha256|shape|geometry'):
        verify_cached_unit(blob, observed, identity)


def policy_fixture(tmp_path, version):
    manifest, _ = rooted(tmp_path)
    # Include a future/data-only rung to catch a hard-coded gamut roster.
    rates = [640, 768, 896, 1152]
    template = copy.deepcopy(manifest['units']['expert'])
    adoption = copy.deepcopy(manifest['encoder_adoptions']['expert'])
    manifest['units'].pop('expert')
    manifest['unit_roots'].pop('expert')
    manifest['encoder_adoptions'].pop('expert')
    for rate in rates:
        name = f'expert{rate}'
        record = copy.deepcopy(template)
        record['file'] = name + '.tessera'
        record['identity'].update(unit=name, recipe={'grid': 'E2M1x2', 'q256': rate})
        manifest['units'][name] = record
        manifest['unit_roots'][name] = 'added'
        entry = copy.deepcopy(adoption)
        entry['reference_pair'][0] = name
        entry['reference_encoding_identity']['unit'] = name
        entry['candidate_encoding_identity'] = record['identity']
        manifest['encoder_adoptions'][name] = entry
    names = set(manifest['encoder_adoptions'])
    formats = [f'TESSERA_E2M1_K2_R{rate}' for rate in (640, 768, 1152)]
    policy = {'schema': f'prismaquant.joint_served_activation_policy.v{version}',
              **({'format': 'TESSERA_E2M1_K2_R896'} if version == 1 else {'formats': sorted(formats)}),
              'executed_grouping': {'groups': {'group': {'members': sorted(names), 'input_global_scale': 0.5}}}}
    chosen = {896} if version == 1 else {640, 768, 1152}
    manifest['served_activation_policy'] = bound(tmp_path, 'policy', policy)
    manifest['served_activations'] = {f'expert{rate}': {'group': 'group', 'input_global_scale': 0.5}
                                      for rate in chosen}
    return manifest, policy


@pytest.mark.parametrize('version', [1, 2])
def test_served_policy_scope_is_data_and_v1_keeps_exact_single_rung(tmp_path, version):
    manifest, _ = policy_fixture(tmp_path, version)
    bundle = CachedUnitBundle(manifest, tmp_path, set(manifest['units']), manifest['source'],
                              encoder_source_proof_mode='strict')
    assert bundle.warnings == []
    assert bundle.served_activations == manifest['served_activations']
    scales = {name + '.input_global_scale': 0.5 for name in bundle.served_activations}
    bundle.require_served_scales(scales)
    with pytest.raises(ValueError, match='exported activation scale differs'):
        bundle.require_served_scales(dict.fromkeys(scales, 0.25))


@pytest.mark.parametrize('damage', ['missing', 'extra', 'scale', 'group', 'member', 'digest'])
def test_v2_policy_coverage_and_values_remain_exact(tmp_path, damage):
    manifest, policy = policy_fixture(tmp_path, 2)
    kwargs = {'encoder_source_proof_mode': 'permissive'}
    CachedUnitBundle(manifest, tmp_path, set(manifest['units']), manifest['source'], **kwargs)
    if damage == 'missing':
        del manifest['served_activations']['expert640']
    elif damage == 'extra':
        manifest['served_activations']['expert896'] = {'group': 'group', 'input_global_scale': 0.5}
    elif damage in ('scale', 'group'):
        manifest['served_activations']['expert640'][{'scale': 'input_global_scale', 'group': 'group'}[damage]] = 99
    elif damage == 'member':
        policy['executed_grouping']['groups']['group']['members'].remove('expert640')
        manifest['served_activation_policy'] = bound(tmp_path, 'policy', policy)
    else:
        manifest['served_activation_policy']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='served activations differ|absent from served activation policy|SHA256 differs'):
        CachedUnitBundle(manifest, tmp_path, set(manifest['units']), manifest['source'], **kwargs)


@pytest.mark.parametrize('formats', [None, [], ['TESSERA_E2M1_K2_R768'] * 2,
                                     ['TESSERA_E2M1_K2_R896', 'TESSERA_E2M1_K2_R640'],
                                     [None], ['TESSERA_E4M3_K1_R1024'], ['TESSERA_E2M1_K2_R0768']])
def test_v2_policy_refuses_malformed_format_scope(tmp_path, formats):
    manifest, policy = policy_fixture(tmp_path, 2)
    # Establish the new API before exercising a refusal (fails before #644).
    CachedUnitBundle(manifest, tmp_path, set(manifest['units']), manifest['source'],
                     encoder_source_proof_mode='strict')
    policy['formats'] = formats
    manifest['served_activation_policy'] = bound(tmp_path, 'policy', policy)
    with pytest.raises(ValueError, match='policy .*differs'):
        CachedUnitBundle(manifest, tmp_path, set(manifest['units']), manifest['source'])
