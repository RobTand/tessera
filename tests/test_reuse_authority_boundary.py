"""Tessera reads no client's records by itself (tessera#599).

A rooted cached-unit bundle and a Hessian reference both bind documents their
producer wrote.  Tessera used to name one producer's schemas and accept its
records unasked.  Now the producer supplies the reader: without it, a bundle or
reference that binds producer records refuses by name, and is never accepted
unjudged.  The client records below are PrismaQuant's, as it writes them; the
schema names appear only in this test, as data.
"""
import hashlib
import json

import pytest

from tessera.cached_unit import CachedUnitBundle
from tessera.errors import GrammarError
from tessera.export import ActivationSource
from tessera.hessian_capture import ReferenceHessianCollection, ReferenceHessians
from test_hessian_reference_capture import build_reference
from test_rooted_cached_bundle import rooted
from reuse_authority_fixture import AUTHORITY

CLIENT_SCHEMAS = {'catalog_extension': 'prismaquant.joint_catalog_extension.v3',
                  'candidate_overlay': 'prismaquant.t4_adopted_catalog.v1',
                  'adoption': 'prismaquant.joint_catalog_source_adoption.v1',
                  'proof': 'prismaquant.reseal_proof_bundle.v1'}
CLIENT_CANONICAL_CAPTURE = ('prismaquant.tessera_calibration_cache.v2', 'tessera_campaign_prefix_f32_v1')


def client_bundle(tmp_path):
    return rooted(tmp_path, CLIENT_SCHEMAS['catalog_extension'], schemas=CLIENT_SCHEMAS)[0]


@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_client_bundle_without_an_authority_refuses_by_name(tmp_path, mode):
    manifest = client_bundle(tmp_path)
    with pytest.raises(ValueError, match=r'need a producer reuse authority \(CachedUnitBundle\(authority=\.\.\.\)\)'):
        CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'],
                         encoder_source_proof_mode=mode)


def test_composed_client_bundle_without_an_authority_refuses_by_name(tmp_path):
    manifest = client_bundle(tmp_path)
    path = tmp_path / 'child.json'
    path.write_text(json.dumps(manifest))
    composed = {'schema': 'tessera.cached_units.v3', 'source': manifest['source'],
                'children': [{'manifest': {'path': str(path),
                                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                              'producer_package': None}]}
    with pytest.raises(ValueError, match='need a producer reuse authority'):
        CachedUnitBundle(composed, tmp_path, {'dense', 'expert'}, manifest['source'])


def test_an_authority_that_does_not_know_the_client_refuses_its_records(tmp_path):
    manifest = client_bundle(tmp_path)
    with pytest.raises(ValueError, match='authority schema differs'):
        CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'], authority=AUTHORITY)


@pytest.mark.parametrize('authority', [object(), 'prismaquant', {'check_document': None}])
def test_a_non_authority_is_refused_before_anything_is_read(tmp_path, authority):
    manifest = client_bundle(tmp_path)
    with pytest.raises(TypeError, match='ReuseAuthority'):
        CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'], authority=authority)


def test_client_reference_without_its_canonical_capture_refuses_by_name(tmp_path):
    handoff = build_reference(tmp_path, CLIENT_CANONICAL_CAPTURE)[0]
    message = r'need the producer canonical capture \(canonical_capture=\(schema, source\)\)'
    with pytest.raises(GrammarError, match=message):
        ReferenceHessians(handoff)
    with pytest.raises(GrammarError, match=message):
        ActivationSource.from_capture(handoff)


def test_client_reference_collection_without_its_canonical_capture_refuses_by_name(tmp_path):
    from test_hessian_reference_capture import write_json
    paths = []
    for part in ('one', 'two'):
        root = tmp_path / part
        root.mkdir()
        paths.append(build_reference(root, CLIENT_CANONICAL_CAPTURE)[0])
    # The first child refuses while it is opened, before any roster check.
    document = tmp_path / 'both.collection.references.json'
    write_json(document, {'schema': 'tessera.hessian_capture.collection.v1',
                          'references': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                         for p in sorted(paths)],
                          'units': ['a', 'b'], 'capture_sha256': '0' * 64})
    with pytest.raises(GrammarError, match='need the producer canonical capture'):
        ReferenceHessianCollection(document)


def test_the_supplied_canonical_capture_is_the_one_judged(tmp_path):
    handoff = build_reference(tmp_path, CLIENT_CANONICAL_CAPTURE)[0]
    with ReferenceHessians(handoff, canonical_capture=CLIENT_CANONICAL_CAPTURE) as owner:
        assert set(owner) == {'a', 'b'}
        assert owner['a'].shape == (4, 4)
    with pytest.raises(GrammarError, match='one complete canonical capture'):
        ReferenceHessians(handoff, canonical_capture=('other.calibration_cache.v1', CLIENT_CANONICAL_CAPTURE[1]))
    for malformed in ('prismaquant.tessera_calibration_cache.v2', ('only-one',), ('', 'x'), (1, 2)):
        with pytest.raises(GrammarError, match='canonical_capture must be'):
            ReferenceHessians(handoff, canonical_capture=malformed)
