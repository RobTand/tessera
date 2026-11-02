"""Shared rooted-bundle documents with independently bound producers."""
import hashlib
import json

import reuse_authority_fixture as fixture


SCHEMAS = {'catalog_extension': fixture.CATALOG_EXTENSION_SCHEMA,
           'candidate_overlay': fixture.CANDIDATE_OVERLAY_SCHEMA,
           'adoption': fixture.ADOPTION_SCHEMA, 'proof': fixture.PROOF_SCHEMA}


def rooted(tmp_path, extension_schema=fixture.CATALOG_EXTENSION_SCHEMA, *, schemas=SCHEMAS):
    """A rooted bundle whose producer documents carry ``schemas`` (the fixture producer's)."""
    roots = {key: tmp_path / key for key in ('old', 'added')}
    for path in roots.values():
        path.mkdir()
        (path / 'unit.tessera').write_bytes(b'unchanged wire')
    bound = {}
    for name, schema in [('catalog_extension', extension_schema),
                         ('candidate_overlay', schemas['candidate_overlay'])]:
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps({'schema': schema}))
        bound[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    units = {name: {'file': 'unit.tessera', 'blob_bytes': 14,
                    'blob_sha256': hashlib.sha256(b'unchanged wire').hexdigest(),
                    'identity': {'unit': name, 'encoder_source_sha256': seal,
                                 'source': {'sha256': 'c'*64}, 'calibration': None,
                                 'encoder_fixture_id': 'f'*64}}
             for name, seal in [('dense', 'a' * 64), ('expert', 'b' * 64)]}
    manifest = {'schema': 'tessera.cached_units.v2', 'source': {'sha256': 'source'},
                'units': units, 'wire_roots': {k: str(v) for k, v in roots.items()},
                'unit_roots': {'dense': 'old', 'expert': 'added'},
                'producer_packages': {seal: {'path': str(tmp_path / ('producer-' + seal[0])),
                                             'sha256': seal} for seal in ('a'*64, 'b'*64)},
                'reuse_authority': {**bound, 'checkpoint_encoder_source_sha256': 'a'*64,
                                    'encoder_source_proofs': []},
                'encoder_adoptions': {}, 'served_activation_policy': None, 'served_activations': {}}
    proof = {'schema': schemas['proof'], 'ok': True,
             'encoder_fixture_id_equal': True,
             'pins': {'old': {'encoder_source_sha256': 'a'*64},
                      'new': {'encoder_source_sha256': 'b'*64}},
             'fixture_id': {'ids': {'old': 'f'*64, 'new': 'f'*64}}}
    path = tmp_path / 'encoder-proof.json'
    path.write_text(json.dumps(proof))
    proof_bound = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest['reuse_authority']['encoder_source_proofs'] = [proof_bound]
    candidate = units['expert']['identity']
    manifest['encoder_adoptions']['expert'] = {
        'schema': schemas['adoption'],
        'reference_pair': ['expert', 'old-format'],
        'reference_encoding_identity': {**candidate, 'encoder_source_sha256': 'a'*64},
        'candidate_encoding_identity': candidate, 'encoder_source_proof': proof_bound}
    return manifest, roots
