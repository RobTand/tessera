"""An explicitly bound mixed catalog retains original files and producers."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tessera.cached_unit import CachedUnitBundle


def rooted(tmp_path):
    roots = {key: tmp_path / key for key in ('old', 'added')}
    for path in roots.values():
        path.mkdir()
        (path / 'unit.tessera').write_bytes(b'unchanged wire')
    bound = {}
    for name, schema in [('catalog_extension', 'prismaquant.joint_catalog_extension.v1'),
                         ('candidate_overlay', 'prismaquant.t4_adopted_catalog.v1')]:
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
    proof = {'schema': 'prismaquant.reseal_proof_bundle.v1', 'ok': True,
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
        'schema': 'prismaquant.joint_catalog_source_adoption.v1',
        'reference_pair': ['expert', 'old-format'],
        'reference_encoding_identity': {**candidate, 'encoder_source_sha256': 'a'*64},
        'candidate_encoding_identity': candidate, 'encoder_source_proof': proof_bound}
    return manifest, roots


def test_roots_allow_same_leaf_without_copying_or_relabeling(tmp_path):
    manifest, roots = rooted(tmp_path)
    stats = {k: (p / 'unit.tessera').stat() for k, p in roots.items()}
    bundle = CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'])
    for name in manifest['units']:
        blob, record = bundle.read(name)
        assert blob == b'unchanged wire' and record == manifest['units'][name]
    for key, path in roots.items():
        after = (path / 'unit.tessera').stat()
        assert (after.st_ino, after.st_ctime_ns) == (stats[key].st_ino, stats[key].st_ctime_ns)


@pytest.mark.parametrize('change', ['missing_root', 'extra_unit', 'aliased_roots', 'unbound_producer', 'missing_authority'])
def test_rooted_coverage_and_authority_are_closed(tmp_path, change):
    manifest, _ = rooted(tmp_path)
    if change == 'missing_root':
        del manifest['unit_roots']['expert']
    elif change == 'extra_unit':
        manifest['unit_roots']['foreign'] = 'old'
    elif change == 'aliased_roots':
        manifest['wire_roots']['added'] = manifest['wire_roots']['old']
    elif change == 'unbound_producer':
        del manifest['producer_packages']['b'*64]
    else:
        del manifest['reuse_authority']
    with pytest.raises(ValueError):
        CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'])


def test_root_and_wire_symlink_changes_refuse(tmp_path):
    manifest, roots = rooted(tmp_path)
    bundle = CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'])
    wire = roots['added'] / 'unit.tessera'
    wire.unlink()
    wire.symlink_to(roots['old'] / 'unit.tessera')
    with pytest.raises(ValueError, match='escapes'):
        bundle.read('expert')


def test_actual_mixed_producers_export_complete_dense_and_expert_roster(tmp_path, monkeypatch):
    """Real historical factories and exporter consume both roots, never encode."""
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open
    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear
    from tessera.fused import parse_fused
    from tessera.historical_producer import load_historical_producer
    from tessera.serving_parts import source_identity
    from test_cached_producer import _distinct_producer, _exporter, STACK

    exporter = _exporter()
    source = tmp_path / 'checkpoint'; source.mkdir()
    roots = {key: tmp_path / key for key in ('old', 'added')}
    producers, packages = {}, {}
    for key in roots:
        roots[key].mkdir()
        parent = tmp_path / ('producer-' + key); parent.mkdir()
        package, seal = _distinct_producer(parent)
        producers[key] = load_historical_producer(package, seal)
        packages[seal] = {'path': str(package), 'sha256': seal}
    weight = torch.randn(32, 32, generator=torch.Generator().manual_seed(74)).bfloat16()
    blob = encode_linear(weight.float(), grid=E4M3_GRID, q256=1024,
                         name='TESSERA_E4M3_K1_R1024', verify=False).blob
    dense = {f'model.layers.0.feed_forward.{role}.weight': weight for role in ('w1', 'w3', 'w2')}
    tensors = {**dense, **{f'{STACK}.0.{role}.weight': weight for role in ('w1', 'w3', 'w2')}}
    save_file({name: value.clone() for name, value in tensors.items()}, str(source / 'model.safetensors'))
    config = {'architectures': ['Lfm2MoeForCausalLM'], 'hidden_size': 32,
              'moe_intermediate_size': 32, 'num_experts': 1}
    (source / 'config.json').write_text(json.dumps(config))
    choices = {name: {'grid': 'E4M3', 'q256': 1024} for name in dense}
    choices[STACK] = {'grid': 'E4M3', 'q256': 1024}
    projected = exporter.project_expert_plan({k: list(v.shape) for k, v in tensors.items()},
                                             config, {STACK: choices[STACK]})
    units = [(name, None, 'old') for name in dense]
    units += [(unit['tensor'], unit, 'added') for unit in projected['stacks'][STACK]['units']]
    records, owners, adoptions, identities = {}, {}, {}, {}
    for index, (name, unit, owner) in enumerate(units):
        identity = exporter.cached_input_identity(producers[owner], weight, name, unit, E4M3_GRID, 1024)
        key = identity['unit']; identities[owner] = identity
        filename = f'unit-{index % 3}.tessera'  # same leaves in independent immutable roots
        (roots[owner] / filename).write_bytes(blob)
        records[key] = {'file': filename, 'blob_bytes': len(blob),
                        'blob_sha256': hashlib.sha256(blob).hexdigest(), 'identity': identity}
        owners[key] = owner
        if owner == 'added':
            reference = exporter.cached_input_identity(producers['old'], weight, name, unit, E4M3_GRID, 1024)
            adoptions[key] = {'schema': 'prismaquant.joint_catalog_source_adoption.v1',
                'reference_pair': [key, 'old-format'], 'reference_encoding_identity': reference,
                'candidate_encoding_identity': identity}
    def bound(name, value):
        path = tmp_path / (name + '.json'); path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    old, new = (identities[key]['encoder_source_sha256'] for key in ('old', 'added'))
    fixture = identities['old']['encoder_fixture_id']
    proof = bound('proof', {'schema': 'prismaquant.reseal_proof_bundle.v1', 'ok': True,
        'encoder_fixture_id_equal': True, 'pins': {'old': {'encoder_source_sha256': old},
        'new': {'encoder_source_sha256': new}}, 'fixture_id': {'ids': {'old': fixture, 'new': fixture}}})
    for adoption in adoptions.values(): adoption['encoder_source_proof'] = proof
    authority = {'catalog_extension': bound('extension', {'schema': 'prismaquant.joint_catalog_extension.v1'}),
        'candidate_overlay': bound('overlay', {'schema': 'prismaquant.t4_adopted_catalog.v1'}),
        'checkpoint_encoder_source_sha256': old, 'encoder_source_proofs': [proof]}
    manifest = {'schema': 'tessera.cached_units.v2', 'source': source_identity(source), 'units': records,
        'wire_roots': {key: str(path) for key, path in roots.items()}, 'unit_roots': owners,
        'producer_packages': packages, 'reuse_authority': authority, 'encoder_adoptions': adoptions,
        'served_activation_policy': None, 'served_activations': {}}
    manifest_path = tmp_path / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    plan_path = tmp_path / 'plan.json'; plan_path.write_text(json.dumps(choices))
    before = {p: (p.stat().st_ino, p.stat().st_ctime_ns) for root in roots.values() for p in root.iterdir()}
    def forbidden(*args, **kwargs): raise AssertionError('mixed cached export encoded a wire')
    monkeypatch.setattr(exporter, 'encode_linear_planes', forbidden)
    monkeypatch.setattr(exporter, 'output_partitions', lambda census, module: [32, 32] if module.endswith('.w13') else [32])
    out = tmp_path / 'out'
    monkeypatch.setattr('sys.argv', ['export', str(source), str(out), '--plan-json', str(plan_path),
        '--cached-units', str(manifest_path), '--device', 'cpu', '--allow-unrouted', '--allow-unserveable'])
    exporter.main()
    with safe_open(str(out / 'model.safetensors'), framework='pt') as handle:
        actual = [member.blob for name in handle.keys() if name.endswith(('.wire', '.wire_bytes'))
                  for member in parse_fused(handle.get_tensor(name).numpy().tobytes())]
    assert len(actual) == len(tensors) == 6 and all(item == blob for item in actual)
    assert all((p.stat().st_ino, p.stat().st_ctime_ns) == fence for p, fence in before.items())
    receipt = json.loads((out / 'tessera_serving_manifest.json').read_text())
    assert receipt['cached_units']['planned_units'] == 6
    assert receipt['cached_units']['producer_packages'] == packages
    assert receipt['cached_units']['served_activation_policy'] == manifest['served_activation_policy']
    assert receipt['cached_units']['served_activations'] == manifest['served_activations']


def test_added_a4_cannot_omit_served_activation_policy(tmp_path):
    manifest, _ = rooted(tmp_path)
    identity = manifest['units']['expert']['identity']
    identity['recipe'] = {'grid': 'E2M1x2', 'q256': 896}
    with pytest.raises(ValueError, match='bound policy'):
        CachedUnitBundle(manifest, tmp_path, {'dense', 'expert'}, manifest['source'])
