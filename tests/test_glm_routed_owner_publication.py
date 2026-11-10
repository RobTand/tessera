"""The real routed-owner producer publishes both tiny-source receipts."""
import json
from pathlib import Path

import pytest
import torch

from experiments import glm_routed_owner_inputs as inputs


@pytest.mark.parametrize('mode', ['source', 'rate'])
def test_each_mode_reaches_its_published_receipt_on_a_tiny_source(tmp_path, monkeypatch, mode):
    from datetime import datetime
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from tessera import cached_unit, encoder_identity
    unit = 'model.language_model.layers.3.mlp.experts.0.gate_proj'
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    model = tmp_path / 'model'
    model.mkdir()
    save_file({unit + '.weight': weight}, str(model / 'shard.safetensors'))
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {unit + '.weight': 'shard.safetensors'}}))
    shared = tmp_path / 'source.safetensors'
    save_file({inputs.SOURCE_PREFIX + unit: weight}, str(shared))
    wire = tmp_path / 'wire'
    wire.mkdir()
    (wire / 'unit.wire').write_bytes(b'controlled cached unit')
    seal = 'a' * 64
    identity = {'source': cached_unit.tensor_identity(weight), 'encoder_source_sha256': seal,
                'encoder_fixture_id': 'b' * 64}
    manifest = {'units': {unit: {'identity': identity, 'file': 'unit.wire',
                                'blob_sha256': inputs.sha256_file(wire / 'unit.wire'),
                                'blob_bytes': (wire / 'unit.wire').stat().st_size}}}
    served = {'cached_units': {'manifest_sha256': 'c' * 64, 'historical_producer': {'source_sha256': seal},
                               'hessian_identity': {}}, 'activation_aware': {}}
    plan = {'grid': 'E4M3', 'q256': 1024, 'source_layout': 'unpacked_per_expert'}
    entry = {'experts': 1, 'hidden_size': 4, 'intermediate_size': 8, 'unit_of': {unit: {}}}
    member = {'unit': unit, 'rows': 8, 'cols': 4, 'expert': 0, 'role': 'w1'}
    monkeypatch.setattr(inputs, 'load_served_manifest', lambda path: served)
    monkeypatch.setattr(inputs, 'resolve_plan', lambda document, stack: plan)
    monkeypatch.setattr(inputs, 'layer_units', lambda *args: entry)
    monkeypatch.setattr(inputs, 'member_roster', lambda value: [member])
    monkeypatch.setattr(inputs, 'bundle_manifest', lambda *args: manifest)
    closed = []
    monkeypatch.setattr(inputs, 'activation_source', lambda *args: SimpleNamespace(
        hessians=SimpleNamespace(close=lambda: closed.append(True))))
    def identify(source, *args):
        return {**identity, 'source': cached_unit.tensor_identity(source)}
    monkeypatch.setattr(inputs, 'identity_factory', lambda *args: (identify, None))
    # Codec acceptance is covered separately; this control drives the real producer's publication path.
    monkeypatch.setattr(cached_unit, 'make_unit_record', lambda blob, identity, filename: {
        'file': filename, 'blob_bytes': len(blob), 'identity': identity})
    monkeypatch.setattr(cached_unit, 'encoder_source_sha256', lambda: seal)
    monkeypatch.setattr(encoder_identity, 'encoder_fixture_id', lambda: bytes.fromhex('b' * 64))
    args = SimpleNamespace(model=model, export=tmp_path, bundle=tmp_path / 'bundle.json',
        out=tmp_path / 'out', stack='stack', layer=3, source_file=shared,
        producer_authority=None, wire_dir=wire, rate='a8', format='test-format')
    assert getattr(inputs, mode + '_mode')(args) == 0
    receipt = args.out / ('source-receipt.json' if mode == 'source' else 'a8/rate-receipt.json')
    observed = json.loads(receipt.read_text())
    assert observed['mode'] == mode
    datetime.strptime(observed['written'], '%Y-%m-%dT%H:%M:%SZ')
    assert observed['verified']
    if mode == 'rate':
        assert closed == [True]
        assert len(observed['member_records']) == 1
    else:
        assert observed['verified']['tensor_identity_checked_against_the_bundles_sealed_source'] == 1
