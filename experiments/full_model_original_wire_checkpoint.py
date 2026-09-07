"""Frame every measured E4M3/R1024 campaign original into a research checkpoint."""
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
import export_tessera_serving as exporter
from tessera.cached_unit import encoder_source_sha256
from tessera.fused import parse_fused
from tessera.serving_parts import source_identity, sha256_file

ROOT = Path('/mnt/shared/tessera-clean-runtime-20260907/full-model-original-r1024')
SRC = Path('/mnt/shared/models/LFM2.5-8B-A1B-BF16')
MERGED = Path('/mnt/shared/tessera-measurements/first-model-20260907/full-model-anchors-merged-01')
CENSUS = Path('/mnt/shared/tessera-measurements/first-model-20260907/full-model-anchors-02/census.json')
FORMAT = 'TESSERA_E4M3_K1_R1024'
ANCHORS_SHA = 'ff3fc7dd8397833f5e40e4d974b817145fc0678794ad840ecdcb1e98f7d30259'
CAMPAIGN_ID = '257f6fc9cbb3df0b9ff0f476905da1dd717654d9e0a9ae6b9ce7974189b17cb1'


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'artifact': str(path), 'sha256': sha256_file(path),
                      'bytes': path.stat().st_size}), flush=True)


def main():
    started = time.time()
    torch.set_num_threads(4)
    assert encoder_source_sha256() == '57809bff862b880dc397e6d271a80c04d6c87d1af3bba12c076648fc5443355c'
    anchors_path = MERGED / 'cost.anchors.json'
    assert sha256_file(anchors_path) == ANCHORS_SHA
    anchors = json.loads(anchors_path.read_text())
    assert anchors['identity_sha256'] == CAMPAIGN_ID
    assert sha256_file(CENSUS) == '62d41825f84edd280de46bbb89893676fae8203563834dc95d8ad29c05bad04d'
    census = json.loads(CENSUS.read_text())
    expected = set(census['dense_targets']) | set(census['expert_targets'])
    assert len(expected) == 2142 and len(census['dense_targets']) == 30
    assert {entry['qname'] for entry in anchors['units']} == expected
    bundle = ROOT / 'cached-units'
    bundle.mkdir(exist_ok=False)
    records, origins = {}, {}
    for entry in anchors['units']:
        name = entry['qname']
        part = MERGED / 'cost.anchors.json.parts' / entry['file']
        envelope = pickle.loads(part.read_bytes())
        assert envelope['qname'] == name and envelope['identity_sha256'] == CAMPAIGN_ID
        assert hashlib.sha256(envelope['payload']).hexdigest() == envelope['payload_sha256']
        payload = pickle.loads(envelope['payload'])
        selected = [a for a in payload['anchors'] if a['format_name'] == FORMAT]
        assert len(selected) == 1 and selected[0]['qname'] == name and selected[0]['hessian_applied']
        record = payload['wire_records'][FORMAT]
        assert record['identity']['unit'] == name
        assert record['identity']['recipe']['grid'] == 'E4M3' and record['identity']['recipe']['q256'] == 1024
        filename = record['file']
        assert Path(filename).name == filename
        wire = MERGED / 'cache/wire' / filename
        assert sha256_file(wire) == record['blob_sha256']
        assert wire.stat().st_size == record['blob_bytes'] == selected[0]['wire_bytes']
        shutil.copyfile(wire, bundle / filename)
        records[name] = record
        origins[name] = {'part_sha256': sha256_file(part), 'payload_sha256': envelope['payload_sha256'],
                         'original_wire': str(wire), 'original_blob_sha256': record['blob_sha256']}
    source = source_identity(SRC)
    write('cached-units/manifest.json', {'schema': 'tessera.cached_units.v1', 'source': source, 'units': records})
    write('originals.json', {'anchors_sha256': ANCHORS_SHA, 'campaign_identity_sha256': CAMPAIGN_ID,
                           'census_sha256': sha256_file(CENSUS), 'units': origins})
    _, dense, _, _ = exporter.quantizable(SRC)
    plan = {name: 'PASSTHROUGH' for name in dense if not exporter.MOE_ROUTER.match(name)}
    for name in census['dense_targets']:
        plan[name + '.weight'] = {'grid': 'E4M3', 'q256': 1024}
    stacks = {records[name]['identity']['projection']['tensor'].rsplit('.', 3)[0]
              for name in census['expert_targets']}
    # The projection's logical tensor ends .<expert>.<role>.weight.
    assert len(stacks) == 22
    for stack in stacks:
        plan[stack] = {'grid': 'E4M3', 'q256': 1024, 'source_layout': 'unpacked_per_expert'}
    write('plan.json', plan)
    write('source-identity.json', source)
    hessian = MERGED / 'cache/hessian_capture.pt'
    hp = hessian.with_name(hessian.name + '.provenance.json')
    assert json.loads(hp.read_text())['capture_sha256'] == 'e2781e6c12097b50d2edb4d57fa48112eb9782e7ce776f161546d25b3ac654c2'
    write('hessian-input.json', {'path': str(hessian), 'sha256': sha256_file(hessian),
                                'provenance_sha256': sha256_file(hp)})
    cmd = [sys.executable, str(Path(exporter.__file__)), str(SRC), str(ROOT / 'checkpoint'),
           '--grid', 'E4M3', '--q256', '1024', '--device', 'cuda', '--plan-json', str(ROOT / 'plan.json'),
           '--cached-units', str(bundle / 'manifest.json'), '--hessian', str(hessian)]
    write('export-command.json', {'argv': cmd})
    subprocess.run(cmd, check=True)
    out = ROOT / 'checkpoint'
    manifest = json.loads((out / 'tessera_serving_manifest.json').read_text())
    assert manifest['cached_units']['planned_units'] == len(expected)
    assert json.loads((out / 'config.json').read_text())['quantization_config']['quant_method'] == 'tessera'
    replacements = {name + '.weight' for name in expected}
    wire_proofs, passthrough, output_wires = {}, {}, set()
    with safe_open(str(SRC / 'model.safetensors'), framework='pt') as before, \
            safe_open(str(out / 'model.safetensors'), framework='pt') as after:
        for module, info in manifest['modules'].items():
            if module in stacks:
                for role in info['roles']:
                    name = role['tensor'].removesuffix('.weight')
                    key = name + '.wire'
                    members = parse_fused(after.get_tensor(key).numpy().tobytes())
                    assert len(members) == 1
                    digest = hashlib.sha256(members[0].blob).hexdigest()
                    assert digest == records[name]['blob_sha256']
                    wire_proofs[name] = {'sha256': digest, 'tensor': key}
                    output_wires.add(key)
            else:
                key = module + '.wire_bytes'
                members = parse_fused(after.get_tensor(key).numpy().tobytes())
                assert len(members) == len(info['roles'])
                for member, role in zip(members, info['roles']):
                    name = role['tensor'].removesuffix('.weight')
                    assert member.name == role['role'] and member.rows == role['rows']
                    digest = hashlib.sha256(member.blob).hexdigest()
                    assert digest == records[name]['blob_sha256']
                    wire_proofs[name] = {'sha256': digest, 'tensor': key, 'role': member.name}
                output_wires.add(key)
        assert set(wire_proofs) == expected
        keep = set(before.keys()) - replacements
        assert set(after.keys()) == keep | output_wires
        for name in sorted(keep):
            a, b = before.get_tensor(name), after.get_tensor(name)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
            passthrough[name] = {'shape': list(a.shape), 'dtype': str(a.dtype)}
    write('export-proof.json', {'schema': 'tessera.full_model_original_wire_checkpoint.v1', 'status': 'passed',
        'scope': 'Uniform measured E4M3/R1024 research reference. All 2142 original campaign blobs; every other tensor bit-exact source precision. Not allocator-derived or a production default.',
        'wires': wire_proofs, 'passthrough': passthrough, 'anchors_sha256': ANCHORS_SHA,
        'originals_sha256': sha256_file(ROOT / 'originals.json'),
        'plan_sha256': sha256_file(ROOT / 'plan.json'),
        'source_identity_sha256': sha256_file(ROOT / 'source-identity.json'),
        'checkpoint_files': {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                             for p in sorted(out.iterdir()) if p.is_file()},
        'started_epoch': started, 'finished_epoch': time.time()})


if __name__ == '__main__':
    main()
