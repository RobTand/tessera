"""PB-only bounded frame sealer for the existing routed diagnosis readset.

Independent publisher cached-unit identities authorize canonical outer frames;
unknown outer hashes are recorded only after checking the SAME owned bytes.
This performs no tensor/GPU decode, timings, cache or origin fallback.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

from pb_staged_store import StagedInputs

ROOT = '/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported'
MODULE = 'model.language_model.layers.10.mlp.experts'


def seal(manifest_path):
    reader = StagedInputs(manifest_path, allow_unsealed_wires=True)
    try:
        reader.json(ROOT + '/config.json')
        index = reader.json(ROOT + '/model.safetensors.index.json')['weight_map']
        published = reader.json(ROOT + '/tessera_serving_manifest.json')
        reader.bind_roles(ROOT, published['modules'][MODULE]['roles'])
        proof = {}
        for name in reader.roles:
            raw = reader.wire(ROOT, name, index=index)
            proof[name] = dict(reader.reads[-1]['cached_member'])
            del raw
        # Every declared range, including the routing capture, must be owned and
        # hash-checked before emitting a complete readset. No staged path reread.
        seen = {(r['path'],r['offset']) for r in reader.reads}
        for identity in reader.entries.keys() - seen:
            reader.read(*identity)
        doc = json.loads(json.dumps(reader.manifest))
        observed = {(r['path'],r['offset']):r for r in reader.reads}
        for entry in doc['entries']:
            got = observed[(entry['path'],entry['offset'])]
            if got['bytes'] != entry['bytes']:
                raise ValueError('sealed length changed')
            entry['sha256'] = got['sha256']
        doc['produced_by'] = {'tool':'PB pinned cached-inner/canonical-outer sealer',
                              'commit':os.environ['TESSERA_HEAD'],
                              'action_key':os.environ['PRISMABUILD_ACTION_KEY']}
        doc['annotations']['row_id'] = 'routed-gate826-sealed-one-BMT128'
        evidence = {'source':os.environ['TESSERA_HEAD'],
                    'action_key':os.environ['PRISMABUILD_ACTION_KEY'],
                    'provisional_manifest_sha256':reader.manifest_sha256,
                    'wire_count':len(proof), 'entries':len(observed),
                    'total_bytes':doc['total_bytes'], 'members':proof,
                    'reads':reader.reads, 'timing_claim':None, 'gpu_decode':False}
    finally:
        reader.close()
    return doc,evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    args = ap.parse_args()
    manifest,proof = seal(args.manifest)
    files = {'routed_gate_826_inputs.json':json.dumps(manifest,indent=2).encode()+b'\n',
             'sealed-wire-proof.json':json.dumps(proof,indent=2).encode()+b'\n'}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name,raw in files.items():
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0
            archive.addfile(info,io.BytesIO(raw))
    print(json.dumps({'wire_count':proof['wire_count'],'entries':proof['entries'],
                      'bytes':proof['total_bytes'],
                      'manifest_sha256':hashlib.sha256(files['routed_gate_826_inputs.json']).hexdigest()}))
    print('ROUTED_GATE_SEAL_BEGIN')
    print(base64.b64encode(stream.getvalue()).decode())
    print('ROUTED_GATE_SEAL_END')


if __name__ == '__main__':
    main()
