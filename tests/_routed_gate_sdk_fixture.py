"""Public SDK schema fixture; import-time dependency belongs to SDK population."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from prismabuild import client
from test_routed_gate_staged_store import reader_class

def fixture(tmp_path, payload=b'owned-wire', expected=None, *, origin='/forbidden-origin/wire', offset=9):
    staged = tmp_path / 'staged'
    staged.write_bytes(payload)
    entry = {'path': origin, 'offset': offset, 'bytes': len(b'owned-wire'),
             'sha256': expected or hashlib.sha256(b'owned-wire').hexdigest()}
    manifest = {'entries': [entry], 'entry_count': 1, 'total_bytes': entry['bytes']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    key = client.residency_map_key(origin, offset)
    action = 'a'*64
    opened = []
    sdk = SimpleNamespace(
        read_data_manifest=lambda p: (json.loads(Path(p).read_text()), 'identity'),
        injected_context=lambda: {'ok': True, 'ctx': {'action_key': action,
            'queue_root': str(tmp_path/'queue'), 'map_path': str(tmp_path/'queue'/client.RESIDENCY/'a.map.json')}},
        read_residency_map=lambda p: client.validate_residency_map({
            'schema': client.RESIDENCY_MAP_SCHEMA_V1, 'stage_root': str(tmp_path),
            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'tier_id': 'fixture', 'leads': ['f'*64], 'generation': 0,
            'entries': {key: {'stage_path':str(staged),'offset':offset,
                'bytes':entry['bytes'],'sha256':entry['sha256']}}}),
        residency_map_key=client.residency_map_key,
        PoolQueue=client.PoolQueue, RESIDENCY=client.RESIDENCY,
        covers_for_keys=lambda *a,**k: {'ok': True, 'covers': [],
            'expected':{key:{'bytes':entry['bytes'],'sha256':entry['sha256']}}},
        acquire_for=lambda *a,**k: {'ok': True, 'pin_id': 'pin', 'ref_id': 'ref',
                                  'pin': {'stage_root': str(tmp_path)}},
        release=lambda *a,**k: True,
    )
    def pinned(*args, **kwargs):
        fd = os.open(staged, os.O_RDONLY)
        opened.append(fd)
        return fd, {'range_ref': key}
    sdk.open_pinned = pinned
    return reader_class()(path, sdk=sdk), staged, opened, sdk


