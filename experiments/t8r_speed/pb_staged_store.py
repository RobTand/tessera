"""Opt-in pinned inputs for the existing single-case kernel benchmark Store.

No origin fallback, cache, dispatcher, or tensor materialization framework.
Only PrismaBuild's public client SDK admits and opens the declared ranges.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
from pathlib import Path


def verify_cached_frame(raw, role):
    """Verify the producer's cached unit inside the actual exported frame.

    export_serving.pack_cached_expert_unit adds canonical TSRFUSE1 framing;
    its cached_blob_sha256 covers the inner unit, not the outer wire tensor.
    """
    from tessera.fused import parse_fused, pack_fused
    members = parse_fused(raw)
    if len(members) != 1:
        raise ValueError('cached expert wire must have exactly one member')
    member = members[0]
    if member.name != role['role'] or member.rows != role['rows']:
        raise ValueError('cached expert frame role/rows differ')
    inner = hashlib.sha256(member.blob).hexdigest()
    if inner != role['cached_blob_sha256']:
        raise ValueError('cached inner unit digest differs')
    if pack_fused([(member.name, member.rows, member.blob)]) != raw:
        raise ValueError('cached expert frame is not canonical')
    if len(raw) != role['blob_bytes']:
        raise ValueError('cached expert outer length differs')
    return {'inner_sha256':inner, 'inner_bytes':len(member.blob),
            'outer_sha256':hashlib.sha256(raw).hexdigest(), 'outer_bytes':len(raw),
            'role':member.name, 'rows':member.rows}


class StagedInputs:
    def __init__(self, manifest_path, *, sdk=None, allow_unsealed_wires=False):
        if sdk is None:
            from prismabuild import client as sdk
        self.sdk = sdk
        raw = Path(manifest_path).read_bytes()  # immutable checkout/CAS input
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest, encoding = sdk.read_data_manifest(manifest_path)
        if encoding != "identity" or manifest != json.loads(raw):
            raise ValueError("sealed replay manifest changed during validation")
        self.manifest = manifest
        self.entries = {(e['path'], e['offset']): e for e in manifest['entries']}
        if len(self.entries) != manifest['entry_count']:
            raise ValueError('duplicate input ranges')
        for (path, offset), entry in self.entries.items():
            digest = entry.get('sha256')
            if digest is None and allow_unsealed_wires and offset > 0 and path.endswith('.safetensors'):
                continue
            if not isinstance(digest,str) or len(digest)!=64:
                raise ValueError('single replay requires every input digest')
        bound = sdk.injected_context()
        if not bound.get('ok'):
            raise ValueError(f"pinned inputs refused: {bound.get('refusal')}")
        self.ctx = bound['ctx']
        self.queue = sdk.PoolQueue(Path(self.ctx['queue_root']))
        mapping = sdk.read_residency_map(self.ctx['map_path'])
        if mapping['manifest_sha256'] != self.manifest_sha256:
            raise ValueError('residency map does not bind this action/readset')
        self.keys = {(p, off): sdk.residency_map_key(p, off) for p, off in self.entries}
        if any(k not in mapping['entries'] for k in self.keys.values()):
            raise ValueError('residency map does not cover every declared range')
        expected = {self.keys[k]: {'bytes':e['bytes'],
                    'sha256':e['sha256'] or mapping['entries'][self.keys[k]]['sha256']}
                    for k,e in self.entries.items()}
        self.actual_digests = {k: expected[key]['sha256'] for k,key in self.keys.items()}
        epoch = mapping.get('epoch', '')
        root = self.queue.root / sdk.RESIDENCY
        covers = sdk.covers_for_keys(root, self.ctx['action_key'], list(expected),
                                    tier_id=mapping['tier_id'], epoch=epoch,
                                    manifest_sha256=self.manifest_sha256)
        if not covers.get('ok'):
            raise ValueError(f"pinned covers refused: {covers.get('refusal')}")
        held = sdk.acquire_for(self.ctx, tier_id=mapping['tier_id'], epoch=epoch,
                               covers=covers['covers'], expected=expected,
                               span={'start_bytes': 0, 'end_bytes': manifest['total_bytes']},
                               acquire_token=secrets.token_hex(16))
        if not held.get('ok'):
            raise ValueError(f"pinned inputs refused: {held.get('refusal')}")
        self.held = held
        self.closed = False
        self.reads = []
        self.headers = {}
        self.roles = {}

    def read(self, path, offset=0):
        if self.closed:
            raise ValueError('pinned inputs already released')
        identity = (str(path), int(offset))
        entry = self.entries.get(identity)
        if entry is None:
            raise ValueError(f'undeclared input range: {identity}')
        key = self.keys[identity]
        fd, serving = self.sdk.open_pinned(self.queue, self.held['pin'],
                                         self.held['ref_id'], key)
        try:
            # A staged range starts at byte zero. No source-path reread follows
            # authentication: this owned bytearray is exactly what the consumer gets.
            data = bytearray()
            while len(data) < entry['bytes']:
                chunk = os.read(fd, min(1 << 20, entry['bytes'] - len(data)))
                if not chunk:
                    raise ValueError('short pinned range')
                data.extend(chunk)
            if os.read(fd, 1):
                raise ValueError('oversized pinned range')
            digest = hashlib.sha256(data).hexdigest()
            if digest != self.actual_digests[identity]:
                raise ValueError(f'pinned range digest differs: {identity}')
            self.reads.append({'path': identity[0], 'offset': identity[1],
                               'bytes': len(data), 'sha256': digest, 'serving_tier': serving})
            return data
        finally:
            os.close(fd)

    def json(self, path):
        return json.loads(self.read(path))

    def bind_roles(self, root, roles):
        """Publisher metadata is already pinned; retain its cached-unit authority."""
        self.roles = {r['tensor'].removesuffix('.weight')+'.wire':r for r in roles}
        roster = {(r['expert'],r['role']) for r in roles}
        if (len(roles)!=864 or len(self.roles)!=864 or
            roster != {(e,r) for e in range(288) for r in ('gate_proj','up_proj','down_proj')} or
            any(not n.startswith('model.language_model.layers.10.mlp.experts.') for n in self.roles)):
            raise ValueError('single replay requires exactly the L10 expert-role roster')

    def wire(self, root, name, *, index):
        """One authenticated outer frame and its independently checked inner unit."""
        path = str(Path(root) / index[name])
        if path not in self.headers:
            raw = self.read(path)
            n = struct.unpack('<Q', raw[:8])[0]
            if len(raw) != n + 8:
                raise ValueError('pinned safetensors header length differs')
            self.headers[path] = (n, json.loads(raw[8:]))
        n, header = self.headers[path]
        record = header[name]
        if record['dtype'] != 'U8' or len(record['shape']) != 1:
            raise ValueError('single replay accepts one-dimensional U8 wires only')
        start, end = record['data_offsets']
        raw = self.read(path, 8 + n + start)
        if len(raw) != end - start or len(raw) != record['shape'][0]:
            raise ValueError('wire header differs from owned bytes')
        if name not in self.roles:
            raise ValueError('wire lacks independent cached-unit authority')
        identity = verify_cached_frame(raw, self.roles[name])
        self.reads[-1]['cached_member'] = identity
        return raw

    def tensor(self, root, name, *, index):
        import torch
        raw = self.wire(root,name,index=index)
        # frombuffer retains THIS authenticated owner; no mutable path is opened
        # again. The unchanged intake consumes this exact tensor.
        return torch.frombuffer(raw, dtype=torch.uint8)

    def close(self):
        if not self.closed:
            result = self.sdk.release(self.queue, self.held['pin_id'], self.held['ref_id'],
                                      consumer_action_key=self.ctx['action_key'],
                                      stage_root=self.held['pin']['stage_root'])
            if result is not True:
                raise ValueError(f'pinned input release failed: {result}')
            self.closed = True
