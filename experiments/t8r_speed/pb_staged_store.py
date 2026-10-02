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
from types import SimpleNamespace


class StagedInputs:
    def __init__(self, manifest_path, *, sdk=None):
        if sdk is None:
            from prismabuild import client as sdk
        self.sdk = sdk
        raw = Path(manifest_path).read_bytes()  # immutable checkout/CAS input
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest, encoding = sdk.read_data_manifest(manifest_path)
        if encoding != "identity" or manifest != json.loads(raw):
            raise ValueError("sealed replay manifest changed during validation")
        self.entries = {(e['path'], e['offset']): e for e in manifest['entries']}
        if len(self.entries) != manifest['entry_count']:
            raise ValueError('duplicate input ranges')
        if any(not isinstance(e.get('sha256'), str) or len(e['sha256']) != 64
               for e in self.entries.values()):
            raise ValueError('single replay requires every input digest')
        bound = sdk.injected_context()
        if not bound.get('ok'):
            raise ValueError(f"pinned inputs refused: {bound.get('refusal')}")
        self.ctx = bound['ctx']
        # The public descriptor/release APIs consume queue.root; admission itself
        # constructs its authoritative PoolQueue inside acquire_for/injected_context.
        self.queue = SimpleNamespace(root=Path(self.ctx['queue_root']))
        mapping = sdk.read_residency_map(self.ctx['map_path'])
        if (mapping['consumer_action_key'] != self.ctx['action_key']
                or mapping['manifest_sha256'] != self.manifest_sha256):
            raise ValueError('residency map does not bind this action/readset')
        self.keys = {(p, off): sdk.residency_map_key(p, off) for p, off in self.entries}
        expected = {self.keys[k]: {'bytes': e['bytes'], 'sha256': e['sha256']}
                    for k, e in self.entries.items()}
        if any(k not in mapping['entries'] for k in expected):
            raise ValueError('residency map does not cover every declared range')
        epoch = mapping.get('epoch', '')
        root = Path(self.ctx['map_path']).parent.parent
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
            if digest != entry['sha256']:
                raise ValueError(f'pinned range digest differs: {identity}')
            self.reads.append({'path': identity[0], 'offset': identity[1],
                               'bytes': len(data), 'sha256': digest, 'serving_tier': serving})
            return data
        finally:
            os.close(fd)

    def json(self, path):
        return json.loads(self.read(path))

    def tensor(self, root, name, *, index=None):
        import torch
        if index is None:
            index = self.json(Path(root) / 'model.safetensors.index.json')['weight_map']
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
