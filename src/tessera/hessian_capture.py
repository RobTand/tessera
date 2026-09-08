"""Opt-in Hessian commitments over existing canonical capture files.

This is an input reader, not a tensor cache. Metadata is bound through held
regular files; every value lookup verifies and copies one bounded H, retaining
no tensor. Legacy capture loading and the Hessian content-seal grammar do not
change. Commitments are not reported as verification of unconsumed payloads.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import threading

from .grammar import GrammarError

REFERENCE_SCHEMA = 'tessera.hessian_capture.references.v1'
LOAD_SCHEMA = 'tessera.hessian_reference_load.v1'
BINDING_SCHEMA = 'tessera.canonical_hessian_binding.v1'
CANONICAL_SCHEMA = 'prismaquant.tessera_calibration_cache.v2'
CANONICAL_SOURCE = 'tessera_campaign_prefix_f32_v1'
# The first JSON cannot declare the bound under which it is first parsed.
MAX_METADATA_BYTES = 128*1024**2
READ_BYTES = 8*1024**2


def _sha(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value))


def _positive(value):
    return type(value) is int and value > 0


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise GrammarError(f'duplicate Hessian reference JSON key: {key}')
        result[key] = value
    return result


def _signature(value):
    return tuple(getattr(value, name) for name in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


class _HeldFile:
    """A verified regular object whose pathname must keep naming it."""
    def __init__(self, path, *, cap):
        self.path = Path(path)
        self.fd = None
        if not self.path.is_absolute() or self.path.resolve() != self.path:
            raise GrammarError('Hessian reference paths must be absolute and nonsymlink')
        try:
            # A replaced FIFO must reach fstat rather than wait for a writer.
            # O_NONBLOCK does not change regular-file reads; the same held
            # descriptor must still pass the regular-file and size checks.
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            self.stat = os.fstat(self.fd)
            if not stat.S_ISREG(self.stat.st_mode) or not 0 < self.stat.st_size <= cap:
                raise GrammarError('Hessian reference file exceeds its regular-file byte bound')
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.fd is None:
            raise GrammarError('Hessian reference owner is closed')
        if any(_signature(value) != _signature(self.stat) for value in
               (os.fstat(self.fd), self.path.lstat())):
            raise GrammarError(f'Hessian reference source changed or was replaced: {self.path}')

    def read(self, *, expected=None, keep=False):
        digest = hashlib.sha256()
        pieces = [] if keep else None
        offset = 0
        while offset < self.stat.st_size:
            self.check()
            block = os.pread(self.fd, min(READ_BYTES, self.stat.st_size-offset), offset)
            if not block:
                raise GrammarError('Hessian reference source ended during its verified read')
            digest.update(block)
            if keep:
                pieces.append(block)
            offset += len(block)
            # This is read-only input, with no dirty pages to fsync.
            os.posix_fadvise(self.fd, 0, offset, os.POSIX_FADV_DONTNEED)
        self.check()
        actual = digest.hexdigest()
        if expected is not None and actual != expected:
            raise GrammarError(f'Hessian reference file checksum mismatch: {self.path}')
        return (b''.join(pieces) if keep else None), actual

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def normalize_reference_binding(value):
    if (not isinstance(value, Mapping) or set(value) !=
            {'schema', 'canonical_capture_sha256', 'census_sha256'} or
            value.get('schema') != BINDING_SCHEMA or
            not all(_sha(value.get(key)) for key in ('canonical_capture_sha256', 'census_sha256'))):
        raise GrammarError('canonical Hessian binding requires exact manifest and census SHA-256 values')
    return dict(value)


def normalize_reference_load_policy(policy):
    if (not isinstance(policy, dict) or set(policy) !=
            {'schema', 'max_metadata_bytes', 'max_file_bytes', 'max_hessian_bytes'} or
            policy['schema'] != LOAD_SCHEMA or
            not all(_positive(policy[k]) for k in policy if k != 'schema') or
            policy['max_metadata_bytes'] > MAX_METADATA_BYTES or
            policy['max_hessian_bytes'] > policy['max_file_bytes']):
        raise ValueError('Hessian reference load policy has invalid byte bounds')
    return dict(policy)


def capture_sha256_from_units(provenance, units):
    """The existing v1 seal, given already bound per-unit tensor commitments."""
    from .export import CAPTURE_CONTEXT, HESSIAN_IDENTITY
    identity = {field: provenance.get(field) for field in (*HESSIAN_IDENTITY, *CAPTURE_CONTEXT)}
    digest = hashlib.sha256()
    digest.update(json.dumps({'schema': 'tessera.hessian_capture.v1', 'identity': identity},
                             sort_keys=True, default=str).encode())
    for name in sorted(units):
        value = units[name]
        if not isinstance(name, str) or not name or not _sha(value):
            raise GrammarError('Hessian capture commitments require named SHA-256 values')
        digest.update(b'\0'+name.encode()+b'\0')
        digest.update(value.encode())
    return digest.hexdigest()


class ReferenceHessians(Mapping):
    """A bounded, checked Mapping over one complete canonical capture.

    The owner retains JSON metadata and three file descriptors, never H/X.
    A lookup owns at most one capped source mapping and one returned H copy;
    the existing tensor identity helper also stages at most one H-sized byte
    string. Caller-owned H values and encoder workspaces are additional.
    """
    def __init__(self, path):
        self._held = []
        self._closed = False
        self._lock = threading.RLock()
        self._verified = set()
        self._reads = self._read_bytes = self._peak_file = self._peak_h = 0
        self._live_payloads = 0
        try:
            self._document, self.document_sha256 = self._metadata(path, MAX_METADATA_BYTES)
            p = self._document
            if not isinstance(p, dict) or set(p) != {
                    'schema', 'canonical_capture', 'census', 'provenance', 'counts',
                    'hessians', 'capture_sha256', 'rows', 'load_policy'} or p['schema'] != REFERENCE_SCHEMA:
                raise GrammarError('unsupported or non-closed Hessian reference capture')
            policy = normalize_reference_load_policy(p['load_policy'])
            if self._held[0].stat.st_size > policy['max_metadata_bytes']:
                raise GrammarError('Hessian reference metadata exceeds its byte bound')
            self._policy = policy
            for key in ('canonical_capture', 'census'):
                if not isinstance(p[key], dict) or set(p[key]) != {'path', 'sha256'} or not _sha(p[key]['sha256']):
                    raise GrammarError('Hessian reference must bind its canonical manifest and census')
            self._canonical, _ = self._metadata(p['canonical_capture']['path'],
                policy['max_metadata_bytes'], p['canonical_capture']['sha256'])
            self._census, _ = self._metadata(p['census']['path'],
                policy['max_metadata_bytes'], p['census']['sha256'])
            self._validate()
        except BaseException:
            self.close()
            raise

    def _metadata(self, path, cap, expected=None):
        owned = _HeldFile(path, cap=cap)
        self._held.append(owned)
        raw, digest = owned.read(expected=expected, keep=True)
        try:
            return json.loads(raw, object_pairs_hook=_pairs), digest
        except (ValueError, UnicodeError) as error:
            raise GrammarError(f'invalid Hessian reference metadata: {error}') from error

    def _validate(self):
        p, canonical, census = self._document, self._canonical, self._census
        identity = canonical.get('identity') or {}
        units, entries = identity.get('units'), canonical.get('entries')
        if (canonical.get('schema') != CANONICAL_SCHEMA or canonical.get('status') != 'complete' or
                identity.get('schema') != CANONICAL_SCHEMA or
                identity.get('storage_source') != CANONICAL_SOURCE or
                identity.get('census_sha256') != p['census']['sha256'] or
                not isinstance(units, dict) or not units or not isinstance(entries, dict) or
                set(entries) != set(units) or census.get('unit_shapes') != units or
                not _positive(identity.get('max_act_rows'))):
            raise GrammarError('Hessian references require one complete canonical capture and its exact census')
        if (not isinstance(p['counts'], dict) or p['counts'] != census.get('counts') or
                set(p['counts']) != set(units) or
                not all(_positive(n) for n in p['counts'].values()) or
                not isinstance(census.get('max_abs'), dict) or set(census['max_abs']) != set(units)):
            raise GrammarError('Hessian reference counts/roster differ from the canonical census')
        provenance = p['provenance']
        if (not isinstance(provenance, dict) or provenance.get('hessian_role') != 'fit' or
                {k:v for k,v in provenance.items() if k != 'hessian_role'} != identity.get('calibration')):
            raise GrammarError('Hessian reference provenance differs from the fit canonical capture')
        hessians = p['hessians']
        if not isinstance(hessians, dict) or not hessians or not set(hessians) <= set(units):
            raise GrammarError('Hessian reference has a missing or foreign unit roster')
        root = Path(p['canonical_capture']['path']).parent
        paths = set()
        for name, item in hessians.items():
            if (not isinstance(item, dict) or set(item) != {'algorithm', 'dtype', 'shape', 'sha256'} or
                    item['algorithm'] != 'sha256.dtype_shape_contiguous.v1' or
                    item['dtype'] != 'torch.float32' or not _sha(item['sha256'])):
                raise GrammarError(f'{name}: unsupported Hessian tensor commitment')
            shape = units[name]
            if (not isinstance(shape, list) or len(shape) != 2 or not all(_positive(v) for v in shape) or
                    item['shape'] != [shape[1], shape[1]] or
                    4*shape[1]**2 > self._policy['max_hessian_bytes']):
                raise GrammarError(f'{name}: Hessian geometry exceeds the declared byte bound')
            entry = entries[name]
            relative = PurePosixPath(entry.get('path', ''))
            if (set(entry) != {'path', 'sha256'} or not _sha(entry['sha256']) or
                    relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != 'inputs' or
                    '..' in relative.parts or str(relative) != entry['path'] or
                    not str(relative).endswith('.pt')):
                raise GrammarError(f'{name}: noncanonical Hessian source reference')
            path = root/str(relative)
            if path in paths:
                raise GrammarError('two Hessian units name one canonical source file')
            paths.add(path)
        commitments = {name: item['sha256'] for name,item in hessians.items()}
        if not _sha(p['capture_sha256']) or capture_sha256_from_units(provenance, commitments) != p['capture_sha256']:
            raise GrammarError('Hessian reference commitments disagree with the capture seal')
        rows = p['rows']
        seen = set()
        if not isinstance(rows, list) or not rows:
            raise GrammarError('Hessian reference requires its priced row commitment proofs')
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'units', 'capture_sha256'} or
                    not isinstance(row['units'], list) or not row['units'] or
                    any(not isinstance(n, str) for n in row['units']) or
                    row['units'] != sorted(set(row['units'])) or
                    not set(row['units']) <= set(commitments) or seen & set(row['units'])):
                raise GrammarError('Hessian reference rows have duplicate, foreign or malformed coverage')
            if capture_sha256_from_units(provenance, {n:commitments[n] for n in row['units']}) != row['capture_sha256']:
                raise GrammarError('Hessian reference row commitments disagree with their priced seal')
            seen.update(row['units'])
        if seen != set(commitments):
            raise GrammarError('Hessian reference row proofs do not cover the exact committed roster')

    def require_current(self):
        if self._closed:
            raise GrammarError('Hessian reference owner is closed')
        for held in self._held:
            held.check()

    @property
    def descriptor(self):
        self.require_current()
        return copy.deepcopy(self._document)

    @property
    def provenance(self):
        self.require_current()
        return dict(self._document['provenance'])

    @property
    def counts(self):
        self.require_current()
        return dict(self._document['counts'])

    def canonical_manifest(self):
        self.require_current()
        return copy.deepcopy(self._canonical)

    def require_census(self, census):
        self.require_current()
        if census != self._census:
            raise GrammarError('Hessian reference census differs from the supplied campaign census')

    def binding(self):
        self.require_current()
        return dict(schema=BINDING_SCHEMA,
                    canonical_capture_sha256=self._document['canonical_capture']['sha256'],
                    census_sha256=self._document['census']['sha256'])

    def committed_units(self):
        self.require_current()
        return {name:item['sha256'] for name,item in self._document['hessians'].items()}

    def require_provenance(self, provenance):
        self.require_current()
        if {k:v for k,v in provenance.items() if k != 'path'} != self._document['provenance']:
            raise GrammarError('Hessian reference provenance moved from its canonical commitment')

    def __iter__(self):
        self.require_current()
        return iter(self._document['hessians'])

    def __len__(self):
        self.require_current()
        return len(self._document['hessians'])

    def __contains__(self, name):
        self.require_current()
        return name in self._document['hessians']

    def __getitem__(self, name):
        import torch
        from .cached_unit import tensor_identity
        with self._lock:
            self.require_current()
            expected = self._document['hessians'][name]
            entry = self._canonical['entries'][name]
            path = Path(self._document['canonical_capture']['path']).parent/entry['path']
            held = _HeldFile(path, cap=self._policy['max_file_bytes'])
            payload = H = X = tensor = None
            try:
                held.read(expected=entry['sha256'])
                self._live_payloads = 1
                # Torch maps the authenticated held object, never a second pathname.
                payload = torch.load(f'/proc/self/fd/{held.fd}', map_location='cpu',
                                     weights_only=True, mmap=True)
                if not isinstance(payload, dict) or set(payload) != {
                        'inputs', 'hessian', 'name', 'source', 'count', 'max_abs'}:
                    raise GrammarError(f'{name}: canonical H source has unexpected owners')
                H, X = payload['hessian'], payload['inputs']
                count = self._document['counts'][name]
                columns = expected['shape'][0]
                rows = min(count, self._canonical['identity']['max_act_rows'])
                if (payload['name'] != name or payload['source'] != CANONICAL_SOURCE or
                        type(payload['count']) is not int or payload['count'] != count or
                        payload['max_abs'] != self._census['max_abs'][name] or
                        not math.isfinite(float(payload['max_abs']))):
                    raise GrammarError(f'{name}: canonical H source metadata differs from census')
                for tensor, shape in ((H, [columns,columns]), (X,[rows,columns])):
                    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
                            list(tensor.shape) != shape or not tensor.is_contiguous() or
                            tensor.storage_offset()*4+tensor.numel()*4 > tensor.untyped_storage().nbytes()):
                        raise GrammarError(f'{name}: canonical H source geometry or precision changed')
                storages = {t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in (H,X)}
                if sum(storages.values()) > 4*(columns**2+rows*columns):
                    raise GrammarError(f'{name}: canonical H source storage exceeds its geometry')
                # Detach the returned H from the source mapping before page release.
                result = H.clone()
                if tensor_identity(result) != expected:
                    raise GrammarError(f'{name}: consumed Hessian tensor differs from its commitment')
                low, high = torch.aminmax(result)
                if not (math.isfinite(float(low)) and math.isfinite(float(high))):
                    raise GrammarError(f'{name}: consumed Hessian is nonfinite')
                held.check()
                self.require_current()
                self._verified.add(name)
                self._reads += 1
                self._read_bytes += held.stat.st_size
                self._peak_file = max(self._peak_file, held.stat.st_size)
                self._peak_h = max(self._peak_h, result.numel()*result.element_size())
                return result
            finally:
                payload = H = X = tensor = None
                self._live_payloads = 0
                os.posix_fadvise(held.fd, 0, 0, os.POSIX_FADV_DONTNEED)
                held.close()

    def receipt(self):
        return dict(schema='tessera.hessian_reference_consumption.v1',
            committed_units=len(self._document['hessians']), verified_units=sorted(self._verified),
            loaded_entries=self._reads, source_read_bytes=self._read_bytes,
            peak_file_bytes=self._peak_file, peak_hessian_bytes=self._peak_h,
            live_payloads=self._live_payloads, closed=self._closed)

    def close(self):
        self._closed = True
        for held in self._held:
            held.close()

    def __enter__(self):
        self.require_current()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
