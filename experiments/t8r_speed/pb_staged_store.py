"""Opt-in pinned inputs for the existing single-case kernel benchmark Store.

No origin fallback, cache, dispatcher, or tensor materialization framework.
PrismaBuild's public client SDK owns admitted staged ranges. The explicit
direct-vLLM mode uses the same sealed ranges through held original FDs;
it acquires no PB context, residency state or lease.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
import json
import os
import secrets
import stat
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
    def __init__(self, manifest_path, *, sdk=None, allow_unsealed_wires=False, direct_vllm=False):
        if sdk is None:
            from prismabuild import client as sdk
        self.sdk = sdk
        raw = Path(manifest_path).read_bytes()  # immutable checkout/CAS input
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest, encoding = sdk.read_data_manifest(manifest_path)
        if encoding != "identity" or manifest != json.loads(raw):
            raise ValueError("sealed replay manifest changed during validation")
        self.manifest = manifest
        self.direct_vllm = bool(direct_vllm)
        self.closed = False
        self.reads = []
        self.headers = {}
        self.roles = {}
        self.entries = {(e['path'], e['offset']): e for e in manifest['entries']}
        if len(self.entries) != manifest['entry_count']:
            raise ValueError('duplicate input ranges')
        for (path, offset), entry in self.entries.items():
            digest = entry.get('sha256')
            if digest is None and allow_unsealed_wires and not direct_vllm and offset > 0 and path.endswith('.safetensors'):
                continue
            if not isinstance(digest,str) or len(digest)!=64:
                raise ValueError('single replay requires every input digest')
        if direct_vllm:
            self.actual_digests = {identity:entry['sha256'] for identity,entry in self.entries.items()}
            self.direct_files = {}
            self.direct_record = {'transport':'direct-vllm-held-original-fds',
                'manifest_sha256':self.manifest_sha256,'before':{},'after':None}
            try:
                for path in sorted({path for path,_offset in self.entries}):
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                    self.direct_files[path] = {'fd':fd}
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError('direct input is not a regular file')
                    identity = self._file_identity(path,info)
                    self.direct_files[path]['identity'] = identity
                    self.direct_record['before'][path] = identity
                for (path,offset),entry in self.entries.items():
                    if offset + entry['bytes'] > self.direct_files[path]['identity']['size']:
                        raise ValueError('direct range exceeds held file')
            except BaseException:
                for item in self.direct_files.values():os.close(item['fd'])
                self.closed = True
                raise
            return
        bound = sdk.injected_context()
        if not bound.get('ok'):
            raise ValueError(f"pinned inputs refused: {bound.get('refusal')}")
        self.ctx = bound['ctx']
        self.queue = sdk.PoolQueue(Path(self.ctx['queue_root']))
        mapping = sdk.read_residency_map(self.ctx['map_path'])
        if mapping['manifest_sha256'] != self.manifest_sha256:
            raise ValueError('residency map does not bind this action/readset')
        self.keys = {(p, off): sdk.residency_map_key(p, off) for p, off in self.entries}
        epoch = mapping.get('epoch', '')
        root = self.queue.root / sdk.RESIDENCY
        # The composed map may lag the mover's final incremental fragment.
        # Public covers resolves dated material for the complete requested set;
        # it owns completeness and refuses unpublished/contradictory ranges.
        covers = sdk.covers_for_keys(root, self.ctx['action_key'], list(self.keys.values()),
                                    tier_id=mapping['tier_id'], epoch=epoch,
                                    manifest_sha256=self.manifest_sha256)
        if not covers.get('ok'):
            raise ValueError(f"pinned covers refused: {covers.get('refusal')}")
        expected = {}
        for identity,entry in self.entries.items():
            key = self.keys[identity]
            proof = covers['expected'].get(key)
            if proof is None or proof['bytes'] != entry['bytes']:
                raise ValueError('pinned covers do not prove every declared range')
            expected[key] = {'bytes':entry['bytes'],
                             'sha256':entry['sha256'] or proof['sha256']}
        self.actual_digests = {k:expected[key]['sha256'] for k,key in self.keys.items()}
        held = sdk.acquire_for(self.ctx, tier_id=mapping['tier_id'], epoch=epoch,
                               covers=covers['covers'], expected=expected,
                               span={'start_bytes': 0, 'end_bytes': manifest['total_bytes']},
                               acquire_token=secrets.token_hex(16))
        if not held.get('ok'):
            raise ValueError(f"pinned inputs refused: {held.get('refusal')}")
        self.held = held

    @staticmethod
    def _file_identity(path, info):
        return {'path':path,'device':info.st_dev,'inode':info.st_ino,'size':info.st_size,
                'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns}

    def _direct_identity(self, path):
        item = self.direct_files[path]
        identity = self._file_identity(path,os.fstat(item['fd']))
        if identity != item['identity']:
            raise ValueError('direct input file identity changed: '+path)
        return identity

    def read(self, path, offset=0):
        if self.closed:
            raise ValueError('pinned inputs already released')
        identity = (str(path), int(offset))
        entry = self.entries.get(identity)
        if entry is None:
            raise ValueError(f'undeclared input range: {identity}')
        if self.direct_vllm:
            self._direct_identity(identity[0])
            fd = self.direct_files[identity[0]]['fd']
            serving = {'transport':'direct-vllm-held-original-fd'}
        else:
            key = self.keys[identity]
            fd, serving = self.sdk.open_pinned(self.queue, self.held['pin'],
                                             self.held['ref_id'], key)
        try:
            # A staged range starts at byte zero. No source-path reread follows
            # authentication: this owned bytearray is exactly what the consumer gets.
            data = bytearray()
            while len(data) < entry['bytes']:
                length = min(1 << 20, entry['bytes'] - len(data))
                chunk = (os.pread(fd,length,identity[1]+len(data)) if self.direct_vllm
                         else os.read(fd,length))
                if not chunk:
                    raise ValueError('short pinned range')
                data.extend(chunk)
            if not self.direct_vllm and os.read(fd, 1):
                raise ValueError('oversized pinned range')
            if self.direct_vllm:self._direct_identity(identity[0])
            digest = hashlib.sha256(data).hexdigest()
            if digest != self.actual_digests[identity]:
                raise ValueError(f'pinned range digest differs: {identity}')
            self.reads.append({'path': identity[0], 'offset': identity[1],
                               'bytes': len(data), 'sha256': digest, 'serving_tier': serving})
            return data
        finally:
            if not self.direct_vllm:os.close(fd)

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

    def native_artifact(self, path):
        """Experimental trusted code artifact, held through load/profile/fence.

        A public pinned FD and pre/post hashes do not establish the immutable
        original tensor-provider contract for mutable native-code file bytes.
        """
        identity = (str(path),0)
        entry = self.entries.get(identity)
        if self.closed or entry is None or not str(path).endswith('.so'):
            raise ValueError('undeclared native code artifact')
        if self.direct_vllm:
            self._direct_identity(str(path))
            return os.dup(self.direct_files[str(path)]['fd']),dict(entry),{'transport':'direct-vllm-held-original-fd'}
        fd,serving = self.sdk.open_pinned(self.queue,self.held['pin'],
                                        self.held['ref_id'],self.keys[identity])
        return fd,dict(entry),serving

    def close(self):
        if not self.closed:
            if self.direct_vllm:
                try:
                    self.direct_record['after'] = {path:self._direct_identity(path)
                                                   for path in self.direct_files}
                finally:
                    for item in self.direct_files.values():os.close(item['fd'])
                    self.closed = True
                return
            result = self.sdk.release(self.queue, self.held['pin_id'], self.held['ref_id'],
                                      consumer_action_key=self.ctx['action_key'],
                                      stage_root=self.held['pin']['stage_root'])
            if result is not True:
                raise ValueError(f'pinned input release failed: {result}')
            self.closed = True


class NativeCallback:
    """One diagnostic code artifact through the existing build callback.

    Model-wire admission stays in StagedInputs.wire. This experimental code
    path trusts the native artifact owner; it makes no immutable-provider claim.
    """
    MODULE = 'tessera_routed_fused_mma_e4m3'

    def __init__(self, reader, path, rf, out, *, expected_sha256, source_sha256,
                 module=None, source_module=None, install_build_callback=True):
        self.MODULE = self.MODULE if module is None else module
        self.source_module = self.MODULE if source_module is None else source_module
        if (self.MODULE, self.source_module) not in {
            ("tessera_routed_fused_mma_e4m3", "tessera_routed_fused_mma_e4m3"),
            ("tessera_routed_fused_value", "tessera_routed_fused_value"),
            ("tessera_routed_fused_value_prefetch4", "tessera_routed_fused_value"),
            ("tessera_routed_fused_e4m3", "tessera_routed_fused_e4m3"),
            ("tessera_routed_fused_e2m1", "tessera_routed_fused_value"),
            ("tessera_routed_fused_e2m1_apf4", "tessera_routed_fused_value"),
            ("tessera_routed_fused_e2m1_dev_apf0", "tessera_routed_fused_value"),
            ("tessera_routed_fused_e2m1_dev_apf4", "tessera_routed_fused_value"),
        }:
            raise ValueError("unqualified retained native family")
        self.reader,self.rf,self.out = reader,rf,Path(out)
        self.original = rf.build_library
        self.install_build_callback = install_build_callback
        self.fd,self.entry,self.serving = reader.native_artifact(path)
        self.module = None
        self.closed = False
        info=os.fstat(self.fd)
        self.file_id=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        self.record = {'code_artifact_contract':'experimental trusted binary owner; not immutable original tensor provider',
                       'declared_path':str(path),'expected_sha256':expected_sha256,
                       'serving_tier':self.serving,'source_sha256':source_sha256}
        if reader.direct_vllm:
            self.record['input_transport'] = 'direct-vllm-held-original-fd'
        else:
            self.record.update(pin_id=reader.held['pin_id'],ref_id=reader.held['ref_id'])
        self.source_sha256 = source_sha256
        try:
            cache_info = getattr(rf._ext, "cache_info", None)
            cached = cache_info().currsize if cache_info else getattr(rf, "_LIB", None) is not None
            if self.MODULE in sys.modules or cached:
                raise ValueError('foreign native module already loaded')
            self.record['before_load_sha256'] = self._hash()
            if self.entry['sha256'] != expected_sha256 or self.record['before_load_sha256']!=expected_sha256:
                raise ValueError('native code artifact digest differs')
            # Retain exact bytes before the first profile; no source-path copy.
            self.out.mkdir(parents=True,exist_ok=True)
            target=self.out/(self.MODULE+'.so')
            with target.open('xb') as stream:
                offset=0
                while offset<self.entry['bytes']:
                    chunk=os.pread(self.fd,min(1<<20,self.entry['bytes']-offset),offset)
                    if not chunk: raise ValueError('short native code artifact')
                    stream.write(chunk);offset+=len(chunk)
                stream.flush();os.fsync(stream.fileno())
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected_sha256:
                raise ValueError('retained native code artifact differs')
            self.record['retained_path']=str(target)
            if self.install_build_callback:
                rf.build_library=self._build
        except BaseException:
            os.close(self.fd);self.closed=True
            raise

    def _hash(self):
        info=os.fstat(self.fd)
        if (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)!=self.file_id:
            raise ValueError('native code artifact file identity changed')
        if info.st_size!=self.entry['bytes']:
            raise ValueError('native code artifact length differs')
        digest=hashlib.sha256();offset=0
        while offset<info.st_size:
            data=os.pread(self.fd,min(1<<20,info.st_size-offset),offset)
            if not data: raise ValueError('short native code artifact')
            digest.update(data);offset+=len(data)
        return digest.hexdigest()

    def load_declared(self, src):
        """Map one held code artifact; direct diagnostics do not attest serving admission."""
        if self.closed:
            raise ValueError('native artifact already released')
        if hashlib.sha256(Path(src).read_bytes()).hexdigest()!=self.source_sha256:
            raise ValueError('native artifact source owner differs')
        if self.module is not None or self.MODULE in sys.modules:
            raise ValueError('native module already loaded')
        if self._hash()!=self.record['expected_sha256']:
            raise ValueError('native artifact changed before load')
        origin=f'/proc/self/fd/{self.fd}'
        loader=importlib.machinery.ExtensionFileLoader(self.MODULE,origin)
        spec=importlib.util.spec_from_file_location(self.MODULE,origin,loader=loader)
        self.module=importlib.util.module_from_spec(spec)
        sys.modules[self.MODULE]=self.module
        loader.exec_module(self.module)
        self.bind(self.module)
        self.record['module_origin']=origin
        self.record['stage_fd_target']=os.readlink(origin)
        self.record['after_load_sha256']=self._hash()
        self.record['through_build_owner']=self.install_build_callback
        if self.record['after_load_sha256']!=self.record['expected_sha256']:
            raise ValueError('native artifact changed during load')
        return self.module

    def _build(self, module, source_module, compile_fn):
        if not self.install_build_callback or (module,source_module)!=(self.MODULE,self.source_module):
            raise ValueError('foreign native library requested')
        def retained(src,build,token,verbose):
            return self.load_declared(src)
        # Existing serving owner still checks platform, lock and exported constants.
        return self.original(module,source_module,retained)

    def bind(self, module):
        origin=f'/proc/self/fd/{self.fd}'
        if (module is not self.module or sys.modules.get(self.MODULE) is not module or
            module.__name__!=self.MODULE or module.__file__!=origin or
            module.__spec__.origin!=origin):
            raise ValueError('actual native module origin/identity differs')

    def attest_mapped(self, module):
        """Diagnostic mapping proof using the held artifact FD, never path rereads."""
        self.bind(module)
        from tessera._dev.native_identity import mapped_file_device
        info = os.fstat(self.fd)
        device = mapped_file_device(self.fd)
        mapped_path = os.readlink(f"/proc/self/fd/{self.fd}")
        matches = []
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and fields[5] == mapped_path and "x" in fields[1]:
                major, minor = (int(v, 16) for v in fields[3].split(":"))
                if (major, minor, int(fields[4])) == (*device, info.st_ino):
                    matches.append(line)
        if not matches:
            raise ValueError("native module is not mapped from its held ELF inode")
        self.record["executable_mappings"] = matches
        self.record["mapped_sha256"] = self._hash()

    def finish(self, fence, *, keep_load_fd=False):
        if self.closed: return
        try:
            if self.module is not None:
                fence()
                self.bind(self.module)
                self.record['after_profile_sha256']=self._hash()
                if self.record['after_profile_sha256']!=self.record['expected_sha256']:
                    raise ValueError('native artifact changed during profile')
        finally:
            if self.install_build_callback:
                self.rf.build_library=self.original
            # Diagnostic multi-arm callers hold every loaded pathname alive
            # until teardown: CPython/dlopen can cache /proc/self/fd names.
            if not keep_load_fd or self.module is None:
                os.close(self.fd)
            self.closed=True
            if self.module is not None and sys.modules.get(self.MODULE) is self.module:
                del sys.modules[self.MODULE]
